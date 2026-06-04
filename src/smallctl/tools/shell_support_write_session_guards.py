from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from ..state import LoopState
from .fs_sessions import _same_target_path, _write_session_can_finalize
from .fs_write_session_policy import _write_session_resume_metadata
from .shell_parsing import _split_shell_words
from .shell_path_utils import (
    _path_alias_mentioned,
    _target_path_aliases,
    _token_path_candidates,
)


def _guard_fail(
    message: str,
    *,
    reason: str,
    command: str,
    error_kind: str | None = None,
    next_required_tool: dict[str, Any] | None = None,
    next_required_action: dict[str, Any] | str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent guard failure result."""
    from .common import fail
    metadata: dict[str, Any] = {
        "reason": reason,
        "command": command,
    }
    if error_kind is not None:
        metadata["error_kind"] = error_kind
    if next_required_tool is not None:
        metadata["next_required_tool"] = next_required_tool
    if next_required_action is not None:
        metadata["next_required_action"] = next_required_action
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return fail(message, metadata=metadata)


def _shell_write_session_target_path_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    session = getattr(state, "write_session", None)
    if session is None:
        return None

    status = str(getattr(session, "status", "") or "").strip().lower()
    if status == "complete":
        return None

    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    if not target_path:
        return None

    has_unpromoted_changes = bool(
        getattr(session, "write_sections_completed", [])
        or str(getattr(session, "write_last_staged_hash", "") or "").strip()
        or bool(getattr(session, "write_pending_finalize", False))
    )
    if not has_unpromoted_changes:
        return None

    if not _command_targets_path(command, target_path=target_path, cwd=state.cwd):
        return None

    from ..harness.tool_visibility import _has_finalizable_write_session
    can_finalize = _has_finalizable_write_session(state)
    if can_finalize:
        next_required_tool = {
            "tool_name": "finalize_write_session",
            "required_fields": [],
            "required_arguments": {},
            "optional_fields": [],
            "notes": [
                "Promote the staged file to the target path before running shell verification on the target.",
            ],
        }
        finalize_hint = "or finalize it with `finalize_write_session` "
    else:
        next_required_tool = _write_session_resume_metadata(session, path=target_path)
        finalize_hint = ""

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    return _guard_fail(
        f"Write Session `{session_id}` for `{target_path}` is still {status or 'open'} and staged-only. "
        f"The command targets the unpromoted target path. Resume the write session with `file_write` "
        f"{finalize_hint}before running shell checks on `{target_path}`.",
        reason="write_session_unpromoted_target_path",
        command=command,
        next_required_tool=next_required_tool,
        extra_metadata={
            "write_session_id": session_id,
            "write_session_status": status or "open",
            "write_session_mode": str(getattr(session, "write_session_mode", "") or "").strip(),
            "target_path": target_path,
            "staging_path": staging_path,
            "next_section_name": str(getattr(session, "write_next_section", "") or "").strip(),
        },
    )


def _shell_write_session_artifact_delete_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    session = getattr(state, "write_session", None)
    if session is None:
        return None

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if not session_id:
        return None

    destructive_targets = _write_session_delete_targets(command)
    if not destructive_targets:
        return None

    cwd = str(getattr(state, "cwd", "") or "")
    protected_paths = _protected_write_session_paths(session, cwd=cwd)
    matched_targets = [
        target
        for target in destructive_targets
        if _targets_write_session_artifact(
            target,
            session_id=session_id,
            protected_paths=protected_paths,
            cwd=cwd,
        )
    ]
    if not matched_targets:
        return None

    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    return _guard_fail(
        f"Shell command blocked: `{command}` would delete Write Session `{session_id}` recovery artifacts. "
        "Staging, original, and attempt snapshots are recovery state, not disposable scratch files. "
        "Finalize or resume the write session, or ask explicitly for cleanup once recovery is no longer needed.",
        reason="write_session_artifact_delete_blocked",
        command=command,
        error_kind="write_session_artifact_delete_blocked",
        next_required_tool={
            "tool_name": "finalize_write_session" if _write_session_can_finalize(session) else "file_write",
            "required_fields": [] if _write_session_can_finalize(session) else ["path", "content", "write_session_id", "section_name"],
            "required_arguments": {} if _write_session_can_finalize(session) else _write_session_resume_metadata(session, path=target_path).get("required_arguments", {}),
            "optional_fields": [] if _write_session_can_finalize(session) else ["next_section_name"],
            "notes": [
                "Do not delete .smallctl/write_sessions artifacts while recovery may still need them.",
                "If cleanup is intentional, ask the user for explicit approval after completion.",
            ],
        },
        extra_metadata={
            "write_session_id": session_id,
            "write_session_status": str(getattr(session, "status", "") or "").strip(),
            "target_path": target_path,
            "staging_path": staging_path,
            "matched_targets": matched_targets,
        },
    )


def _shell_execution_authoring_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    if plan is not None and not getattr(plan, "approved", False):
        return _guard_fail(
            "Shell execution is blocked until the spec contract is approved.",
            reason="spec_not_approved",
            command=command,
            extra_metadata={
                "plan_id": getattr(plan, "plan_id", ""),
            },
        )

    if state.contract_phase() == "author":
        if not state.files_changed_this_cycle:
            return _guard_fail(
                "Shell execution is blocked until the authoring contract has produced a target artifact.",
                reason="authoring_target_missing",
                command=command,
                extra_metadata={
                    "contract_phase": state.contract_phase(),
                    "files_changed_this_cycle": state.files_changed_this_cycle,
                },
            )
    return None


def _write_session_delete_targets(command: str) -> list[str]:
    commands = [str(command or "")]
    tokens = _split_shell_words(command)
    if len(tokens) >= 3 and tokens[0] in {"bash", "sh", "/bin/bash", "/bin/sh"} and tokens[1] in {"-c", "-lc"}:
        commands.append(tokens[2])

    targets: list[str] = []
    for raw_command in commands:
        words = _split_shell_words(raw_command)
        for index, word in enumerate(words):
            if word not in {"rm", "/bin/rm"}:
                continue
            for candidate in words[index + 1 :]:
                if candidate in {"&&", "||", ";", "|"}:
                    break
                if candidate.startswith("-"):
                    continue
                for path_candidate in _token_path_candidates(candidate):
                    targets.append(path_candidate)
    return targets


def _protected_write_session_paths(session: Any, *, cwd: str | None = None) -> set[str]:
    protected: set[str] = set()
    for raw_path in (
        getattr(session, "write_staging_path", ""),
        getattr(session, "write_original_snapshot_path", ""),
        getattr(session, "write_last_attempt_snapshot_path", ""),
    ):
        path_text = str(raw_path or "").strip()
        if not path_text:
            continue
        protected.add(path_text)
        try:
            base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
            resolved = (Path(path_text) if Path(path_text).is_absolute() else base / path_text).resolve()
            protected.add(str(resolved))
        except Exception:
            pass
    return protected


def _targets_write_session_artifact(
    target: str,
    *,
    session_id: str,
    protected_paths: set[str],
    cwd: str | None = None,
) -> bool:
    raw = str(target or "").strip().strip("'\"`")
    if not raw:
        return False

    normalized = raw.replace("\\", "/")
    if ".smallctl/write_sessions/" in normalized and session_id in normalized:
        return True
    if normalized.endswith(".smallctl/write_sessions") or normalized.endswith(".smallctl/write_sessions/"):
        return True

    if any("*" in part or "?" in part or "[" in part for part in (raw, normalized)):
        if ".smallctl/write_sessions/" in normalized and session_id in normalized:
            return True

    for protected in protected_paths:
        if not protected:
            continue
        if _same_target_path(raw, protected, cwd):
            return True
        if any(mark in raw for mark in ("*", "?", "[")):
            prefix = raw.split("*", 1)[0].split("?", 1)[0].split("[", 1)[0]
            if prefix and str(protected).startswith(prefix):
                return True
    return False


def _command_targets_path(command: str, *, target_path: str, cwd: str | None = None) -> bool:
    raw_command = str(command or "")
    target = str(target_path or "").strip()
    if not raw_command.strip() or not target:
        return False

    aliases = _target_path_aliases(target, cwd=cwd)
    if not aliases:
        return False

    for alias in aliases:
        if _path_alias_mentioned(raw_command, alias):
            return True

    tokens: list[str]
    try:
        tokens = shlex.split(raw_command, posix=True)
    except ValueError:
        tokens = raw_command.split()

    for token in tokens:
        for candidate in _token_path_candidates(token):
            if any(candidate == alias for alias in aliases):
                return True
            if _same_target_path(candidate, target, cwd):
                return True
    return False
