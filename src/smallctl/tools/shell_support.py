from __future__ import annotations

import shlex
import re
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail
from .fs_sessions import _same_target_path, _write_session_can_finalize
from .fs_write_session_policy import _write_session_resume_metadata


_ARGPARSE_REQUIRED_ARGS_PATTERN = re.compile(
    r"(?:error:\s*)?the following arguments are required:\s*(.+)",
    re.IGNORECASE,
)


def _extract_missing_argparse_arguments(error_text: str) -> list[str]:
    match = _ARGPARSE_REQUIRED_ARGS_PATTERN.search(str(error_text or ""))
    if not match:
        return []

    missing = match.group(1).strip()
    if not missing:
        return []

    missing = missing.replace(" and ", ", ")
    values = [part.strip(" .`'\"") for part in missing.split(",")]
    return [value for value in values if value]


def _build_argparse_missing_args_question(command: str, missing_args: list[str]) -> str:
    missing_text = ", ".join(missing_args) if missing_args else "required arguments"
    return (
        f"The command `{command}` is missing required arguments: {missing_text}. "
        "What values should I use?"
    )


def _detect_unsupported_shell_syntax(command: str) -> str | None:
    if "<<<" in command:
        return (
            "Command uses Bash-only here-string redirection (`<<<`), but smallctl runs shell "
            "commands through /bin/sh on Unix. Rewrite it with POSIX syntax (for example, "
            "use `printf` piped into the command) or wrap the whole command in `bash -lc`."
        )
    return None


def _shell_workspace_relative_hint(command: str, cwd: str | None = None) -> str | None:
    raw_command = str(command or "")
    match = re.search(r"(?<![\w/])(/temp(?:/[^\s\"'`]+)*)", raw_command)
    if match is None:
        return None

    suspicious_path = match.group(1)
    trimmed = suspicious_path.lstrip("/")
    if not trimmed:
        return None

    base = Path(cwd) if cwd else Path.cwd()
    workspace_candidate = (base / Path(trimmed)).resolve()
    if not (workspace_candidate.exists() or workspace_candidate.parent.exists()):
        return None

    return (
        f"That command used the root-level `{suspicious_path}` path. "
        f"If you meant the workspace copy, retry with `{('./' + trimmed)}` instead."
    )


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

    next_section = str(getattr(session, "write_next_section", "") or "").strip()
    if (
        _write_session_can_finalize(session)
        and not next_section
        and bool(getattr(session, "write_sections_completed", []))
    ):
        next_required_tool = {
            "tool_name": "finalize_write_session",
            "required_fields": [],
            "required_arguments": {},
            "optional_fields": [],
            "notes": [
                "Promote the staged file to the target path before running shell verification on the target.",
            ],
        }
    else:
        next_required_tool = _write_session_resume_metadata(session, path=target_path)

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    return fail(
        f"Write Session `{session_id}` for `{target_path}` is still {status or 'open'} and staged-only. "
        f"The command targets the unpromoted target path. Resume the write session with `file_write` "
        f"or finalize it with `finalize_write_session` before running shell checks on `{target_path}`.",
        metadata={
            "command": command,
            "reason": "write_session_unpromoted_target_path",
            "write_session_id": session_id,
            "write_session_status": status or "open",
            "write_session_mode": str(getattr(session, "write_session_mode", "") or "").strip(),
            "target_path": target_path,
            "staging_path": staging_path,
            "next_section_name": next_section,
            "next_required_tool": next_required_tool,
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
    return fail(
        f"Shell command blocked: `{command}` would delete Write Session `{session_id}` recovery artifacts. "
        "Staging, original, and attempt snapshots are recovery state, not disposable scratch files. "
        "Finalize or resume the write session, or ask explicitly for cleanup once recovery is no longer needed.",
        metadata={
            "command": command,
            "reason": "write_session_artifact_delete_blocked",
            "error_kind": "write_session_artifact_delete_blocked",
            "write_session_id": session_id,
            "write_session_status": str(getattr(session, "status", "") or "").strip(),
            "target_path": target_path,
            "staging_path": staging_path,
            "matched_targets": matched_targets,
            "next_required_tool": {
                "tool_name": "finalize_write_session" if _write_session_can_finalize(session) else "file_write",
                "required_fields": [] if _write_session_can_finalize(session) else ["path", "content", "write_session_id", "section_name"],
                "required_arguments": {} if _write_session_can_finalize(session) else _write_session_resume_metadata(session, path=target_path).get("required_arguments", {}),
                "optional_fields": [] if _write_session_can_finalize(session) else ["next_section_name"],
                "notes": [
                    "Do not delete .smallctl/write_sessions artifacts while recovery may still need them.",
                    "If cleanup is intentional, ask the user for explicit approval after completion.",
                ],
            },
        },
    )


def _shell_execution_authoring_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    if plan is not None and not getattr(plan, "approved", False):
        return fail(
            "Shell execution is blocked until the spec contract is approved.",
            metadata={
                "command": command,
                "reason": "spec_not_approved",
                "plan_id": getattr(plan, "plan_id", ""),
            },
        )

    if state.contract_phase() == "author":
        if not state.files_changed_this_cycle:
            return fail(
                "Shell execution is blocked until the authoring contract has produced a target artifact.",
                metadata={
                    "command": command,
                    "reason": "authoring_target_missing",
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


def _split_shell_words(command: str) -> list[str]:
    try:
        return shlex.split(str(command or ""), posix=True)
    except ValueError:
        return str(command or "").split()


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


def _target_path_aliases(target_path: str, *, cwd: str | None = None) -> list[str]:
    aliases: set[str] = set()
    raw = str(target_path or "").strip()
    if not raw:
        return []

    aliases.add(raw)
    if raw.startswith("./"):
        aliases.add(raw[2:])
    elif not raw.startswith("/"):
        aliases.add(f"./{raw}")

    try:
        base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
        resolved = (Path(raw) if Path(raw).is_absolute() else (base / raw)).resolve()
        aliases.add(str(resolved))
        try:
            rel = resolved.relative_to(base)
        except Exception:
            rel = None
        if rel is not None:
            rel_str = str(rel)
            if rel_str:
                aliases.add(rel_str)
                aliases.add(f"./{rel_str}")
    except Exception:
        pass

    return [alias for alias in aliases if alias]


def _path_alias_mentioned(command: str, alias: str) -> bool:
    if not alias:
        return False
    pattern = rf"(?<![A-Za-z0-9_./-]){re.escape(alias)}(?![A-Za-z0-9_./-])"
    return bool(re.search(pattern, command))


def _token_path_candidates(token: str) -> list[str]:
    normalized = str(token or "").strip().strip("'\"`")
    if not normalized:
        return []

    while normalized.startswith("("):
        normalized = normalized[1:].strip()
    while normalized.endswith((";", "|", "&", ",", ")")):
        normalized = normalized[:-1].strip()
    if not normalized:
        return []

    candidates = [normalized]
    if "=" in normalized and not normalized.startswith("="):
        _, value = normalized.split("=", 1)
        value = value.strip().strip("'\"`")
        while value.endswith((";", "|", "&", ",", ")")):
            value = value[:-1].strip()
        if value:
            candidates.append(value)
    return candidates


def _shell_status_update_interval(timeout_sec: int) -> float:
    return max(1.0, min(max(1, timeout_sec) / 3.0, 10.0))


def _build_shell_status_update(command: str, *, elapsed_sec: float, timeout_sec: int) -> str:
    elapsed_text = f"{elapsed_sec:.0f}s"
    timeout_text = f"{max(1, timeout_sec)}s"
    return f"[still running after {elapsed_text} of {timeout_text}] {command}"
