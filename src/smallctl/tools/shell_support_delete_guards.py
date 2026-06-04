from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail
from .shell_path_utils import _is_within_path, _safe_resolve_path
from .shell_support_constants import (
    _DISPOSABLE_PATH_NAMES,
    _DISPOSABLE_PATH_SUFFIXES,
    _SHELL_CONTROL_TOKENS,
    _SOURCE_OR_TEST_DIR_NAMES,
    _SOURCE_OR_TEST_SUFFIXES,
)


def _extract_shell_delete_targets(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        return []
    try:
        parts = shlex.split(raw)
    except ValueError:
        return []
    targets: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if token in {"rm", "rmdir"}:
            index += 1
            while index < len(parts):
                current = parts[index]
                if current in _SHELL_CONTROL_TOKENS:
                    break
                if current.startswith("-"):
                    index += 1
                    continue
                targets.append(current)
                index += 1
            continue
        if token == "find":
            segment: list[str] = []
            index += 1
            while index < len(parts) and parts[index] not in _SHELL_CONTROL_TOKENS:
                segment.append(parts[index])
                index += 1
            if "-delete" in segment and segment:
                targets.append(segment[0])
            continue
        if token == "git" and index + 1 < len(parts) and parts[index + 1] == "clean":
            targets.append(".")
            index += 2
            continue
        index += 1
    return targets


def _is_disposable_delete_target(path: Path) -> bool:
    if path.name in _DISPOSABLE_PATH_NAMES:
        return True
    if path.suffix in _DISPOSABLE_PATH_SUFFIXES:
        return True
    return any(part in _DISPOSABLE_PATH_NAMES for part in path.parts)


def _looks_like_source_or_test_artifact(path: Path) -> bool:
    if path.suffix in _SOURCE_OR_TEST_SUFFIXES:
        return True
    return any(part in _SOURCE_OR_TEST_DIR_NAMES for part in path.parts)


def _protected_working_set_paths(state: LoopState) -> set[str]:
    protected: set[str] = set()
    cwd = str(getattr(state, "cwd", "") or Path.cwd())

    def add_path(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            protected.add(str(_safe_resolve_path(text, cwd=cwd)))

    challenge_progress = getattr(state, "challenge_progress", None)
    if challenge_progress is not None:
        for path in getattr(challenge_progress, "last_code_change_paths", []) or []:
            add_path(path)
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        for key in ("protected_working_set", "_protected_working_set"):
            raw = scratchpad.get(key)
            if isinstance(raw, dict):
                for path in raw.keys():
                    add_path(path)
            elif isinstance(raw, (list, tuple, set)):
                for path in raw:
                    add_path(path)
        for key in ("generated_paths", "_generated_paths", "_task_target_paths"):
            raw = scratchpad.get(key)
            if isinstance(raw, (list, tuple, set)):
                for path in raw:
                    add_path(path)
    return protected


def _target_contains_protected_path(target: Path, protected_paths: set[str]) -> bool:
    for protected in protected_paths:
        protected_path = Path(protected)
        if protected_path == target or _is_within_path(protected_path, target):
            return True
    return False


def _explicit_delete_requested(state: LoopState, target: Path) -> bool:
    task_text_parts = [
        str(getattr(getattr(state, "run_brief", None), "original_task", "") or ""),
        str(getattr(getattr(state, "working_memory", None), "current_goal", "") or ""),
    ]
    text = "\n".join(task_text_parts).lower()
    if not any(word in text for word in ("delete", "remove", "rm -rf", "clean up", "cleanup")):
        return False
    target_name = target.name.lower()
    target_text = str(target).lower()
    return target_name in text or target_text in text


def _shell_workspace_destructive_delete_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    targets = _extract_shell_delete_targets(command)
    if not targets:
        return None
    cwd = str(getattr(state, "cwd", "") or Path.cwd())
    workspace = _safe_resolve_path(cwd)
    protected_paths = _protected_working_set_paths(state)
    blocked: list[dict[str, Any]] = []
    allowed: list[str] = []
    for raw_target in targets:
        resolved = _safe_resolve_path(raw_target, cwd=cwd)
        if _is_disposable_delete_target(resolved):
            allowed.append(str(resolved))
            continue
        if not _is_within_path(resolved, workspace):
            continue
        if _explicit_delete_requested(state, resolved):
            allowed.append(str(resolved))
            continue
        reasons: list[str] = []
        if str(resolved) in protected_paths:
            reasons.append("protected_working_set")
        if _target_contains_protected_path(resolved, protected_paths):
            reasons.append("contains_protected_working_set_path")
        if _looks_like_source_or_test_artifact(resolved):
            reasons.append("source_or_test_artifact")
        # Unknown workspace directory deletes are a common destructive reset pattern.
        if not reasons and ("rm -r" in command or "rm -rf" in command or "rm -fr" in command or "rmdir" in command):
            reasons.append("unknown_workspace_delete")
        if reasons:
            blocked.append(
                {
                    "raw_target": raw_target,
                    "resolved_target": str(resolved),
                    "reasons": reasons,
                }
            )
    if not blocked:
        return None
    return fail(
        "Shell command blocked: destructive delete targets implementation artifacts. "
        "Deletion is not an acceptable repair/reset operation for code changes unless "
        "the user explicitly requested that exact deletion.",
        metadata={
            "reason": "workspace_destructive_delete_blocked",
            "command": command,
            "error_kind": "workspace_destructive_delete_blocked",
            "next_required_tool": {
                "tool_name": "file_read",
                "required_fields": ["path"],
                "notes": [
                    "Read the target and repair it with file_patch.",
                    "For generated code you own, use file_write with replace_strategy='overwrite'.",
                    "If deletion is intentional, ask_human with the exact path and reason.",
                ],
            },
            "blocked_targets": blocked,
            "allowed_targets": allowed,
        },
    )
