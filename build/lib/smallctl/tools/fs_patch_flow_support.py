from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..state import LoopState
from .fs_sessions import _same_target_path
from .fs_write_sessions import _content_hash, _read_text_file

_REPEAT_SENSITIVE_PATCHES_KEY = "_repeat_sensitive_file_patches"


def _repeat_sensitive_patch_signature(
    *,
    source_path: Path,
    target_text: str,
    replacement_text: str,
    expected_occurrences: int,
) -> str:
    return "|".join(
        [
            str(source_path),
            _content_hash(target_text),
            _content_hash(replacement_text),
            str(expected_occurrences),
        ]
    )


def _repeat_sensitive_patch_records(state: LoopState | None) -> dict[str, dict[str, Any]]:
    if state is None:
        return {}
    records = state.scratchpad.get(_REPEAT_SENSITIVE_PATCHES_KEY)
    if not isinstance(records, dict):
        records = {}
        state.scratchpad[_REPEAT_SENSITIVE_PATCHES_KEY] = records
    return records


def _verifier_traceback_focus(
    state: LoopState | None,
    *,
    source_path: Path,
    requested_path: str,
) -> dict[str, Any] | None:
    if state is None:
        return None
    verdict_fn = getattr(state, "current_verifier_verdict", None)
    verifier = verdict_fn() if callable(verdict_fn) else getattr(state, "last_verifier_verdict", None)
    if not isinstance(verifier, dict):
        return None
    text = "\n".join(
        str(verifier.get(key) or "")
        for key in ("key_stderr", "key_stdout")
        if str(verifier.get(key) or "").strip()
    )
    if not text:
        return None

    source_resolved = source_path.resolve()
    matches = re.findall(r'File "([^"]+)", line (\d+)', text)
    for filename, line_text in reversed(matches):
        try:
            traceback_path = Path(filename).resolve()
        except OSError:
            continue
        if traceback_path != source_resolved:
            continue
        line = int(line_text)
        start_line = max(1, line - 5)
        end_line = line + 5
        return {
            "traceback_path": str(traceback_path),
            "traceback_line": line,
            "next_required_tool": {
                "tool_name": "file_read",
                "required_arguments": {
                    "path": requested_path,
                    "start_line": start_line,
                    "end_line": end_line,
                },
                "reason": "live_verifier_traceback_requires_current_slice",
            },
            "recovery_hint": (
                "Do not retry the prior patch; the current verifier traceback names this line. "
                "Read that slice and patch the live failure."
            ),
        }
    return None


def _empty_target_patch_file_write_metadata(
    *,
    path: str,
    target: Path,
    cwd: str | None,
    encoding: str,
    state: LoopState | None,
    write_session_id: str | None,
    replacement_text: str,
) -> dict[str, Any] | None:
    if not replacement_text.strip():
        return None

    session = getattr(state, "write_session", None) if state is not None else None
    active_session_id = str(getattr(session, "write_session_id", "") or "").strip()
    session_status = str(getattr(session, "status", "") or "").strip().lower()
    has_active_session = session is not None and session_status not in {"complete", "finalized", "aborted"}
    if write_session_id and active_session_id and write_session_id != active_session_id:
        has_active_session = False

    session_matches_target = False
    staged_empty = False
    no_completed_sections = False
    if has_active_session:
        session_target = str(getattr(session, "write_target_path", "") or "").strip()
        try:
            session_matches_target = bool(session_target) and _same_target_path(session_target, path, cwd)
        except Exception:
            session_matches_target = False

        if session_matches_target:
            no_completed_sections = not bool(getattr(session, "write_sections_completed", []) or [])
            staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
            if staging_path:
                try:
                    staged_empty = _read_text_file(Path(staging_path), encoding=encoding) == ""
                except Exception:
                    staged_empty = False

    try:
        target_missing = not target.exists()
        target_empty = not target_missing and target.is_file() and _read_text_file(target, encoding=encoding) == ""
    except OSError:
        target_missing = False
        target_empty = False

    should_write = target_missing or target_empty or (session_matches_target and (staged_empty or no_completed_sections))
    if not should_write:
        return None

    required_arguments: dict[str, Any] = {
        "path": path,
        "content": replacement_text,
        "replace_strategy": "overwrite",
    }
    required_fields = ["path", "content"]
    if session_matches_target and active_session_id:
        required_arguments["section_name"] = (
            str(getattr(session, "write_next_section", "") or "").strip() or "initial_content"
        )
        required_fields.extend(["section_name", "replace_strategy"])

    return {
        "path": str(target),
        "requested_path": path,
        "error_kind": "patch_target_empty_for_new_file",
        "recovery_hint": (
            "This looks like initial file authoring, not an exact-text patch. "
            "Use `file_write` with `replace_strategy='overwrite'` for the first content."
        ),
        "suggested_tools": ["file_write"],
        "next_required_tool": {
            "tool_name": "file_write",
            "required_fields": required_fields,
            "required_arguments": required_arguments,
            "reason": "empty_or_new_file_requires_file_write_not_file_patch",
        },
    }
