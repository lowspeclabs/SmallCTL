from __future__ import annotations

from pathlib import Path
from typing import Any

_LOOPISH_TOOL_MODES = {"loop", "execute"}


def _tool_name(entry: dict[str, Any]) -> str:
    function = entry.get("function") if isinstance(entry, dict) else None
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _normalize_turn_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in _LOOPISH_TOOL_MODES:
        return "loop"
    if normalized in {"chat", "planning", "indexer"}:
        return normalized
    return "loop"


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for entry in tools:
        name = _tool_name(entry)
        if name:
            names.append(name)
    return names


def _schema_for_name(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {},
        },
    }


def _scratchpad(state: Any) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    return scratchpad


def _has_runtime_code_index(cwd: str | None) -> bool:
    return bool(cwd and (Path(cwd) / ".smallctl" / "code_index.json").exists())


def _has_finalizable_write_session(state: Any) -> bool:
    session = getattr(state, "write_session", None)
    if session is None:
        return False
    status = str(getattr(session, "status", "") or "").strip().lower()
    if status == "complete":
        return False
    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    if not target_path:
        return False
    has_unpromoted_changes = bool(
        getattr(session, "write_sections_completed", [])
        or str(getattr(session, "write_last_staged_hash", "") or "").strip()
        or bool(getattr(session, "write_pending_finalize", False))
    )
    if not has_unpromoted_changes:
        return False
    # A session is only finalizable when all sections are written
    # (no next section pending) or explicitly pending finalize.
    next_section = str(getattr(session, "write_next_section", "") or "").strip()
    pending_finalize = bool(getattr(session, "write_pending_finalize", False))
    return pending_finalize or not next_section


def _has_artifacts(state: Any) -> bool:
    artifacts = getattr(state, "artifacts", None)
    return isinstance(artifacts, dict) and bool(artifacts)


def _has_plan(state: Any) -> bool:
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    return plan is not None


def _has_background_jobs(state: Any) -> bool:
    jobs = getattr(state, "background_jobs", None)
    return isinstance(jobs, list) and len(jobs) > 0


def _has_planning_file_patch_context(state: Any) -> bool:
    from .task_boundary_support import parse_inline_task_wrapper
    active_plan = getattr(state, "active_plan", None)
    if active_plan is not None:
        tasks = getattr(active_plan, "tasks", None)
        if isinstance(tasks, list) and tasks:
            for task in tasks:
                wrapper = parse_inline_task_wrapper(task)
                if wrapper and wrapper.get("tool_name") in {"file_patch", "ast_patch"}:
                    return True
    draft_plan = getattr(state, "draft_plan", None)
    if draft_plan is not None:
        tasks = getattr(draft_plan, "tasks", None)
        if isinstance(tasks, list) and tasks:
            for task in tasks:
                wrapper = parse_inline_task_wrapper(task)
                if wrapper and wrapper.get("tool_name") in {"file_patch", "ast_patch"}:
                    return True
    # Recent file read or write provides enough context for patching
    records = getattr(state, "tool_execution_records", None)
    if isinstance(records, dict) and records:
        for record in reversed(list(records.values())):
            if not isinstance(record, dict):
                continue
            tool_name = str(record.get("tool_name") or "").strip()
            if tool_name in {"file_read", "file_write", "file_append"}:
                result = record.get("result")
                if isinstance(result, dict) and bool(result.get("success")):
                    return True
            if tool_name in {"file_patch", "ast_patch"}:
                # Stop at the first patch-related tool; if we reached here it wasn't successful enough to return above
                break
    return False
