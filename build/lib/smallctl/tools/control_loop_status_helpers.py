from __future__ import annotations

from typing import Any

from ..write_session_fsm import recent_write_session_events
from .fs_write_sessions import write_session_contract


def max_steps_progress(state: Any) -> tuple[int, float]:
    max_steps = getattr(state, "scratchpad", {}).get("_max_steps")
    try:
        max_steps_int = int(max_steps) if max_steps is not None else 0
    except (TypeError, ValueError):
        max_steps_int = 0

    if max_steps_int > 0:
        progress_pct = min(1.0, max(0.0, getattr(state, "step_count", 0) / max_steps_int))
    else:
        progress_pct = 0.0
    return max_steps_int, progress_pct


def write_session_status_payload(
    state: Any,
    *,
    schema_failure: dict[str, Any] | None,
    resume_action: dict[str, Any] | None,
) -> dict[str, Any] | None:
    session = getattr(state, "write_session", None)
    payload = session.to_dict() if session else None
    if payload is not None and schema_failure is not None:
        payload = dict(payload)
        payload["last_schema_failure"] = schema_failure
    if payload is not None and resume_action is not None:
        payload = dict(payload)
        payload["resume_action"] = resume_action
    if payload is not None and session is not None:
        payload = dict(payload)
        payload["contract"] = write_session_contract(session)
    return payload


def write_session_status_events(state: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    return recent_write_session_events(state, limit=limit)


def subtask_ledger_status(state: Any) -> dict[str, Any] | None:
    ledger = getattr(state, "subtask_ledger", None)
    if ledger is None:
        return None
    subtasks = []
    for task in getattr(ledger, "subtasks", []) or []:
        subtasks.append(
            {
                "subtask_id": str(getattr(task, "subtask_id", "") or ""),
                "title": str(getattr(task, "title", "") or ""),
                "goal": str(getattr(task, "goal", "") or ""),
                "status": str(getattr(task, "status", "") or ""),
                "acceptance": list(getattr(task, "acceptance", []) or []),
                "evidence": list(getattr(task, "evidence", []) or [])[-3:],
                "blockers": list(getattr(task, "blockers", []) or [])[-3:],
                "next_action": getattr(task, "next_action", None),
                "attempts": int(getattr(task, "attempts", 0) or 0),
            }
        )
    return {
        "task_id": getattr(ledger, "task_id", None),
        "active_subtask_id": getattr(ledger, "active_subtask_id", None),
        "active_subtask": next(
            (item for item in subtasks if item["subtask_id"] == getattr(ledger, "active_subtask_id", None)),
            None,
        ),
        "subtasks": subtasks[-12:],
        "done_subtask_ids": [item["subtask_id"] for item in subtasks if item["status"] == "done"],
        "pending_subtask_ids": [
            item["subtask_id"] for item in subtasks if item["status"] in {"pending", "active", "blocked"}
        ],
    }
