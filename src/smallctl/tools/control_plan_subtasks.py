from __future__ import annotations

from typing import Any

from ..state import LoopState


def open_plan_subtasks(state: LoopState) -> list[dict[str, Any]]:
    plan = state.active_plan or state.draft_plan
    ledger = getattr(state, "subtask_ledger", None)
    if plan is None or ledger is None:
        return []
    step_ids = {
        str(getattr(step, "step_id", "") or "").strip()
        for step in plan.iter_steps()
        if str(getattr(step, "step_id", "") or "").strip()
    }
    if not step_ids:
        return []
    open_items: list[dict[str, Any]] = []
    for task in getattr(ledger, "subtasks", []) or []:
        subtask_id = str(getattr(task, "subtask_id", "") or "").strip()
        if subtask_id not in step_ids:
            continue
        status = str(getattr(task, "status", "") or "").strip().lower()
        if status in {"done", "abandoned"}:
            continue
        open_items.append(
            {
                "subtask_id": subtask_id,
                "title": str(getattr(task, "title", "") or ""),
                "goal": str(getattr(task, "goal", "") or ""),
                "status": status or "pending",
                "acceptance": list(getattr(task, "acceptance", []) or []),
                "evidence": list(getattr(task, "evidence", []) or [])[-3:],
                "blockers": list(getattr(task, "blockers", []) or [])[-3:],
                "next_action": getattr(task, "next_action", None),
                "attempts": int(getattr(task, "attempts", 0) or 0),
            }
        )
    return open_items


def plan_subtask_completion_block(
    state: LoopState,
    *,
    verifier_verdict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    open_items = open_plan_subtasks(state)
    if not open_items:
        return None
    first = open_items[0]
    status = str(first.get("status") or "").lower()
    blocked = status in {"blocked", "failed"}
    if blocked:
        next_required_action = {
            "tool_names": ["escalate_to_bigger_model", "ask_human", "task_fail"],
            "notes": [
                "Escalate to a bigger model if stronger debugging or planning is needed.",
                "Ask the human if progress requires missing information, approval, credentials, or an ambiguous choice.",
                "Fail the task only if the blocker is terminal.",
            ],
        }
    else:
        next_required_action = {
            "tool_names": ["loop_status"],
            "notes": [
                "Continue the active plan subtask with concrete tool evidence.",
                "Do not call task_complete until all plan subtasks are done or explicitly abandoned/failed.",
            ],
        }
    return {
        "open_plan_subtasks": open_items,
        "next_required_subtask": first,
        "next_required_action": next_required_action,
        "last_verifier_verdict": verifier_verdict,
    }
