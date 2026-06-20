from __future__ import annotations

from typing import Any

from ..context.retrieval import LexicalRetriever


def build_goal_recap(harness: Any) -> str:
    state = getattr(harness, "state", None)
    if state is None:
        return ""

    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)

    original_task = str(getattr(run_brief, "original_task", "") or "").strip()
    current_goal = LexicalRetriever._effective_current_goal(state)
    phase_focus = str(getattr(run_brief, "current_phase_objective", "") or "").strip()

    parts: list[str] = []
    if original_task:
        parts.append(f"Original task: {original_task}")
    if current_goal and current_goal != original_task:
        parts.append(f"Current goal: {current_goal}")
    if phase_focus and phase_focus not in {original_task, current_goal}:
        parts.append(f"Phase focus: {phase_focus}")

    transaction = _get_task_transaction(state)
    if transaction is not None:
        prev_goal = transaction.get("previous_goal", "")
        if prev_goal and prev_goal != current_goal:
            parts.append(f"Previous goal was: {prev_goal}")
        turn_type = transaction.get("turn_type", "")
        if turn_type:
            parts.append(f"Transition type: {turn_type}")

    if not parts:
        return ""
    return "Goal recap: " + " | ".join(parts)


def build_goal_recap_from_state(state: Any) -> str:
    if state is None:
        return ""
    run_brief = getattr(state, "run_brief", None)
    if run_brief is None:
        return ""

    original_task = str(getattr(run_brief, "original_task", "") or "").strip()
    current_goal = LexicalRetriever._effective_current_goal(state)
    phase_focus = str(getattr(run_brief, "current_phase_objective", "") or "").strip()

    parts: list[str] = []
    if original_task:
        parts.append(f"Original task: {original_task}")
    if current_goal and current_goal != original_task:
        parts.append(f"Current goal: {current_goal}")
    if phase_focus and phase_focus not in {original_task, current_goal}:
        parts.append(f"Phase focus: {phase_focus}")

    transaction = _get_task_transaction(state)
    if transaction is not None:
        prev_goal = transaction.get("previous_goal", "")
        if prev_goal and prev_goal != current_goal:
            parts.append(f"Previous goal was: {prev_goal}")
        turn_type = transaction.get("turn_type", "")
        if turn_type:
            parts.append(f"Transition type: {turn_type}")

    if not parts:
        return ""
    return "Goal recap: " + " | ".join(parts)


def _get_task_transaction(state: Any) -> dict | None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    transaction = scratchpad.get("_task_transaction")
    if isinstance(transaction, dict):
        return transaction
    handoff = scratchpad.get("_last_task_handoff")
    if isinstance(handoff, dict) and str(handoff.get("status") or "").strip().lower() in {
        "closed", "failed", "aborted", "superseded",
    }:
        return handoff
    return None
