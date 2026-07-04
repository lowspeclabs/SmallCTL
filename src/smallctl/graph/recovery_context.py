from __future__ import annotations

from typing import Any

from ..context.retrieval import LexicalRetriever


def build_goal_recap(harness: Any) -> str:
    state = getattr(harness, "state", None)
    if state is None:
        return ""

    run_brief = getattr(state, "run_brief", None)

    original_task = str(getattr(run_brief, "original_task", "") or "").strip()
    current_goal = LexicalRetriever._effective_current_goal(state)
    phase_focus = str(getattr(run_brief, "current_phase_objective", "") or "").strip()

    parts: list[str] = []
    if original_task:
        parts.append(f"Original task: {original_task}")
    if current_goal and current_goal != original_task:
        parts.append(f"Current goal: {current_goal}")
    if phase_focus and not _is_redundant_focus(phase_focus, original_task, current_goal):
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


def build_concise_goal_hint(harness: Any, *, max_chars: int = 200) -> str:
    """Return a very short goal hint for recovery nudges.

    Long recovery messages that repeat the full task objective can confuse
    reasoning-heavy models (especially Gemma-4 served by llama.cpp), causing
    them to recap the plan instead of emitting the next tool call.  This
    helper returns only a one-line summary so the nudge stays actionable.
    """
    state = getattr(harness, "state", None)
    if state is None:
        return ""
    run_brief = getattr(state, "run_brief", None)
    if run_brief is None:
        return ""

    original_task = str(getattr(run_brief, "original_task", "") or "").strip()
    if not original_task:
        return ""

    # Keep just the first sentence / line, trimmed to a small budget.
    one_liner = original_task.split("\n")[0].strip()
    if len(one_liner) > max_chars:
        one_liner = one_liner[: max_chars - 1].rsplit(" ", 1)[0] + "..."
    return f"Task: {one_liner}"


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
    if phase_focus and not _is_redundant_focus(phase_focus, original_task, current_goal):
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


def _is_redundant_focus(focus: str, original_task: str, current_goal: str) -> bool:
    """Return True when *focus* just prefixes or restates an already-listed goal."""
    if not focus:
        return True
    focus_norm = " ".join(focus.split()).lower()
    for other in (original_task, current_goal):
        if not other:
            continue
        other_norm = " ".join(other.split()).lower()
        if focus_norm == other_norm:
            return True
        # Phase focus is often "<phase>: <original task>".
        if focus_norm.endswith(other_norm) or other_norm.endswith(focus_norm):
            return True
        # Also treat a phase-prefixed focus as redundant if the task follows the colon.
        if ":" in focus_norm:
            after_colon = focus_norm.split(":", 1)[1].strip()
            if after_colon and (after_colon == other_norm or other_norm.endswith(after_colon)):
                return True
    return False


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
