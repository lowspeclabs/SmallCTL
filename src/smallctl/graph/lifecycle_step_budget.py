from __future__ import annotations

from typing import Any

from ..guards import is_seven_b_or_under_model_name
from ..models.conversation import ConversationMessage


STEP_BUDGET_NUDGE_THRESHOLD = 40
_STEP_BUDGET_NUDGE_INJECTED_KEY = "_step_budget_nudge_injected"


def _maybe_inject_step_budget_nudge(harness: Any, graph_state: Any) -> bool:
    """Inject a hard step-budget nudge for small models after the threshold is exceeded.

    The nudge is latched per task/run: it fires at most once per task sequence
    and is re-armed only when a new task/run boundary rolls in.
    Returns True if a nudge was injected.
    """
    model_name = str(
        getattr(harness.state, "scratchpad", {}).get("_model_name")
        or getattr(getattr(harness, "client", None), "model", "")
        or ""
    ).strip()
    if not is_seven_b_or_under_model_name(model_name):
        return False
    if graph_state.final_result is not None:
        return False
    if getattr(harness.state, "write_session", None) is not None:
        return False
    scratchpad = getattr(harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        harness.state.scratchpad = scratchpad
    # Latch per task/run: fire at most once until a new task boundary bumps
    # `_task_sequence` (or the scratchpad is replaced), which re-arms the latch.
    task_sequence = scratchpad.get("_task_sequence") or 0
    latch = scratchpad.get(_STEP_BUDGET_NUDGE_INJECTED_KEY)
    if isinstance(latch, dict) and latch.get("task_sequence") == task_sequence:
        return False
    scratchpad[_STEP_BUDGET_NUDGE_INJECTED_KEY] = {
        "task_sequence": task_sequence,
        "step_count": int(getattr(harness.state, "step_count", 0) or 0),
    }
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Step budget exceeded (more than {STEP_BUDGET_NUDGE_THRESHOLD} steps). You must now call `task_complete` "
                "with your best synthesis of findings, or call `task_fail` if blocked. "
                "Do not issue any more discovery commands."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "step_budget_exceeded",
            },
        )
    )
    harness._runlog(
        "step_budget_exceeded",
        "injected hard step-budget nudge for small model",
        step=harness.state.step_count,
        model_name=model_name,
    )
    return True
