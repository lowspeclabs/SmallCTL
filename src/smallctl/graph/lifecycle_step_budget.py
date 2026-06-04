from __future__ import annotations

from typing import Any

from ..harness.messages import ConversationMessage


def _maybe_inject_step_budget_nudge(harness: Any, graph_state: Any) -> bool:
    """Inject a hard step-budget nudge for small models after >40 steps.

    Returns True if a nudge was injected.
    """
    model_name = str(
        getattr(harness.state, "scratchpad", {}).get("_model_name")
        or getattr(getattr(harness, "client", None), "model", "")
        or ""
    ).strip()
    from ..model_config import is_seven_b_or_under_model_name

    if not is_seven_b_or_under_model_name(model_name):
        return False
    if graph_state.final_result is not None:
        return False
    if getattr(harness.state, "write_session", None) is not None:
        return False
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Step budget exceeded (40+ steps). You must now call `task_complete` "
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
