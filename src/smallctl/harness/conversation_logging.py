from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage


def log_conversation_state(harness: Any, event: str) -> None:
    if harness.run_logger is None:
        return
    harness.run_logger.log(
        "chat",
        "conversation_history",
        "conversation snapshot",
        conversation_id=harness.conversation_id,
        snapshot_event=event,
        step=harness.state.step_count,
        history=[m.to_dict() for m in harness.state.conversation_history],
        recent_messages=[m.to_dict() for m in harness.state.recent_messages],
        prompt_budget=harness.state.prompt_budget.__dict__,
    )


def record_assistant_message(
    harness: Any,
    *,
    assistant_text: str,
    tool_calls: list[dict[str, Any]],
    speaker: str | None = None,
    hidden_from_prompt: bool = False,
) -> None:
    metadata: dict[str, Any] = {}
    normalized_speaker = str(speaker or "").strip().lower()
    if normalized_speaker:
        metadata["speaker"] = normalized_speaker
    if hidden_from_prompt:
        metadata["hidden_from_prompt"] = True
    message = ConversationMessage(
        role="assistant",
        content=assistant_text or None,
        tool_calls=tool_calls,
        metadata=metadata,
    )
    harness.state.append_message(message)
    if harness.state.plan_execution_mode and harness.state.active_step_id:
        harness.state.step_sandbox_history.append(message)
    if hidden_from_prompt or not assistant_text:
        return
    refresh_options = getattr(harness, "_refresh_task_handoff_action_options", None)
    if callable(refresh_options):
        refresh_options(assistant_text)
