from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType


async def _check_interactive_session_progress(
    harness: Any,
    graph_state: Any,
    event_handler: Any,
) -> str | None:
    """Check transparent interactive session state and inject nudges or return guard error.

    Detects:
    - Stalled session: AWAITING_INPUT for N turns with no input (Strategy B).
    - Zombie session: process exited but final output never read.
    - Orphan session: model started a new file/command without closing a prior session.
    - Runaway input: model keeps sending to an already-exited process.
    """
    state = getattr(harness, "state", None)
    if state is None:
        return None

    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None

    session = scratchpad.get("_transparent_interactive_session")
    if not isinstance(session, dict):
        return None

    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        return None

    status = str(session.get("status") or "").strip()
    awaiting = bool(session.get("awaiting_input"))
    send_count = int(session.get("send_count", 0) or 0)
    max_sends = int(session.get("max_sends", 12) or 12)

    # Runaway input: too many sends
    if send_count >= max_sends:
        # Clear the transparent session
        scratchpad.pop("_transparent_interactive_session", None)
        state.scratchpad = scratchpad
        state.append_message(
            ConversationMessage(
                role="user",
                content=(
                    "The interactive session has exceeded the maximum number of turns. "
                    "Please call task_complete with the current results, or describe what happened."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "interactive_session_max_turns",
                },
            )
        )
        return (
            "Interactive session guard: exceeded maximum sends. "
            "The model should call task_complete or task_fail."
        )

    # Stalled session: awaiting input but no progress
    if awaiting:
        turns_awaiting = int(session.get("turns_awaiting", 0) or 0) + 1
        session["turns_awaiting"] = turns_awaiting
        if turns_awaiting >= 3:
            scratchpad.pop("_transparent_interactive_session", None)
            state.scratchpad = scratchpad
            state.append_message(
                ConversationMessage(
                    role="user",
                    content=(
                        "The interactive program is waiting for input but none was provided. "
                        "Reply with just the value to send, or call task_complete if the task is done."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "interactive_session_stalled",
                    },
                )
            )
            return (
                "Interactive session guard: stalled awaiting input for multiple turns."
            )
    else:
        session["turns_awaiting"] = 0

    # Zombie session: exited but not cleaned up
    if status == "exited" and session.get("final_read_done") is not True:
        session["final_read_done"] = True
        scratchpad.pop("_transparent_interactive_session", None)
        state.scratchpad = scratchpad
        state.append_message(
            ConversationMessage(
                role="user",
                content=(
                    "The interactive program has exited. "
                    "Call task_complete with the results, or restart if needed."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "interactive_session_exited",
                },
            )
        )
        return None

    return None


async def _maybe_nudge_interactive_input(
    harness: Any,
    graph_state: Any,
) -> None:
    """If a transparent session is awaiting input, render the prompt into the conversation."""
    state = getattr(harness, "state", None)
    if state is None:
        return

    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return

    session = scratchpad.get("_transparent_interactive_session")
    if not isinstance(session, dict):
        return

    if not session.get("awaiting_input"):
        return

    # Already nudged this turn?
    if session.get("nudge_sent"):
        return

    prompt_text = str(session.get("last_prompt") or "").strip()
    if not prompt_text:
        return

    session["nudge_sent"] = True
    state.append_message(
        ConversationMessage(
            role="user",
            content=f"[AUTOMATED INTERACTION] The program is waiting for input:\n{prompt_text}",
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "interactive_input_request",
            },
        )
    )
