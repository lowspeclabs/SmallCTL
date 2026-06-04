from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from .state import PendingToolCall


def _dispatch_stagnation_recovery(harness: Any, guard_error: str) -> None:
    """Inject stagnation recovery nudge and reset counters."""
    from ..harness.task_transactions import recovery_context_lines, transaction_from_scratchpad

    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
    tx_lines = recovery_context_lines(transaction)
    tx_note = (" " + " ".join(tx_lines)) if tx_lines else ""
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    active_intent = str(getattr(state, "active_intent", "") or "").strip().lower()
    if phase == "repair" and active_intent in {"requested_file_patch", "requested_write_file"}:
        repair_directive = (
            " You MUST now call a concrete mutation tool (`file_patch`, `file_write`, or `ast_patch`) "
            "or run a focused verifier (`shell_exec`). Do not read, analyze, or plan further."
        )
    else:
        repair_directive = " Try a different tool, check permissions, or rethink your approach instead of repeating the same action."
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"System: {guard_error}."
                f"{repair_directive}"
                f"{tx_note}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "stagnation",
                "guard_error": guard_error,
            },
        )
    )
    harness._runlog(
        "stagnation_recovery",
        "injected strategy pivot nudge",
        step=harness.state.step_count,
    )
    if "no_progress" in harness.state.stagnation_counters:
        harness.state.stagnation_counters["no_progress"] = 0
    if "repeat_command" in harness.state.stagnation_counters:
        harness.state.stagnation_counters["repeat_command"] = 0
    if "repeat_patch" in harness.state.stagnation_counters:
        harness.state.stagnation_counters["repeat_patch"] = 0
    phase = str(getattr(harness.state, "current_phase", "") or "").strip().lower()
    if phase != "repair":
        harness.state.tool_history.clear()
    else:
        harness.state.tool_history = harness.state.tool_history[-6:]


def _dispatch_artifact_read_recovery(harness: Any, graph_state: Any, recovery_hint: tuple[str, str]) -> None:
    """Dispatch artifact_grep as recovery for repeated artifact_read."""
    artifact_id, query = recovery_hint
    from .tool_call_parser import _clear_artifact_read_guard_state
    _clear_artifact_read_guard_state(harness, artifact_id)
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="artifact_grep",
            args={
                "artifact_id": artifact_id,
                "query": query,
            },
            source="system",
        )
    ]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Auto-advancing repeated `artifact_read` on artifact {artifact_id} "
                f"to `artifact_grep` with query `{query}`."
            ),
            metadata={
                "recovery_kind": "artifact_read",
                "artifact_id": artifact_id,
                "query": query,
                "recovery_mode": "direct_dispatch",
            },
        )
    )
    harness._runlog(
        "artifact_read_recovery",
        "scheduled recovery dispatch",
        step=harness.state.step_count,
        artifact_id=artifact_id,
        query=query,
    )


def _inject_artifact_read_recovery_nudge(harness: Any, recovery_hint: tuple[str, str]) -> None:
    """Inject a user nudge for artifact_read recovery in chat mode."""
    artifact_id, query = recovery_hint
    recovery_armed = harness.state.scratchpad.get("_artifact_read_recovery_nudged")
    if recovery_armed == artifact_id:
        return
    msg = (
        f"You are repeating `artifact_read` on artifact {artifact_id}. "
        f"Use `artifact_grep` with query `{query}` instead of reading the same artifact again."
    )
    harness.state.scratchpad["_artifact_read_recovery_nudged"] = artifact_id
    harness.state.scratchpad["_artifact_read_recovery_query"] = query
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=msg,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "artifact_read",
                "artifact_id": artifact_id,
                "query": query,
            },
        )
    )
    harness._runlog(
        "artifact_read_recovery",
        "injected recovery nudge",
        step=harness.state.step_count,
        artifact_id=artifact_id,
        query=query,
    )
