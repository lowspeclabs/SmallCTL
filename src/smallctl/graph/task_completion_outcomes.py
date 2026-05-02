from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..state import clip_text_value
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord


def _maybe_schedule_task_complete_verifier_loop_status(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "task_complete" or record.result.success:
        return False

    error_text = str(record.result.error or "").strip().lower()
    if "latest verifier verdict is still failing" not in error_text:
        return False

    signature = "|".join(
        [
            str(record.tool_call_id or ""),
            str(record.operation_id or ""),
            "task_complete_verifier_loop_status",
        ]
    )
    if harness.state.scratchpad.get("_task_complete_verifier_loop_status_autocontinue") == signature:
        return False
    harness.state.scratchpad["_task_complete_verifier_loop_status_autocontinue"] = signature

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="loop_status",
            args={},
            raw_arguments="{}",
            source="system",
        )
    ]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content="Auto-continuing verifier recovery with `loop_status` before requesting another completion attempt.",
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "task_complete_verifier_loop_status_autocontinue",
            },
        )
    )
    harness._runlog(
        "task_complete_verifier_loop_status_autocontinue",
        "scheduled automatic loop_status after blocked task completion",
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
    )
    return True


def _maybe_schedule_task_complete_repair_loop_status(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "task_complete" or record.result.success:
        return False

    error_text = str(record.result.error or "").strip().lower()
    if "not allowed in phase 'repair'" not in error_text:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    blocked_phase = str(
        metadata.get("phase")
        or metadata.get("dispatch_phase")
        or ""
    ).strip().lower()
    if blocked_phase and blocked_phase != "repair":
        return False

    signature = "|".join(
        [
            str(record.tool_call_id or ""),
            str(record.operation_id or ""),
            "task_complete_repair_loop_status",
        ]
    )
    if harness.state.scratchpad.get("_task_complete_repair_loop_status_autocontinue") == signature:
        return False
    harness.state.scratchpad["_task_complete_repair_loop_status_autocontinue"] = signature

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="loop_status",
            args={},
            raw_arguments="{}",
            source="system",
        )
    ]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Auto-continuing repair recovery with `loop_status` after blocked "
                "`task_complete` in the REPAIR phase. The verifier still has a "
                "non-zero exit code. Fix the failing command (re-run it and "
                "achieve exit_code 0) to exit the repair phase."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "task_complete_repair_loop_status_autocontinue",
                "phase": "repair",
            },
        )
    )
    harness._runlog(
        "task_complete_repair_loop_status_autocontinue",
        "scheduled automatic loop_status after repair-phase task completion block",
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
    )
    return True


def _maybe_emit_task_complete_verifier_nudge(harness: Any, record: ToolExecutionRecord) -> bool:
    if record.tool_name != "task_complete" or record.result.success:
        return False

    error_text = str(record.result.error or "").strip().lower()
    if "latest verifier verdict is still failing" not in error_text:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    verifier = metadata.get("last_verifier_verdict")
    if not isinstance(verifier, dict) or not verifier:
        current_verifier = getattr(harness.state, "current_verifier_verdict", None)
        verifier = current_verifier() if callable(current_verifier) else None
    if not isinstance(verifier, dict) or not verifier:
        return False

    target_text, clipped = clip_text_value(
        str(verifier.get("command") or verifier.get("target") or "").strip(),
        limit=180,
    )
    note = ""
    acceptance_delta = verifier.get("acceptance_delta")
    if isinstance(acceptance_delta, dict):
        notes = acceptance_delta.get("notes")
        if isinstance(notes, list):
            note = next((str(item).strip() for item in notes if str(item).strip()), "")
    if not note:
        note = str(verifier.get("key_stderr") or verifier.get("key_stdout") or "").strip()
    note_text, note_clipped = clip_text_value(note, limit=180)

    signature_bits = [
        str(record.tool_call_id or ""),
        str(verifier.get("verdict") or ""),
        target_text,
        note_text,
    ]
    signature = "|".join(signature_bits)
    if harness.state.scratchpad.get("_task_complete_verifier_retry_nudge") == signature:
        return False
    harness.state.scratchpad["_task_complete_verifier_retry_nudge"] = signature

    message = "Do not repeat `task_complete` yet."
    verifier_bits: list[str] = []
    if target_text:
        suffix = " [truncated]" if clipped else ""
        verifier_bits.append(f"latest verifier: `{target_text}{suffix}`")
    if note_text:
        suffix = " [truncated]" if note_clipped else ""
        verifier_bits.append(f"result: {note_text}{suffix}")
    if verifier_bits:
        message += " " + " | ".join(verifier_bits) + "."
    message += (
        " Use `loop_status` to inspect the blocker, then either run one focused repair step "
        "or rerun the check in a zero-exit diagnostic form if the failure itself is the proof you need."
    )

    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "task_complete_verifier_retry",
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "verifier_verdict": str(verifier.get("verdict") or ""),
            },
        )
    )
    harness._runlog(
        "task_complete_verifier_retry_nudge",
        "injected recovery nudge after blocked task completion",
        tool_call_id=record.tool_call_id,
        verifier_verdict=str(verifier.get("verdict") or ""),
        verifier_target=str(verifier.get("command") or verifier.get("target") or ""),
    )
    return True
