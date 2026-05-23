from __future__ import annotations

from typing import Any

from ..models.events import UIEvent, UIEventType
from ..models.conversation import ConversationMessage
from ..recovery_metrics import record_terminal_success_metrics
from ..state import clip_text_value
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord

_REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"
_REMOTE_VERIFIER_PENDING_COMPLETE_KEY = "_task_complete_remote_mutation_verifier_pending_complete"
_REMOTE_VERIFIER_TOOLS = {"ssh_file_read", "ssh_exec"}


def _maybe_schedule_task_complete_remote_mutation_verifier(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "task_complete" or record.result.success:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("reason") or "").strip() != "remote_mutation_requires_verification":
        return False

    action = metadata.get("next_required_action")
    if not isinstance(action, dict):
        return False
    tool_names = action.get("tool_names")
    normalized_tool_names = {str(name).strip() for name in tool_names} if isinstance(tool_names, list) else set()
    if not normalized_tool_names.intersection({"ssh_file_read", "ssh_exec"}):
        return False
    required_arguments = action.get("required_arguments")
    if not isinstance(required_arguments, dict):
        return False

    args: dict[str, Any] = {}
    for key in ("target", "host", "user", "path", "command"):
        value = str(required_arguments.get(key) or "").strip()
        if value:
            args[key] = value
    tool_name = "ssh_file_read" if "ssh_file_read" in normalized_tool_names and args.get("path") else "ssh_exec"
    if tool_name == "ssh_file_read" and not args.get("path"):
        return False
    if tool_name == "ssh_exec" and not args.get("command"):
        return False
    if not (args.get("host") or args.get("target")):
        return False

    signature = "|".join(
        [
            str(record.tool_call_id or ""),
            str(record.operation_id or ""),
            f"task_complete_remote_mutation_verifier:{tool_name}",
            str(args),
        ]
    )
    if harness.state.scratchpad.get("_task_complete_remote_mutation_verifier_autocontinue") == signature:
        return False
    harness.state.scratchpad["_task_complete_remote_mutation_verifier_autocontinue"] = signature
    harness.state.scratchpad[_REMOTE_VERIFIER_PENDING_COMPLETE_KEY] = {
        "tool_name": tool_name,
        "args": args,
        "message": str(record.args.get("message") or ""),
        "tool_call_id": str(record.tool_call_id or ""),
        "operation_id": str(record.operation_id or ""),
    }

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name=tool_name,
            args=args,
            raw_arguments="{}",
            source="system",
        )
    ]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Auto-continuing remote mutation verification with `ssh_file_read` "
                "or `ssh_exec` because `task_complete` was blocked. For deletion "
                "tasks, a `not found` / `no such file` read or an empty directory "
                "listing is valid proof. For binary or key files, a read-only "
                "presence/hash check is valid proof. "
                "Verification already completed means the model may call `task_complete` "
                "immediately without additional chatter."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "task_complete_remote_mutation_verifier_autocontinue",
                "tool_name": tool_name,
                "required_arguments": args,
            },
        )
    )
    harness._runlog(
        "task_complete_remote_mutation_verifier_autocontinue",
        "scheduled automatic remote verifier after remote mutation task completion block",
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
        path=str(args.get("path") or ""),
        command=str(args.get("command") or ""),
        host=str(args.get("host") or args.get("target") or ""),
    )
    return True


async def maybe_auto_complete_after_remote_mutation_verifier(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
    event_handler: Any,
) -> bool:
    if record.tool_name not in _REMOTE_VERIFIER_TOOLS:
        return False

    scratchpad = getattr(harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    pending = scratchpad.get(_REMOTE_VERIFIER_PENDING_COMPLETE_KEY)
    if not isinstance(pending, dict):
        return False
    if scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY):
        return False
    if str(pending.get("tool_name") or "") != str(record.tool_name or ""):
        return False
    if not _arguments_match(pending.get("args"), record.args):
        return False
    if not _verifier_result_satisfies_remote_gate(record):
        return False

    message = str(pending.get("message") or "").strip() or "Remote mutation verified; task complete."
    scratchpad.pop(_REMOTE_VERIFIER_PENDING_COMPLETE_KEY, None)
    scratchpad["_task_complete"] = True
    scratchpad["_task_complete_message"] = message
    harness.state.touch()
    record_terminal_success_metrics(harness.state)

    emit = getattr(harness, "_emit", None)
    if callable(emit):
        maybe_awaitable = emit(
            event_handler,
            UIEvent(
                event_type=UIEventType.SYSTEM,
                content="Task marked complete after automatic remote verification.",
                data={"status_activity": "remote mutation verified"},
            ),
        )
        if hasattr(maybe_awaitable, "__await__"):
            await maybe_awaitable

    graph_state.final_result = {
        "status": "completed",
        "message": {"status": "complete", "message": message},
        "assistant": str(graph_state.last_assistant_text or "").strip() or message,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
    }
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "task_complete_remote_mutation_verifier_autoaccepted",
            "auto-accepted task completion after remote mutation verifier cleared the gate",
            verifier_tool=record.tool_name,
            original_tool_call_id=str(pending.get("tool_call_id") or ""),
            original_operation_id=str(pending.get("operation_id") or ""),
        )
    return True


def _verifier_result_satisfies_remote_gate(record: ToolExecutionRecord) -> bool:
    if bool(record.result.success):
        if record.tool_name != "ssh_exec":
            return True
        output = record.result.output if isinstance(record.result.output, dict) else {}
        command = str(record.args.get("command") or "").strip()
        stdout = str(output.get("stdout") or "").strip()
        return not command.startswith("find ") or not stdout

    if record.tool_name != "ssh_file_read":
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    failure_markers = " ".join(
        [
            str(record.result.error or ""),
            str(metadata.get("message") or ""),
            str(metadata.get("error_kind") or ""),
        ]
    ).lower()
    return (
        "no such file" in failure_markers
        or "not found" in failure_markers
        or "file_not_found" in failure_markers
    )


def _arguments_match(expected: Any, actual: Any) -> bool:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    for key, expected_value in expected.items():
        if str(actual.get(key) or "").strip() != str(expected_value or "").strip():
            return False
    return True


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
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    verifier = metadata.get("last_verifier_verdict")
    if isinstance(verifier, dict) and (
        bool(verifier.get("approval_denied")) or str(verifier.get("verdict") or "").strip() == "needs_human"
    ):
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
                "`task_complete` in the REPAIR phase. If the failure is expected "
                "or diagnostic (e.g. 'not found'), you may call `task_complete` "
                "with the finding. Otherwise, fix the failing command before completing."
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
    if bool(verifier.get("approval_denied")) or str(verifier.get("verdict") or "").strip() == "needs_human":
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
