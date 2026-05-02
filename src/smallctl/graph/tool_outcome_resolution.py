from __future__ import annotations

from typing import Any

from ..models.events import UIEvent, UIEventType
from ..tools.fs_loop_guard import build_loop_guard_outline_interrupt_payload
from .chat_progress import _clear_chat_progress_guard
from .interrupts import build_interrupt_payload
from .shell_outcomes import _clear_shell_human_retry_state, _remember_shell_human_retry_state
from .state import GraphRunState, ToolExecutionRecord
from .write_session_outcomes import _auto_update_active_plan_step
from .task_completion_outcomes import (
    _maybe_emit_task_complete_verifier_nudge,
    _maybe_schedule_task_complete_repair_loop_status,
    _maybe_schedule_task_complete_verifier_loop_status,
)


def _terminal_message_text(output: Any) -> str:
    if isinstance(output, dict):
        for key in ("message", "output", "text", "question"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(output, str):
        return output.strip()
    if output is None:
        return ""
    return str(output).strip()


def _clear_shell_retry_state_if_applicable(harness: Any, record: ToolExecutionRecord) -> None:
    if record.tool_name != "shell_exec":
        return
    if record.result.success:
        _clear_shell_human_retry_state(harness)
        return
    if getattr(record.result, "status", None) != "needs_human" and record.result.metadata.get("status") != "needs_human":
        _clear_shell_human_retry_state(harness)


async def _emit_ui_event(harness: Any, event_handler: Any, event: UIEvent) -> None:
    emit = getattr(harness, "_emit", None)
    if not callable(emit):
        return
    maybe_awaitable = emit(event_handler, event)
    if hasattr(maybe_awaitable, "__await__"):
        await maybe_awaitable


async def maybe_apply_terminal_tool_outcome(
    graph_state: GraphRunState,
    deps: Any,
    record: ToolExecutionRecord,
    *,
    chat_mode: bool,
) -> bool:
    harness = deps.harness
    _clear_shell_retry_state_if_applicable(harness, record)

    if record.tool_name == "task_complete" and record.result.success:
        _auto_update_active_plan_step(harness, status="completed", note=str(record.result.output or ""))
        message = _terminal_message_text(record.result.output)
        assistant_text = str(graph_state.last_assistant_text or "").strip() or message
        if chat_mode:
            _clear_chat_progress_guard(harness)
            graph_state.final_result = {
                "status": "chat_completed",
                "message": message,
                "assistant": assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
        else:
            await _emit_ui_event(
                harness,
                deps.event_handler,
                UIEvent(event_type=UIEventType.SYSTEM, content="Task marked complete."),
            )
            graph_state.final_result = {
                "status": "completed",
                "message": record.result.output,
                "assistant": assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
        return True

    if record.tool_name == "task_complete" and not record.result.success:
        scheduled_loop_status = _maybe_schedule_task_complete_repair_loop_status(graph_state, harness, record)
        if not scheduled_loop_status:
            scheduled_loop_status = _maybe_schedule_task_complete_verifier_loop_status(graph_state, harness, record)
        _maybe_emit_task_complete_verifier_nudge(harness, record)
        if scheduled_loop_status:
            await _emit_ui_event(
                harness,
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content="Auto-continuing recovery with `loop_status` after blocked task completion.",
                    data={"status_activity": "auto-continuing verifier recovery"},
                ),
            )
        return False

    if record.tool_name == "task_fail" and record.result.success:
        _auto_update_active_plan_step(harness, status="blocked", note=str(record.result.output or ""))
        if chat_mode:
            _clear_chat_progress_guard(harness)
            message = str(record.result.output.get("message") if isinstance(record.result.output, dict) else record.result.output)
            graph_state.final_result = {
                "status": "chat_failed",
                "message": message,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
        else:
            await _emit_ui_event(
                harness,
                deps.event_handler,
                UIEvent(event_type=UIEventType.ERROR, content="Task marked failed."),
            )
            graph_state.final_result = harness._failure(
                "Task marked failed.",
                error_type="tool",
                details={
                    "tool_name": record.tool_name,
                    "output": record.result.output,
                    "assistant": graph_state.last_assistant_text,
                    "thinking": graph_state.last_thinking_text,
                    "usage": graph_state.last_usage,
                },
            )
            graph_state.error = graph_state.final_result["error"]
        return True

    if record.tool_name == "ask_human" and record.result.success:
        if chat_mode:
            _clear_chat_progress_guard(harness)
        question = ""
        if isinstance(record.result.output, dict):
            question = str(record.result.output.get("question", "")).strip()
        payload = build_loop_guard_outline_interrupt_payload(
            state=harness.state,
            thread_id=graph_state.thread_id,
            question=question,
            current_phase=harness.state.current_phase,
            active_profiles=list(harness.state.active_tool_profiles),
            recent_tool_outcomes=[r.to_summary_dict() for r in graph_state.last_tool_results],
        ) or build_interrupt_payload(
            harness=harness,
            graph_state=graph_state,
            record=record,
        )
        graph_state.interrupt_payload = payload
        harness.state.pending_interrupt = payload
        if not chat_mode:
            await _emit_ui_event(
                harness,
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content="Human input requested by model.",
                    data={"interrupt": payload},
                ),
            )
        graph_state.final_result = {
            "status": "needs_human",
            "message": payload.get("question", "Human input requested."),
            "assistant": graph_state.last_assistant_text,
            "thinking": graph_state.last_thinking_text,
            "usage": graph_state.last_usage,
            "interrupt": payload,
        }
        return True

    if (
        getattr(record.result, "status", None) == "needs_human"
        or record.result.metadata.get("status") == "needs_human"
    ):
        if chat_mode:
            _clear_chat_progress_guard(harness)
        if record.tool_name == "shell_exec":
            _remember_shell_human_retry_state(harness, record)
        question = record.result.metadata.get("question", "Human input required for tool.")
        payload = {
            "question": question,
            "tool_name": record.tool_name,
            "tool_call_id": record.tool_call_id,
            "metadata": {**record.result.metadata, "interrupt_type": "tool_request"},
            "current_phase": "explore",
            "active_profiles": list(harness.state.active_tool_profiles),
            "thread_id": graph_state.thread_id,
            "operation_id": record.operation_id,
            "recent_tool_outcomes": [r.to_summary_dict() for r in graph_state.last_tool_results],
        }
        graph_state.interrupt_payload = payload
        harness.state.pending_interrupt = payload
        if not chat_mode:
            await _emit_ui_event(
                harness,
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content=f"Tool '{record.tool_name}' requires human input: {question}",
                    data={"interrupt": payload},
                ),
            )
        graph_state.final_result = {
            "status": "needs_human",
            "message": record.result.error,
            "assistant": graph_state.last_assistant_text,
            "thinking": graph_state.last_thinking_text,
            "usage": graph_state.last_usage,
            "interrupt": payload,
        }
        return True

    return False
