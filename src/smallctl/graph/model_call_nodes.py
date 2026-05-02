from __future__ import annotations

import json
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from . import nodes as _nodes
from .model_stream import process_model_stream
from .state import GraphRunState, PendingToolCall
from .tool_call_parser import parse_tool_calls


def _conversation_tool_calls_from_pending(
    pending_calls: list[PendingToolCall],
    *,
    thread_id: str,
    step_count: int,
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for index, pending in enumerate(pending_calls):
        if str(getattr(pending, "source", "model") or "model").strip().lower() != "model":
            continue
        tool_name = str(pending.tool_name or "").strip()
        if not tool_name:
            continue
        call_id = str(pending.tool_call_id or "").strip()
        if not call_id:
            call_id = f"call_inline_{thread_id}_{step_count}_{index}"
            pending.tool_call_id = call_id
        raw_arguments = str(getattr(pending, "raw_arguments", "") or "").strip()
        if not raw_arguments:
            raw_arguments = json.dumps(json_safe_value(pending.args or {}), ensure_ascii=True, sort_keys=True)
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": raw_arguments,
                },
            }
        )
    return tool_calls


async def model_call(
    graph_state: GraphRunState,
    deps: Any,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    harness = deps.harness
    if graph_state.recorded_tool_call_ids:
        undispatched = list(graph_state.recorded_tool_call_ids)
        graph_state.recorded_tool_call_ids = []
        err_msg = (
            f"Recorded tool calls from the previous turn were never dispatched: {undispatched}. "
            "This indicates a boundary gap between model_call and dispatch_tools."
        )
        graph_state.final_result = harness._failure(
            err_msg,
            error_type="dispatch_boundary_gap",
            details={"undispatched_tool_call_ids": undispatched},
        )
        graph_state.error = graph_state.final_result["error"]
        harness._runlog(
            "dispatch_boundary_gap",
            err_msg,
            undispatched_tool_call_ids=undispatched,
        )
        return
    result = await process_model_stream(graph_state, deps, messages=messages, tools=tools)
    if graph_state.final_result is not None:
        return

    halt_detected = bool(getattr(result, "halted", False))
    halt_reason = str(getattr(result, "halt_reason", "") or "").strip()
    halt_details = json_safe_value(getattr(result, "halt_details", {}) or {})
    if halt_detected:
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        if halt_reason:
            harness.state.scratchpad["_last_stream_halt_reason"] = halt_reason
        if isinstance(halt_details, dict):
            harness.state.scratchpad["_last_stream_halt_details"] = halt_details
    else:
        harness.state.scratchpad.pop("_last_stream_halted_without_done", None)
        harness.state.scratchpad.pop("_last_stream_halt_reason", None)
        harness.state.scratchpad.pop("_last_stream_halt_details", None)

    usage_payload = json_safe_value(result.usage)
    if not isinstance(usage_payload, dict):
        usage_payload = {}
    if usage_payload:
        harness._apply_usage(usage_payload)

    graph_state.last_usage = usage_payload
    graph_state.last_assistant_text = result.stream.assistant_text
    graph_state.last_thinking_text = result.stream.thinking_text

    duration = result.duration
    ttft = result.ttft

    graph_state.latency_metrics["model_call_duration_sec"] = round(duration, 3)
    graph_state.latency_metrics["ttft_sec"] = round(ttft, 3)

    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.METRICS,
            content=f"Model call: {duration:.2f}s (TTFT: {ttft:.2f}s)",
            data={
                "duration_sec": duration,
                "ttft_sec": ttft,
                "usage": usage_payload,
            }
        ),
    )

    parse_result = parse_tool_calls(
        result.stream,
        result.timeline,
        graph_state,
        deps,
        model_name=getattr(harness.client, "model", None),
    )
    graph_state.pending_tool_calls = parse_result.pending_tool_calls
    conversation_tool_calls = _conversation_tool_calls_from_pending(
        graph_state.pending_tool_calls,
        thread_id=graph_state.thread_id,
        step_count=harness.state.step_count,
    )
    graph_state.last_assistant_text = parse_result.final_assistant_text
    graph_state.last_thinking_text = parse_result.final_thinking_text

    if parse_result.final_assistant_text.strip() != result.stream.assistant_text.strip():
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ASSISTANT,
                content=parse_result.final_assistant_text.strip(),
                data=_nodes._planner_speaker_data(
                    graph_state,
                    {"kind": "replace"},
                ),
            ),
        )
    if parse_result.final_thinking_text.strip() != result.stream.thinking_text.strip():
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.THINKING,
                content=parse_result.final_thinking_text,
                data=_nodes._planner_speaker_data(
                    graph_state,
                    {"kind": "replace"},
                ),
            ),
        )

    if not graph_state.pending_tool_calls:
        harness.state.inactive_steps += 1
        harness.state.scratchpad["_consecutive_idle"] = int(harness.state.scratchpad.get("_consecutive_idle", 0)) + 1
    else:
        harness.state.scratchpad["_consecutive_idle"] = 0

    if int(harness.state.scratchpad.get("_consecutive_idle", 0)) >= 2:
        nudge = (
            "\n[SYSTEM NUDGE]: You have provided 2 consecutive turns without any tool actions. "
            "Please focus on making concrete progress towards the goal (explore/execute) "
            "rather than providing high-level summaries or explanation. "
            "If you are finished, use the task_complete tool."
        )
        harness.state.append_message(ConversationMessage(role="system", content=nudge))
        harness.state.scratchpad["_consecutive_idle"] = 1

    if parse_result.final_assistant_text:
        harness._runlog(
            "model_output",
            "assistant output complete",
            assistant_text=parse_result.final_assistant_text,
        )
    if parse_result.final_assistant_text or conversation_tool_calls:
        harness._record_assistant_message(
            assistant_text=parse_result.final_assistant_text,
            tool_calls=conversation_tool_calls,
            speaker="planner" if graph_state.run_mode == "planning" or harness.state.planning_mode_enabled else None,
            hidden_from_prompt=_nodes._model_uses_gpt_oss_commentary_rules(harness),
        )
        harness._log_conversation_state("assistant_message")
        graph_state.recorded_tool_call_ids = [tc["id"] for tc in conversation_tool_calls]
    if parse_result.final_thinking_text:
        harness._runlog(
            "model_thinking",
            "thinking output complete",
            thinking_text=parse_result.final_thinking_text,
        )
    for entry in result.timeline:
        if entry.kind == "tool_call":
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.TOOL_CALL,
                    content=entry.content,
                    data=_nodes._planner_speaker_data(graph_state, entry.data),
                ),
            )
