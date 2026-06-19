from __future__ import annotations

import json
import re
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
        else:
            # Validate that raw_arguments is parseable JSON; if not, replace with a safe
            # placeholder so downstream servers (e.g. llama.cpp) don't choke on malformed
            # tool-call arguments in the conversation history.
            try:
                json.loads(raw_arguments)
            except Exception:
                raw_arguments = json.dumps(
                    {"_error": "model generated invalid JSON arguments"},
                    ensure_ascii=True,
                    sort_keys=True,
                )
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


def _conclusion_signature(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r'```.*?```', '', lowered, flags=re.DOTALL)
    lowered = re.sub(r'\{.*?\}', '', lowered)
    markers = [
        "enough evidence", "sufficient evidence", "all evidence gathered",
        "now write report", "prepare report", "create artifact",
        "final answer ready", "call task_complete", "complete task",
    ]
    for m in markers:
        if m in lowered:
            return m
    return ""


def _apply_terminal_conclusion_tracking(harness: Any, graph_state: GraphRunState, assistant_text: str) -> bool:
    sig = _conclusion_signature(assistant_text)
    scratchpad = harness.state.scratchpad
    if not sig:
        scratchpad["_terminal_conclusion_signatures"] = []
        return False
    sigs = scratchpad.setdefault("_terminal_conclusion_signatures", [])
    if isinstance(sigs, list) and len(sigs) >= 1 and sigs[-1] == sig and not graph_state.pending_tool_calls:
        from ..challenge_progress import terminal_readiness_state
        readiness = terminal_readiness_state(harness.state)
        if readiness:
            nudge_msg = "The required artifact exists and verification is complete. Call task_complete now."
        else:
            nudge_msg = "You have repeated the same conclusion twice without tool action. Either write the required artifact or call task_complete."
        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=nudge_msg,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "terminal_conclusion_repetition",
                },
            )
        )
        scratchpad["_terminal_conclusion_signatures"] = []
        harness._runlog("terminal_state_breaker", "injected terminal conclusion nudge", signature=sig)
        return True
    scratchpad["_terminal_conclusion_signatures"] = [sig]
    return False


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
        from ..recovery_metrics import increment_metric

        increment_metric(harness.state, "model_stream_halt_count")
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

    # Timestamp tracking for pre-tool stall classification
    import time
    scratchpad = harness.state.scratchpad
    if "_first_assistant_token_time" not in scratchpad:
        scratchpad["_first_assistant_token_time"] = time.time()
    if result.stream.assistant_text and "_first_assistant_text_time" not in scratchpad:
        scratchpad["_first_assistant_text_time"] = time.time()

    # Conclusion signature tracking for same-conclusion repetition breaker
    _apply_terminal_conclusion_tracking(harness, graph_state, result.stream.assistant_text)

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

    # If the stream was halted due to a degenerate loop, do not let the
    # malformed assistant text pollute conversation history or retrieval.
    # Keep any tool calls that were already emitted, but replace the prose
    # with a compact placeholder.
    if halt_reason == "model_output_degenerate_loop":
        parse_result.final_assistant_text = (
            "[Previous assistant output was halted because it entered a repetition loop.]"
        )
        parse_result.final_thinking_text = ""

    conversation_tool_calls = _conversation_tool_calls_from_pending(
        graph_state.pending_tool_calls,
        thread_id=graph_state.thread_id,
        step_count=harness.state.step_count,
    )
    graph_state.last_assistant_text = parse_result.final_assistant_text
    graph_state.last_thinking_text = parse_result.final_thinking_text

    _maybe_inject_file_truncation_hallucination_nudge(harness, parse_result.final_thinking_text)

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

    assistant_text = str(parse_result.final_assistant_text or "").strip()
    if not graph_state.pending_tool_calls:
        harness.state.inactive_steps += 1
        # A substantive answer turn (e.g. a completed research summary) is not
        # "idle" even though it emitted no tool call. Reset the counter so the
        # loop doesn't accuse the model of stalling.
        if assistant_text and len(assistant_text) >= 80:
            harness.state.scratchpad["_consecutive_idle"] = 0
        else:
            harness.state.scratchpad["_consecutive_idle"] = int(harness.state.scratchpad.get("_consecutive_idle", 0)) + 1
    else:
        harness.state.scratchpad["_consecutive_idle"] = 0

    if int(harness.state.scratchpad.get("_consecutive_idle", 0)) >= 2:
        nudge = (
            "[HARNESS NOTICE]: You have provided 2 consecutive turns without any tool actions. "
            "Please focus on making concrete progress towards the goal (explore/execute) "
            "rather than providing high-level summaries or explanation. "
            "If you are finished, use the task_complete tool."
        )
        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=nudge,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "consecutive_idle",
                },
            )
        )
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


def _maybe_inject_file_truncation_hallucination_nudge(harness: Any, thinking_text: str) -> None:
    """Inject a recovery nudge when the model hallucinates that a fully-read file is truncated."""
    if not thinking_text:
        return
    lowered = thinking_text.lower()
    hallucination_markers = (
        "truncated",
        "incomplete",
        "only shows lines",
        "only shows the",
        "file appears to be",
        "missing content",
    )
    if not any(marker in lowered for marker in hallucination_markers):
        return

    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    history = scratchpad.get("_progress_read_history", [])
    if not isinstance(history, list):
        return

    for item in reversed(history[-4:]):
        if not isinstance(item, dict):
            continue
        if str(item.get("tool_name", "")) != "file_read":
            continue
        if not bool(item.get("complete_file")):
            continue
        if bool(item.get("file_content_truncated")):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue

        nudge_key = f"_file_truncation_hallucination_nudged:{path}"
        if scratchpad.get(nudge_key):
            continue
        scratchpad[nudge_key] = True

        state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"The file `{path}` was fully read and is complete. "
                    "Do not assume it is truncated or incomplete. "
                    "Use the evidence already in context to proceed with the next action."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "file_truncation_hallucination",
                    "path": path,
                },
            )
        )
        harness._runlog(
            "file_truncation_hallucination_nudge",
            "injected recovery nudge for file truncation hallucination",
            path=path,
        )
        break
