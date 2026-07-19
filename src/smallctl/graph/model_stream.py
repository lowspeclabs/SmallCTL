from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..client import OpenAICompatClient, StreamResult
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from .deps import GraphRuntimeDeps
from .state import GraphRunState, PendingToolCall
from .tool_call_parser import _detect_empty_file_write_payload
from .write_recovery import (
    can_safely_synthesize,
    infer_write_target_path,
    maybe_finalize_recovered_assistant_write,
    recover_write_intent,
    _maybe_prepend_existing_content,
    write_recovery_kind,
    write_recovery_metadata,
)
from .model_stream_fallback import (
    StreamProcessingResult,
    _attempt_text_write_fallback,
)
from .model_stream_fallback_support import (
    _classify_model_call_error,
    _format_partial_tool_calls,
    _should_attempt_empty_payload_text_fallback,
)
from .model_stream_fallback_recovery import (
    _active_text_write_fallback_session,
    _build_incomplete_tool_call_recovery_message,
    _is_sub4b_write_timeout,
    _with_speaker,
)
from .model_stream_loop import run_model_stream_loop
from .model_stream_loop_rendering import (
    StreamTagState,
    flush_model_stream_buffer,
    handle_model_stream_chunk,
)
from .model_stream_resolution import resolve_model_stream_result


async def _run_nonstream_model_call(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    harness: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    echo_to_stdout: bool,
    start_tag: str,
    end_tag: str,
    start_time: float,
    suppress_ui_events: bool = False,
) -> dict[str, Any]:
    """Execute a single non-streaming model call and return loop-shaped result."""
    chunks: list[dict[str, Any]] = []
    first_token_time: float | None = None
    stream_state = StreamTagState()
    last_chunk_error_details: dict[str, Any] | None = None
    stream_completed_cleanly = True
    try:
        async for event in harness.client.stream_chat(messages=messages, tools=tools, force_nonstream=True):
            if event.get("type") == "chunk":
                if first_token_time is None:
                    first_token_time = time.monotonic()
                stream_state, first_token_time = await handle_model_stream_chunk(
                    harness=harness,
                    deps=deps,
                    event=event,
                    start_tag=start_tag,
                    end_tag=end_tag,
                    echo_to_stdout=echo_to_stdout,
                    chunks=chunks,
                    stream_state=stream_state,
                    first_token_time=first_token_time,
                    suppress_ui_events=suppress_ui_events,
                )
            elif event.get("type") == "done":
                break
            elif event.get("type") == "chunk_error":
                stream_completed_cleanly = False
                last_chunk_error_details = {
                    "error": str(event.get("error") or ""),
                    "details": event.get("details"),
                }
                harness._runlog(
                    "nonstream_chunk_error",
                    "non-stream fallback returned a chunk error",
                    error=event.get("error"),
                    details=event.get("details"),
                )
                break
    except Exception as exc:
        stream_completed_cleanly = False
        last_chunk_error_details = {
            "error": str(exc),
            "exception_type": exc.__class__.__name__,
        }
        harness.log.exception("nonstream model call failed")
        harness._runlog("nonstream_model_call_error", "non-stream fallback failed", error=str(exc))

    await flush_model_stream_buffer(
        harness=harness,
        deps=deps,
        stream_state=stream_state,
        start_tag=start_tag,
        end_tag=end_tag,
        echo_to_stdout=echo_to_stdout,
        suppress_ui_events=suppress_ui_events,
    )
    return {
        "chunks": chunks,
        "stream_completed_cleanly": stream_completed_cleanly,
        "trigger_early_4b_fallback": False,
        "salvage_partial_stream": None,
        "last_chunk_error_details": last_chunk_error_details,
        "stream_ended_without_done": False,
        "stream_ended_without_done_details": {},
        "partial_assistant_text": "",
        "first_token_time": first_token_time,
    }


async def process_model_stream(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    suppress_ui_events: bool = False,
) -> StreamProcessingResult:
    """Stream a model call, process chunks, and return assembled results."""
    harness = deps.harness
    if hasattr(harness, "state") and harness.state is not None:
        scratchpad = getattr(harness.state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            scratchpad["_model_calls"] = int(scratchpad.get("_model_calls", 0)) + 1
    event_handler = getattr(harness, "event_handler", None)
    echo_to_stdout = event_handler is None and not suppress_ui_events
    graph_state.pending_tool_calls = []
    graph_state.last_tool_results = []
    graph_state.last_assistant_text = ""
    graph_state.last_thinking_text = ""
    graph_state.last_usage = {}
    start_time = time.monotonic()

    start_tag = str(harness.thinking_start_tag or "<think>")
    end_tag = str(harness.thinking_end_tag or "</think>")
    harness.state.scratchpad.pop("_last_incomplete_tool_call", None)
    harness.state.scratchpad.pop("_last_text_write_fallback_assistant_text", None)

    try_nonstream = bool(harness.state.scratchpad.pop("_try_nonstream_next_turn", False))
    if try_nonstream:
        loop_result = await _run_nonstream_model_call(
            graph_state,
            deps,
            harness=harness,
            messages=messages,
            tools=tools,
            echo_to_stdout=echo_to_stdout,
            start_tag=start_tag,
            end_tag=end_tag,
            start_time=start_time,
            suppress_ui_events=suppress_ui_events,
        )
    else:
        loop_result = await run_model_stream_loop(
            graph_state,
            deps,
            harness=harness,
            messages=messages,
            tools=tools,
            echo_to_stdout=echo_to_stdout,
            start_tag=start_tag,
            end_tag=end_tag,
            start_time=start_time,
            suppress_ui_events=suppress_ui_events,
        )
    if graph_state.final_result is not None:
        # Preserve provider-classified failures from the stream loop. Resolution
        # should only infer a chunk-exhaustion error when the loop ended
        # ambiguously without already finalizing the run.
        return StreamProcessingResult(chunks=loop_result["chunks"])
    return await resolve_model_stream_result(
        graph_state,
        deps,
        harness=harness,
        chunks=loop_result["chunks"],
        salvage_partial_stream=loop_result["salvage_partial_stream"],
        last_chunk_error_details=loop_result["last_chunk_error_details"],
        stream_ended_without_done=loop_result["stream_ended_without_done"],
        stream_ended_without_done_details=loop_result["stream_ended_without_done_details"],
        partial_assistant_text=loop_result.get("partial_assistant_text", ""),
        trigger_early_4b_fallback=loop_result["trigger_early_4b_fallback"],
        stream_completed_cleanly=loop_result["stream_completed_cleanly"],
        echo_to_stdout=echo_to_stdout,
        messages=messages,
        start_time=start_time,
        first_token_time=loop_result["first_token_time"],
    )
