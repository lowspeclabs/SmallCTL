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
from .model_stream_resolution import resolve_model_stream_result


async def process_model_stream(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> StreamProcessingResult:
    """Stream a model call, process chunks, and return assembled results."""
    harness = deps.harness
    event_handler = getattr(harness, "event_handler", None)
    echo_to_stdout = event_handler is None
    graph_state.pending_tool_calls = []
    graph_state.last_tool_results = []
    graph_state.last_assistant_text = ""
    graph_state.last_thinking_text = ""
    graph_state.last_usage = {}
    start_time = time.perf_counter()

    start_tag = str(harness.thinking_start_tag or "<think>")
    end_tag = str(harness.thinking_end_tag or "</think>")
    harness.state.scratchpad.pop("_last_incomplete_tool_call", None)
    harness.state.scratchpad.pop("_last_text_write_fallback_assistant_text", None)
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
        trigger_early_4b_fallback=loop_result["trigger_early_4b_fallback"],
        stream_completed_cleanly=loop_result["stream_completed_cleanly"],
        echo_to_stdout=echo_to_stdout,
        messages=messages,
        start_time=start_time,
        first_token_time=loop_result["first_token_time"],
    )
