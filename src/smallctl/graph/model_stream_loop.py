from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from ..client import OpenAICompatClient, StreamResult
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from .deps import GraphRuntimeDeps
from .state import GraphRunState
from .tool_call_parser import _detect_empty_file_write_payload
from .model_stream_fallback import StreamProcessingResult
from .model_stream_fallback_support import (
    _classify_model_call_error,
    _format_partial_tool_calls,
)
from .model_stream_fallback_recovery import (
    _build_incomplete_tool_call_recovery_message,
    _is_sub4b_write_timeout,
)
from .model_stream_loop_rendering import flush_model_stream_buffer
from .model_stream_loop_rendering import handle_model_stream_chunk
from .model_stream_loop_rendering import StreamTagState
from .model_stream_loop_recovery import handle_model_stream_chunk_error


async def run_model_stream_loop(
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
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    first_token_time: float | None = None
    stream_state = StreamTagState()
    timeout_recovery_nudges = 0
    last_chunk_error_details: dict[str, Any] | None = None
    salvage_partial_stream: StreamResult | None = None
    stream_ended_without_done = False
    stream_ended_without_done_details: dict[str, Any] = {}

    _CHUNK_ERROR_MAX_RETRIES = 2
    _trigger_early_4b_fallback: bool = bool(harness.state.scratchpad.get("_sub4b_chat_fallback_active"))
    _stream_completed_cleanly: bool = False

    for _model_attempt in range(_CHUNK_ERROR_MAX_RETRIES + 1):
        try:
            chunks = []
            stream_state = StreamTagState()
            _retry_immediately = False
            async for event in harness.client.stream_chat(messages=messages, tools=tools):
                if harness._cancel_requested:
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
                    )
                    graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
                    return {
                        "chunks": chunks,
                        "stream_completed_cleanly": False,
                        "trigger_early_4b_fallback": _trigger_early_4b_fallback,
                        "salvage_partial_stream": salvage_partial_stream,
                        "last_chunk_error_details": last_chunk_error_details,
                        "stream_ended_without_done": stream_ended_without_done,
                        "stream_ended_without_done_details": stream_ended_without_done_details,
                        "first_token_time": first_token_time,
                    }
                if event.get("type") == "chunk_error":
                    err_msg = event.get("error", "unknown upstream error")
                    details = event.get("details")
                    if not isinstance(details, dict):
                        details = {}
                    last_chunk_error_details = details
                    retrying = _model_attempt < _CHUNK_ERROR_MAX_RETRIES
                    overflow = _parse_context_window_overflow(err_msg, details)
                    if overflow is not None:
                        n_keep, n_ctx = overflow
                        rebuild_messages = getattr(harness, "_rebuild_messages_after_context_overflow", None)
                        if callable(rebuild_messages):
                            try:
                                replacement_messages = await rebuild_messages(
                                    n_ctx=n_ctx,
                                    n_keep=n_keep,
                                    error_message=err_msg,
                                    event_handler=deps.event_handler,
                                )
                            except RuntimeError as exc:
                                await harness._emit(
                                    deps.event_handler,
                                    UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
                                )
                                graph_state.final_result = harness._failure(
                                    str(exc),
                                    error_type="prompt_budget",
                                )
                                graph_state.error = graph_state.final_result["error"]
                                return {
                                    "chunks": chunks,
                                    "stream_completed_cleanly": False,
                                    "trigger_early_4b_fallback": _trigger_early_4b_fallback,
                                    "salvage_partial_stream": salvage_partial_stream,
                                    "last_chunk_error_details": last_chunk_error_details,
                                    "stream_ended_without_done": stream_ended_without_done,
                                    "stream_ended_without_done_details": stream_ended_without_done_details,
                                    "first_token_time": first_token_time,
                            }
                            if replacement_messages:
                                messages = replacement_messages
                                _retry_immediately = True
                                harness._runlog(
                                    "stream_context_shrink",
                                    "shrinking prompt context after upstream n_keep overflow",
                                    error=err_msg,
                                    attempt=_model_attempt + 1,
                                    retrying=retrying,
                                    details=details,
                                    n_keep=n_keep,
                                    n_ctx=n_ctx,
                                    max_prompt_tokens=getattr(harness.context_policy, "max_prompt_tokens", None),
                                )
                                if retrying:
                                    await harness._emit(
                                        deps.event_handler,
                                        UIEvent(
                                            event_type=UIEventType.ALERT,
                                            content=f"Stream chunk error (shrinking context and retrying): {err_msg}",
                                            data={
                                                "is_api_error": True,
                                                "retrying": True,
                                                "attempt": _model_attempt + 1,
                                                "details": details,
                                                "recovery": "context_shrink",
                                            },
                                        ),
                                    )
                                break
                    recovery_result = await handle_model_stream_chunk_error(
                        harness=harness,
                        deps=deps,
                        graph_state=graph_state,
                        messages=messages,
                        chunks=chunks,
                        err_msg=err_msg,
                        details=details,
                        model_attempt=_model_attempt,
                        chunk_error_max_retries=_CHUNK_ERROR_MAX_RETRIES,
                        timeout_recovery_nudges=timeout_recovery_nudges,
                        trigger_early_4b_fallback=_trigger_early_4b_fallback,
                        salvage_partial_stream=salvage_partial_stream,
                    )
                    timeout_recovery_nudges = recovery_result["timeout_recovery_nudges"]
                    _trigger_early_4b_fallback = recovery_result["trigger_early_4b_fallback"]
                    salvage_partial_stream = recovery_result["salvage_partial_stream"]
                    retrying = recovery_result["retrying"]
                    if _trigger_early_4b_fallback:
                        break
                    break
                if event.get("type") == "backend_wedged":
                    details = event.get("details")
                    if not isinstance(details, dict):
                        details = {}
                    graph_state.latency_metrics["backend_wedged_count"] = (
                        int(graph_state.latency_metrics.get("backend_wedged_count", 0) or 0) + 1
                    )
                    harness._runlog(
                        "backend_wedged",
                        "backend did not emit a first token before timeout",
                        details=details,
                    )
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(
                            event_type=UIEventType.ERROR,
                            content="Backend did not emit a first token before timeout. Automatic recovery did not succeed.",
                            data={"is_api_error": True, "details": details},
                        ),
                    )
                    graph_state.final_result = harness._failure(
                        "Backend did not emit a first token before timeout",
                        error_type="provider",
                        details=details,
                    )
                    graph_state.error = graph_state.final_result["error"]
                    return {
                        "chunks": chunks,
                        "stream_completed_cleanly": False,
                        "trigger_early_4b_fallback": _trigger_early_4b_fallback,
                        "salvage_partial_stream": salvage_partial_stream,
                        "last_chunk_error_details": last_chunk_error_details,
                        "stream_ended_without_done": stream_ended_without_done,
                        "stream_ended_without_done_details": stream_ended_without_done_details,
                        "first_token_time": first_token_time,
                    }
                if event.get("type") == "stream_ended_without_done":
                    stream_ended_without_done = True
                    details = event.get("details")
                    if isinstance(details, dict):
                        stream_ended_without_done_details = dict(details)
                    continue
                if event.get("type") == "chunk":
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
                    )
                    continue
            else:
                await flush_model_stream_buffer(
                    harness=harness,
                    deps=deps,
                    stream_state=stream_state,
                    start_tag=start_tag,
                    end_tag=end_tag,
                    echo_to_stdout=echo_to_stdout,
                )
                _stream_completed_cleanly = True
                break

            if _trigger_early_4b_fallback:
                break
            if _retry_immediately:
                continue
            if _model_attempt < _CHUNK_ERROR_MAX_RETRIES:
                await asyncio.sleep(float(_model_attempt + 1))
        except asyncio.CancelledError:
            await harness._emit(
                deps.event_handler,
                UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
            )
            raise
        except Exception as exc:
            harness.log.exception("stream_chat failed")
            log_kv(harness.log, logging.ERROR, "harness_stream_error", error=str(exc))
            error_type, details = _classify_model_call_error(exc)
            is_api = error_type == "provider"
            content_prefix = "Provider error" if is_api else "Stream error"

            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ERROR,
                    content=f"{content_prefix}: {exc}",
                    data={"is_api_error": is_api},
                ),
            )
            err_msg = str(exc) or type(exc).__name__
            graph_state.final_result = harness._failure(err_msg, error_type=error_type, details=details)
            graph_state.error = graph_state.final_result["error"]
            return {
                "chunks": chunks,
                "stream_completed_cleanly": False,
                "trigger_early_4b_fallback": _trigger_early_4b_fallback,
                "salvage_partial_stream": salvage_partial_stream,
                "last_chunk_error_details": last_chunk_error_details,
                "stream_ended_without_done": stream_ended_without_done,
                "stream_ended_without_done_details": stream_ended_without_done_details,
                "first_token_time": first_token_time,
            }

    return {
        "chunks": chunks,
        "stream_completed_cleanly": _stream_completed_cleanly,
        "trigger_early_4b_fallback": _trigger_early_4b_fallback,
        "salvage_partial_stream": salvage_partial_stream,
        "last_chunk_error_details": last_chunk_error_details,
        "stream_ended_without_done": stream_ended_without_done,
        "stream_ended_without_done_details": stream_ended_without_done_details,
        "first_token_time": first_token_time,
    }


def _parse_context_window_overflow(err_msg: str, details: dict[str, Any]) -> tuple[int, int] | None:
    match = re.search(r"n_keep=(\d+).*?n_ctx=(\d+)", f"{err_msg} {details!r}")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
