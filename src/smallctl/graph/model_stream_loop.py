from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Iterable

from ..client import OpenAICompatClient, StreamResult
from ..client.chunk_parser import chunk_contains_tool_call_delta
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

_REASONING_ONLY_MAX_SECONDS = 60.0
_REASONING_ONLY_MAX_CHUNKS = 4096
_REASONING_ONLY_TOOL_MAX_SECONDS = 25.0
_REASONING_ONLY_TOOL_MAX_CHUNKS = 1500
_REASONING_ONLY_MAX_RETRIES = 1
_REASONING_PROGRESS_MIN_FRAGMENTS = 4
_REASONING_PROGRESS_MIN_UNIQUE_RATIO = 0.35
_REASONING_PROGRESS_MIN_DISTINCT_WORDS = 12
_REASONING_PROGRESS_MIN_NOVEL_WORDS = 4
_REASONING_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")


@dataclass(frozen=True)
class _ReasoningProgressAssessment:
    progress: bool
    fragment_count: int
    unique_ratio: float
    distinct_word_count: int
    novel_word_count: int
    novel_word_ratio: float

    def to_log_dict(self) -> dict[str, int | float | bool]:
        return {
            "progress": self.progress,
            "fragment_count": self.fragment_count,
            "unique_ratio": round(self.unique_ratio, 3),
            "distinct_word_count": self.distinct_word_count,
            "novel_word_count": self.novel_word_count,
            "novel_word_ratio": round(self.novel_word_ratio, 3),
        }


def _chunk_delta(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data", {})
    if not isinstance(data, dict):
        return {}
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    delta = choices[0].get("delta", {})
    return delta if isinstance(delta, dict) else {}


def _chunk_has_assistant_content(event: dict[str, Any]) -> bool:
    delta = _chunk_delta(event)
    content = delta.get("content")
    if isinstance(content, str) and bool(content.strip()):
        return True
    # Some backends (llama.cpp, OpenRouter/DeepInfra) populate reasoning_content
    # while leaving content null. Treat reasoning as valid assistant output so the
    # stream is not incorrectly flagged as a reasoning-only stall.
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    return isinstance(reasoning, str) and bool(reasoning.strip())


def _chunk_has_reasoning(event: dict[str, Any]) -> bool:
    return bool(_chunk_reasoning_text(event))


def _chunk_reasoning_text(event: dict[str, Any]) -> str:
    delta = _chunk_delta(event)
    for key in ("reasoning_content", "reasoning"):
        reasoning = delta.get(key)
        if isinstance(reasoning, str) and reasoning:
            return reasoning
    return ""


def _assess_reasoning_fragments_progress(fragments: Iterable[str]) -> _ReasoningProgressAssessment:
    normalized = [" ".join(fragment.lower().split()) for fragment in fragments if fragment.strip()]
    fragment_count = len(normalized)
    unique_ratio = len(set(normalized)) / max(1, fragment_count)
    midpoint = max(1, fragment_count // 2)
    early_words = set(_REASONING_WORD_RE.findall(" ".join(normalized[:midpoint])))
    late_words = set(_REASONING_WORD_RE.findall(" ".join(normalized[midpoint:])))
    words = early_words | late_words
    novel_word_count = len(late_words - early_words)
    novel_word_ratio = novel_word_count / max(1, len(late_words))
    progress = (
        fragment_count >= _REASONING_PROGRESS_MIN_FRAGMENTS
        and unique_ratio >= _REASONING_PROGRESS_MIN_UNIQUE_RATIO
        and len(words) >= _REASONING_PROGRESS_MIN_DISTINCT_WORDS
        and novel_word_count >= _REASONING_PROGRESS_MIN_NOVEL_WORDS
    )
    return _ReasoningProgressAssessment(
        progress=progress,
        fragment_count=fragment_count,
        unique_ratio=unique_ratio,
        distinct_word_count=len(words),
        novel_word_count=novel_word_count,
        novel_word_ratio=novel_word_ratio,
    )


def _reasoning_fragments_show_progress(fragments: Iterable[str]) -> bool:
    return _assess_reasoning_fragments_progress(fragments).progress


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _reasoning_only_limits(tools: list[dict[str, Any]]) -> tuple[float, int]:
    if not tools:
        return _REASONING_ONLY_MAX_SECONDS, _REASONING_ONLY_MAX_CHUNKS
    return (
        min(_REASONING_ONLY_MAX_SECONDS, _REASONING_ONLY_TOOL_MAX_SECONDS),
        min(_REASONING_ONLY_MAX_CHUNKS, _REASONING_ONLY_TOOL_MAX_CHUNKS),
    )


def _build_reasoning_only_nudge(tools: list[dict[str, Any]], *, phase: str = "") -> str:
    names = _tool_names(tools)
    base = (
        "The prior response stream spent too long in reasoning without producing assistant content "
        "or a tool call."
    )
    if phase == "repair":
        if "escalate_to_bigger_model" in names:
            return (
                f"{base} You are in REPAIR phase. Stop analyzing and emit ONE concrete action now: "
                "call file_patch/file_write/ast_patch to fix the code, call shell_exec to re-run the verifier, "
                "or call escalate_to_bigger_model if you are stuck. Do not continue hidden reasoning only."
            )
        return (
            f"{base} You are in REPAIR phase. Stop analyzing and emit ONE concrete action now: "
            "call file_patch/file_write/ast_patch to fix the code, or call shell_exec to re-run the verifier. "
            "Do not continue hidden reasoning only."
        )
    if "escalate_to_bigger_model" in names:
        return (
            f"{base} Continue now with a concrete action: call file/read tools if you can make "
            "progress, or call escalate_to_bigger_model if you are stuck or need stronger reasoning. "
            "Do not continue hidden reasoning only."
        )
    return (
        f"{base} Continue now by calling an appropriate available tool, or answer directly if "
        "no tool is needed. Do not continue hidden reasoning only."
    )


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
    run_logger = getattr(harness, "run_logger", None)
    set_trace_id = getattr(run_logger, "set_trace_id", None)
    if callable(set_trace_id):
        thread_id = str(getattr(harness.state, "thread_id", "") or getattr(harness, "conversation_id", "") or "run")
        set_trace_id(f"{thread_id}:{getattr(harness.state, 'step_count', 0)}")

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
    reasoning_only_retries = 0

    for _model_attempt in range(_CHUNK_ERROR_MAX_RETRIES + 1):
        try:
            chunks = []
            stream_state = StreamTagState()
            _retry_immediately = False
            _stop_after_reasoning_only_stall = False
            attempt_started_at = time.monotonic()
            reasoning_only_chunks = 0
            reasoning_only_max_seconds, reasoning_only_max_chunks = _reasoning_only_limits(tools)
            reasoning_only_base_seconds = reasoning_only_max_seconds
            reasoning_only_base_chunks = reasoning_only_max_chunks
            reasoning_only_progress_deferred = False
            reasoning_only_fragments: deque[str] = deque(maxlen=512)
            saw_assistant_content = False
            saw_tool_call = False
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
                        last_chunks=details.get("last_chunks", []),
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
                    saw_tool_call = saw_tool_call or chunk_contains_tool_call_delta(event.get("data", {}))
                    saw_assistant_content = saw_assistant_content or _chunk_has_assistant_content(event)
                    reasoning_text = _chunk_reasoning_text(event)
                    if reasoning_text and not saw_assistant_content and not saw_tool_call:
                        reasoning_only_chunks += 1
                        reasoning_only_fragments.append(reasoning_text)
                    if (
                        tools
                        and not saw_assistant_content
                        and not saw_tool_call
                        and (
                            reasoning_only_chunks >= reasoning_only_max_chunks
                            or (time.monotonic() - attempt_started_at) >= reasoning_only_max_seconds
                        )
                    ):
                        elapsed_seconds = round(time.monotonic() - attempt_started_at, 3)
                        tool_names = _tool_names(tools)
                        hard_max_seconds = max(reasoning_only_max_seconds, _REASONING_ONLY_MAX_SECONDS)
                        hard_max_chunks = max(reasoning_only_max_chunks, _REASONING_ONLY_MAX_CHUNKS)
                        progress_assessment = _assess_reasoning_fragments_progress(reasoning_only_fragments)
                        progress_details = progress_assessment.to_log_dict()
                        if (
                            not reasoning_only_progress_deferred
                            and (
                                reasoning_only_max_seconds < hard_max_seconds
                                or reasoning_only_max_chunks < hard_max_chunks
                            )
                            and progress_assessment.progress
                        ):
                            reasoning_only_progress_deferred = True
                            reasoning_only_max_seconds = hard_max_seconds
                            reasoning_only_max_chunks = hard_max_chunks
                            harness._runlog(
                                "reasoning_only_stream_progress_defer",
                                "reasoning-only stream is still changing; extending to hard guard budget",
                                attempt=_model_attempt + 1,
                                reasoning_only_chunks=reasoning_only_chunks,
                                elapsed_seconds=elapsed_seconds,
                                base_max_seconds=reasoning_only_base_seconds,
                                base_max_chunks=reasoning_only_base_chunks,
                                hard_max_seconds=hard_max_seconds,
                                hard_max_chunks=hard_max_chunks,
                                tool_count=len(tools),
                                tools_available=tool_names,
                                **progress_details,
                            )
                        else:
                            if reasoning_only_retries >= _REASONING_ONLY_MAX_RETRIES:
                                stream_ended_without_done = True
                                stream_ended_without_done_details = {
                                    "reason": "reasoning_only_stream_stall",
                                    "retrying": False,
                                    "attempt": _model_attempt + 1,
                                    "reasoning_only_chunks": reasoning_only_chunks,
                                    "elapsed_seconds": elapsed_seconds,
                                    "tools_available": tool_names,
                                    "progress": progress_details,
                                }
                                harness.state.scratchpad["_last_reasoning_only_stall"] = dict(
                                    stream_ended_without_done_details
                                )
                                harness._runlog(
                                    "reasoning_only_stream_halt",
                                    "halting model call after repeated reasoning-only stream stall",
                                    attempt=_model_attempt + 1,
                                    reasoning_only_chunks=reasoning_only_chunks,
                                    elapsed_seconds=elapsed_seconds,
                                    tool_count=len(tools),
                                    tools_available=tool_names,
                                    **progress_details,
                                )
                                await harness._emit(
                                    deps.event_handler,
                                    UIEvent(
                                        event_type=UIEventType.ALERT,
                                        content=(
                                            "Model stream stayed in reasoning after recovery; handing off to "
                                            "halt recovery/escalation logic."
                                        ),
                                        data=stream_ended_without_done_details,
                                    ),
                                )
                                _stop_after_reasoning_only_stall = True
                                break
                            reasoning_only_retries += 1
                            current_phase = str(getattr(harness.state, "current_phase", "") or "").strip().lower()
                            nudge = _build_reasoning_only_nudge(tools, phase=current_phase)
                            messages = list(messages) + [ConversationMessage(role="system", content=nudge).to_dict()]
                            harness.state.scratchpad["_last_reasoning_only_retry"] = {
                                "attempt": _model_attempt + 1,
                                "reasoning_only_chunks": reasoning_only_chunks,
                                "elapsed_seconds": elapsed_seconds,
                                "tools_available": tool_names,
                                "progress": progress_details,
                            }
                            harness._runlog(
                                "reasoning_only_stream_retry",
                                "retrying model call after reasoning-only stream stall",
                                attempt=_model_attempt + 1,
                                reasoning_only_chunks=reasoning_only_chunks,
                                elapsed_seconds=elapsed_seconds,
                                tool_count=len(tools),
                                tools_available=tool_names,
                                **progress_details,
                            )
                            await harness._emit(
                                deps.event_handler,
                                UIEvent(
                                    event_type=UIEventType.ALERT,
                                    content="Model stream stalled in reasoning; retrying with a tool/action nudge.",
                                    data={
                                        "reason": "reasoning_only_stream_stall",
                                        "retrying": True,
                                        "reasoning_only_chunks": reasoning_only_chunks,
                                        "elapsed_seconds": elapsed_seconds,
                                    },
                                ),
                            )
                            _retry_immediately = True
                            break
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
            if _stop_after_reasoning_only_stall:
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
            is_api = error_type in ("provider", "content_policy_violation")
            if error_type == "content_policy_violation":
                content_prefix = "Content policy violation"
            else:
                content_prefix = "Provider error" if is_api else "Stream error"

            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ERROR,
                    content=f"{content_prefix}: {exc}",
                    data={"is_api_error": is_api, "error_type": error_type, "details": details},
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
    if match is not None:
        return int(match.group(1)), int(match.group(2))
    if details.get("context_overflow") is True:
        try:
            request_tokens = int(details.get("request_tokens") or 0)
            context_limit = int(details.get("context_limit") or 0)
        except Exception:
            return None
        if request_tokens > 0 and context_limit > 0:
            return request_tokens, context_limit
    return None
