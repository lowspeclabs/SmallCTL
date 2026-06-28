from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import time
from typing import Any, Iterable

from ..client import StreamResult
from ..client.chunk_parser import chunk_contains_tool_call_delta, normalize_sentencepiece_whitespace
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from .deps import GraphRuntimeDeps
from .state import GraphRunState
from .tool_model_rules import _model_is_gemma_4, _model_is_lfm25_8b_a1b
from .model_stream_fallback_support import (
    _classify_model_call_error,
)
from .model_stream_loop_rendering import flush_model_stream_buffer
from .model_stream_loop_rendering import handle_model_stream_chunk
from .model_stream_loop_rendering import StreamTagState
from .model_stream_loop_recovery import handle_model_stream_chunk_error

_REASONING_ONLY_MAX_SECONDS = 60.0
_REASONING_ONLY_MAX_CHUNKS = 4096
_REASONING_ONLY_TOOL_MAX_SECONDS = 25.0
_REASONING_ONLY_TOOL_MAX_CHUNKS = 1500
_LFM25_REASONING_ONLY_TOOL_MAX_SECONDS = 12.0
_LFM25_REASONING_ONLY_TOOL_MAX_CHUNKS = 512
# Gemma-4 variants (including 12b) often emit long reasoning traces before
# producing a tool call, especially when recovering from a prior tool failure.
# Give them a wider chunk/time budget so the harness can nudge them into action
# instead of halting immediately.
_GEMMA4_REASONING_ONLY_TOOL_MAX_SECONDS = 40.0
_GEMMA4_REASONING_ONLY_TOOL_MAX_CHUNKS = 2500
_REASONING_ONLY_MAX_RETRIES = 1
_REASONING_ONLY_GEMMA4_MAX_RETRIES = 2
_REASONING_PROGRESS_MIN_FRAGMENTS = 4
_REASONING_PROGRESS_MIN_UNIQUE_RATIO = 0.35
_REASONING_PROGRESS_MIN_DISTINCT_WORDS = 12
_REASONING_PROGRESS_MIN_NOVEL_WORDS = 4
_REASONING_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")

# Degenerate-repetition guard: detect models that fall into a loop emitting the
# same short phrase/token many times within a single completion.
_DEGENERATE_REPETITION_MIN_REPEAT = 6
_DEGENERATE_REPETITION_WINDOW_CHARS = 400
_DEGENERATE_REPETITION_MIN_REPEATED_PHRASE_LENGTH = 4


def _next_model_call_trace_id(harness: Any) -> str:
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        if state is not None:
            state.scratchpad = scratchpad

    call_sequence = int(scratchpad.get("_model_call_sequence", 0) or 0) + 1
    scratchpad["_model_call_sequence"] = call_sequence
    thread_id = str(
        getattr(state, "thread_id", "")
        or getattr(harness, "conversation_id", "")
        or "run"
    ).strip()
    task_id = str(scratchpad.get("_active_task_id") or scratchpad.get("_task_sequence") or "task").strip()
    step_count = int(getattr(state, "step_count", 0) or 0)
    return f"{thread_id}:{task_id}:step-{step_count}:call-{call_sequence}"


class _ModelOutputDegenerate(Exception):
    """Raised when the model stream is detected to be a degenerate repetition loop."""

    def __init__(self, *, repeated_phrase: str, repeat_count: int, window: str) -> None:
        self.repeated_phrase = repeated_phrase
        self.repeat_count = repeat_count
        self.window = window
        super().__init__(f"Degenerate repetition detected: {repeated_phrase!r} x{repeat_count}")


def _phrase_is_semantic_repetition(phrase: str) -> bool:
    """Return True when a repeated phrase carries enough semantic content to be a loop signal.

    Very short phrases that are pure markdown formatting (e.g. code-fence transitions,
    bullet markers) can appear many times in legitimate structured output. Require at
    least one alphanumeric character for short phrases so real list formatting is not
    misclassified as a degenerate loop.
    """
    if len(phrase) >= 16:
        return True
    return bool(re.search(r"[a-z0-9]", phrase, re.IGNORECASE))


def _detect_degenerate_repetition(buffer: str) -> tuple[str, int, str] | None:
    """Return (phrase, count, window) if buffer contains a repetitive phrase loop."""
    text = str(buffer or "")
    if len(text) < _DEGENERATE_REPETITION_WINDOW_CHARS:
        return None
    window = text[-_DEGENERATE_REPETITION_WINDOW_CHARS:]
    lowered = window.lower()
    # Try progressively shorter phrase windows, looking for a substring that
    # appears many times consecutively or near-consecutively.
    for phrase_len in range(80, _DEGENERATE_REPETITION_MIN_REPEATED_PHRASE_LENGTH - 1, -1):
        if phrase_len * 2 > len(lowered):
            continue
        start = 0
        end = phrase_len
        while end <= len(lowered):
            phrase = lowered[start:end]
            # Skip whitespace-only or very low-entropy phrases
            if len(phrase.strip()) < _DEGENERATE_REPETITION_MIN_REPEATED_PHRASE_LENGTH:
                start += 1
                end += 1
                continue
            # Skip short pure-formatting phrases that occur in normal markdown lists
            if not _phrase_is_semantic_repetition(phrase):
                start += 1
                end += 1
                continue
            # Count non-overlapping occurrences in the window
            indices = []
            scan = 0
            while True:
                idx = lowered.find(phrase, scan)
                if idx == -1:
                    break
                indices.append(idx)
                scan = idx + len(phrase)
            
            count = len(indices)
            if count >= _DEGENERATE_REPETITION_MIN_REPEAT:
                # Require density (total length of occurrences over their total span)
                # to be high (e.g. >= 0.70) to ensure the repetitions are consecutive
                # or near-consecutive, preventing false positives on lists or common terms.
                span = (indices[-1] + len(phrase)) - indices[0]
                if span > 0 and (count * len(phrase)) / span >= 0.70:
                    return phrase, count, window
            start += 1
            end += 1
    return None


def _trim_degenerate_suffix(buffer: str, window: str, repeated_phrase: str) -> str:
    """Return the portion of buffer before the repetitive suffix, if any."""
    text = str(buffer or "")
    if not text:
        return text
    # Remove the trailing window that contained the loop; this is the most
    # conservative trim and avoids leaving dangling repeated fragments.
    if window and text.endswith(window):
        return text[: -len(window)].rstrip()
    # Fallback: remove everything from the last occurrence of the phrase.
    phrase = str(repeated_phrase or "").lower()
    if phrase:
        lowered = text.lower()
        last_idx = lowered.rfind(phrase)
        if last_idx > 0:
            return text[:last_idx].rstrip()
    return text


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
    return False


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




def _partial_tool_call_summaries(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    partials: dict[int, dict[str, Any]] = {}
    for event in chunks:
        if not isinstance(event, dict) or event.get("type") != "chunk":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or {}
        if not isinstance(delta, dict):
            continue
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            try:
                index = int(call.get("index") or 0)
            except (TypeError, ValueError):
                index = 0
            item = partials.setdefault(index, {"index": index, "tool_name": "", "argument_chars": 0})
            function = call.get("function") or {}
            if isinstance(function, dict):
                name = str(function.get("name") or "").strip()
                if name:
                    item["tool_name"] = name
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    item["argument_chars"] = int(item.get("argument_chars", 0) or 0) + len(arguments)
            call_id = str(call.get("id") or "").strip()
            if call_id:
                item["tool_call_id"] = call_id
    return [item for item in partials.values() if item.get("tool_name") or item.get("argument_chars")]


async def _emit_cancelled_partial_tool_calls(
    *,
    harness: Any,
    deps: GraphRuntimeDeps,
    chunks: list[dict[str, Any]],
) -> None:
    partials = _partial_tool_call_summaries(chunks)
    if not partials:
        return
    first = partials[0]
    tool_name = str(first.get("tool_name") or "tool").strip()
    argument_chars = int(first.get("argument_chars", 0) or 0)
    harness._runlog(
        "partial_tool_call_cancelled",
        "model stream contained a partial tool call when cancellation was requested",
        tool_name=tool_name,
        argument_chars=argument_chars,
        partial_tool_calls=partials,
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.SYSTEM,
            content=(
                f"Model had started streaming `{tool_name}` when the run was cancelled "
                f"({argument_chars} argument chars received, not dispatched)."
            ),
            data={
                "ui_kind": "partial_tool_call_cancelled",
                "event": "partial_tool_call_cancelled",
                "tool_name": tool_name,
                "argument_chars": argument_chars,
                "partial_tool_calls": partials,
            },
        ),
    )

def _reasoning_only_limits(
    tools: list[dict[str, Any]],
    *,
    model_name: str | None = None,
) -> tuple[float, int]:
    if not tools:
        return _REASONING_ONLY_MAX_SECONDS, _REASONING_ONLY_MAX_CHUNKS
    if _model_is_lfm25_8b_a1b(model_name):
        return (
            min(_REASONING_ONLY_MAX_SECONDS, _LFM25_REASONING_ONLY_TOOL_MAX_SECONDS),
            min(_REASONING_ONLY_MAX_CHUNKS, _LFM25_REASONING_ONLY_TOOL_MAX_CHUNKS),
        )
    if _model_is_gemma_4(model_name):
        return (
            min(_REASONING_ONLY_MAX_SECONDS, _GEMMA4_REASONING_ONLY_TOOL_MAX_SECONDS),
            min(_REASONING_ONLY_MAX_CHUNKS, _GEMMA4_REASONING_ONLY_TOOL_MAX_CHUNKS),
        )
    return (
        min(_REASONING_ONLY_MAX_SECONDS, _REASONING_ONLY_TOOL_MAX_SECONDS),
        min(_REASONING_ONLY_MAX_CHUNKS, _REASONING_ONLY_TOOL_MAX_CHUNKS),
    )


def _build_reasoning_only_nudge(tools: list[dict[str, Any]], *, phase: str = "", harness: Any = None) -> str:
    names = _tool_names(tools)
    base = (
        "The prior response stream spent too long in reasoning without producing assistant content "
        "or a tool call."
    )
    context_hint = _build_reasoning_only_context_hint(harness)
    if context_hint:
        base = f"{base} {context_hint}"
    ssh_actions = [
        name for name in ("ssh_exec", "ssh_file_read", "ssh_file_write") if name in names
    ]
    if phase == "repair":
        if ssh_actions:
            action_list = "/".join(ssh_actions)
            return (
                f"{base} You are in REPAIR phase for a remote target. Stop analyzing and emit ONE concrete "
                f"remote action now: call {action_list} to inspect or repair the remote host, or call "
                "task_complete only if the objective is fully verified. Do not continue hidden reasoning only."
            )
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
    if ssh_actions:
        action_list = "/".join(ssh_actions)
        return (
            f"{base} Continue now with one concrete remote action: call {action_list}, or call "
            "task_complete only if the objective is fully verified. Do not continue hidden reasoning only."
        )
    return (
        f"{base} Continue now by calling an appropriate available tool, or answer directly if "
        "no tool is needed. Do not continue hidden reasoning only."
    )


def _build_reasoning_only_context_hint(harness: Any) -> str:
    """Return a short, actionable hint from the most recent error/tool failure."""
    state = getattr(harness, "state", None)
    if state is None:
        return ""
    recent_errors = getattr(state, "recent_errors", None) or []
    if not recent_errors:
        return ""
    last_error = str(recent_errors[-1]).strip()
    if not last_error:
        return ""
    # Keep the hint concise; long hints consume context and can confuse small models.
    snipped = last_error[:220]
    if len(last_error) > 220:
        snipped = snipped + "..."
    return (
        f"The most recent issue was: {snipped} "
        "Address it with your very next action; do not re-analyze."
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
    suppress_ui_events: bool = False,
) -> dict[str, Any]:
    run_logger = getattr(harness, "run_logger", None)
    set_trace_id = getattr(run_logger, "set_trace_id", None)
    if callable(set_trace_id):
        trace_id = _next_model_call_trace_id(harness)
        set_trace_id(trace_id)
        if hasattr(run_logger, "set_task_id"):
            parts = trace_id.split(":")
            if len(parts) >= 2:
                run_logger.set_task_id(parts[1])
        if hasattr(run_logger, "set_step_count"):
            run_logger.set_step_count(getattr(harness.state, "step_count", 0))
        if hasattr(run_logger, "set_call_count"):
            call_part = trace_id.split(":")[-1]
            if call_part.startswith("call-"):
                try:
                    run_logger.set_call_count(int(call_part[5:]))
                except ValueError:
                    pass

    if run_logger is not None and hasattr(run_logger, "handle_debug_signal"):
        signal_path = Path(getattr(harness.state, "cwd", ".") or ".") / ".smallctl" / "debug-signal"
        run_logger.handle_debug_signal(signal_path)

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
            active_model_name = str(getattr(getattr(harness, "client", None), "model", "") or "")
            lfm25_reasoning_guard = _model_is_lfm25_8b_a1b(active_model_name)
            gemma4_reasoning_guard = _model_is_gemma_4(active_model_name)
            reasoning_only_max_seconds, reasoning_only_max_chunks = _reasoning_only_limits(
                tools,
                model_name=active_model_name,
            )
            # Shrink the reasoning-only window on each retry. A model that already
            # stalled once is unlikely to need the full initial budget again;
            # tightening the limit keeps the TUI responsive and avoids long waits
            # before halting/escalation.
            _reasoning_retry_scale = 0.5 ** reasoning_only_retries
            reasoning_only_max_seconds = max(
                1.0, reasoning_only_max_seconds * _reasoning_retry_scale
            )
            reasoning_only_max_chunks = max(
                1, int(reasoning_only_max_chunks * _reasoning_retry_scale)
            )
            max_reasoning_only_retries = (
                _REASONING_ONLY_GEMMA4_MAX_RETRIES
                if gemma4_reasoning_guard
                else _REASONING_ONLY_MAX_RETRIES
            )
            reasoning_only_base_seconds = reasoning_only_max_seconds
            reasoning_only_base_chunks = reasoning_only_max_chunks
            reasoning_only_progress_deferred = False
            reasoning_only_fragments: deque[str] = deque(maxlen=512)
            saw_assistant_content = False
            saw_tool_call = False
            assistant_text_buffer = ""
            reasoning_text_buffer = ""
            partial_assistant_text = ""
            async for event in harness.client.stream_chat(messages=messages, tools=tools):
                if harness._cancel_requested:
                    await _emit_cancelled_partial_tool_calls(
                        harness=harness,
                        deps=deps,
                        chunks=chunks,
                    )
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
                        "partial_assistant_text": "",
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
                        "partial_assistant_text": "",
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
                        hard_max_seconds = max(
                            reasoning_only_max_seconds,
                            _REASONING_ONLY_MAX_SECONDS * _reasoning_retry_scale,
                        )
                        hard_max_chunks = max(
                            reasoning_only_max_chunks,
                            int(_REASONING_ONLY_MAX_CHUNKS * _reasoning_retry_scale),
                        )
                        progress_assessment = _assess_reasoning_fragments_progress(reasoning_only_fragments)
                        progress_details = progress_assessment.to_log_dict()
                        if (
                            not reasoning_only_progress_deferred
                            and (
                                reasoning_only_max_seconds < hard_max_seconds
                                or reasoning_only_max_chunks < hard_max_chunks
                            )
                            and progress_assessment.progress
                            and not lfm25_reasoning_guard
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
                            if reasoning_only_retries >= max_reasoning_only_retries:
                                stream_ended_without_done = True
                                stream_ended_without_done_details = {
                                    "reason": "reasoning_only_stream_stall",
                                    "retrying": False,
                                    "attempt": _model_attempt + 1,
                                    "reasoning_only_chunks": reasoning_only_chunks,
                                    "elapsed_seconds": elapsed_seconds,
                                    "tools_available": tool_names,
                                    "progress": progress_details,
                                    "max_reasoning_only_retries": max_reasoning_only_retries,
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
                            nudge = _build_reasoning_only_nudge(tools, phase=current_phase, harness=harness)
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
                        suppress_ui_events=suppress_ui_events,
                    )
                    delta = _chunk_delta(event)
                    content = delta.get("content")
                    if isinstance(content, str):
                        assistant_text_buffer += normalize_sentencepiece_whitespace(content)
                        repetition = _detect_degenerate_repetition(assistant_text_buffer)
                        if repetition is not None:
                            phrase, count, window = repetition
                            raise _ModelOutputDegenerate(
                                repeated_phrase=phrase,
                                repeat_count=count,
                                window=window,
                            )
                    reasoning_text = _chunk_reasoning_text(event)
                    if isinstance(reasoning_text, str) and reasoning_text:
                        reasoning_text_buffer += normalize_sentencepiece_whitespace(reasoning_text)
                        repetition = _detect_degenerate_repetition(reasoning_text_buffer)
                        if repetition is not None:
                            phrase, count, window = repetition
                            raise _ModelOutputDegenerate(
                                repeated_phrase=phrase,
                                repeat_count=count,
                                window=window,
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
                    suppress_ui_events=suppress_ui_events,
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
        except _ModelOutputDegenerate as exc:
            harness._runlog(
                "model_output_degenerate_loop",
                "detected degenerate repetition loop in model output",
                repeated_phrase=exc.repeated_phrase,
                repeat_count=exc.repeat_count,
                buffer_chars=len(assistant_text_buffer),
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content="Model output entered a repetition loop; halting this turn and requesting recovery.",
                    data={
                        "ui_kind": "model_output_degenerate_loop_exhausted",
                        "event": "model_output_degenerate_loop_exhausted",
                        "reason": "model_output_degenerate_loop",
                        "repeated_phrase": exc.repeated_phrase,
                        "repeat_count": exc.repeat_count,
                    },
                ),
            )
            # Salvage any assistant text produced before the repetitive suffix so
            # that a partial inline tool call (e.g. an ssh_exec JSON block that
            # degenerated while escaping a string) can still be parsed and acted
            # on instead of being discarded as an empty/action-stall turn.
            partial_assistant_text = _trim_degenerate_suffix(
                assistant_text_buffer,
                exc.window,
                exc.repeated_phrase,
            )
            stream_ended_without_done = True
            stream_ended_without_done_details = {
                "reason": "model_output_degenerate_loop",
                "repeated_phrase": exc.repeated_phrase,
                "repeat_count": exc.repeat_count,
                "buffer_chars": len(assistant_text_buffer),
                "partial_assistant_text": partial_assistant_text,
            }
            harness.state.scratchpad["_last_stream_halted_without_done"] = True
            harness.state.scratchpad["_last_stream_halt_reason"] = "model_output_degenerate_loop"
            harness.state.scratchpad["_last_stream_halt_details"] = dict(stream_ended_without_done_details)
            break
        except asyncio.CancelledError:
            await _emit_cancelled_partial_tool_calls(
                harness=harness,
                deps=deps,
                chunks=chunks,
            )
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
                "partial_assistant_text": "",
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
        "partial_assistant_text": partial_assistant_text,
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
