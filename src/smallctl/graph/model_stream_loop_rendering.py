from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Literal

from ..models.events import UIEvent, UIEventType
from .model_stream_fallback_recovery import _with_speaker


@dataclass
class _WrapperFrame:
    kind: Literal["assistant", "thinking"]
    end_tag: str


@dataclass
class _StreamBatchState:
    kind: Literal["assistant", "thinking"] | None = None
    text: str = ""
    last_emit_at: float = field(default_factory=time.monotonic)


@dataclass
class StreamTagState:
    pending: str = ""
    wrapper_stack: list[_WrapperFrame] = field(default_factory=list)
    field_reasoning_seen: bool = False
    batch: _StreamBatchState = field(default_factory=_StreamBatchState)


def _stream_wrapper_pairs(
    *,
    start_tag: str,
    end_tag: str,
) -> list[tuple[str, str, Literal["assistant", "thinking"]]]:
    pairs: list[tuple[str, str, Literal["assistant", "thinking"]]] = [
        (start_tag, end_tag, "thinking"),
        ("<analysis>", "</analysis>", "thinking"),
        ("<plan>", "</plan>", "thinking"),
        ("<response>", "</response>", "assistant"),
        ("<execution>", "</execution>", "assistant"),
    ]
    if start_tag == "<think>" and end_tag == "</think>":
        pairs.append(("<thinking>", "</thinking>", "thinking"))

    seen: set[tuple[str, str, str]] = set()
    unique_pairs: list[tuple[str, str, Literal["assistant", "thinking"]]] = []
    for start, end, kind in pairs:
        key = (start.lower(), end.lower(), kind)
        if not start or not end or key in seen:
            continue
        seen.add(key)
        unique_pairs.append((start, end, kind))
    return unique_pairs


def _split_partial_tag_suffix(text: str, candidate_tags: list[str]) -> tuple[str, str]:
    if not text:
        return "", ""
    max_suffix = min(len(text), max((len(tag) for tag in candidate_tags), default=0) - 1)
    lowered = text.lower()
    for length in range(max_suffix, 0, -1):
        suffix = lowered[-length:]
        if any(tag.startswith(suffix) for tag in candidate_tags):
            return text[:-length], text[-length:]
    return text, ""


async def _emit_stream_text(
    *,
    harness: Any,
    deps: Any,
    kind: Literal["assistant", "thinking"],
    text: str,
    echo_to_stdout: bool,
    stream_state: StreamTagState,
    suppress_duplicate_thinking: bool = False,
) -> None:
    if not text:
        return
    if kind == "assistant" and echo_to_stdout:
        harness._stream_print(text)
    if kind == "thinking":
        harness._runlog("model_token", "thinking token", token=text)
    else:
        harness._runlog("model_token", "assistant token", token=text)
    if kind == "thinking":
        if suppress_duplicate_thinking:
            return
    batch = stream_state.batch
    if batch.kind is not None and batch.kind != kind:
        await _flush_stream_batch(harness=harness, deps=deps, stream_state=stream_state)
    batch = stream_state.batch
    batch.kind = kind
    batch.text += text
    if len(batch.text) >= 512 or (time.monotonic() - batch.last_emit_at) >= 0.05:
        await _flush_stream_batch(harness=harness, deps=deps, stream_state=stream_state)


async def _flush_stream_batch(
    *,
    harness: Any,
    deps: Any,
    stream_state: StreamTagState,
) -> None:
    batch = stream_state.batch
    if batch.kind is None or not batch.text:
        return
    text = batch.text
    kind = batch.kind
    batch.kind = None
    batch.text = ""
    batch.last_emit_at = time.monotonic()
    if kind == "thinking":
        if harness.thinking_visibility:
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.THINKING,
                    content=text,
                    data=_with_speaker(harness),
                ),
            )
        return
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ASSISTANT,
            content=text,
            data=_with_speaker(harness),
        ),
    )


async def _route_stream_content(
    *,
    harness: Any,
    deps: Any,
    text: str,
    start_tag: str,
    end_tag: str,
    echo_to_stdout: bool,
    stream_state: StreamTagState,
) -> None:
    wrapper_pairs = _stream_wrapper_pairs(start_tag=start_tag, end_tag=end_tag)
    start_map = [(start.lower(), end, kind) for start, end, kind in wrapper_pairs]
    all_tags = [tag.lower() for start, end, _kind in wrapper_pairs for tag in (start, end)]

    pending = f"{stream_state.pending}{text}"
    stream_state.pending = ""
    while pending:
        lowered = pending.lower()
        current_frame = stream_state.wrapper_stack[-1] if stream_state.wrapper_stack else None
        current_kind: Literal["assistant", "thinking"] = current_frame.kind if current_frame is not None else "assistant"
        candidates: list[tuple[int, int, int, str, str | None, Literal["assistant", "thinking"] | None]] = []

        if current_frame is not None:
            current_end = current_frame.end_tag.lower()
            end_index = lowered.find(current_end)
            if end_index != -1:
                candidates.append((end_index, 0, -len(current_end), "end", current_end, None))
        else:
            for _start, end, _kind in wrapper_pairs:
                end_lower = end.lower()
                end_index = lowered.find(end_lower)
                if end_index != -1:
                    candidates.append((end_index, 2, -len(end_lower), "stray_end", end_lower, None))

        for start_lower, end, kind in start_map:
            start_index = lowered.find(start_lower)
            if start_index != -1:
                candidates.append((start_index, 1, -len(start_lower), "start", start_lower, kind))

        if not candidates:
            emittable, suffix = _split_partial_tag_suffix(pending, all_tags)
            await _emit_stream_text(
                harness=harness,
                deps=deps,
                kind=current_kind,
                text=emittable,
                echo_to_stdout=echo_to_stdout,
                stream_state=stream_state,
                suppress_duplicate_thinking=current_kind == "thinking" and stream_state.field_reasoning_seen,
            )
            stream_state.pending = suffix
            return

        index, _priority, _neg_len, action, matched_tag, next_kind = min(candidates)
        prefix = pending[:index]
        await _emit_stream_text(
            harness=harness,
            deps=deps,
            kind=current_kind,
            text=prefix,
            echo_to_stdout=echo_to_stdout,
            stream_state=stream_state,
            suppress_duplicate_thinking=current_kind == "thinking" and stream_state.field_reasoning_seen,
        )
        pending = pending[index + len(matched_tag):]

        if action == "start" and next_kind is not None:
            matching_end = ""
            for start, end, kind in wrapper_pairs:
                if start.lower() == matched_tag and kind == next_kind:
                    matching_end = end
                    break
            if matching_end:
                stream_state.wrapper_stack.append(_WrapperFrame(kind=next_kind, end_tag=matching_end))
        elif action == "end" and stream_state.wrapper_stack:
            stream_state.wrapper_stack.pop()


async def handle_model_stream_chunk(
    *,
    harness: Any,
    deps: Any,
    event: dict[str, Any],
    start_tag: str,
    end_tag: str,
    echo_to_stdout: bool,
    chunks: list[dict[str, Any]],
    stream_state: StreamTagState,
    first_token_time: float | None,
) -> tuple[StreamTagState, float | None]:
    data = event.get("data", {})
    choices = data.get("choices") or []
    if not choices:
        chunks.append(event)
        return stream_state, first_token_time

    delta = choices[0].get("delta", {})
    reason_field = delta.get("reasoning_content") or delta.get("reasoning")
    content_field = delta.get("content")

    if (content_field or reason_field) and first_token_time is None:
        first_token_time = time.perf_counter()

    if reason_field:
        stream_state.field_reasoning_seen = True
        await _emit_stream_text(
            harness=harness,
            deps=deps,
            kind="thinking",
            text=str(reason_field),
            echo_to_stdout=echo_to_stdout,
            stream_state=stream_state,
            suppress_duplicate_thinking=False,
        )

    if content_field:
        await _route_stream_content(
            harness=harness,
            deps=deps,
            text=str(content_field),
            start_tag=start_tag,
            end_tag=end_tag,
            echo_to_stdout=echo_to_stdout,
            stream_state=stream_state,
        )

    chunks.append(event)
    return stream_state, first_token_time


async def flush_model_stream_buffer(
    *,
    harness: Any,
    deps: Any,
    stream_state: StreamTagState,
    start_tag: str = "<think>",
    end_tag: str = "</think>",
    echo_to_stdout: bool,
) -> None:
    if stream_state.pending:
        known_tags = [
            tag.lower()
            for start, end, _kind in _stream_wrapper_pairs(start_tag=start_tag, end_tag=end_tag)
            for tag in (start, end)
        ]
        pending_lower = stream_state.pending.lower()
        if any(tag.startswith(pending_lower) for tag in known_tags):
            stream_state.pending = ""
        else:
            current_kind: Literal["assistant", "thinking"] = (
                stream_state.wrapper_stack[-1].kind if stream_state.wrapper_stack else "assistant"
            )
            await _emit_stream_text(
                harness=harness,
                deps=deps,
                kind=current_kind,
                text=stream_state.pending,
                echo_to_stdout=echo_to_stdout,
                stream_state=stream_state,
                suppress_duplicate_thinking=current_kind == "thinking" and stream_state.field_reasoning_seen,
            )
            stream_state.pending = ""
    await _flush_stream_batch(harness=harness, deps=deps, stream_state=stream_state)
