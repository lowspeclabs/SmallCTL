from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import re

from .chunk_parser import (
    extract_content_fragments,
    extract_thinking_from_tags,
    find_protocol_control_marker,
    format_tool_call_text,
    max_thinking_tag_alias_length,
    maybe_parse_tool_args,
    merge_reasoning_text,
    normalize_sentencepiece_whitespace,
    normalize_thinking_tag_aliases,
    strip_protocol_control_markers,
)

_REASONING_HALLUCINATION_PATTERNS = [
    re.compile(r"<tool_call\b[^>]*>"),
    re.compile(r"<function\b[^>]*>"),
    re.compile(r"<call\b[^>]*>"),
    re.compile(r"</tool_call>"),
    re.compile(r"</function>"),
    re.compile(r"</call>"),
]


def _scrub_reasoning_hallucinations(text: str) -> str:
    """Strip unauthorized tool-call-like XML tokens that models hallucinate in reasoning."""
    cleaned = str(text or "")
    for pattern in _REASONING_HALLUCINATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


@dataclass
class StreamResult:
    """Result of collecting a stream."""
    assistant_text: str = ""
    thinking_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEntry:
    """Single entry in a collected timeline."""
    kind: Literal["assistant", "thinking", "tool_call"]
    content: str
    data: dict[str, Any] = field(default_factory=dict)


def _stream_chunk_data(item: dict[str, Any]) -> dict[str, Any] | None:
    """Return provider chunk data from either harness-wrapped or raw chunks."""
    if not isinstance(item, dict):
        return None
    if item.get("type") == "chunk":
        data = item.get("data", {})
        return data if isinstance(data, dict) else None
    if isinstance(item.get("choices"), list):
        return item
    return None


def collect_stream(
    chunks: list[dict[str, Any]],
    *,
    reasoning_mode: Literal["auto", "tags", "field", "off"] = "auto",
    thinking_start_tag: str = "<think>",
    thinking_end_tag: str = "</think>",
) -> StreamResult:
    """Collect chunks into a structured result."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}

    for item in chunks:
        data = _stream_chunk_data(item)
        if data is None:
            continue
        # Capture usage/model from the final chunk even when choices is empty.
        next_usage = data.get("usage")
        if isinstance(next_usage, dict):
            usage = next_usage
        backend_model = data.get("model")
        if isinstance(backend_model, str) and backend_model.strip():
            usage["_backend_model_name"] = backend_model.strip()
        choices = data.get("choices") or []
        if not choices:
            continue
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            continue
        delta = first_choice.get("delta", {})
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        for kind, fragment in extract_content_fragments(content):
            if kind == "thinking":
                reasoning_parts.append(fragment)
            else:
                text_parts.append(fragment)
        field_reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(field_reasoning, str) and field_reasoning:
            reasoning_parts.append(_scrub_reasoning_hallucinations(field_reasoning))

        tc_deltas = delta.get("tool_calls") or []
        for tc in tc_deltas:
            if not isinstance(tc, dict):
                continue
            try:
                idx = int(tc.get("index", 0))
            except (TypeError, ValueError):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue
            existing = tool_calls.setdefault(
                idx,
                {
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {"name": "", "arguments": ""},
                },
            )
            if "id" in tc:
                existing["id"] = tc["id"]
            if "name" in fn and isinstance(fn["name"], str):
                existing["function"]["name"] += normalize_sentencepiece_whitespace(fn["name"])
            if "arguments" in fn and isinstance(fn["arguments"], str):
                existing["function"]["arguments"] += normalize_sentencepiece_whitespace(fn["arguments"])

    assistant_text = "".join(text_parts)
    tag_assistant, tag_thinking = extract_thinking_from_tags(
        assistant_text,
        thinking_start_tag=thinking_start_tag,
        thinking_end_tag=thinking_end_tag,
    )
    field_thinking = "".join(reasoning_parts)

    thinking_text = ""
    if reasoning_mode == "tags":
        assistant_text = tag_assistant
        thinking_text = merge_reasoning_text(field_thinking, tag_thinking)
    elif reasoning_mode == "field":
        thinking_text = field_thinking
    elif reasoning_mode == "auto":
        assistant_text = tag_assistant
        thinking_text = merge_reasoning_text(field_thinking, tag_thinking)
    elif reasoning_mode == "off":
        thinking_text = merge_reasoning_text(field_thinking, tag_thinking)

    ordered_tool_calls = [tool_calls[i] for i in sorted(tool_calls.keys())]
    return StreamResult(
        assistant_text=assistant_text,
        thinking_text=thinking_text,
        tool_calls=ordered_tool_calls,
        usage=usage,
    )


class _TimelineCollector:
    """Collects stream chunks into a timeline of entries."""

    def __init__(
        self,
        *,
        reasoning_mode: Literal["auto", "tags", "field", "off"],
        thinking_start_tag: str,
        thinking_end_tag: str,
    ) -> None:
        self.reasoning_mode = reasoning_mode
        self.thinking_start_tag = thinking_start_tag
        self.thinking_end_tag = thinking_end_tag
        self.entries: list[TimelineEntry] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self._tool_entry_index: dict[int, int] = {}
        self._tag_pending = ""
        self._inside_thinking_tag = False
        self._max_tag_length = max_thinking_tag_alias_length(
            thinking_start_tag=thinking_start_tag,
            thinking_end_tag=thinking_end_tag,
        )
        self._auto_used_tag_thinking = False

    def feed(self, item: dict[str, Any]) -> None:
        data = _stream_chunk_data(item)
        if data is None:
            return
        choices = data.get("choices") or []
        if not choices:
            return
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return
        delta = first_choice.get("delta", {})
        if not isinstance(delta, dict):
            return

        content = delta.get("content")
        for kind, fragment in extract_content_fragments(content):
            if kind == "thinking":
                self._flush_pending_text()
                if self.reasoning_mode in {"field", "auto", "tags"}:
                    self._append_text("thinking", fragment)
            else:
                self._feed_content(fragment)

        field_reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(field_reasoning, str) and field_reasoning:
            self._flush_pending_text()
            if self.reasoning_mode in {"field", "auto", "tags"}:
                self._append_text("thinking", _scrub_reasoning_hallucinations(field_reasoning))

        tc_deltas = delta.get("tool_calls") or []
        if tc_deltas:
            self._flush_pending_text()
        for tc in tc_deltas:
            if not isinstance(tc, dict):
                continue
            try:
                idx = int(tc.get("index", 0))
            except (TypeError, ValueError):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue
            existing = self.tool_calls.setdefault(
                idx,
                {
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {"name": "", "arguments": ""},
                },
            )
            if "id" in tc:
                existing["id"] = tc["id"]
            if "name" in fn and isinstance(fn["name"], str):
                existing["function"]["name"] += normalize_sentencepiece_whitespace(fn["name"])
            if "arguments" in fn and isinstance(fn["arguments"], str):
                existing["function"]["arguments"] += normalize_sentencepiece_whitespace(fn["arguments"])
            self._upsert_tool_call_entry(idx)

    def finalize(self) -> list[TimelineEntry]:
        self._flush_pending_text()
        return self.entries

    def _feed_content(self, text: str) -> None:
        if self.reasoning_mode in {"field", "off"}:
            self._append_text("assistant", text)
            return
        self._feed_tagged_content(text)

    def _feed_tagged_content(self, text: str) -> None:
        pending = normalize_thinking_tag_aliases(
            self._tag_pending + text,
            thinking_start_tag=self.thinking_start_tag,
            thinking_end_tag=self.thinking_end_tag,
        )
        self._tag_pending = ""
        cursor = 0
        while cursor < len(pending):
            if self._inside_thinking_tag:
                end = pending.find(self.thinking_end_tag, cursor)
                control_marker = find_protocol_control_marker(pending, cursor)
                if control_marker is not None and (end == -1 or control_marker[0] < end):
                    control_start, control_length = control_marker
                    if control_start > cursor:
                        self._append_text("thinking", pending[cursor:control_start])
                        self._auto_used_tag_thinking = True
                    cursor = control_start + control_length
                    self._inside_thinking_tag = False
                    continue
                if end == -1:
                    safe_end = max(cursor, len(pending) - max(0, self._max_tag_length - 1))
                    if safe_end > cursor:
                        self._append_text("thinking", pending[cursor:safe_end])
                        self._auto_used_tag_thinking = True
                        cursor = safe_end
                    self._tag_pending = pending[cursor:]
                    return
                if end > cursor:
                    self._append_text("thinking", pending[cursor:end])
                    self._auto_used_tag_thinking = True
                cursor = end + len(self.thinking_end_tag)
                self._inside_thinking_tag = False
                continue

            start = pending.find(self.thinking_start_tag, cursor)
            if start == -1:
                safe_end = max(cursor, len(pending) - max(0, self._max_tag_length - 1))
                if safe_end > cursor:
                    self._append_text("assistant", pending[cursor:safe_end])
                    cursor = safe_end
                self._tag_pending = pending[cursor:]
                return
            if start > cursor:
                self._append_text("assistant", pending[cursor:start])
            cursor = start + len(self.thinking_start_tag)
            self._inside_thinking_tag = True

    def _append_text(self, kind: Literal["assistant", "thinking"], text: str) -> None:
        if not text:
            return
        if kind == "assistant":
            text = strip_protocol_control_markers(text)
            if not text:
                return
        if self.entries and self.entries[-1].kind == kind:
            self.entries[-1].content += text
            return
        self.entries.append(TimelineEntry(kind=kind, content=text))

    def _upsert_tool_call_entry(self, idx: int) -> None:
        payload = self.tool_calls[idx]
        function = payload.get("function", {})
        if not isinstance(function, dict):
            return
        tool_name = str(function.get("name") or "").strip() or "tool_call"
        args_text = str(function.get("arguments") or "")
        args = maybe_parse_tool_args(args_text)
        data: dict[str, Any] = {
            "tool_call_id": payload.get("id"),
            "args_text": args_text,
            "display_text": format_tool_call_text(tool_name, args_text, args),
            "source": "model",
        }
        if isinstance(args, dict):
            data["args"] = args
        entry = TimelineEntry(kind="tool_call", content=tool_name, data=data)
        existing_index = self._tool_entry_index.get(idx)
        if existing_index is None:
            self._tool_entry_index[idx] = len(self.entries)
            self.entries.append(entry)
            return
        self.entries[existing_index] = entry

    def _flush_pending_text(self) -> None:
        if not self._tag_pending:
            return
        kind = "thinking" if self._inside_thinking_tag and self.reasoning_mode != "off" else "assistant"
        self._append_text(kind, strip_protocol_control_markers(self._tag_pending))
        if kind == "thinking":
            self._auto_used_tag_thinking = True
        self._tag_pending = ""


def collect_timeline(
    chunks: list[dict[str, Any]],
    *,
    reasoning_mode: Literal["auto", "tags", "field", "off"] = "auto",
    thinking_start_tag: str = "<think>",
    thinking_end_tag: str = "</think>",
) -> list[TimelineEntry]:
    collector = _TimelineCollector(
        reasoning_mode=reasoning_mode,
        thinking_start_tag=thinking_start_tag,
        thinking_end_tag=thinking_end_tag,
    )
    for item in chunks:
        collector.feed(item)
    return collector.finalize()
