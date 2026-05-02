from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .chunk_parser import extract_content_fragments, extract_thinking_from_tags, format_tool_call_text, maybe_parse_tool_args, merge_reasoning_text


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
        if not isinstance(item, dict) or item.get("type") != "chunk":
            continue
        data = item.get("data", {})
        if not isinstance(data, dict):
            continue
        next_usage = data.get("usage")
        if isinstance(next_usage, dict):
            usage = next_usage
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
            reasoning_parts.append(field_reasoning)

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
                existing["function"]["name"] += fn["name"]
            if "arguments" in fn and isinstance(fn["arguments"], str):
                existing["function"]["arguments"] += fn["arguments"]

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
        self._auto_prefers_field = False
        self._auto_used_tag_thinking = False

    def feed(self, item: dict[str, Any]) -> None:
        if not isinstance(item, dict) or item.get("type") != "chunk":
            return
        data = item.get("data", {})
        if not isinstance(data, dict):
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
                    if self.reasoning_mode == "auto" and not self._auto_used_tag_thinking:
                        self._auto_prefers_field = True
                    self._append_text("thinking", fragment)
            else:
                self._feed_content(fragment)

        field_reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(field_reasoning, str) and field_reasoning:
            self._flush_pending_text()
            if self.reasoning_mode in {"field", "auto", "tags"}:
                if self.reasoning_mode == "auto" and not self._auto_used_tag_thinking:
                    self._auto_prefers_field = True
                self._append_text("thinking", field_reasoning)

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
                existing["function"]["name"] += fn["name"]
            if "arguments" in fn and isinstance(fn["arguments"], str):
                existing["function"]["arguments"] += fn["arguments"]
            self._upsert_tool_call_entry(idx)

    def finalize(self) -> list[TimelineEntry]:
        self._flush_pending_text()
        return self.entries

    def _feed_content(self, text: str) -> None:
        if self.reasoning_mode in {"field", "off"}:
            self._append_text("assistant", text)
            return
        if self.reasoning_mode == "auto" and self._auto_prefers_field:
            self._append_text("assistant", text)
            return
        self._feed_tagged_content(text)

    def _feed_tagged_content(self, text: str) -> None:
        pending = self._tag_pending + text
        self._tag_pending = ""
        cursor = 0
        while cursor < len(pending):
            if self._inside_thinking_tag:
                end = pending.find(self.thinking_end_tag, cursor)
                if end == -1:
                    safe_end = max(cursor, len(pending) - max(0, len(self.thinking_end_tag) - 1))
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
                safe_end = max(cursor, len(pending) - max(0, len(self.thinking_start_tag) - 1))
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
        self._append_text(kind, self._tag_pending)
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
