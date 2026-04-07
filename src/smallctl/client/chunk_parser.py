"""Chunk parsing and content fragment extraction for model streams."""

from __future__ import annotations

import json
from typing import Any, Literal


def extract_thinking_from_tags(
    text: str,
    *,
    thinking_start_tag: str,
    thinking_end_tag: str,
) -> tuple[str, str]:
    """Extract thinking content from tags and return (assistant_text, thinking_text)."""
    if not text or thinking_start_tag not in text:
        if thinking_start_tag == "<think>":
            text = text.replace("<thinking>", thinking_start_tag)
        if thinking_end_tag == "</think>":
            text = text.replace("</thinking>", thinking_end_tag)
        if thinking_start_tag not in text:
            return text, ""
    normalized = text
    if thinking_start_tag == "<think>":
        normalized = normalized.replace("<thinking>", thinking_start_tag)
    if thinking_end_tag == "</think>":
        normalized = normalized.replace("</thinking>", thinking_end_tag)

    assistant_parts: list[str] = []
    thinking_parts: list[str] = []
    cursor = 0

    while True:
        start = normalized.find(thinking_start_tag, cursor)
        if start == -1:
            assistant_parts.append(normalized[cursor:])
            break
        assistant_parts.append(normalized[cursor:start])
        content_start = start + len(thinking_start_tag)
        end = normalized.find(thinking_end_tag, content_start)
        if end == -1:
            thinking_parts.append(normalized[content_start:])
            break
        thinking_parts.append(normalized[content_start:end])
        cursor = end + len(thinking_end_tag)

    return "".join(assistant_parts), "".join(thinking_parts)


def extract_content_fragments(content: Any) -> list[tuple[Literal["assistant", "thinking"], str]]:
    """Extract content fragments from structured content."""
    fragments: list[tuple[Literal["assistant", "thinking"], str]] = []
    _append_content_fragments(fragments, content, default_kind="assistant")
    return fragments


def _append_content_fragments(
    fragments: list[tuple[Literal["assistant", "thinking"], str]],
    content: Any,
    *,
    default_kind: Literal["assistant", "thinking"],
) -> None:
    """Append content fragments recursively."""
    if isinstance(content, str):
        _append_content_fragment(fragments, default_kind, content)
        return
    if isinstance(content, list):
        for item in content:
            _append_content_fragments(fragments, item, default_kind=default_kind)
        return
    if not isinstance(content, dict):
        return

    kind = _content_fragment_kind(content.get("type"), default_kind=default_kind)
    handled = False
    for key in ("text", "value"):
        value = content.get(key)
        if isinstance(value, str) and value:
            _append_content_fragment(fragments, kind, value)
            handled = True

    nested_content = content.get("content")
    if isinstance(nested_content, (str, list, dict)):
        _append_content_fragments(fragments, nested_content, default_kind=kind)
        handled = True

    for key in ("summary", "parts", "items"):
        nested = content.get(key)
        if isinstance(nested, (str, list, dict)):
            _append_content_fragments(fragments, nested, default_kind=kind)
            handled = True

    if handled:
        return

    if isinstance(content.get("reasoning"), str):
        _append_content_fragment(fragments, "thinking", content["reasoning"])
        return
    if isinstance(content.get("output_text"), str):
        _append_content_fragment(fragments, "assistant", content["output_text"])


def _content_fragment_kind(
    raw_type: Any,
    *,
    default_kind: Literal["assistant", "thinking"],
) -> Literal["assistant", "thinking"]:
    """Determine content fragment kind from type string."""
    value = str(raw_type or "").strip().lower()
    if value in {"reasoning", "reasoning_content", "thinking", "summary_text"}:
        return "thinking"
    if value in {"text", "output_text", "input_text", "message", "output_message"}:
        return "assistant"
    return default_kind


def _append_content_fragment(
    fragments: list[tuple[Literal["assistant", "thinking"], str]],
    kind: Literal["assistant", "thinking"],
    text: str,
) -> None:
    """Append a content fragment, avoiding exact duplicates."""
    if not text:
        return
    if fragments and fragments[-1] == (kind, text):
        return
    fragments.append((kind, text))


def chunk_contains_tool_call_delta(payload: Any) -> bool:
    """Check if a chunk contains tool call delta information."""
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def format_tool_call_text(
    tool_name: str,
    args_text: str,
    args: dict[str, Any] | None,
) -> str:
    """Format tool call text for display."""
    if isinstance(args, dict):
        if not args:
            return f"{tool_name}()"
        return f"{tool_name}({json.dumps(args, ensure_ascii=True, sort_keys=True)})"
    stripped = args_text.strip()
    if stripped:
        return f"{tool_name}({stripped})"
    return f"{tool_name}()"


def maybe_parse_tool_args(arguments: str) -> dict[str, Any] | None:
    """Try to parse tool arguments JSON."""
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
