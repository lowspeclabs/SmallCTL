"""Chunk parsing and content fragment extraction for model streams."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

_REASONING_WRAPPER_TAGS = ("analysis", "plan")
_ASSISTANT_WRAPPER_TAGS = ("execution", "response")
_THINKING_START_TAG_ALIASES = ("<thinking>", "<thought>", "<|thought|>")
_THINKING_END_TAG_ALIASES = ("</thinking>", "</thought>", "</|thought|>")
_PROTOCOL_CONTROL_MARKERS = ("<|channel>", "<channel|>", "<|channel|>")

# SentencePiece tokenizers (used by Gemma and others) represent inter-word
# spaces with U+2581 (lower one eighth block). Backends that return partially
# detokenized text leave these markers in the content, which breaks markdown
# rendering and makes words run together.
_SENTENCEPIECE_WHITESPACE_MARKER = "\u2581"


def normalize_sentencepiece_whitespace(text: str) -> str:
    """Convert SentencePiece whitespace markers into regular spaces."""
    if not text:
        return text
    if _SENTENCEPIECE_WHITESPACE_MARKER not in text:
        return text
    return text.replace(_SENTENCEPIECE_WHITESPACE_MARKER, " ")


def _clean_channel_protocol_body(text: str) -> str:
    cleaned = str(text or "").strip()
    stripped = re.sub(
        r"^(?:thought|thinking|analysis|reasoning)\b\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if not stripped:
        return ""
    return cleaned


def _extract_channel_protocol_blocks(text: str) -> tuple[str, str]:
    normalized = str(text or "")
    thinking_parts: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        body = _clean_channel_protocol_body(match.group("body"))
        if body:
            thinking_parts.append(body)
        return ""

    for pattern in (
        r"<\|channel>(?P<body>.*?)<channel\|>",
        r"<\|channel\|>(?P<body>.*?)</\|channel\|>",
        r"<channel\|>(?P<body>.*?)</channel\|>",
    ):
        normalized = re.sub(
            pattern,
            _capture,
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return normalized, "".join(thinking_parts)


def _normalize_thinking_tag_aliases(
    text: str,
    *,
    thinking_start_tag: str,
    thinking_end_tag: str,
) -> str:
    normalized = str(text or "")
    if thinking_start_tag == "<think>":
        for alias in _THINKING_START_TAG_ALIASES:
            normalized = normalized.replace(alias, thinking_start_tag)
    if thinking_end_tag == "</think>":
        for alias in _THINKING_END_TAG_ALIASES:
            normalized = normalized.replace(alias, thinking_end_tag)
    return normalized


def normalize_thinking_tag_aliases(
    text: str,
    *,
    thinking_start_tag: str,
    thinking_end_tag: str,
) -> str:
    """Normalize supported reasoning tag aliases to the configured tags."""
    return _normalize_thinking_tag_aliases(
        text,
        thinking_start_tag=thinking_start_tag,
        thinking_end_tag=thinking_end_tag,
    )


def max_thinking_tag_alias_length(
    *,
    thinking_start_tag: str,
    thinking_end_tag: str,
) -> int:
    """Return the longest known reasoning tag/control marker length."""
    return max(
        len(thinking_start_tag),
        len(thinking_end_tag),
        *[len(alias) for alias in _THINKING_START_TAG_ALIASES],
        *[len(alias) for alias in _THINKING_END_TAG_ALIASES],
        *[len(marker) for marker in _PROTOCOL_CONTROL_MARKERS],
    )


def strip_protocol_control_markers(text: str) -> str:
    """Remove model protocol transition markers from visible text."""
    cleaned = str(text or "")
    for marker in _PROTOCOL_CONTROL_MARKERS:
        cleaned = cleaned.replace(marker, "")
    return cleaned


def find_protocol_control_marker(text: str, start: int = 0) -> tuple[int, int] | None:
    """Find the earliest protocol control marker at or after start."""
    earliest: tuple[int, int] | None = None
    for marker in _PROTOCOL_CONTROL_MARKERS:
        index = str(text or "").find(marker, start)
        if index == -1:
            continue
        candidate = (index, len(marker))
        if earliest is None or candidate[0] < earliest[0]:
            earliest = candidate
    return earliest


def _extract_reasoning_wrapper_blocks(text: str) -> tuple[str, str]:
    normalized = str(text or "")
    reasoning_parts: list[str] = []

    for tag_name in _REASONING_WRAPPER_TAGS:
        pattern = re.compile(
            rf"<{tag_name}>(?P<body>.*?)</{tag_name}>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def _capture(match: re.Match[str]) -> str:
            body = match.group("body")
            if body:
                reasoning_parts.append(body)
            return ""

        normalized = pattern.sub(_capture, normalized)

    for tag_name in _ASSISTANT_WRAPPER_TAGS:
        normalized = re.sub(rf"</?{tag_name}>", "", normalized, flags=re.IGNORECASE)

    return normalized, "".join(reasoning_parts)


def extract_response_from_wrapper_tags(text: str) -> str:
    """Extract assistant response bodies wrapped in protocol-style tags."""
    normalized = str(text or "")
    response_parts: list[str] = []
    pattern = re.compile(
        r"<response>(?P<body>.*?)</response>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(normalized):
        body = match.group("body")
        if body:
            response_parts.append(body)
    return "".join(response_parts)


def merge_reasoning_text(*parts: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = str(part or "")
        if not text.strip():
            continue
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(text)
    return "".join(merged)


def extract_thinking_from_tags(
    text: str,
    *,
    thinking_start_tag: str,
    thinking_end_tag: str,
) -> tuple[str, str]:
    """Extract thinking content from tags and return (assistant_text, thinking_text)."""
    normalized = _normalize_thinking_tag_aliases(
        text,
        thinking_start_tag=thinking_start_tag,
        thinking_end_tag=thinking_end_tag,
    )
    normalized, channel_thinking = _extract_channel_protocol_blocks(normalized)
    normalized, wrapped_reasoning = _extract_reasoning_wrapper_blocks(normalized)
    if not normalized or thinking_start_tag not in normalized:
        return strip_protocol_control_markers(normalized), merge_reasoning_text(
            wrapped_reasoning,
            channel_thinking,
        )

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
            control_marker = find_protocol_control_marker(normalized, content_start)
            if control_marker is None:
                thinking_parts.append(normalized[content_start:])
                break
            control_start, control_length = control_marker
            thinking_parts.append(normalized[content_start:control_start])
            cursor = control_start + control_length
            continue
        control_marker = find_protocol_control_marker(normalized, content_start)
        if control_marker is not None and control_marker[0] < end:
            control_start, control_length = control_marker
            thinking_parts.append(normalized[content_start:control_start])
            cursor = control_start + control_length
            continue
        thinking_parts.append(normalized[content_start:end])
        cursor = end + len(thinking_end_tag)

    return strip_protocol_control_markers("".join(assistant_parts)), merge_reasoning_text(
        wrapped_reasoning,
        channel_thinking,
        "".join(thinking_parts),
    )


def sanitize_assistant_content_for_history(
    text: str,
    *,
    thinking_start_tag: str = "<think>",
    thinking_end_tag: str = "</think>",
) -> tuple[str, str]:
    """Strip common reasoning wrappers from assistant content.

    Returns ``(cleaned_assistant_text, extracted_thinking_text)``. Handles
    ``<think>``/``<thinking>``/``<thought>`` blocks, ``<reasoning>`` blocks,
    and channel protocol wrappers such as ``<|channel>...</channel|>``.
    """
    if not text or not str(text).strip():
        return str(text or ""), ""

    assistant_text = normalize_sentencepiece_whitespace(str(text))
    thinking_parts: list[str] = []

    def _capture_reasoning(match: re.Match[str]) -> str:
        body = normalize_sentencepiece_whitespace(match.group("body"))
        if body:
            thinking_parts.append(body)
        return ""

    # Extract channel protocol wrappers before the protocol-marker strip pass
    # would delete the markers and leave reasoning labels in assistant text.
    assistant_text, channel_thinking = _extract_channel_protocol_blocks(assistant_text)
    if channel_thinking:
        thinking_parts.append(normalize_sentencepiece_whitespace(channel_thinking))

    # Extract <reasoning>...</reasoning> wrappers.
    assistant_text = re.sub(
        r"<reasoning>(?P<body>.*?)</reasoning>",
        _capture_reasoning,
        assistant_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Standard think tags, reasoning wrapper tags (analysis/plan), and any
    # remaining protocol control markers.
    assistant_text, extracted_thinking = extract_thinking_from_tags(
        assistant_text,
        thinking_start_tag=thinking_start_tag,
        thinking_end_tag=thinking_end_tag,
    )
    if extracted_thinking:
        thinking_parts.append(normalize_sentencepiece_whitespace(extracted_thinking))

    return assistant_text.strip(), normalize_sentencepiece_whitespace("".join(thinking_parts)).strip()


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
    if value in {"reasoning", "reasoning_content", "thinking", "summary_text", "analysis", "plan"}:
        return "thinking"
    if value in {"text", "output_text", "input_text", "message", "output_message", "response"}:
        return "assistant"
    return default_kind


def _append_content_fragment(
    fragments: list[tuple[Literal["assistant", "thinking"], str]],
    kind: Literal["assistant", "thinking"],
    text: str,
) -> None:
    """Append a content fragment, avoiding exact duplicates."""
    text = normalize_sentencepiece_whitespace(text)
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
