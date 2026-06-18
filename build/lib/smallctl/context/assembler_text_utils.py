from __future__ import annotations

import re

from ..models.conversation import ConversationMessage


def prune_thinking_tags(message: ConversationMessage) -> ConversationMessage:
    if message.role != "assistant" or "thinking_insight" not in message.metadata:
        return message
    content = message.content or ""
    start_tag = "<think>"
    end_tag = "</think>"
    if start_tag not in content or end_tag not in content:
        return message
    insight = message.metadata["thinking_insight"]
    pattern = f"{re.escape(start_tag)}.*?{re.escape(end_tag)}"
    new_content = re.sub(
        pattern,
        f"{start_tag}[Insight: {insight}]{end_tag}",
        content,
        flags=re.DOTALL,
    )
    return ConversationMessage(
        role=message.role,
        content=new_content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=message.tool_calls,
        metadata=message.metadata,
        retrieval_safe_text=message.retrieval_safe_text,
    )


_CHARS_PER_TOKEN = 2.0
_TRUNCATION_MIN_CHARS = 64
_TRUNCATION_SUFFIX = "... [truncated]"


def truncate_text_for_prompt(text: str, *, token_limit: int, truncation_note: str = "") -> str:
    if not text:
        return text
    char_cap = max(_TRUNCATION_MIN_CHARS, int(token_limit * _CHARS_PER_TOKEN))
    if len(text) <= char_cap:
        return text
    suffix = _TRUNCATION_SUFFIX
    if truncation_note:
        suffix = f"{suffix}\n\n{truncation_note}"
    clipped = text[: max(0, char_cap - len(suffix))].rstrip()
    return f"{clipped}{suffix}" if clipped else suffix
