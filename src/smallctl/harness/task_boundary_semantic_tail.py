from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from ..context.policy import estimate_text_tokens


def message_is_semantic_tail_candidate(message: Any) -> bool:
    role = str(getattr(message, "role", "") or "").strip().lower()
    if role not in {"user", "assistant"}:
        return False
    metadata = getattr(message, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    if metadata.get("hidden_from_prompt") or metadata.get("is_recovery_nudge"):
        return False
    text = str(getattr(message, "content", "") or "").strip()
    if not text:
        return False
    return True


def _copy_message_with_content(message: Any, new_content: str) -> Any:
    if hasattr(message, "__dataclass_fields__"):
        field_names = {f.name for f in fields(message)}
        if "content" in field_names:
            return replace(message, content=new_content)
    copied: Any
    if hasattr(message, "items"):
        copied = dict(message)
        copied["content"] = new_content
        return copied
    copied = message
    if hasattr(copied, "content"):
        copied.content = new_content
    return copied


def _truncate_message_content(message: Any, max_tokens: int) -> Any:
    text = str(getattr(message, "content", "") or "")
    if estimate_text_tokens(text) <= max_tokens:
        return message
    marker = "\n\n... [prior assistant response truncated]"
    marker_tokens = estimate_text_tokens(marker)
    available_tokens = max(max_tokens - marker_tokens, 1)
    available_chars = max(int(available_tokens / 0.4), 1)
    truncated = text[:available_chars]
    if "\n" in truncated:
        truncated = truncated.rsplit("\n", 1)[0]
    new_text = truncated + marker
    return _copy_message_with_content(message, new_text)


def semantic_recent_tail_messages(messages: list[Any], *, token_cap: int) -> list[Any]:
    candidates = [
        message
        for message in messages
        if message_is_semantic_tail_candidate(message)
    ]
    if not candidates:
        return []

    selected: list[Any] = []
    last_assistant_index = max(
        (index for index, message in enumerate(candidates) if str(getattr(message, "role", "")).lower() == "assistant"),
        default=-1,
    )
    if last_assistant_index >= 0:
        assistant_message = candidates[last_assistant_index]
        user_index = max(
            (
                index
                for index in range(last_assistant_index - 1, -1, -1)
                if str(getattr(candidates[index], "role", "")).lower() == "user"
            ),
            default=-1,
        )
        if user_index >= 0:
            selected.append(candidates[user_index])
        selected.append(assistant_message)
    else:
        user_index = max(
            (index for index, message in enumerate(candidates) if str(getattr(message, "role", "")).lower() == "user"),
            default=-1,
        )
        if user_index >= 0:
            selected.append(candidates[user_index])

    if not selected:
        return []

    total_tokens = sum(
        estimate_text_tokens(str(getattr(message, "content", "") or ""))
        for message in selected
    )
    if total_tokens <= token_cap:
        return selected

    if len(selected) == 2:
        user_message, assistant_message = selected
        assistant_tokens = estimate_text_tokens(str(getattr(assistant_message, "content", "") or ""))
        if assistant_tokens <= token_cap:
            return [assistant_message]
        # Prefer a truncated assistant outcome over dropping it, so ordinal/
        # numbered follow-ups like "address #1" still have the prior list.
        return [_truncate_message_content(assistant_message, token_cap)]

    return [_truncate_message_content(selected[-1], token_cap)]
