from __future__ import annotations

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
        user_tokens = estimate_text_tokens(str(getattr(user_message, "content", "") or ""))
        if user_tokens <= token_cap:
            return [user_message]

    return [selected[-1]]
