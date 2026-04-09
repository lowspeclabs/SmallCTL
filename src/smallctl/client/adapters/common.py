"""Shared provider adapter utilities."""

from __future__ import annotations

from typing import Any


def sanitize_message_for_transport(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role", "user"))
    sanitized: dict[str, Any] = {"role": role, "content": message.get("content")}
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            sanitized["tool_calls"] = tool_calls
    elif role == "tool":
        name = message.get("name")
        if name:
            sanitized["name"] = name
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            sanitized["tool_call_id"] = tool_call_id
    else:
        name = message.get("name")
        if name:
            sanitized["name"] = name
    return sanitized


def rewrite_orphan_tool_message_for_openrouter(message: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(message.get("name") or "").strip()
    tool_call_id = str(message.get("tool_call_id") or "").strip()
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        content = raw_content.strip()
    else:
        content = "" if raw_content is None else str(raw_content)

    prefix = "[OpenRouter compatibility] Orphan tool result"
    if tool_name:
        prefix = f"{prefix} from {tool_name}"
    if tool_call_id:
        prefix = f"{prefix} (tool_call_id={tool_call_id})"
    if content:
        content = f"{prefix}:\n{content}"
    else:
        content = f"{prefix}."
    return {"role": "user", "content": content}


def sanitize_messages_with_pending_tool_cleanup(
    messages: list[dict[str, Any]],
    *,
    rewrite_orphan_tool_messages: bool = False,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    pending_tool_call_ids: set[str] = set()
    pending_assistant_index: int | None = None

    def close_pending_tool_block() -> None:
        nonlocal pending_tool_call_ids, pending_assistant_index
        if pending_assistant_index is not None and pending_tool_call_ids:
            sanitized[pending_assistant_index].pop("tool_calls", None)
        pending_tool_call_ids = set()
        pending_assistant_index = None

    for message in messages:
        role = str(message.get("role", "user"))
        if role == "assistant":
            if pending_tool_call_ids:
                close_pending_tool_block()
            sanitized_message = sanitize_message_for_transport(message)
            sanitized.append(sanitized_message)
            tool_calls = sanitized_message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                pending_tool_call_ids = {
                    str(tool_call.get("id"))
                    for tool_call in tool_calls
                    if isinstance(tool_call, dict) and tool_call.get("id")
                }
                if pending_tool_call_ids:
                    pending_assistant_index = len(sanitized) - 1
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            if pending_tool_call_ids and tool_call_id and tool_call_id in pending_tool_call_ids:
                sanitized.append(sanitize_message_for_transport(message))
                pending_tool_call_ids.discard(tool_call_id)
                if not pending_tool_call_ids:
                    pending_assistant_index = None
            else:
                if rewrite_orphan_tool_messages:
                    sanitized.append(rewrite_orphan_tool_message_for_openrouter(message))
                else:
                    sanitized.append(sanitize_message_for_transport(message))
            continue

        if pending_tool_call_ids:
            close_pending_tool_block()
        sanitized.append(sanitize_message_for_transport(message))

    if pending_tool_call_ids:
        close_pending_tool_block()
    return sanitized


def sanitize_messages_for_openrouter(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sanitize_messages_with_pending_tool_cleanup(
        messages,
        rewrite_orphan_tool_messages=True,
    )


def sanitize_messages_for_lmstudio(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = sanitize_messages_with_pending_tool_cleanup(messages)
    if sanitized:
        last_message = sanitized[-1]
        if (
            str(last_message.get("role", "")).strip() == "user"
            and len(str(last_message.get("content") or "").strip()) >= 20
        ):
            return sanitized
    else:
        return sanitized

    latest_user_content = ""
    for message in reversed(sanitized):
        if str(message.get("role", "")).strip() != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            latest_user_content = content
            break

    if not latest_user_content:
        return sanitized

    recap = (
        "Continue working on the user's request. "
        f"User query recap: {latest_user_content}"
    )
    sanitized.append({"role": "user", "content": recap})
    return sanitized


def should_retry_without_stream_options(exc: Any) -> bool:
    try:
        response = exc.response
        if response.status_code not in {400, 404, 422}:
            return False
        try:
            body = response.text.lower()
        except Exception:
            return True
        if not body.strip():
            return True
        return (
            "stream_options" in body
            or "include_usage" in body
            or "unknown field" in body
            or "unexpected field" in body
            or "extra fields not permitted" in body
        )
    except Exception:
        return False
