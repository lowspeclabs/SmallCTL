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


def merge_system_messages_for_single_system_providers(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    non_system_messages: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        if role == "system":
            content = message.get("content")
            text = "" if content is None else str(content).strip()
            if text:
                system_parts.append(text)
            continue
        non_system_messages.append(dict(message))

    if not system_parts:
        return non_system_messages
    merged_system = {"role": "system", "content": "\n\n".join(system_parts)}
    return [merged_system, *non_system_messages]


def sanitize_messages_with_pending_tool_cleanup(
    messages: list[dict[str, Any]],
    *,
    rewrite_orphan_tool_messages: bool = False,
    available_tool_names: set[str] | None = None,
    strip_assistant_content_when_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    pending_tool_call_ids: set[str] = set()
    unavailable_tool_call_ids: set[str] = set()
    pending_assistant_index: int | None = None

    def close_pending_tool_block() -> None:
        nonlocal pending_tool_call_ids, pending_assistant_index
        if pending_assistant_index is not None and pending_tool_call_ids:
            pending_message = sanitized[pending_assistant_index]
            pending_message.pop("tool_calls", None)
            content = pending_message.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                sanitized.pop(pending_assistant_index)
        pending_tool_call_ids = set()
        pending_assistant_index = None

    for message in messages:
        role = str(message.get("role", "user"))
        if role == "assistant":
            if pending_tool_call_ids:
                close_pending_tool_block()
            sanitized_message = sanitize_message_for_transport(message)
            tool_calls = sanitized_message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                filtered_tool_calls: list[dict[str, Any]] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        filtered_tool_calls.append(tool_call)
                        continue
                    tool_name = str(function.get("name") or "").strip()
                    tool_call_id = str(tool_call.get("id") or "").strip()
                    if available_tool_names is not None and tool_name and tool_name not in available_tool_names:
                        if tool_call_id:
                            unavailable_tool_call_ids.add(tool_call_id)
                        continue
                    filtered_tool_calls.append(tool_call)
                if filtered_tool_calls:
                    sanitized_message["tool_calls"] = filtered_tool_calls
                else:
                    sanitized_message.pop("tool_calls", None)

            content = sanitized_message.get("content")
            has_content = not (
                content is None or (isinstance(content, str) and not content.strip())
            )
            tool_calls = sanitized_message.get("tool_calls")
            has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
            if strip_assistant_content_when_tool_calls and has_content and has_tool_calls:
                # OpenRouter rejects some assistant history entries that include both
                # non-empty text content and tool_calls in the same message.
                sanitized_message["content"] = None
                has_content = False
            if not has_content and not has_tool_calls:
                continue

            sanitized.append(sanitized_message)
            if has_tool_calls:
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
            elif tool_call_id and tool_call_id in unavailable_tool_call_ids:
                unavailable_tool_call_ids.discard(tool_call_id)
                if rewrite_orphan_tool_messages:
                    sanitized.append(rewrite_orphan_tool_message_for_openrouter(message))
                else:
                    sanitized.append(sanitize_message_for_transport(message))
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


def sanitize_messages_for_openrouter(
    messages: list[dict[str, Any]],
    *,
    available_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    return sanitize_messages_with_pending_tool_cleanup(
        merge_system_messages_for_single_system_providers(messages),
        rewrite_orphan_tool_messages=True,
        available_tool_names=available_tool_names,
        strip_assistant_content_when_tool_calls=True,
    )


def _assistant_has_single_task_complete_call(message: dict[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return False
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        return False
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return False
    return str(function.get("name") or "").strip() == "task_complete"


def _rewrite_lmstudio_task_complete_pair(
    assistant_message: dict[str, Any],
    tool_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    assistant_content = str(assistant_message.get("content") or "").strip()
    tool_summaries = [
        str(message.get("content") or "").strip()
        for message in tool_messages
        if str(message.get("content") or "").strip()
    ]
    summary_parts = [part for part in [assistant_content, "\n".join(tool_summaries)] if part]
    if summary_parts:
        summary = "\n".join(summary_parts)
        content = f"Previous task summary: {summary}"
    else:
        content = "Previous task completed."
    return {"role": "assistant", "content": content}


def _collapse_completed_task_complete_pairs_for_lmstudio(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        role = str(message.get("role", "")).strip()
        if role != "assistant" or not _assistant_has_single_task_complete_call(message):
            sanitized.append(dict(message))
            index += 1
            continue

        tool_calls = message.get("tool_calls")
        tool_call_ids = {
            str(tool_call.get("id") or "").strip()
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and str(tool_call.get("id") or "").strip()
        }
        if not tool_call_ids:
            sanitized.append(dict(message))
            index += 1
            continue

        cursor = index + 1
        matched_tools: list[dict[str, Any]] = []
        pending_ids = set(tool_call_ids)
        while cursor < total:
            next_message = messages[cursor]
            if str(next_message.get("role", "")).strip() != "tool":
                break
            tool_call_id = str(next_message.get("tool_call_id") or "").strip()
            if tool_call_id in pending_ids and str(next_message.get("name") or "").strip() == "task_complete":
                matched_tools.append(dict(next_message))
                pending_ids.discard(tool_call_id)
                cursor += 1
                continue
            break

        has_later_user_followup = any(
            str(later.get("role", "")).strip() == "user" and str(later.get("content") or "").strip()
            for later in messages[cursor:]
        )
        if matched_tools and not pending_ids and has_later_user_followup:
            sanitized.append(_rewrite_lmstudio_task_complete_pair(message, matched_tools))
            index = cursor
            continue

        sanitized.append(dict(message))
        index += 1

    return sanitized


def sanitize_messages_for_lmstudio(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = sanitize_messages_with_pending_tool_cleanup(messages)
    sanitized = _collapse_completed_task_complete_pairs_for_lmstudio(sanitized)
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
