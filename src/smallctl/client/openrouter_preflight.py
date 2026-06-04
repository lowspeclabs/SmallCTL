from __future__ import annotations

import json
import logging
from typing import Any

from ..logging_utils import log_kv
from .provider_adapters import sanitize_messages_for_openrouter
from .request_budget import (
    approx_token_count as _budget_approx_token_count,
    client_context_limit as _budget_client_context_limit,
    json_size_bytes as _budget_json_size_bytes,
)


def _json_size_bytes(value: Any) -> int:
    return _budget_json_size_bytes(value)


def _approx_token_count(value: Any) -> int:
    return _budget_approx_token_count(value)


def _client_context_limit(client: Any) -> int | None:
    return _budget_client_context_limit(client)


def _context_pressure_diagnostics(payload: dict[str, Any], *, context_limit: int | None) -> dict[str, Any]:
    estimated_payload_tokens = _approx_token_count(payload)
    estimated_tool_schema_tokens = _approx_token_count(payload.get("tools", []))
    diagnostics: dict[str, Any] = {
        "estimated_payload_tokens": estimated_payload_tokens,
        "estimated_tool_schema_tokens": estimated_tool_schema_tokens,
    }
    if context_limit is not None:
        diagnostics["known_context_limit"] = context_limit
        diagnostics["estimated_context_tokens_remaining"] = context_limit - estimated_payload_tokens
        if estimated_payload_tokens >= context_limit:
            diagnostics["likely_provider_rejection"] = "context_overflow"
    return diagnostics


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _tool_schema_diagnostics(payload: dict[str, Any], *, context_limit: int | None = None) -> dict[str, Any]:
    raw_tools = payload.get("tools")
    tool_list = raw_tools if isinstance(raw_tools, list) else []
    invalid_count = 0
    names: list[str] = []
    for tool in tool_list:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            invalid_count += 1
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            invalid_count += 1
            continue
        name = _tool_name(tool)
        if not name:
            invalid_count += 1
            continue
        names.append(name)
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            invalid_count += 1
    return {
        "tool_schema_count": len(tool_list),
        "tool_schema_bytes": _json_size_bytes(tool_list),
        "invalid_tool_schema_count": invalid_count,
        "tool_names": names,
        **_context_pressure_diagnostics(payload, context_limit=context_limit),
    }


def _summarize_400_payload(payload: dict[str, Any], *, context_limit: int | None = None) -> dict[str, Any]:
    messages = payload.get("messages")
    message_list = messages if isinstance(messages, list) else []
    role_counts: dict[str, int] = {}
    assistant_with_tool_calls_count = 0
    assistant_content_tool_calls_coexist = False
    tool_message_count = 0
    for message in message_list:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip() or "unknown"
        role_counts[role] = role_counts.get(role, 0) + 1
        tool_calls = message.get("tool_calls")
        has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
        content = message.get("content")
        has_non_empty_content = not (
            content is None or (isinstance(content, str) and not content.strip())
        )
        if role == "assistant" and has_tool_calls:
            assistant_with_tool_calls_count += 1
            if has_non_empty_content:
                assistant_content_tool_calls_coexist = True
        if role == "tool":
            tool_message_count += 1

    return {
        "model": str(payload.get("model") or ""),
        "payload_bytes": _json_size_bytes(payload),
        "has_stream_options": "stream_options" in payload,
        "message_count": len(message_list),
        "role_counts": role_counts,
        "assistant_with_tool_calls_count": assistant_with_tool_calls_count,
        "tool_message_count": tool_message_count,
        "assistant_content_and_tool_calls_coexist": assistant_content_tool_calls_coexist,
        **_tool_schema_diagnostics(payload, context_limit=context_limit),
    }


def _message_role_counts(messages: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(messages, list):
        return counts
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip() or "unknown"
        counts[role] = counts.get(role, 0) + 1
    return counts


def _normalize_tool_schemas_for_openrouter(tools: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(tools, list):
        return [], []
    normalized: list[dict[str, Any]] = []
    issues: list[str] = []
    seen_names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            issues.append(f"tool[{index}]:not_object")
            continue
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            issues.append(f"tool[{index}]:not_function_schema")
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            issues.append(f"tool[{index}]:missing_name")
            continue
        if name in seen_names:
            issues.append(f"tool[{index}]:duplicate_name:{name}")
            continue
        seen_names.add(name)
        description = function.get("description")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            issues.append(f"tool[{index}]:parameters_repaired:{name}")
            parameters = {"type": "object", "properties": {}}
        elif str(parameters.get("type") or "").strip() != "object":
            issues.append(f"tool[{index}]:parameters_type_repaired:{name}")
            parameters = {**parameters, "type": "object"}
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "" if description is None else str(description),
                    "parameters": parameters,
                },
            }
        )
    return normalized, issues


def _normalize_openrouter_tool_calls(
    message: dict[str, Any],
    *,
    available_tool_names: set[str],
) -> tuple[dict[str, Any], list[str]]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return message, []
    normalized_calls: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            issues.append(f"tool_call[{index}]:not_object")
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            issues.append(f"tool_call[{index}]:missing_function")
            continue
        call_id = str(tool_call.get("id") or "").strip()
        name = str(function.get("name") or "").strip()
        if not call_id or not name:
            issues.append(f"tool_call[{index}]:missing_id_or_name")
            continue
        if name not in available_tool_names:
            issues.append(f"tool_call[{index}]:unavailable_tool:{name}")
            continue
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments)
            except Exception:
                arguments = str(arguments)
            issues.append(f"tool_call[{index}]:arguments_stringified:{name}")
        normalized_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    normalized = dict(message)
    if normalized_calls:
        normalized["tool_calls"] = normalized_calls
    else:
        normalized.pop("tool_calls", None)
    return normalized, issues


def _preflight_openrouter_payload(client: Any, payload: dict[str, Any], *, stage: str) -> dict[str, Any]:
    if client.provider_profile != "openrouter":
        return payload

    repaired = dict(payload)
    issues: list[str] = []
    normalized_tools, tool_issues = _normalize_tool_schemas_for_openrouter(repaired.get("tools"))
    issues.extend(tool_issues)
    if "tools" in repaired:
        if normalized_tools:
            repaired["tools"] = normalized_tools
        else:
            repaired.pop("tools", None)
            issues.append("tools:removed_empty_or_invalid")

    available_tool_names = {
        str((tool.get("function") or {}).get("name") or "").strip()
        for tool in normalized_tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    raw_messages = repaired.get("messages")
    prepared_messages: list[dict[str, Any]] = []
    if isinstance(raw_messages, list):
        raw_system_count = sum(
            1
            for message in raw_messages
            if isinstance(message, dict)
            and str(message.get("role") or "").strip() == "system"
        )
        if raw_system_count > 1:
            issues.append(f"messages:merged_system:{raw_system_count}")
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict):
                issues.append(f"message[{index}]:not_object")
                continue
            role = str(message.get("role") or "user").strip()
            if role not in {"system", "user", "assistant", "tool"}:
                prepared_messages.append({"role": "user", "content": str(message.get("content") or "")})
                issues.append(f"message[{index}]:invalid_role:{role}")
                continue
            prepared = dict(message)
            if role == "assistant":
                prepared, tool_call_issues = _normalize_openrouter_tool_calls(
                    prepared,
                    available_tool_names=available_tool_names,
                )
                issues.extend(f"message[{index}]:{issue}" for issue in tool_call_issues)
            prepared_messages.append(prepared)
    else:
        issues.append("messages:not_list")

    sanitized_available_names: set[str] | None = available_tool_names
    repaired["messages"] = sanitize_messages_for_openrouter(
        prepared_messages,
        available_tool_names=sanitized_available_names,
    )
    repaired.pop("stream_options", None)

    if issues:
        diagnostics = {
            "stage": stage,
            "provider_profile": client.provider_profile,
            "issues": issues[:25],
            **_summarize_400_payload(repaired, context_limit=_client_context_limit(client)),
        }
        log_kv(
            client.log,
            logging.WARNING,
            "chat_payload_preflight_repaired",
            **diagnostics,
        )
        if client.run_logger:
            client.run_logger.log(
                "chat",
                "payload_preflight_repaired",
                "provider payload repaired before request",
                **diagnostics,
            )
    return repaired
