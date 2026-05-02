from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator
from urllib.parse import quote

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from ..logging_utils import log_kv
from .chunk_parser import chunk_contains_tool_call_delta
from .provider_adapters import sanitize_messages_for_openrouter
from .streaming import SSEStreamer
from .usage import extract_context_limit, extract_runtime_context_limit


def _client_key(client: Any) -> tuple[str, str]:
    return (client.base_url, client.api_key)


def _get_async_client(client: Any) -> Any:
    if httpx is None:
        raise RuntimeError("Dependency missing: httpx")
    key = _client_key(client)
    shared_clients = client._shared_clients
    async_client = shared_clients.get(key)
    if async_client is None:
        async_client = httpx.AsyncClient(timeout=None)
        shared_clients[key] = async_client
    return async_client


async def _reset_async_client(client: Any) -> None:
    key = _client_key(client)
    async_client = client._shared_clients.pop(key, None)
    if async_client is None:
        return
    try:
        await async_client.aclose()
    except Exception:
        pass


def _request_first_token_timeout_sec(client: Any, tools: list[dict[str, Any]]) -> float:
    timeout = float(client.first_token_timeout_sec)
    if client.provider_profile != "lmstudio":
        return timeout
    if client.is_small_model and not tools:
        return timeout

    tool_count = len(tools)
    if tool_count >= 12:
        return max(timeout, 60.0)
    if tool_count > 0:
        normalized_model = str(client.model or "").strip().lower()
        if "gemma" in normalized_model:
            return max(timeout, 60.0)
    return timeout


def _is_tool_call_continuation_timeout(exc: Exception) -> bool:
    return "tool call continuation" in str(exc).lower()


def _log_http_error(client: Any, event: str, exc: "httpx.HTTPStatusError") -> None:
    try:
        body = exc.response.text[:1000]
    except Exception:
        body = ""
    log_kv(
        client.log,
        logging.WARNING,
        event,
        status=exc.response.status_code,
        body=body,
    )
    if client.run_logger:
        client.run_logger.log(
            "chat",
            event,
            "chat http error",
            status=exc.response.status_code,
            body=body,
        )


def _parse_retry_after_seconds(response: Any) -> float | None:
    try:
        raw_value = response.headers.get("Retry-After")
    except Exception:
        return None
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    try:
        retry_after = float(value)
    except Exception:
        retry_after = None
    if retry_after is not None:
        return max(0.0, retry_after)
    try:
        retry_at = parsedate_to_datetime(value)
    except Exception:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    delta = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


def _extract_available_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _http_error_body(exc: Any, *, limit: int = 1000) -> str:
    try:
        return str(exc.response.text or "")[:limit]
    except Exception:
        return ""


def _summarize_http_error_body(body: str) -> dict[str, Any]:
    snippet = str(body or "")[:1000]
    lower = snippet.lower()
    summary: dict[str, Any] = {"body": snippet}
    if "together" in lower:
        summary["upstream_provider"] = "Together"
    elif "openrouter" in lower:
        summary["upstream_provider"] = "OpenRouter"
    if "input validation error" in lower:
        summary["provider_error"] = "Input validation error"
    return summary


def _tool_schema_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
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
        name = str(function.get("name") or "").strip()
        if not name:
            invalid_count += 1
            continue
        names.append(name)
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            invalid_count += 1
    return {
        "tool_schema_count": len(tool_list),
        "invalid_tool_schema_count": invalid_count,
        "tool_names": names[:25],
    }


def _summarize_400_payload(payload: dict[str, Any]) -> dict[str, Any]:
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
        "has_stream_options": "stream_options" in payload,
        "message_count": len(message_list),
        "role_counts": role_counts,
        "assistant_with_tool_calls_count": assistant_with_tool_calls_count,
        "tool_message_count": tool_message_count,
        "assistant_content_and_tool_calls_coexist": assistant_content_tool_calls_coexist,
        **_tool_schema_diagnostics(payload),
    }


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

    # With no active tool schemas, assistant/tool-call history is invalid for many
    # OpenRouter-routed backends. Passing an empty set rewrites those entries as
    # plain user context instead of preserving stale tool-call structure.
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
            **_summarize_400_payload(repaired),
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


def _provider_400_chunk_error_details(
    client: Any,
    *,
    payload: dict[str, Any],
    exc: Any,
    attempt: int,
    recovery_stages_attempted: int,
) -> dict[str, Any]:
    body = _http_error_body(exc)
    return {
        "type": "provider_input_validation",
        "attempt": attempt,
        "provider_profile": client.provider_profile,
        "status_code": 400,
        "recovery_stages_attempted": recovery_stages_attempted,
        **_summarize_400_payload(payload),
        **_summarize_http_error_body(body),
    }


def _provider_400_error_message(details: dict[str, Any]) -> str:
    provider = str(details.get("provider_profile") or "provider")
    upstream = str(details.get("upstream_provider") or "").strip()
    provider_error = str(details.get("provider_error") or "").strip()
    label = provider
    if upstream and upstream.lower() != provider.lower():
        label = f"{provider}/{upstream}"
    suffix = f": {provider_error}" if provider_error else ""
    return f"{label} input validation failed after retries (HTTP 400{suffix})"


def _build_openrouter_recovery_payload(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    reduced_features: bool,
) -> dict[str, Any]:
    if reduced_features:
        available_tool_names: set[str] | None = set()
    else:
        available_tool_names = _extract_available_tool_names(tools) or None
    sanitized_messages = sanitize_messages_for_openrouter(
        messages,
        available_tool_names=available_tool_names,
    )
    payload: dict[str, Any] = {
        "model": client.model,
        "messages": sanitized_messages,
        "stream": True,
    }
    if tools and not reduced_features:
        payload["tools"] = tools
    if client.adapter.stream_policy.supports_stream_options and not reduced_features:
        payload["stream_options"] = {"include_usage": True}
    payload = client.adapter.mutate_payload(payload)
    if reduced_features:
        payload.pop("tools", None)
        payload.pop("stream_options", None)
    return payload


def _build_minimal_context_payload(
    client: Any,
    *,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    last_system_msg = None
    for msg in reversed(messages):
        if str(msg.get("role", "")).strip() == "system":
            last_system_msg = msg
            break

    recent_user_msgs = []
    for msg in reversed(messages):
        role = str(msg.get("role", "")).strip()
        if role == "tool" or (role == "assistant" and msg.get("tool_calls")):
            break
        if role == "user":
            recent_user_msgs.insert(0, msg)
            if len(recent_user_msgs) >= 2:
                break

    user_content = ""
    if recent_user_msgs:
        if len(recent_user_msgs) == 2:
            user_content = str(recent_user_msgs[0].get("content") or "") + "\n\n" + str(recent_user_msgs[1].get("content") or "")
        else:
            user_content = str(recent_user_msgs[0].get("content") or "")

    reduced_messages = []
    if last_system_msg:
        reduced_messages.append(last_system_msg)
    if user_content:
        reduced_messages.append({"role": "user", "content": user_content})

    payload: dict[str, Any] = {
        "model": client.model,
        "messages": client.adapter.sanitize_messages(reduced_messages),
        "stream": True,
    }
    return client.adapter.mutate_payload(payload)


async def stream_chat(
    client: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    if httpx is None:
        raise RuntimeError("Dependency missing: httpx")

    original_messages = [dict(message) for message in messages]
    original_tools = [dict(tool) for tool in tools]
    messages = client.adapter.sanitize_messages(messages)

    url = f"{client.base_url}{client.chat_endpoint}"
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
    }
    headers = client.adapter.mutate_headers(headers)

    payload: dict[str, Any] = {
        "model": client.model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    if client.adapter.stream_policy.supports_stream_options:
        payload["stream_options"] = {"include_usage": True}
    payload = client.adapter.mutate_payload(payload)
    payload = _preflight_openrouter_payload(client, payload, stage="initial")

    log_kv(
        client.log,
        logging.INFO,
        "chat_request",
        url=url,
        model=client.model,
        message_count=len(messages),
        tool_count=len(tools),
    )
    log_kv(
        client.log,
        logging.INFO,
        "backend_health_check",
        url=url,
        model=client.model,
        provider_profile=client.provider_profile,
        stream=True,
    )
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "request",
            "chat request started",
            url=url,
            model=client.model,
            message_count=len(messages),
            tool_count=len(tools),
        )

    last_error: Exception | None = None
    current_payload = dict(payload)
    openrouter_400_recovery_stage = 0
    openrouter_nonstream_fallback_attempted = False
    openrouter_minimal_context_attempted = False
    async_client = _get_async_client(client)
    request_first_token_timeout_sec = _request_first_token_timeout_sec(client, tools)
    streamer = SSEStreamer(
        provider_profile=client.provider_profile,
        first_token_timeout_sec=request_first_token_timeout_sec,
        tool_call_continuation_timeout_sec=client._request_tool_call_continuation_timeout_sec(tools),
        aggressive_tool_call_timeout=client.is_small_model,
        run_logger=client.run_logger,
        log=client.log,
    )

    for attempt in range(1, client.STREAM_RETRY_ATTEMPTS + 1):
        retry_after_seconds: float | None = None
        saw_chunk = False
        saw_tool_call_chunk = False
        retry_after_backend_recovery = False
        current_payload = _preflight_openrouter_payload(client, current_payload, stage=f"attempt_{attempt}")
        try:
            async for event in streamer.stream_sse(async_client, url, headers, current_payload):
                if event.get("type") == "backend_first_token_timeout":
                    details = dict(event.get("details") or {})
                    log_kv(
                        client.log,
                        logging.WARNING,
                        "chat_backend_first_token_timeout",
                        attempt=attempt,
                        provider_profile=client.provider_profile,
                        timeout_sec=details.get("timeout_sec"),
                    )
                    if client.run_logger:
                        client.run_logger.log(
                            "chat",
                            "backend_first_token_timeout",
                            "backend stalled before first token",
                            attempt=attempt,
                            provider_profile=client.provider_profile,
                            timeout_sec=details.get("timeout_sec"),
                        )
                    recovery: dict[str, Any] | None = None
                    if client.backend_recovery_handler is not None:
                        recovery = await client.backend_recovery_handler(
                            {
                                "attempt": attempt,
                                "provider_profile": client.provider_profile,
                                "base_url": client.base_url,
                                "model": client.model,
                                "details": details,
                            }
                        )
                    if isinstance(recovery, dict) and recovery.get("status") == "recovered":
                        retry_after_backend_recovery = True
                        log_kv(
                            client.log,
                            logging.WARNING,
                            "chat_backend_recovery_succeeded",
                            attempt=attempt,
                            provider_profile=client.provider_profile,
                            action=recovery.get("action"),
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "backend_recovery_succeeded",
                                "backend recovery succeeded after first-token timeout",
                                attempt=attempt,
                                provider_profile=client.provider_profile,
                                action=recovery.get("action"),
                            )
                        await _reset_async_client(client)
                        async_client = _get_async_client(client)
                        break
                    wedge_details = dict(details)
                    if isinstance(recovery, dict) and recovery:
                        wedge_details["recovery"] = recovery
                    yield {
                        "type": "backend_wedged",
                        "error": "Backend did not emit a first token before timeout",
                        "details": wedge_details,
                    }
                    return
                if event.get("type") == "chunk":
                    saw_chunk = True
                    if chunk_contains_tool_call_delta(event.get("data", {})):
                        saw_tool_call_chunk = True
                yield event
            if retry_after_backend_recovery:
                continue
            return
        except httpx.HTTPStatusError as exc:
            _log_http_error(client, "chat_stream_http_error", exc)
            status_code = int(exc.response.status_code)
            if status_code == 400:
                diagnostics = {
                    **_summarize_400_payload(current_payload),
                    **_summarize_http_error_body(_http_error_body(exc)),
                    "status_code": status_code,
                }
                log_kv(
                    client.log,
                    logging.WARNING,
                    "chat_stream_400_diagnostics",
                    attempt=attempt,
                    provider_profile=client.provider_profile,
                    **diagnostics,
                )
                if client.run_logger:
                    client.run_logger.log(
                        "chat",
                        "stream_400_diagnostics",
                        "stream request returned HTTP 400",
                        attempt=attempt,
                        provider_profile=client.provider_profile,
                        **diagnostics,
                    )
                if client.provider_profile == "openrouter":
                    if openrouter_400_recovery_stage == 0:
                        openrouter_400_recovery_stage = 1
                        current_payload = _build_openrouter_recovery_payload(
                            client,
                            messages=original_messages,
                            tools=original_tools,
                            reduced_features=False,
                        )
                        current_payload = _preflight_openrouter_payload(
                            client,
                            current_payload,
                            stage="strict_sanitize_retry",
                        )
                        log_kv(
                            client.log,
                            logging.WARNING,
                            "chat_stream_400_openrouter_retry",
                            attempt=attempt,
                            strategy="strict_sanitize_retry",
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "stream_400_openrouter_retry",
                                "retrying OpenRouter stream after strict sanitize rebuild",
                                attempt=attempt,
                                strategy="strict_sanitize_retry",
                            )
                        continue
                    if openrouter_400_recovery_stage == 1:
                        openrouter_400_recovery_stage = 2
                        current_payload = _build_openrouter_recovery_payload(
                            client,
                            messages=original_messages,
                            tools=original_tools,
                            reduced_features=True,
                        )
                        current_payload = _preflight_openrouter_payload(
                            client,
                            current_payload,
                            stage="reduced_features_retry",
                        )
                        log_kv(
                            client.log,
                            logging.WARNING,
                            "chat_stream_400_openrouter_retry",
                            attempt=attempt,
                            strategy="reduced_features_retry",
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "stream_400_openrouter_retry",
                                "retrying OpenRouter stream with reduced payload features",
                                attempt=attempt,
                                strategy="reduced_features_retry",
                            )
                        continue
                    if not openrouter_nonstream_fallback_attempted:
                        openrouter_nonstream_fallback_attempted = True
                        log_kv(
                            client.log,
                            logging.WARNING,
                            "chat_stream_400_openrouter_nonstream_fallback",
                            attempt=attempt,
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "stream_400_openrouter_nonstream_fallback",
                                "attempting OpenRouter non-stream fallback after repeated 400 stream errors",
                                attempt=attempt,
                            )
                        try:
                            async for event in streamer.nonstream_chat(async_client, url, headers, current_payload):
                                yield event
                            return
                        except httpx.HTTPStatusError as fallback_exc:
                            last_error = fallback_exc
                            _log_http_error(client, "chat_nonstream_http_error", fallback_exc)
                            await _reset_async_client(client)
                            async_client = _get_async_client(client)
                            continue
                        except (httpx.TimeoutException, httpx.TransportError) as fallback_exc:
                            last_error = fallback_exc
                            await _reset_async_client(client)
                            async_client = _get_async_client(client)
                            continue
                    if not openrouter_minimal_context_attempted:
                        openrouter_minimal_context_attempted = True
                        current_payload = _build_minimal_context_payload(
                            client,
                            messages=original_messages,
                        )
                        current_payload = _preflight_openrouter_payload(
                            client,
                            current_payload,
                            stage="minimal_context_retry",
                        )
                        log_kv(
                            client.log,
                            logging.WARNING,
                            "chat_stream_400_openrouter_minimal_context_retry",
                            attempt=attempt,
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "stream_400_openrouter_minimal_context_retry",
                                "retrying OpenRouter stream with minimal context",
                                attempt=attempt,
                                strategy="minimal_context_retry",
                            )
                        continue

                    log_kv(
                        client.log,
                        logging.ERROR,
                        "openrouter_input_validation_exhausted",
                        attempt=attempt,
                        provider_profile="openrouter",
                        message_count=len(original_messages),
                        recovery_stages_attempted=4,
                    )
                    if client.run_logger:
                        client.run_logger.log(
                            "chat",
                            "openrouter_input_validation_exhausted",
                            "All recovery stages exhausted for OpenRouter HTTP 400",
                            attempt=attempt,
                            provider_profile="openrouter",
                            message_count=len(original_messages),
                            recovery_stages_attempted=4,
                        )
                    exhausted_details = _provider_400_chunk_error_details(
                        client,
                        payload=current_payload,
                        exc=last_error if last_error is not None else exc,
                        attempt=attempt,
                        recovery_stages_attempted=4,
                    )
                    yield {
                        "type": "chunk_error",
                        "error": _provider_400_error_message(exhausted_details),
                        "details": exhausted_details,
                    }
                    return

            if status_code == 429:
                retry_after_seconds = _parse_retry_after_seconds(exc.response)
                if retry_after_seconds is not None:
                    log_kv(
                        client.log,
                        logging.WARNING,
                        "chat_stream_rate_limited",
                        attempt=attempt,
                        retry_after_sec=retry_after_seconds,
                    )
                    if client.run_logger:
                        client.run_logger.log(
                            "chat",
                            "stream_rate_limited",
                            "provider returned 429; retrying after backoff",
                            attempt=attempt,
                            retry_after_sec=retry_after_seconds,
                        )
            if status_code in {429, 502, 503, 504, 530}:
                last_error = exc
                await _reset_async_client(client)
                async_client = _get_async_client(client)
            elif client.adapter.should_retry_without_stream_options(exc):
                if "stream_options" not in current_payload:
                    raise
                current_payload = dict(current_payload)
                current_payload.pop("stream_options", None)
                log_kv(
                    client.log,
                    logging.WARNING,
                    "chat_stream_options_unsupported",
                    status=exc.response.status_code,
                )
                if client.run_logger:
                    client.run_logger.log(
                        "chat",
                        "stream_options_unsupported",
                        "retrying stream without usage options",
                        status=exc.response.status_code,
                    )
                continue
            else:
                raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if saw_chunk:
                if client.provider_profile == "lmstudio" or client.is_small_model:
                    if saw_tool_call_chunk or _is_tool_call_continuation_timeout(exc):
                        log_kv(
                            client.log,
                            logging.WARNING,
                            "chat_stream_incomplete_tool_call",
                            error=str(exc),
                            attempt=attempt,
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "stream_incomplete_tool_call",
                                "treating stalled tool call as retryable chunk error",
                                error=str(exc),
                                attempt=attempt,
                            )
                        yield {
                            "type": "chunk_error",
                            "error": "Incomplete tool call from provider stream",
                            "details": {
                                "reason": "tool_call_continuation_timeout",
                                "attempt": attempt,
                                "provider_profile": client.provider_profile,
                                "message": str(exc),
                            },
                        }
                        return
                    log_kv(
                        client.log,
                        logging.WARNING,
                        "chat_stream_stalled_after_chunks",
                        error=str(exc),
                        attempt=attempt,
                    )
                    if client.run_logger:
                        client.run_logger.log(
                            "chat",
                            "stream_stalled_after_chunks",
                            "treating stalled lmstudio stream as complete after partial output",
                            error=str(exc),
                            attempt=attempt,
                        )
                    yield {
                        "type": "stream_ended_without_done",
                        "details": {
                            "reason": "read_timeout_after_chunks",
                            "attempt": attempt,
                            "provider_profile": client.provider_profile,
                            "message": str(exc),
                            "tool_call_stream_active": saw_tool_call_chunk,
                        },
                    }
                    return
                raise
            last_error = exc
            log_kv(
                client.log,
                logging.WARNING,
                "chat_stream_transport_retry_nonstream",
                error=str(exc),
                attempt=attempt,
            )
            if client.run_logger:
                client.run_logger.log(
                    "chat",
                    "stream_transport_retry_nonstream",
                    "retrying as non-stream chat request",
                    error=str(exc),
                    attempt=attempt,
                )
            try:
                async for event in streamer.nonstream_chat(async_client, url, headers, current_payload):
                    yield event
                return
            except (httpx.TimeoutException, httpx.TransportError) as fallback_exc:
                last_error = fallback_exc
                await _reset_async_client(client)
                async_client = _get_async_client(client)
        if attempt < client.STREAM_RETRY_ATTEMPTS:
            backoff = float(attempt)
            if retry_after_seconds is not None:
                backoff = max(backoff, retry_after_seconds)
            log_kv(
                client.log,
                logging.WARNING,
                "chat_retry_scheduled",
                attempt=attempt + 1,
                delay_sec=backoff,
            )
            await asyncio.sleep(backoff)
    if last_error is not None:
        if (
            isinstance(last_error, httpx.HTTPStatusError)
            and int(getattr(last_error.response, "status_code", 0) or 0) == 400
            and client.provider_profile == "openrouter"
        ):
            exhausted_details = _provider_400_chunk_error_details(
                client,
                payload=current_payload,
                exc=last_error,
                attempt=client.STREAM_RETRY_ATTEMPTS,
                recovery_stages_attempted=3,
            )
            yield {
                "type": "chunk_error",
                "error": _provider_400_error_message(exhausted_details),
                "details": exhausted_details,
            }
            return
        raise last_error


async def fetch_model_context_limit(client: Any) -> int | None:
    if httpx is None:
        raise RuntimeError("Dependency missing: httpx")
    if not client.runtime_context_probe:
        return None
    headers = {"Authorization": f"Bearer {client.api_key}"}
    model_id = client.model
    model_url = f"{client.base_url}/models/{quote(model_id, safe='')}"
    list_url = f"{client.base_url}/models"
    async_client = _get_async_client(client)
    runtime_urls = [f"{client.base_url}/props", f"{client.base_url}/slots"]
    if client.base_url.endswith("/v1"):
        root = client.base_url[: -len("/v1")]
        runtime_urls.extend([f"{root}/props", f"{root}/slots"])
    log_kv(client.log, logging.DEBUG, "context_probe_start", model=model_id, base_url=client.base_url)
    for runtime_url in runtime_urls:
        try:
            response = await async_client.get(runtime_url, headers=headers, timeout=10.0)
            if response.status_code < 400:
                runtime_payload = response.json()
                runtime_limit = extract_runtime_context_limit(runtime_payload)
                if runtime_limit:
                    log_kv(client.log, logging.INFO, "context_probe_success", source="runtime", limit=runtime_limit)
                    return runtime_limit
        except Exception:
            pass
    try:
        response = await async_client.get(model_url, headers=headers, timeout=10.0)
        if response.status_code < 400:
            payload = response.json()
            limit = extract_context_limit(payload)
            if limit:
                log_kv(client.log, logging.INFO, "context_probe_success", source="model_metadata", limit=limit)
                return limit
    except Exception:
        pass
    try:
        response = await async_client.get(list_url, headers=headers, timeout=10.0)
        if response.status_code >= 400:
            return None
        payload = response.json()
    except Exception:
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return extract_context_limit(payload)

    selected: dict[str, Any] | None = None
    for item in data:
        if isinstance(item, dict) and str(item.get("id", "")) == model_id:
            selected = item
            break
    if selected is None:
        for item in data:
            if isinstance(item, dict) and model_id in str(item.get("id", "")):
                selected = item
                break
    if selected is not None:
        return extract_context_limit(selected)
    return extract_context_limit(payload)
