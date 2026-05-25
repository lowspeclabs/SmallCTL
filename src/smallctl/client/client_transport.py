from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator
from urllib.parse import quote

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from ..logging_utils import log_kv
from ..redaction import redact_sensitive_messages, redact_sensitive_text
from .adapters.common import merge_system_messages_for_single_system_providers
from .chunk_parser import chunk_contains_tool_call_delta
from .provider_adapters import sanitize_messages_for_openrouter
from .request_budget import RequestEstimator
from .request_budget import approx_token_count as _budget_approx_token_count
from .request_budget import build_request_budget
from .request_budget import client_context_limit as _budget_client_context_limit
from .request_budget import json_size_bytes as _budget_json_size_bytes
from .streaming import SSEStreamer, summarize_stream_chunk
from .tool_budgeting import ToolBudgetResult, fit_tools_to_context_budget
from .usage import extract_context_limit, extract_max_completion_tokens, extract_runtime_context_limit
from .usage import extract_supported_parameters

_LLAMACPP_CONTEXT_OVERFLOW_RE = re.compile(
    r"request\s*\((?P<request_tokens>\d+)\s+tokens?\)\s+exceeds\s+the\s+available\s+context\s+size\s*\((?P<context_tokens>\d+)\s+tokens?\)",
    re.IGNORECASE,
)
_LOCAL_WRITE_INTENT_RE = re.compile(
    r"\b(build|create|implement|write|generate|add|make)\b.*\b(file|script|module|\.py|\.js|\.ts|\.md|\.txt)\b"
    r"|\b(file|script|module)\b.*\b(build|create|implement|write|generate|add|make)\b",
    re.IGNORECASE | re.DOTALL,
)
_LOCAL_PATCH_INTENT_RE = re.compile(
    r"\b(fix|patch|update|modify|edit|change|repair|refactor)\b",
    re.IGNORECASE,
)
_UNSET = object()
_DEFAULT_MAX_COMPLETION_TOKENS = 2048


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


def _is_llamacpp_model_unloaded_chunk_error(client: Any, event: dict[str, Any]) -> bool:
    if client.provider_profile != "llamacpp" or event.get("type") != "chunk_error":
        return False
    message = str(event.get("error") or "").strip().lower()
    details = event.get("details")
    detail_message = ""
    if isinstance(details, dict):
        detail_message = str(
            details.get("message")
            or details.get("error")
            or details.get("detail")
            or ""
        ).strip().lower()
    return "model is unloaded" in message or "model is unloaded" in detail_message


def _llamacpp_model_unloaded_details(
    client: Any,
    event: dict[str, Any],
    *,
    attempt: int,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = event.get("details")
    normalized = dict(details) if isinstance(details, dict) else {}
    normalized.update(
        {
            "type": "model_unloaded",
            "reason": "model_unloaded",
            "provider_profile": client.provider_profile,
            "model": client.model,
            "attempt": attempt,
            "message": str(
                normalized.get("message")
                or normalized.get("error")
                or event.get("error")
                or "Model is unloaded."
            ),
        }
    )
    if recovery:
        normalized["recovery"] = recovery
    return normalized


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


def _transport_error_details(
    client: Any,
    exc: Exception,
    *,
    url: str,
    attempt: int,
    phase: str,
) -> dict[str, Any]:
    return {
        "url": url,
        "attempt": attempt,
        "phase": phase,
        "provider_profile": client.provider_profile,
        "model": client.model,
        "exception_type": type(exc).__name__,
        "error": str(exc),
    }


def _log_transport_error(
    client: Any,
    event: str,
    exc: Exception,
    *,
    url: str,
    attempt: int,
    phase: str,
    level: int = logging.WARNING,
) -> None:
    details = _transport_error_details(
        client,
        exc,
        url=url,
        attempt=attempt,
        phase=phase,
    )
    log_kv(client.log, level, event, **details)
    if client.run_logger:
        client.run_logger.log(
            "chat",
            event,
            "chat transport error",
            **details,
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


def _repair_llamacpp_system_messages_for_transport(
    client: Any,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if client.provider_profile != "llamacpp":
        return messages

    system_positions = [
        index
        for index, message in enumerate(messages)
        if str(message.get("role") or "").strip().lower() == "system"
    ]
    if not system_positions:
        return messages
    if (
        system_positions == [0]
        and str(messages[0].get("role") or "").strip() == "system"
    ):
        return messages

    repaired = merge_system_messages_for_single_system_providers(messages)
    log_kv(
        client.log,
        logging.WARNING,
        "llamacpp_system_messages_repaired",
        system_count=len(system_positions),
        system_positions=system_positions,
    )
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "llamacpp_system_messages_repaired",
            "repaired llama.cpp system message order before transport",
            system_count=len(system_positions),
            system_positions=system_positions,
        )
    return repaired


def _is_llamacpp_jinja_system_message_error(client: Any, exc: Any) -> bool:
    if client.provider_profile != "llamacpp":
        return False
    try:
        if int(exc.response.status_code) != 500:
            return False
    except Exception:
        return False
    body = _http_error_body(exc).lower()
    return "jinja" in body and "system message" in body


def _is_llamacpp_malformed_tool_call_json_error(client: Any, exc: Any) -> bool:
    if client.provider_profile != "llamacpp":
        return False
    try:
        if int(exc.response.status_code) != 500:
            return False
    except Exception:
        return False
    body = _http_error_body(exc, limit=12000).lower()
    return (
        "failed to parse tool call arguments as json" in body
        and "json.exception.parse_error" in body
        and ("missing closing quote" in body or "invalid string" in body)
    )


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


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


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


def _remember_context_limit(client: Any, limit: int | None) -> int | None:
    if limit is None:
        return None
    try:
        normalized = int(limit)
    except Exception:
        return None
    if normalized <= 0:
        return None
    try:
        client.runtime_context_limit = normalized
    except Exception:
        pass
    return normalized


def _remember_model_metadata(client: Any, payload: Any, *, source: str) -> int | None:
    if isinstance(payload, dict):
        try:
            client.model_metadata = dict(payload)
        except Exception:
            pass
        try:
            client.model_metadata_source = source
        except Exception:
            pass

        max_completion_tokens = extract_max_completion_tokens(payload)
        if max_completion_tokens is not None:
            try:
                client.model_max_completion_tokens = int(max_completion_tokens)
            except Exception:
                pass

        supported_parameters = extract_supported_parameters(payload)
        if supported_parameters is not None:
            try:
                client.model_supported_parameters = list(supported_parameters)
            except Exception:
                pass

    return _remember_context_limit(client, extract_context_limit(payload))


def _http_error_body(exc: Any, *, limit: int = 1000) -> str:
    try:
        return str(exc.response.text or "")[:limit]
    except Exception:
        return ""


def _provider_root(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


def _openrouter_error_message_from_body(body: str) -> str:
    try:
        payload = json.loads(str(body or ""))
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            if message:
                return message
        message = str(payload.get("message") or "").strip()
        if message:
            return message
    return str(body or "").strip()


def _openrouter_auth_failure_details(
    client: Any,
    *,
    status_code: int,
    body: str,
    phase: str,
) -> dict[str, Any]:
    provider_error = _openrouter_error_message_from_body(body)
    return {
        "type": "provider_authentication",
        "reason": "openrouter_authentication_failed",
        "provider_profile": "openrouter",
        "model": client.model,
        "status_code": int(status_code),
        "provider_error": provider_error,
        "body": str(body or "")[:1000],
        "phase": phase,
        "recoverable": False,
        "hint": "Update SMALLCTL_API_KEY with a valid OpenRouter API key from the correct account.",
    }


def _openrouter_auth_failure_message(details: dict[str, Any]) -> str:
    provider_error = str(details.get("provider_error") or "").strip()
    suffix = f" ({provider_error})" if provider_error else ""
    return (
        "OpenRouter authentication failed: API key is invalid, revoked, "
        f"or belongs to a missing account{suffix}. Update SMALLCTL_API_KEY."
    )


def _content_policy_violation_details(
    client: Any,
    *,
    status_code: int,
    body: str,
    phase: str,
) -> dict[str, Any]:
    provider_error = _openrouter_error_message_from_body(body)
    is_empty_body = not bool(str(body or "").strip())
    return {
        "type": "content_policy_violation",
        "reason": "provider_content_policy_block",
        "provider_profile": client.provider_profile,
        "model": client.model,
        "status_code": int(status_code),
        "provider_error": provider_error,
        "body": str(body or "")[:1000],
        "phase": phase,
        "recoverable": False,
        "hint": (
            "The provider rejected this request as a content-policy violation. "
            "This often happens when passwords, credentials, or PII appear in the prompt. "
            "Use .env or tool configuration for secrets instead of including them in task text."
        ),
        "empty_body": is_empty_body,
    }


def _content_policy_violation_message(details: dict[str, Any]) -> str:
    provider_error = str(details.get("provider_error") or "").strip()
    suffix = f" ({provider_error})" if provider_error else ""
    return (
        "Provider content policy violation: the request was blocked by the provider's safety filter"
        f"{suffix}. Remove credentials or sensitive data from the prompt and retry."
    )


async def _preflight_openrouter_auth(client: Any, async_client: Any) -> dict[str, Any] | None:
    if client.provider_profile != "openrouter":
        return None
    normalized_base = str(client.base_url or "").strip().rstrip("/")
    if normalized_base.endswith("/api/v1"):
        credits_url = f"{normalized_base}/credits"
    else:
        credits_url = f"{_provider_root(normalized_base)}/api/v1/credits"
    headers = client.adapter.mutate_headers(
        {
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json",
        }
    )
    try:
        response = await async_client.get(credits_url, headers=headers)
    except Exception as exc:
        log_kv(
            client.log,
            logging.WARNING,
            "openrouter_auth_preflight_error",
            error=str(exc),
            url=credits_url,
        )
        if client.run_logger:
            client.run_logger.log(
                "chat",
                "openrouter_auth_preflight_error",
                "OpenRouter auth preflight could not be completed",
                error=str(exc),
                url=credits_url,
            )
        return None
    if int(response.status_code) != 401:
        return None
    body = str(getattr(response, "text", "") or "")[:1000]
    details = _openrouter_auth_failure_details(
        client,
        status_code=401,
        body=body,
        phase="auth_preflight",
    )
    log_kv(client.log, logging.ERROR, "openrouter_auth_failed", **details)
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "openrouter_auth_failed",
            "OpenRouter authentication failed during preflight",
            **details,
        )
    return details


def _choose_write_tool_hint(payload: dict[str, Any]) -> str:
    tool_names = [_tool_name(tool) for tool in payload.get("tools", []) if _tool_name(tool)]
    for preferred in (
        "file_write",
        "ssh_file_write",
        "file_append",
        "file_patch",
        "ssh_file_patch",
        "ssh_file_replace_between",
        "ast_patch",
    ):
        if preferred in tool_names:
            return preferred
    return tool_names[0] if tool_names else "file_write"


def _extract_llamacpp_last_read_preview(body: str, *, limit: int = 2000) -> str:
    marker = "last read:"
    lower = str(body or "").lower()
    index = lower.rfind(marker)
    if index < 0:
        return ""
    preview = str(body)[index + len(marker) :].strip()
    if len(preview) > limit:
        preview = preview[:limit].rstrip() + "..."
    return preview


def _llamacpp_malformed_tool_call_chunk_error_details(
    client: Any,
    *,
    payload: dict[str, Any],
    exc: Any,
    attempt: int,
) -> dict[str, Any]:
    body = _http_error_body(exc, limit=12000)
    body_summary = _summarize_http_error_body(body)
    return {
        "type": "malformed_tool_call_json",
        # Reuse the existing incomplete-tool-call recovery path: the server saw
        # an unfinished JSON argument string even if the stream had already ended.
        "reason": "tool_call_continuation_timeout",
        "provider_profile": client.provider_profile,
        "model": client.model,
        "status_code": 500,
        "recoverable": True,
        "attempt": attempt,
        "tool_name_hint": _choose_write_tool_hint(payload),
        "partial_tool_call_arguments_preview": _extract_llamacpp_last_read_preview(body),
        **_summarize_400_payload(payload, context_limit=_client_context_limit(client)),
        **body_summary,
    }


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
    context_match = _LLAMACPP_CONTEXT_OVERFLOW_RE.search(snippet)
    if context_match:
        request_tokens = int(context_match.group("request_tokens"))
        context_tokens = int(context_match.group("context_tokens"))
        summary.update(
            {
                "provider_error": "Context overflow",
                "context_overflow": True,
                "request_tokens": request_tokens,
                "context_limit": context_tokens,
                "context_overflow_tokens": request_tokens - context_tokens,
            }
        )
    return summary


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


def _latest_user_message_audit(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"latest_user_present": False}
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip() != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            text = "\n".join(text_parts)
        else:
            text = str(content or "")
        encoded = text.encode("utf-8", errors="replace")
        preview = redact_sensitive_text(text).replace("\n", "\\n")[:240]
        return {
            "latest_user_present": bool(text.strip()),
            "latest_user_sha256": hashlib.sha256(encoded).hexdigest()[:12] if text else "",
            "latest_user_chars": len(text),
            "latest_user_preview": preview,
        }
    return {"latest_user_present": False}


def _log_request_audit(client: Any, *, payload: dict[str, Any], tools: list[dict[str, Any]], stage: str) -> None:
    messages = payload.get("messages")
    payload_tools = payload.get("tools")
    active_tools = payload_tools if isinstance(payload_tools, list) else tools
    tool_names = [_tool_name(tool) for tool in active_tools if _tool_name(tool)]
    details = {
        "stage": stage,
        "provider_profile": client.provider_profile,
        "model": client.model,
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "role_counts": _message_role_counts(messages),
        "tool_count": len(active_tools),
        "tool_names": tool_names,
        **_latest_user_message_audit(messages),
    }
    log_kv(client.log, logging.INFO, "chat_request_payload_audit", **details)
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "request_payload_audit",
            "chat request payload audit",
            **details,
        )


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
        **_summarize_400_payload(payload, context_limit=_client_context_limit(client)),
        **_summarize_http_error_body(body),
    }


def _llamacpp_context_overflow_chunk_error_details(
    client: Any,
    *,
    payload: dict[str, Any],
    body_summary: dict[str, Any],
    status_code: int,
    attempt: int,
) -> dict[str, Any]:
    context_limit = body_summary.get("context_limit")
    request_tokens = body_summary.get("request_tokens")
    budget_limit = context_limit if isinstance(context_limit, int) else _client_context_limit(client)
    budget = build_request_budget(budget_limit) if isinstance(budget_limit, int) else None
    details: dict[str, Any] = {
        "type": "context_budget_exceeded",
        "reason": "context_overflow",
        "provider_profile": client.provider_profile,
        "model": client.model,
        "status_code": status_code,
        "attempt": attempt,
        "recoverable": True,
        **_summarize_400_payload(payload, context_limit=budget_limit),
        **body_summary,
    }
    if budget is not None:
        details.update(
            {
                "effective_prompt_budget": budget.effective_prompt_budget,
                "reserve_completion_tokens": budget.reserve_completion_tokens,
                "safety_margin_tokens": budget.safety_margin_tokens,
                "tokenizer_slop_tokens": budget.tokenizer_slop_tokens,
            }
        )
    if isinstance(request_tokens, int) and isinstance(budget_limit, int):
        details["over_budget_tokens"] = max(0, request_tokens - budget_limit)
    return details


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


def _log_llamacpp_budget_preflight(
    client: Any,
    *,
    stage: str,
    action: str,
    result: ToolBudgetResult | None,
    context_limit: int | None,
    budget_context_limit: int | None = None,
    context_limit_source: str = "observed",
    reduction_reason: str = "",
) -> None:
    details: dict[str, Any] = {
        "stage": stage,
        "provider_profile": client.provider_profile,
        "model": client.model,
        "context_limit": context_limit,
        "context_limit_source": context_limit_source,
        "budget_action": action,
    }
    if reduction_reason:
        details["reduction_reason"] = reduction_reason
    if result is not None:
        budget_limit = context_limit if budget_context_limit is None else budget_context_limit
        budget = (
            build_request_budget(result.footprint.estimated_payload_tokens)
            if budget_limit is None
            else build_request_budget(budget_limit)
        )
        details.update(
            {
                "effective_prompt_budget": budget.effective_prompt_budget,
                "reserve_completion_tokens": budget.reserve_completion_tokens,
                "safety_margin_tokens": budget.safety_margin_tokens,
                "tokenizer_slop_tokens": budget.tokenizer_slop_tokens,
                "estimated_payload_tokens": result.footprint.estimated_payload_tokens,
                "estimated_message_tokens": result.footprint.estimated_message_tokens,
                "estimated_tool_tokens": result.footprint.estimated_tool_tokens,
                "tool_count_before": result.tool_count_before,
                "tool_count_after": result.tool_count_after,
                "dropped_tool_names": list(result.dropped_tool_names),
                "kept_tool_names": list(result.kept_tool_names),
                "over_budget_tokens": result.footprint.over_budget_tokens,
            }
        )
    log_kv(client.log, logging.DEBUG, "chat_payload_preflight_budget", **details)
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "payload_preflight_budget",
            "chat payload budget preflight",
            **details,
        )


def _llamacpp_budget_preflight(
    client: Any,
    *,
    payload: dict[str, Any],
    stage: str,
    context_limit: int | None = None,
    context_limit_source: str = "observed",
    reduction_reason: str = "",
    log_context_limit: Any = _UNSET,
) -> ToolBudgetResult | None:
    if client.provider_profile != "llamacpp":
        return None
    limit = context_limit or _client_context_limit(client)
    raw_tools = payload.get("tools")
    tools = raw_tools if isinstance(raw_tools, list) else []
    if limit is None:
        estimator = RequestEstimator()
        footprint = estimator.footprint(payload)
        log_kv(
            client.log,
            logging.DEBUG,
            "chat_payload_preflight_budget",
            stage=stage,
            provider_profile=client.provider_profile,
            model=client.model,
            context_limit=None,
            budget_action="skipped_unknown_limit",
            estimated_payload_tokens=footprint.estimated_payload_tokens,
            estimated_message_tokens=footprint.estimated_message_tokens,
            estimated_tool_tokens=footprint.estimated_tool_tokens,
            tool_count_before=footprint.tool_count,
            tool_count_after=footprint.tool_count,
            dropped_tool_names=[],
            kept_tool_names=[_tool_name(tool) for tool in tools if _tool_name(tool)],
            over_budget_tokens=0,
        )
        if client.run_logger:
            client.run_logger.log(
                "chat",
                "payload_preflight_budget",
                "chat payload budget preflight",
                stage=stage,
                provider_profile=client.provider_profile,
                model=client.model,
                context_limit=None,
                budget_action="skipped_unknown_limit",
                estimated_payload_tokens=footprint.estimated_payload_tokens,
                estimated_message_tokens=footprint.estimated_message_tokens,
                estimated_tool_tokens=footprint.estimated_tool_tokens,
                tool_count_before=footprint.tool_count,
                tool_count_after=footprint.tool_count,
                dropped_tool_names=[],
                kept_tool_names=[_tool_name(tool) for tool in tools if _tool_name(tool)],
                over_budget_tokens=0,
            )
        return None

    budget = build_request_budget(limit)
    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=budget,
        requested_tool_name=_infer_llamacpp_requested_tool(payload, tools),
        estimator=RequestEstimator(),
    )
    displayed_context_limit = limit if log_context_limit is _UNSET else log_context_limit
    _log_llamacpp_budget_preflight(
        client,
        stage=stage,
        action=result.action,
        result=result,
        context_limit=displayed_context_limit,
        budget_context_limit=limit,
        context_limit_source=context_limit_source,
        reduction_reason=reduction_reason,
    )
    return result


def _payload_text_for_tool_inference(payload: dict[str, Any], *, limit: int = 12000) -> str:
    pieces: list[str] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(str(item.get("text") or ""))
        if sum(len(piece) for piece in pieces) >= limit:
            break
    return "\n".join(pieces)[-limit:]


def _infer_llamacpp_requested_tool(payload: dict[str, Any], tools: list[dict[str, Any]]) -> str:
    """Keep the task-critical mutation tool alive when shrinking llama.cpp payloads."""
    available = {_tool_name(tool) for tool in tools if _tool_name(tool)}
    text = _payload_text_for_tool_inference(payload)
    if {"file_write", "file_patch", "ast_patch"} & available:
        if "file_write" in available and _LOCAL_WRITE_INTENT_RE.search(text):
            return "file_write"
        if "file_patch" in available and _LOCAL_PATCH_INTENT_RE.search(text):
            return "file_patch"
        if "ast_patch" in available and _LOCAL_PATCH_INTENT_RE.search(text):
            return "ast_patch"
    if {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"} & available:
        lower_text = text.lower()
        if "ssh_file_write" in available and ("ssh" in lower_text or "remote" in lower_text):
            if _LOCAL_WRITE_INTENT_RE.search(text):
                return "ssh_file_write"
        if "ssh_file_patch" in available and ("ssh" in lower_text or "remote" in lower_text):
            if _LOCAL_PATCH_INTENT_RE.search(text):
                return "ssh_file_patch"
    return ""


def _build_llamacpp_reduced_tools_payload(
    client: Any,
    *,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    tools = payload.get("tools")
    if not isinstance(tools, list) or len(tools) < 12:
        return None
    limit = _client_context_limit(client)
    limit_was_observed = limit is not None
    if limit is None:
        limit = 1
    result = _llamacpp_budget_preflight(
        client,
        payload=payload,
        stage="http_400_reduced_tools_retry",
        context_limit=limit,
        context_limit_source="unknown",
        reduction_reason="http_400_recovery",
        log_context_limit=None if not limit_was_observed else limit,
    )
    if result is None or result.tool_count_after >= result.tool_count_before:
        return None
    if result.action == "exceeded" and limit_was_observed:
        return None
    return client.adapter.mutate_payload(result.payload)


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
    messages = _repair_llamacpp_system_messages_for_transport(client, messages)
    redacted_messages = redact_sensitive_messages(messages)
    if redacted_messages != messages:
        log_kv(
            client.log,
            logging.WARNING,
            "prompt_sensitive_data_redacted",
            message_count=len(messages),
        )
        if client.run_logger:
            client.run_logger.log(
                "chat",
                "prompt_sensitive_data_redacted",
                "sensitive data redacted from prompt before sending to provider",
                message_count=len(messages),
            )
        messages = redacted_messages

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
    request_max_tokens = client._request_max_completion_tokens(tools)
    if request_max_tokens is not None:
        payload["max_tokens"] = int(request_max_tokens)
    if getattr(client, "temperature", None) is not None:
        payload["temperature"] = float(getattr(client, "temperature"))
    if tools:
        payload["tools"] = tools
    if client.adapter.stream_policy.supports_stream_options:
        payload["stream_options"] = {"include_usage": True}
    payload = client.adapter.mutate_payload(payload)
    payload = _preflight_openrouter_payload(client, payload, stage="initial")
    initial_llamacpp_budget = _llamacpp_budget_preflight(
        client,
        payload=payload,
        stage="initial",
    )
    if initial_llamacpp_budget is not None:
        if initial_llamacpp_budget.action == "exceeded":
            known_limit = _client_context_limit(client)
            budget = build_request_budget(known_limit) if known_limit is not None else None
            yield {
                "type": "chunk_error",
                "error": "llamacpp context budget exceeded before request",
                "details": {
                    "type": "context_budget_exceeded",
                    "provider_profile": client.provider_profile,
                    "model": client.model,
                    "context_limit": known_limit,
                    "effective_prompt_budget": budget.effective_prompt_budget if budget else None,
                    "reserve_completion_tokens": budget.reserve_completion_tokens if budget else None,
                    "safety_margin_tokens": budget.safety_margin_tokens if budget else None,
                    "tokenizer_slop_tokens": budget.tokenizer_slop_tokens if budget else None,
                    "estimated_payload_tokens": initial_llamacpp_budget.footprint.estimated_payload_tokens,
                    "estimated_message_tokens": initial_llamacpp_budget.footprint.estimated_message_tokens,
                    "estimated_tool_tokens": initial_llamacpp_budget.footprint.estimated_tool_tokens,
                    "tool_count_before": initial_llamacpp_budget.tool_count_before,
                    "tool_count_after": initial_llamacpp_budget.tool_count_after,
                    "dropped_tool_names": list(initial_llamacpp_budget.dropped_tool_names),
                    "kept_tool_names": list(initial_llamacpp_budget.kept_tool_names),
                    "over_budget_tokens": initial_llamacpp_budget.footprint.over_budget_tokens,
                    "recoverable": False,
                },
            }
            return
        payload = initial_llamacpp_budget.payload
    _log_request_audit(client, payload=payload, tools=tools, stage="initial")

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
    llamacpp_reduced_tools_attempted = False
    llamacpp_jinja_repair_attempted = False
    async_client = _get_async_client(client)
    openrouter_auth_failure = await _preflight_openrouter_auth(client, async_client)
    if openrouter_auth_failure is not None:
        yield {
            "type": "chunk_error",
            "error": _openrouter_auth_failure_message(openrouter_auth_failure),
            "details": openrouter_auth_failure,
        }
        return
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
        recent_chunks: list[dict[str, Any]] = []
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
                        "details": {**wedge_details, "last_chunks": wedge_details.get("last_chunks", recent_chunks)},
                    }
                    return
                if event.get("type") == "chunk":
                    saw_chunk = True
                    recent_chunks = (recent_chunks + [summarize_stream_chunk(event.get("data", {}))])[-5:]
                    if chunk_contains_tool_call_delta(event.get("data", {})):
                        saw_tool_call_chunk = True
                if _is_llamacpp_model_unloaded_chunk_error(client, event):
                    details = _llamacpp_model_unloaded_details(client, event, attempt=attempt)
                    log_kv(
                        client.log,
                        logging.WARNING,
                        "chat_stream_llamacpp_model_unloaded",
                        attempt=attempt,
                        provider_profile=client.provider_profile,
                        model=client.model,
                    )
                    if client.run_logger:
                        client.run_logger.log(
                            "chat",
                            "stream_llamacpp_model_unloaded",
                            "llama.cpp reported the model is unloaded",
                            attempt=attempt,
                            provider_profile=client.provider_profile,
                            model=client.model,
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
                            reason="model_unloaded",
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "backend_recovery_succeeded",
                                "backend recovery succeeded after llama.cpp model-unloaded error",
                                attempt=attempt,
                                provider_profile=client.provider_profile,
                                action=recovery.get("action"),
                                reason="model_unloaded",
                            )
                        await _reset_async_client(client)
                        async_client = _get_async_client(client)
                        break
                    yield {
                        "type": "chunk_error",
                        "error": "llama.cpp model is unloaded",
                        "details": _llamacpp_model_unloaded_details(
                            client,
                            event,
                            attempt=attempt,
                            recovery=recovery if isinstance(recovery, dict) else None,
                        ),
                    }
                    return
                yield event
            if retry_after_backend_recovery:
                continue
            return
        except httpx.HTTPStatusError as exc:
            _log_http_error(client, "chat_stream_http_error", exc)
            status_code = int(exc.response.status_code)
            if client.provider_profile == "openrouter" and status_code == 401:
                details = _openrouter_auth_failure_details(
                    client,
                    status_code=status_code,
                    body=_http_error_body(exc),
                    phase="chat_completion",
                )
                log_kv(client.log, logging.ERROR, "openrouter_auth_failed", **details)
                if client.run_logger:
                    client.run_logger.log(
                        "chat",
                        "openrouter_auth_failed",
                        "OpenRouter authentication failed during chat completion",
                        **details,
                    )
                yield {
                    "type": "chunk_error",
                    "error": _openrouter_auth_failure_message(details),
                    "details": details,
                }
                return
            if status_code == 403:
                body = _http_error_body(exc)
                details = _content_policy_violation_details(
                    client,
                    status_code=status_code,
                    body=body,
                    phase="chat_completion",
                )
                log_kv(client.log, logging.ERROR, "content_policy_violation", **details)
                if client.run_logger:
                    client.run_logger.log(
                        "chat",
                        "content_policy_violation",
                        "provider rejected request as content policy violation",
                        **details,
                    )
                yield {
                    "type": "chunk_error",
                    "error": _content_policy_violation_message(details),
                    "details": details,
                }
                return
            if (
                _is_llamacpp_jinja_system_message_error(client, exc)
                and not llamacpp_jinja_repair_attempted
            ):
                llamacpp_jinja_repair_attempted = True
                current_payload = dict(current_payload)
                current_payload["messages"] = _repair_llamacpp_system_messages_for_transport(
                    client,
                    list(current_payload.get("messages") or []),
                )
                log_kv(
                    client.log,
                    logging.WARNING,
                    "chat_stream_500_llamacpp_jinja_retry",
                    attempt=attempt,
                    strategy="repair_system_messages",
                )
                if client.run_logger:
                    client.run_logger.log(
                        "chat",
                        "stream_500_llamacpp_jinja_retry",
                        "retrying llama.cpp stream after Jinja system-message error",
                        attempt=attempt,
                        strategy="repair_system_messages",
                    )
                await _reset_async_client(client)
                async_client = _get_async_client(client)
                continue
            if _is_llamacpp_malformed_tool_call_json_error(client, exc):
                details = _llamacpp_malformed_tool_call_chunk_error_details(
                    client,
                    payload=current_payload,
                    exc=exc,
                    attempt=attempt,
                )
                log_kv(
                    client.log,
                    logging.WARNING,
                    "chat_stream_500_llamacpp_malformed_tool_call",
                    attempt=attempt,
                    provider_profile=client.provider_profile,
                    tool_name_hint=details.get("tool_name_hint"),
                    estimated_payload_tokens=details.get("estimated_payload_tokens"),
                    estimated_context_tokens_remaining=details.get("estimated_context_tokens_remaining"),
                )
                if client.run_logger:
                    client.run_logger.log(
                        "chat",
                        "stream_500_llamacpp_malformed_tool_call",
                        "recovering from malformed llama.cpp tool-call JSON",
                        attempt=attempt,
                        provider_profile=client.provider_profile,
                        tool_name_hint=details.get("tool_name_hint"),
                        estimated_payload_tokens=details.get("estimated_payload_tokens"),
                        estimated_context_tokens_remaining=details.get("estimated_context_tokens_remaining"),
                    )
                yield {
                    "type": "chunk_error",
                    "error": "llama.cpp returned malformed tool-call JSON",
                    "details": details,
                }
                return
            if status_code == 400:
                body_summary = _summarize_http_error_body(_http_error_body(exc))
                observed_context_limit = body_summary.get("context_limit")
                if client.provider_profile == "llamacpp":
                    _remember_context_limit(
                        client,
                        observed_context_limit if isinstance(observed_context_limit, int) else None,
                    )
                    if body_summary.get("context_overflow") is True:
                        details = _llamacpp_context_overflow_chunk_error_details(
                            client,
                            payload=current_payload,
                            body_summary=body_summary,
                            status_code=status_code,
                            attempt=attempt,
                        )
                        yield {
                            "type": "chunk_error",
                            "error": "llamacpp context window exceeded",
                            "details": details,
                        }
                        return
                diagnostics = {
                    **_summarize_400_payload(current_payload, context_limit=_client_context_limit(client)),
                    **body_summary,
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
                            if int(fallback_exc.response.status_code) == 403:
                                body = _http_error_body(fallback_exc)
                                details = _content_policy_violation_details(
                                    client,
                                    status_code=403,
                                    body=body,
                                    phase="nonstream_fallback",
                                )
                                log_kv(client.log, logging.ERROR, "content_policy_violation_nonstream", **details)
                                if client.run_logger:
                                    client.run_logger.log(
                                        "chat",
                                        "content_policy_violation_nonstream",
                                        "provider rejected non-stream request as content policy violation",
                                        **details,
                                    )
                                yield {
                                    "type": "chunk_error",
                                    "error": _content_policy_violation_message(details),
                                    "details": details,
                                }
                                return
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
                if client.provider_profile == "llamacpp" and not llamacpp_reduced_tools_attempted:
                    fallback_limit = _client_context_limit(client)
                    fallback_limit_was_observed = fallback_limit is not None
                    if fallback_limit is None:
                        fallback_limit = 1
                    budget_result = _llamacpp_budget_preflight(
                        client,
                        payload=current_payload,
                        stage="http_400_reduced_tools_retry",
                        context_limit=fallback_limit,
                        context_limit_source="observed" if fallback_limit_was_observed else "unknown",
                        reduction_reason="http_400_recovery",
                        log_context_limit=None if not fallback_limit_was_observed else fallback_limit,
                    )
                    if (
                        budget_result is not None
                        and budget_result.action == "exceeded"
                        and fallback_limit_was_observed
                    ):
                        yield {
                            "type": "chunk_error",
                            "error": "llamacpp context budget exceeded before request",
                            "details": {
                                "type": "context_budget_exceeded",
                                "provider_profile": client.provider_profile,
                                "model": client.model,
                                "context_limit": fallback_limit,
                                "effective_prompt_budget": build_request_budget(
                                    fallback_limit
                                ).effective_prompt_budget,
                                "tokenizer_slop_tokens": build_request_budget(
                                    fallback_limit
                                ).tokenizer_slop_tokens,
                                "estimated_payload_tokens": budget_result.footprint.estimated_payload_tokens,
                                "estimated_message_tokens": budget_result.footprint.estimated_message_tokens,
                                "estimated_tool_tokens": budget_result.footprint.estimated_tool_tokens,
                                "tool_count_before": budget_result.tool_count_before,
                                "tool_count_after": budget_result.tool_count_after,
                                "dropped_tool_names": list(budget_result.dropped_tool_names),
                                "kept_tool_names": list(budget_result.kept_tool_names),
                                "over_budget_tokens": budget_result.footprint.over_budget_tokens,
                                "recoverable": False,
                            },
                        }
                        return
                    if (
                        budget_result is not None
                        and budget_result.tool_count_after < budget_result.tool_count_before
                    ):
                        previous_tool_count = int(diagnostics.get("tool_schema_count") or 0)
                        current_payload = client.adapter.mutate_payload(budget_result.payload)
                        llamacpp_reduced_tools_attempted = True
                        reduced_diagnostics = _summarize_400_payload(
                            current_payload,
                            context_limit=_client_context_limit(client),
                        )
                        log_kv(
                            client.log,
                            logging.WARNING,
                            "chat_stream_400_llamacpp_retry",
                            attempt=attempt,
                            strategy="reduced_tools_retry",
                            previous_tool_count=previous_tool_count,
                            reduced_tool_count=reduced_diagnostics["tool_schema_count"],
                            reduced_tool_names=reduced_diagnostics["tool_names"],
                            reduced_tool_schema_bytes=reduced_diagnostics["tool_schema_bytes"],
                            reduced_payload_bytes=reduced_diagnostics["payload_bytes"],
                        )
                        if client.run_logger:
                            client.run_logger.log(
                                "chat",
                                "stream_400_llamacpp_retry",
                                "retrying llama.cpp stream with reduced tool schemas",
                                attempt=attempt,
                                strategy="reduced_tools_retry",
                                previous_tool_count=previous_tool_count,
                                reduced_tool_count=reduced_diagnostics["tool_schema_count"],
                                reduced_tool_names=reduced_diagnostics["tool_names"],
                                reduced_tool_schema_bytes=reduced_diagnostics["tool_schema_bytes"],
                                reduced_payload_bytes=reduced_diagnostics["payload_bytes"],
                            )
                        continue

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
                                "last_chunks": recent_chunks,
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
                            "last_chunks": recent_chunks,
                        },
                    }
                    return
                raise
            last_error = exc
            _log_transport_error(
                client,
                "chat_stream_transport_retry_nonstream",
                exc,
                url=url,
                attempt=attempt,
                phase="stream",
            )
            try:
                async for event in streamer.nonstream_chat(async_client, url, headers, current_payload):
                    yield event
                return
            except (httpx.TimeoutException, httpx.TransportError) as fallback_exc:
                last_error = fallback_exc
                _log_transport_error(
                    client,
                    "chat_nonstream_transport_error",
                    fallback_exc,
                    url=url,
                    attempt=attempt,
                    phase="nonstream_fallback",
                )
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
        if isinstance(last_error, (httpx.TimeoutException, httpx.TransportError)):
            _log_transport_error(
                client,
                "chat_transport_exhausted",
                last_error,
                url=url,
                attempt=client.STREAM_RETRY_ATTEMPTS,
                phase="exhausted",
                level=logging.ERROR,
            )
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
                    return _remember_context_limit(client, runtime_limit)
        except Exception:
            pass
    try:
        response = await async_client.get(model_url, headers=headers, timeout=10.0)
        if response.status_code < 400:
            payload = response.json()
            limit = _remember_model_metadata(client, payload, source="model_metadata")
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
        return _remember_model_metadata(client, payload, source="model_list")

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
        return _remember_model_metadata(client, selected, source="model_list")
    return _remember_model_metadata(client, payload, source="model_list")
