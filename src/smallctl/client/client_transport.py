from __future__ import annotations

import asyncio
import hashlib
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
from .transport_constants import (
    _DEFAULT_MAX_COMPLETION_TOKENS,
    _LOCAL_PATCH_INTENT_RE,
    _LOCAL_WRITE_INTENT_RE,
    _UNSET,
)
from .transport_error_classification import (
    _content_policy_violation_details,
    _content_policy_violation_message,
    _http_error_body,
    _is_llamacpp_jinja_system_message_error,
    _is_llamacpp_malformed_tool_call_json_error,
    _is_llamacpp_model_unloaded_chunk_error,
    _is_tool_call_continuation_timeout,
    _log_http_error,
    _log_transport_error,
    _openrouter_auth_failure_details,
    _openrouter_auth_failure_message,
    _openrouter_error_message_from_body,
    _summarize_http_error_body,
    _transport_error_details,
)
from .llamacpp_preflight import (
    _build_llamacpp_reduced_tools_payload,
    _build_minimal_context_payload,
    _llamacpp_budget_preflight,
)
from .openrouter_preflight import (
    _message_role_counts,
    _normalize_openrouter_tool_calls,
    _normalize_tool_schemas_for_openrouter,
    _preflight_openrouter_payload,
    _summarize_400_payload,
    _tool_name,
)
from .usage import extract_context_limit, extract_max_completion_tokens, extract_runtime_context_limit
from .usage import extract_supported_parameters


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


def _provider_root(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


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


# Re-export for backward compatibility
from .context_limit_probe import fetch_model_context_limit
