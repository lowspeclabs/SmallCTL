from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..logging_utils import log_kv
from .transport_constants import _LLAMACPP_CONTEXT_OVERFLOW_RE


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


def _llamacpp_malformed_tool_call_chunk_error_details(
    client: Any,
    *,
    payload: dict[str, Any] | None,
    exc: Any,
    attempt: int,
) -> dict[str, Any]:
    body = _http_error_body(exc, limit=2000)
    tool_name_hint = ""
    # Try to find tool name in error body first
    tool_match = re.search(r'"name"\s*:\s*"([^"]+)"', body)
    if tool_match:
        tool_name_hint = tool_match.group(1)
    # Fallback: extract from payload tools
    if not tool_name_hint and isinstance(payload, dict):
        tools = payload.get("tools") or []
        if isinstance(tools, list) and tools:
            first_tool = tools[0]
            if isinstance(first_tool, dict):
                tool_name_hint = str(first_tool.get("name") or first_tool.get("function", {}).get("name") or "").strip()
    # Extract partial tool call arguments from error message
    partial_preview = ""
    json_match = re.search(r'last read:\s*\'([^\']+)\'', body)
    if json_match:
        partial_preview = json_match.group(1)
    estimated_payload_tokens = 0
    if isinstance(payload, dict):
        try:
            import json
            estimated_payload_tokens = len(json.dumps(payload)) // 4
        except Exception:
            pass
    estimated_context_tokens_remaining = 0
    if hasattr(client, "context_limit") and client.context_limit:
        try:
            estimated_context_tokens_remaining = max(0, client.context_limit - estimated_payload_tokens)
        except Exception:
            pass
    return {
        "type": "malformed_tool_call_json",
        "reason": "tool_call_continuation_timeout",
        "recoverable": True,
        "tool_name_hint": tool_name_hint,
        "partial_tool_call_arguments_preview": partial_preview,
        "estimated_payload_tokens": estimated_payload_tokens,
        "estimated_context_tokens_remaining": estimated_context_tokens_remaining,
        "attempt": attempt,
        "error_kind": "llamacpp_malformed_tool_call_json",
    }


def _llamacpp_context_overflow_chunk_error_details(
    client: Any,
    *,
    payload: dict[str, Any] | None,
    body_summary: dict[str, Any],
    status_code: int,
    attempt: int,
) -> dict[str, Any]:
    return {
        "type": "context_budget_exceeded",
        "reason": "context_overflow",
        "request_tokens": body_summary.get("request_tokens", 0),
        "context_limit": body_summary.get("context_limit", 0),
        "recoverable": True,
        "attempt": attempt,
        "status_code": status_code,
        "error_kind": "llamacpp_context_overflow",
    }


def _provider_400_chunk_error_details(
    client: Any,
    *,
    payload: dict[str, Any],
    exc: Any,
    attempt: int,
    recovery_stages_attempted: int = 0,
) -> dict[str, Any]:
    body = _http_error_body(exc, limit=2000)
    body_summary = _summarize_http_error_body(body)
    role_counts: dict[str, int] = {}
    messages = payload.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role", "")).strip() or "unknown"
                role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "status_code": 400,
        "provider_profile": client.provider_profile,
        "upstream_provider": body_summary.get("upstream_provider", ""),
        "provider_error": body_summary.get("provider_error", ""),
        "model": client.model,
        "role_counts": role_counts,
        "attempt": attempt,
        "recovery_stages_attempted": recovery_stages_attempted,
        "error_kind": "provider_400_exhausted",
    }


def _provider_400_error_message(details: dict[str, Any]) -> str:
    profile = str(details.get("provider_profile") or "").strip()
    upstream = str(details.get("upstream_provider") or "").strip()
    provider_error = str(details.get("provider_error") or "").strip()
    parts = [f"{profile}/{upstream} input validation failed" if upstream else f"{profile} input validation failed"]
    if provider_error:
        parts.append(f": {provider_error}")
    return "".join(parts)


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


def _log_http_error(client: Any, event: str, exc: Any) -> None:
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
