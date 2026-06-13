from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from ..redaction import redact_sensitive_text
from .request_budget import approx_token_count as _budget_approx_token_count


def request_first_token_timeout_sec(client: Any, tools: list[dict[str, Any]]) -> float:
    timeout = float(client.first_token_timeout_sec)
    if _request_has_write_heavy_tool(client, tools):
        multiplier = float(getattr(client, "WRITE_HEAVY_FIRST_TOKEN_TIMEOUT_MULTIPLIER", 2.0))
        return max(timeout, timeout * multiplier)
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


def _request_has_write_heavy_tool(client: Any, tools: list[dict[str, Any]]) -> bool:
    if not tools:
        return False
    write_tool_names = set(getattr(client, "_WRITE_HEAVY_TOOL_NAMES", set()) or set())
    write_argument_fields = set(getattr(client, "_WRITE_HEAVY_ARGUMENT_FIELDS", set()) or set())
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name") or "").strip()
        parameters = function.get("parameters")
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        property_names = {str(name).strip() for name in properties} if isinstance(properties, dict) else set()
        if tool_name in write_tool_names or property_names & write_argument_fields:
            return True
    return False


def llamacpp_model_unloaded_details(
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


def parse_retry_after_seconds(response: Any) -> float | None:
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


def extract_available_tool_names(tools: list[dict[str, Any]]) -> set[str]:
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


def tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def context_pressure_diagnostics(payload: dict[str, Any], *, context_limit: int | None) -> dict[str, Any]:
    estimated_payload_tokens = _budget_approx_token_count(payload)
    estimated_tool_schema_tokens = _budget_approx_token_count(payload.get("tools", []))
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


def provider_root(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


def latest_user_message_audit(messages: Any) -> dict[str, Any]:
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
