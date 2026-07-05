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


def resolve_prompt_processing_timeout_sec(client: Any) -> float:
    """Return the prompt-processing watchdog for this request, or 0 to disable."""
    adapter_timeout = float(getattr(getattr(getattr(client, "adapter", None), "stream_policy", None), "prompt_processing_timeout_sec", 0.0) or 0.0)
    if adapter_timeout <= 0:
        return 0.0
    # Only enable the prompt-processing watchdog for small Gemma-4 on llama.cpp,
    # where repeated SWA/cache invalidation turns long contexts into wall-clock
    # stalls before the first generation token.
    if client.provider_profile != "llamacpp":
        return 0.0
    from ..graph.tool_model_rules_model_detection import _model_is_gemma_4_small
    if not _model_is_gemma_4_small(getattr(client, "model", None)):
        return 0.0
    return adapter_timeout


def _request_has_write_heavy_tool(client: Any, tools: list[dict[str, Any]]) -> bool:
    if not tools:
        return False
    write_tool_names = set(getattr(client, "_WRITE_HEAVY_TOOL_NAMES", set()) or set())
    write_argument_fields = set(getattr(client, "_WRITE_HEAVY_ARGUMENT_FIELDS", set()) or set())
    write_argument_tool_allowlist = set(getattr(client, "_WRITE_HEAVY_ARGUMENT_TOOL_ALLOWLIST", set()) or set())
    readonly_tool_denylist = set(getattr(client, "_READONLY_TOOL_NAME_DENYLIST", set()) or set())
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
        if tool_name in write_tool_names:
            return True
        if tool_name in readonly_tool_denylist:
            continue
        if tool_name not in write_argument_tool_allowlist:
            continue
        if property_names & write_argument_fields:
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
    latest_any: dict[str, Any] | None = None
    latest_human: dict[str, Any] | None = None
    latest_synthetic: dict[str, Any] | None = None
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
        audit = _user_text_audit(text)
        if latest_any is None:
            latest_any = audit
        if _looks_like_synthetic_user_context(text):
            if latest_synthetic is None:
                latest_synthetic = audit
            continue
        if latest_human is None:
            latest_human = audit
            break
    if latest_any is None:
        return {"latest_user_present": False}
    result = {
        "latest_user_present": latest_any["present"],
        "latest_user_sha256": latest_any["sha256"],
        "latest_user_chars": latest_any["chars"],
        "latest_user_preview": latest_any["preview"],
        "latest_user_is_synthetic_context": bool(
            latest_synthetic is not None
            and latest_synthetic.get("sha256") == latest_any.get("sha256")
        ),
    }
    if latest_human is not None:
        result.update(
            {
                "latest_human_user_present": latest_human["present"],
                "latest_human_user_sha256": latest_human["sha256"],
                "latest_human_user_chars": latest_human["chars"],
                "latest_human_user_preview": latest_human["preview"],
            }
        )
    else:
        result["latest_human_user_present"] = False
    if latest_synthetic is not None:
        result.update(
            {
                "latest_synthetic_user_present": latest_synthetic["present"],
                "latest_synthetic_user_chars": latest_synthetic["chars"],
                "latest_synthetic_user_preview": latest_synthetic["preview"],
            }
        )
    return result


def _user_text_audit(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8", errors="replace")
    return {
        "present": bool(text.strip()),
        "sha256": hashlib.sha256(encoded).hexdigest()[:12] if text else "",
        "chars": len(text),
        "preview": redact_sensitive_text(text).replace("\n", "\\n")[:240],
    }


def _looks_like_synthetic_user_context(text: str) -> bool:
    stripped = str(text or "").lstrip()
    if stripped.startswith("<retrieved-knowledge-base>"):
        return True
    if stripped.startswith("### SYSTEM ALERT:") or stripped.startswith("### FORMAT ERROR:"):
        return True
    return bool(stripped.startswith("RELEVANT CONTEXT (RETRIEVED)"))
