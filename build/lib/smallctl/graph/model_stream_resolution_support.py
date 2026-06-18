from __future__ import annotations

from typing import Any

from ..client.chunk_parser import chunk_contains_tool_call_delta


def _provider_chunk_data(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if item.get("type") == "chunk":
        data = item.get("data", {})
        return data if isinstance(data, dict) else None
    if isinstance(item.get("choices"), list):
        return item
    return None


def _tool_call_stream_boundary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    saw_delta = False
    saw_finish = False
    finish_reasons: list[str] = []
    chunk_count = 0
    for item in chunks:
        data = _provider_chunk_data(item)
        if data is None:
            continue
        chunk_count += 1
        saw_delta = saw_delta or chunk_contains_tool_call_delta(data)
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        finish_reason = choices[0].get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            finish_reasons.append(finish_reason)
            if finish_reason == "tool_calls":
                saw_finish = True
    return {
        "saw_tool_call_delta": saw_delta,
        "saw_tool_calls_finish": saw_finish,
        "finish_reasons": finish_reasons,
        "chunk_count": chunk_count,
    }


def _chunk_error_failure_message(details: dict[str, Any] | None) -> str:
    details = details if isinstance(details, dict) else {}
    if details.get("reason") == "model_unloaded" or details.get("type") == "model_unloaded":
        provider = str(details.get("provider_profile") or "provider").strip() or "provider"
        model = str(details.get("model") or "").strip()
        suffix = f" for {model}" if model else ""
        return f"{provider} model is unloaded{suffix}"
    if details.get("reason") == "openrouter_authentication_failed":
        provider_error = str(details.get("provider_error") or "").strip()
        suffix = f" ({provider_error})" if provider_error else ""
        return (
            "OpenRouter authentication failed: API key is invalid, revoked, "
            f"or belongs to a missing account{suffix}. Update SMALLCTL_API_KEY."
        )
    if details.get("type") == "context_budget_exceeded":
        provider = str(details.get("provider_profile") or "provider").strip() or "provider"
        over_budget = int(details.get("over_budget_tokens", 0) or 0)
        suffix = f" by {over_budget} estimated tokens" if over_budget > 0 else ""
        return f"{provider} prompt exceeded the local context budget before request{suffix}"
    if int(details.get("status_code", 0) or 0) == 400:
        provider = str(details.get("provider_profile") or "provider").strip() or "provider"
        upstream = str(details.get("upstream_provider") or "").strip()
        provider_error = str(details.get("provider_error") or "").strip()
        label = provider
        if upstream and upstream.lower() != provider.lower():
            label = f"{provider}/{upstream}"
        suffix = f": {provider_error}" if provider_error else ""
        return f"{label} input validation failed after retries (HTTP 400{suffix})"
    return "Upstream chunk error after retries"


def _chunk_error_failure_type(details: dict[str, Any] | None) -> str:
    details = details if isinstance(details, dict) else {}
    if details.get("reason") == "backend_stream_failure":
        return "backend_stream_failure"
    if details.get("reason") == "model_unloaded" or details.get("type") == "model_unloaded":
        return "provider"
    if details.get("reason") == "openrouter_authentication_failed":
        return "provider"
    if details.get("type") == "context_budget_exceeded":
        return "prompt_budget"
    if int(details.get("status_code", 0) or 0) == 400:
        return "provider"
    return "stream"
