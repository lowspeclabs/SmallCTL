"""Usage tracking and context limit extraction utilities."""

from __future__ import annotations

from typing import Any

from ..provider_profiles import detect_provider_profile as _detect_provider_profile
from ..state import json_safe_value


def extract_context_limit(payload: Any) -> int | None:
    """Extract context limit from model metadata payload.
    
    Searches through the payload for various context limit keys and returns
    the largest discovered value.
    """
    keys = (
        "context_length",
        "max_context_length",
        "max_position_embeddings",
        "max_model_len",
        "max_seq_len",
        "context_window",
        "num_ctx",
        "n_ctx",
        "ctx_len",
        "max_input_tokens",
        "input_token_limit",
        "prompt_token_limit",
    )
    found: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys:
                    parsed = _parse_positive_int(value)
                    if parsed is not None:
                        found.append(parsed)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    if not found:
        return None
    # Use the largest discovered token window value.
    return max(found)


def extract_runtime_context_limit(payload: Any) -> int | None:
    """Extract context limit from runtime server payload (e.g., llama.cpp).
    
    Looks for runtime-specific context settings in default_generation_settings
    and other runtime-specific locations.
    """
    candidates: list[int] = []

    if isinstance(payload, dict):
        settings = payload.get("default_generation_settings")
        if isinstance(settings, dict):
            direct = _parse_positive_int(settings.get("n_ctx"))
            if direct is not None:
                candidates.append(direct)
            params = settings.get("params")
            if isinstance(params, dict):
                nested = _parse_positive_int(params.get("n_ctx"))
                if nested is not None:
                    candidates.append(nested)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"n_ctx", "num_ctx", "context_length", "max_context_length"}:
                    parsed = _parse_positive_int(value)
                    if parsed is not None:
                        candidates.append(parsed)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    if not candidates:
        return None
    # Runtime slot/context values should be bounded and generally consistent.
    return min(candidates)


def detect_provider_profile(
    base_url: str | None,
    model: str | None = None,
    *,
    default: str = "generic",
) -> str:
    """Best-effort provider/backend detection from endpoint and model hints."""
    return _detect_provider_profile(base_url, model, default=default)


def coerce_usage_token_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def apply_usage_metrics(harness: Any, usage: dict[str, Any]) -> None:
    normalized_usage = json_safe_value(usage or {})
    if not isinstance(normalized_usage, dict) or not normalized_usage:
        return
    prompt_tokens = coerce_usage_token_count(normalized_usage.get("prompt_tokens"))
    completion_tokens = coerce_usage_token_count(normalized_usage.get("completion_tokens"))
    total_tokens = coerce_usage_token_count(normalized_usage.get("total_tokens"))
    harness.state.token_usage += total_tokens
    if prompt_tokens > 0:
        harness.state.scratchpad["context_used_tokens"] = prompt_tokens
    elif total_tokens > 0:
        harness.state.scratchpad["context_used_tokens"] = total_tokens
    harness.state.last_completion_tokens = completion_tokens
    harness.state.scratchpad["last_completion_tokens"] = completion_tokens
    harness._runlog("usage", "token usage update", usage=normalized_usage)


def _parse_positive_int(value: Any) -> int | None:
    """Parse a value as a positive integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if 0 < parsed < 10_000_000:
        return parsed
    return None
