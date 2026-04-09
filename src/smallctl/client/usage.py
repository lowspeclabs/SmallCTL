"""Usage tracking and context limit extraction utilities."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


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
    endpoint = str(base_url or "").strip().lower()
    model_name = str(model or "").strip().lower()
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}", scheme="http")

    host = str(parsed.hostname or "").lower()
    path = str(parsed.path or "").lower()
    port = parsed.port
    blob = " ".join(
        part for part in (endpoint, host, path, model_name) if part
    )

    if "openrouter" in blob:
        return "openrouter"
    if "api.openai.com" in blob or host.endswith(".openai.com") or "openai" in blob:
        return "openai"
    if "vllm" in blob:
        return "vllm"
    if "lmstudio" in blob or "lm-studio" in blob:
        return "lmstudio"
    if "ollama" in blob or port == 11434:
        return "ollama"
    if "llamacpp" in blob or "llama.cpp" in blob or "llama-cpp" in blob:
        return "llamacpp"
    if port == 1234 and host in {"localhost", "127.0.0.1", "::1"}:
        return "lmstudio"
    if "lmstudio" in model_name:
        return "lmstudio"
    return default


def _parse_positive_int(value: Any) -> int | None:
    """Parse a value as a positive integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if 0 < parsed < 10_000_000:
        return parsed
    return None
