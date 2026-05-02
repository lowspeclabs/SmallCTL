from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

_SUPPORTED_PROVIDER_PROFILES: tuple[str, ...] = (
    "auto",
    "generic",
    "openai",
    "ollama",
    "vllm",
    "lmstudio",
    "openrouter",
    "llamacpp",
)

_PROVIDER_PROFILE_ALIASES: dict[str, str] = {
    "default": "generic",
    "open-router": "openrouter",
    "open_router": "openrouter",
    "openrouterai": "openrouter",
    "open-routerai": "openrouter",
    "openaiapi": "openai",
    "open-ai": "openai",
    "open_ai": "openai",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "llama.cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "llama_cpp": "llamacpp",
}


def supported_provider_profiles() -> tuple[str, ...]:
    return _SUPPORTED_PROVIDER_PROFILES


def normalize_provider_profile_alias(profile: Any) -> tuple[str, str | None]:
    raw = str(profile or "").strip().lower()
    if not raw:
        return "auto", None
    if raw in _SUPPORTED_PROVIDER_PROFILES:
        return raw, None
    if raw in _PROVIDER_PROFILE_ALIASES:
        return _PROVIDER_PROFILE_ALIASES[raw], raw
    collapsed = raw.replace(" ", "").replace("_", "").replace("-", "")
    if collapsed in _SUPPORTED_PROVIDER_PROFILES:
        return collapsed, raw
    return raw, None


def _validate_provider_profile_or_raise(profile: str, *, context: str) -> str:
    if profile in _SUPPORTED_PROVIDER_PROFILES:
        return profile
    options = ", ".join(_SUPPORTED_PROVIDER_PROFILES)
    raise ValueError(
        f"Invalid {context} '{profile}'. Supported provider_profile values: {options}."
    )


def resolve_provider_profile(
    endpoint: str | None,
    model: str | None,
    provider_profile: Any,
    *,
    enforce_openrouter_for_endpoint: bool = True,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    normalized, alias_used = normalize_provider_profile_alias(provider_profile)
    if alias_used is not None and normalized in _SUPPORTED_PROVIDER_PROFILES:
        warnings.append(
            f"Normalized provider_profile alias '{alias_used}' to '{normalized}'."
        )

    if normalized == "auto":
        detected_raw = detect_provider_profile(endpoint, model)
        detected, _ = normalize_provider_profile_alias(detected_raw)
        resolved = _validate_provider_profile_or_raise(detected, context="detected provider profile")
    else:
        resolved = _validate_provider_profile_or_raise(normalized, context="provider profile")

    detected_endpoint_raw = detect_provider_profile(endpoint, model)
    detected_endpoint, _ = normalize_provider_profile_alias(detected_endpoint_raw)
    detected_endpoint = _validate_provider_profile_or_raise(
        detected_endpoint,
        context="endpoint-detected provider profile",
    )
    if enforce_openrouter_for_endpoint and detected_endpoint == "openrouter" and resolved != "openrouter":
        warnings.append(
            "OpenRouter endpoint detected; overriding provider_profile to 'openrouter' so OpenRouter adapter sanitation is always applied."
        )
        resolved = "openrouter"

    return resolved, warnings


def detect_provider_profile(
    base_url: str | None,
    model: str | None = None,
    *,
    default: str = "generic",
) -> str:
    endpoint = str(base_url or "").strip().lower()
    model_name = str(model or "").strip().lower()
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}", scheme="http")

    host = str(parsed.hostname or "").lower()
    path = str(parsed.path or "").lower()
    port = parsed.port
    blob = " ".join(part for part in (endpoint, host, path, model_name) if part)

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
