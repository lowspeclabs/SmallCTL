from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EscalationModelConfig:
    endpoint: str
    model: str
    provider_profile: str
    api_key: str | None
    chat_endpoint: str
    max_prompt_chars: int
    max_response_tokens: int
    temperature: float
    timeout_sec: int


class EscalationConfigError(ValueError):
    pass


def build_escalation_model_config(config: Any) -> EscalationModelConfig:
    endpoint = str(getattr(config, "escalation_endpoint", "") or "").strip().rstrip("/")
    model = str(getattr(config, "escalation_model", "") or "").strip()
    if not endpoint or not model:
        raise EscalationConfigError(
            "Escalation requires escalation_endpoint and escalation_model; it does not inherit the main model config."
        )

    api_key = getattr(config, "escalation_api_key", None)
    api_key_env = str(getattr(config, "escalation_api_key_env", "") or "").strip()
    if api_key_env:
        api_key = os.getenv(api_key_env) or api_key

    chat_endpoint = str(getattr(config, "escalation_chat_endpoint", "/chat/completions") or "/chat/completions").strip()
    if not chat_endpoint.startswith("/"):
        chat_endpoint = f"/{chat_endpoint}"

    return EscalationModelConfig(
        endpoint=endpoint,
        model=model,
        provider_profile=str(getattr(config, "escalation_provider_profile", "auto") or "auto").strip() or "auto",
        api_key=str(api_key) if api_key not in (None, "") else None,
        chat_endpoint=chat_endpoint,
        max_prompt_chars=max(1000, _safe_int(getattr(config, "escalation_max_prompt_chars", 48000), 48000)),
        max_response_tokens=max(1, _safe_int(getattr(config, "escalation_max_response_tokens", 1600), 1600)),
        temperature=max(0.0, _safe_float(getattr(config, "escalation_temperature", 0.2), 0.2)),
        timeout_sec=max(1, _safe_int(getattr(config, "escalation_timeout_sec", 120), 120)),
    )


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
