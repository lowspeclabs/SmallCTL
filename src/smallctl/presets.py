"""Named configuration presets for common run profiles."""

from __future__ import annotations

from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "safe-small-model": {
        "provider_profile": "generic",
        "reasoning_mode": "off",
        "max_prompt_tokens": 4096,
        "reserve_completion_tokens": 768,
        "reserve_tool_tokens": 384,
        "first_token_timeout_sec": 20,
    },
    "coding-local": {
        "provider_profile": "generic",
        "reasoning_mode": "tags",
        "max_prompt_tokens": 8192,
        "reserve_completion_tokens": 1024,
        "reserve_tool_tokens": 512,
    },
    "lmstudio-small-model": {
        "provider_profile": "lmstudio",
        "reasoning_mode": "tags",
        "max_prompt_tokens": 4096,
        "reserve_completion_tokens": 768,
        "reserve_tool_tokens": 384,
        "first_token_timeout_sec": 25,
    },
}


def list_presets() -> list[str]:
    return sorted(PRESETS.keys())


def get_preset_defaults(name: str | None) -> dict[str, Any]:
    key = str(name or "").strip().lower()
    if not key:
        return {}
    return dict(PRESETS.get(key, {}))
