from __future__ import annotations

from typing import Any

PHASE_TAG_PREFIX = "phase_"

_GENERIC_EXPERIENCE_TAGS = {
    "auto",
    "execute",
    "explore",
    "generic",
    "llamacpp",
    "lmstudio",
    "localhost",
    "ollama",
    "openai",
    "openrouter",
    "repair",
    "scripts",
    "verify",
    "vllm",
}


def is_generic_experience_tag(tag: str) -> bool:
    normalized = str(tag or "").strip().lower()
    if not normalized:
        return True
    if normalized in _GENERIC_EXPERIENCE_TAGS:
        return True
    if normalized.startswith(PHASE_TAG_PREFIX):
        return False
    if "/" in normalized or ":" in normalized:
        return True
    return False


def normalize_experience_tags(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [values]
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = [values]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        tag = str(value or "").strip().lower()
        if not tag or is_generic_experience_tag(tag):
            continue
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized
