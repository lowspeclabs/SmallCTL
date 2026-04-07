from __future__ import annotations

import json
from typing import Any

REDACTED = "[REDACTED]"

_EXACT_SENSITIVE_KEYS = {
    "password",
    "passphrase",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "auth_token",
    "ssh_password",
    "sshpass",
    "conn_pass",
}

_SENSITIVE_SUFFIXES = (
    "_password",
    "_passphrase",
    "_secret",
    "_token",
    "_api_key",
    "_pass",
)


def _normalize_key(key: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(key)).strip("_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if not normalized:
        return False
    if normalized in _EXACT_SENSITIVE_KEYS:
        return True
    return any(normalized.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def redact_sensitive_data(value: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text) and item not in (None, "", [], {}):
                redacted[key] = REDACTED
                continue
            redacted[key] = redact_sensitive_data(item, parent_key=key_text)
        return redacted

    if isinstance(value, list):
        return [redact_sensitive_data(item, parent_key=parent_key) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item, parent_key=parent_key) for item in value)

    if parent_key == "arguments" and isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] not in {"{", "["}:
            return value
        try:
            parsed = json.loads(value)
        except Exception:
            return value
        return json.dumps(redact_sensitive_data(parsed), ensure_ascii=True, sort_keys=True)

    return value
