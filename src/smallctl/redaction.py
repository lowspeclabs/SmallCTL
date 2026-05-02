from __future__ import annotations

import hashlib
import json
import re
from typing import Any

REDACTED = "[REDACTED]"
_ARGUMENT_PREVIEW_LIMIT = 160
_LARGE_ARGUMENT_COMPACTION_THRESHOLD = 240
_WRITE_CONTENT_ARGUMENT_TOOLS = {
    "ssh_file_write",
}
_LARGE_TEXT_ARGUMENT_FIELDS_BY_TOOL = {
    "ssh_file_patch": ("target_text", "replacement_text"),
    "ssh_file_replace_between": ("start_text", "end_text", "replacement_text"),
    "file_patch": ("target_text", "replacement_text", "patch", "diff"),
    "ast_patch": ("patch", "diff", "replacement_text", "target_text"),
}
_LOCAL_WRITE_ARGUMENT_TOOLS = {
    "file_write",
    "file_append",
}

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

_PASSWORD_TEXT_PATTERNS = (
    re.compile(r'(\bpassword\s*(?:is\s+|=|:)?\s*")([^"\r\n]+)(")', re.IGNORECASE),
    re.compile(r"(\bpassword\s*(?:is\s+|=|:)?\s*')([^'\r\n]+)(')", re.IGNORECASE),
    re.compile(r"(\bpassword\s*(?:is\s+|=|:)?\s+)([^\s,;]+)", re.IGNORECASE),
    re.compile(r'(("|\')password("|\')\s*:\s*("|\'))([^"\r\n]+)(("|\'))', re.IGNORECASE),
)


def _hash_credential(value: Any) -> str:
    val_str = str(value)
    if not val_str:
        return REDACTED
    h = hashlib.sha256(val_str.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{REDACTED} [sha256={h}]"


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
                redacted[key] = _hash_credential(item)
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


def compact_tool_arguments_for_metadata(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep persisted tool arguments auditable without duplicating large write bodies."""
    if not isinstance(arguments, dict):
        return {}

    normalized_tool = str(tool_name or "").strip()
    compacted = dict(arguments)
    if normalized_tool in _WRITE_CONTENT_ARGUMENT_TOOLS:
        _replace_text_argument(compacted, "content", force=True)
        return compacted

    if normalized_tool in _LOCAL_WRITE_ARGUMENT_TOOLS:
        _replace_text_argument(compacted, "content", force=False)

    for field_name in _LARGE_TEXT_ARGUMENT_FIELDS_BY_TOOL.get(normalized_tool, ()):
        _replace_text_argument(compacted, field_name, force=False)
    return compacted


def _replace_text_argument(arguments: dict[str, Any], field_name: str, *, force: bool) -> None:
    value = arguments.get(field_name)
    if not isinstance(value, str):
        return
    if not force and len(value) <= _LARGE_ARGUMENT_COMPACTION_THRESHOLD:
        return

    encoded = value.encode("utf-8", errors="replace")
    preview = value[:_ARGUMENT_PREVIEW_LIMIT]
    arguments.pop(field_name, None)
    arguments[f"{field_name}_sha256"] = hashlib.sha256(encoded).hexdigest()
    arguments[f"{field_name}_chars"] = len(value)
    arguments[f"{field_name}_bytes"] = len(encoded)
    arguments[f"{field_name}_preview"] = preview
    arguments[f"{field_name}_truncated"] = True


def redact_sensitive_text(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""

    redacted = value
    for pattern in _PASSWORD_TEXT_PATTERNS:
        def _get_repl(pat_groups: int):
            def _repl(m: re.Match[str]) -> str:
                if pat_groups >= 5:
                    val = m.group(5)
                    if "REDACTED" in (val or ""):
                        return m.group(0)
                    return m.group(1) + _hash_credential(val) + m.group(6)
                elif pat_groups == 3:
                    val = m.group(2)
                    if "REDACTED" in (val or ""):
                        return m.group(0)
                    return m.group(1) + _hash_credential(val) + m.group(3)
                else:
                    val = m.group(2)
                    if "REDACTED" in (val or ""):
                        return m.group(0)
                    return m.group(1) + _hash_credential(val)
            return _repl

        redacted = pattern.sub(_get_repl(pattern.groups), redacted)
    return redacted
