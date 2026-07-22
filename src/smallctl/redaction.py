"""Redaction utilities for transport, logging, and exported boundaries.

Live runtime state may temporarily contain plaintext secrets when needed for task
continuity, but provider payloads and observability/export surfaces must redact
them before serialization or transport.
"""

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
    "passphrase",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "auth_token",
}

_SENSITIVE_SUFFIXES = (
    "_passphrase",
    "_secret",
    "_token",
    "_api_key",
    "_access_key",
    "_key",
)

_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?P<name>(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|token|secret|authorization)"
    r"(?:[_-][A-Za-z0-9]+)*)(?P<key_quote>[\"']?)(?P<before>[ \t]*)"
    r"(?P<separator>[=:])(?P<after>[ \t]*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,\"'\]}]+)"
    r"(?P<trailing_quote>[\"']?)",
    re.IGNORECASE,
)
_AUTHORIZATION_SCHEME_RE = re.compile(
    r"(\bauthorization[ \t]*(?:=|:)[ \t]*[\"']?(?:Bearer|Token|Basic|ApiKey)[ \t]+)"
    r"([^\s,\"'\]}]+)([\"']?)",
    re.IGNORECASE,
)
_PYTHON_ANNOTATION_ROOTS = {
    "annotated",
    "any",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "dict",
    "float",
    "frozenset",
    "int",
    "iterable",
    "list",
    "literal",
    "mapping",
    "none",
    "object",
    "optional",
    "sequence",
    "set",
    "str",
    "tuple",
    "type",
    "union",
}

_SENSITIVE_TEXT_PATTERNS = (
    # Bearer tokens first so a generic authorization handler never strands the JWT.
    re.compile(r"(\bBearer\s+)([A-Za-z0-9._~+/=-]+)", re.IGNORECASE),
    _AUTHORIZATION_SCHEME_RE,
    re.compile(r"(\B--(?:api[-_]?key|token|access[-_]?token|refresh[-_]?token|secret)\s+)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(\B--(?:api[-_]?key|token|access[-_]?token|refresh[-_]?token|secret)=)([^\s,;]+)", re.IGNORECASE),
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


def _is_sensitive_assignment_name(name: str) -> bool:
    normalized = _normalize_key(name)
    parts = [part for part in normalized.split("_") if part]
    if normalized in {"apikey", "authorization"}:
        return True
    if any(part in {"token", "secret"} for part in parts):
        return True
    return any(parts[index : index + 2] == ["api", "key"] for index in range(len(parts) - 1))


def _looks_like_python_annotation(value: str) -> bool:
    candidate = value.strip().rstrip("):")
    root = re.split(r"[\[|]", candidate, maxsplit=1)[0]
    leaf = root.rsplit(".", maxsplit=1)[-1]
    lowered = leaf.lower()
    if lowered in _PYTHON_ANNOTATION_ROOTS or lowered.endswith(("_type", "_t")):
        return True
    return bool(leaf and leaf[0].isupper() and leaf.replace("_", "").isalnum())


def _looks_like_code_assignment(match: re.Match[str]) -> bool:
    value = match.group("value")
    if value.startswith(("\"", "'")):
        return False

    separator = match.group("separator")
    if separator == ":" and _looks_like_python_annotation(value):
        return True

    if re.fullmatch(r"(?:None|True|False|Ellipsis|NotImplemented)", value):
        return True
    if re.match(r"(?:await[ \t]+)?[A-Za-z_][A-Za-z0-9_.]*[\[(]", value):
        return True
    if re.fullmatch(r"(?:self|cls|typing|os|sys|config|settings|env)(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value):
        return True

    if separator == ":" and match.group("key_quote"):
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))

    if separator == "=" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        if match.group("before") or match.group("after"):
            return True
        return _is_sensitive_assignment_name(value)
    return False


def _redact_sensitive_assignment(match: re.Match[str]) -> str:
    name = match.group("name")
    value = match.group("value")
    if not _is_sensitive_assignment_name(name) or "REDACTED" in value:
        return match.group(0)
    if _looks_like_code_assignment(match):
        return match.group(0)
    prefix = (
        name
        + match.group("key_quote")
        + match.group("before")
        + match.group("separator")
        + match.group("after")
    )
    return prefix + _hash_credential(value) + match.group("trailing_quote")


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
            return redact_sensitive_text(value)
        try:
            parsed = json.loads(value)
        except Exception:
            return redact_sensitive_text(value)
        return json.dumps(redact_sensitive_data(parsed), ensure_ascii=True, sort_keys=True)

    if isinstance(value, str):
        return redact_sensitive_text(value)

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
    for pattern in _SENSITIVE_TEXT_PATTERNS:
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
    return _SENSITIVE_ASSIGNMENT_RE.sub(_redact_sensitive_assignment, redacted)


def _redact_arguments_text(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    stripped = arguments.strip()
    if stripped[:1] not in {"{", "["}:
        return redact_sensitive_text(arguments)
    try:
        parsed = json.loads(arguments)
    except Exception:
        return redact_sensitive_text(arguments)
    redacted = redact_sensitive_data(parsed)
    if redacted == parsed:
        return arguments
    return json.dumps(redacted, ensure_ascii=True, sort_keys=True)


def _redact_tool_calls(tool_calls: Any) -> Any:
    if not isinstance(tool_calls, list):
        return tool_calls
    redacted_calls: list[Any] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            redacted_calls.append(call)
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            redacted_calls.append(call)
            continue
        arguments = function.get("arguments")
        redacted_arguments = _redact_arguments_text(arguments)
        if redacted_arguments == arguments:
            redacted_calls.append(call)
            continue
        redacted_call = dict(call)
        redacted_call["function"] = {**function, "arguments": redacted_arguments}
        redacted_calls.append(redacted_call)
    return redacted_calls


def redact_sensitive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of messages with sensitive text redacted from content fields."""
    redacted_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            redacted_messages.append(message)
            continue
        redacted = dict(message)
        content = redacted.get("content")
        if isinstance(content, str):
            redacted["content"] = redact_sensitive_text(content)
        elif isinstance(content, list):
            redacted["content"] = redact_sensitive_data(content)
        if "tool_calls" in redacted:
            redacted["tool_calls"] = _redact_tool_calls(redacted["tool_calls"])
        redacted_messages.append(redacted)
    return redacted_messages
