from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .normalization import (
    coerce_datetime as _coerce_datetime,
    coerce_dict_payload as _coerce_dict_payload,
    coerce_float as _coerce_float,
    coerce_int,
    coerce_json_dict_payload,
    coerce_list_payload as _coerce_list_payload,
    coerce_string_list,
    coerce_timestamp_string as _coerce_timestamp_string,
)

LOOP_STATE_SCHEMA_VERSION = 2


def _coerce_int(value: Any, *, default: int = 0) -> int:
    return coerce_int(value, default=default)


def _coerce_string_list(value: Any) -> list[str]:
    return coerce_string_list(value)


def _coerce_list_payload(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _coerce_dict_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        sanitized = [json_safe_value(item) for item in value]
        return sorted(sanitized, key=str)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_safe_value(getattr(value, field.name)) for field in fields(value)}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe_value(to_dict())
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _migrate_loop_state_payload(payload: dict[str, Any], *, incoming_version: int) -> dict[str, Any]:
    migrated = dict(payload)
    if incoming_version >= LOOP_STATE_SCHEMA_VERSION:
        return migrated

    write_session = migrated.get("write_session")
    if isinstance(write_session, dict):
        ws = dict(write_session)
        if ws.get("session_id") and not ws.get("write_session_id"):
            ws["write_session_id"] = ws.get("session_id")
        if ws.get("mode") and not ws.get("write_session_mode"):
            ws["write_session_mode"] = ws.get("mode")
        if ws.get("lifecycle_status") and not ws.get("status"):
            ws["status"] = ws.get("lifecycle_status")
        migrated["write_session"] = ws
    if incoming_version < LOOP_STATE_SCHEMA_VERSION and "reasoning_graph" not in migrated:
        migrated["reasoning_graph"] = {}
    return migrated


def _coerce_json_dict_payload(value: Any) -> dict[str, Any]:
    return coerce_json_dict_payload(value, json_safe_func=json_safe_value)


def _coerce_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if str(key).strip()}


def _coerce_int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, item in value.items():
        try:
            normalized[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return normalized


def _coerce_write_section_ranges(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, int]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            continue
        start = _coerce_int(item.get("start"), default=-1)
        end = _coerce_int(item.get("end"), default=-1)
        if start < 0 or end < start:
            continue
        normalized[str(key)] = {"start": start, "end": end}
    return normalized


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_intent_label(value: Any) -> str:
    normalized = str(value or "").strip()
    if normalized.startswith("use_") and len(normalized) > 4:
        return f"requested_{normalized[4:]}"
    return normalized


def _filter_dataclass_payload(dataclass_type: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in payload.items() if key in allowed_fields}


def _coerce_conversation_message_payload(value: Any) -> dict[str, Any] | None:
    from .models.conversation import ConversationMessage

    if not isinstance(value, dict):
        return None
    normalized = json_safe_value(value)
    if not isinstance(normalized, dict):
        return None
    filtered = _filter_dataclass_payload(ConversationMessage, normalized)
    filtered["role"] = str(filtered.get("role", "tool"))
    for key in ("content", "name", "tool_call_id", "retrieval_safe_text"):
        if key in filtered and filtered[key] is not None:
            filtered[key] = str(filtered[key])
    if "tool_calls" in filtered:
        tool_calls = filtered.get("tool_calls")
        filtered["tool_calls"] = tool_calls if isinstance(tool_calls, list) else []
    if "metadata" in filtered:
        metadata = filtered.get("metadata")
        filtered["metadata"] = metadata if isinstance(metadata, dict) else {}
    return filtered


def clip_text_value(
    value: Any,
    *,
    limit: int | None,
    marker: str = " [truncated]",
) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if limit is None or limit <= 0 or len(text) <= limit:
        return text, False
    trimmed_limit = max(1, limit - len(marker))
    clipped = text[:trimmed_limit].rstrip()
    if clipped:
        clipped = f"{clipped}{marker}"
    else:
        clipped = marker.strip()
    return clipped, True


def clip_string_list(
    values: Any,
    *,
    limit: int | None,
    item_char_limit: int | None = None,
    keep_tail: bool = True,
    marker: str = " [truncated]",
) -> tuple[list[str], bool]:
    items = _coerce_string_list(values)
    normalized: list[str] = []
    clipped = False
    for item in items:
        clipped_item, item_was_clipped = clip_text_value(item, limit=item_char_limit, marker=marker)
        if item_was_clipped:
            clipped = True
        if clipped_item and clipped_item not in normalized:
            normalized.append(clipped_item)
    if limit is not None and limit > 0 and len(normalized) > limit:
        clipped = True
        normalized = normalized[-limit:] if keep_tail else normalized[:limit]
    return normalized, clipped
