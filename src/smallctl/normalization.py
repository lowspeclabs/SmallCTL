from __future__ import annotations

import json
import re
import ast
from datetime import datetime, timezone
from typing import Any


def _strip_summary_markup(text: str) -> str:
    cleaned = text.strip()
    for marker in ("**", "__", "`"):
        if cleaned.startswith(marker) and cleaned.endswith(marker) and len(cleaned) > len(marker) * 2:
            cleaned = cleaned[len(marker) : -len(marker)].strip()
    if cleaned.startswith("#"):
        cleaned = cleaned.lstrip("#").strip()
    cleaned = cleaned.replace("`", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    return cleaned


def clean_subtask_summary(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if isinstance(value, dict):
        nested = value.get("message") or value.get("output") or value.get("status")
        if nested is not None and nested is not value:
            return clean_subtask_summary(nested)
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, dict):
            nested = parsed.get("message") or parsed.get("output") or parsed.get("status")
            if nested is not None:
                return clean_subtask_summary(nested)
    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
        if tail:
            text = tail
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) >= 24 or len(lines) == 1:
            return _strip_summary_markup(last_line)
    return _strip_summary_markup(text)


def normalize_subtask_status(*, result: dict[str, Any], summary: str) -> str:
    status = str(result.get("status", "unknown"))
    if status != "stopped":
        return status
    if str(result.get("reason", "")) != "no_tool_calls":
        return status
    if summary:
        return "completed"
    return status


def extract_subtask_summary_value(result: dict[str, Any]) -> Any:
    return (
        result.get("message")
        or result.get("assistant")
        or result.get("reason")
        or result.get("status", "")
    )


def coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return [stripped]
        if isinstance(parsed, list):
            result: list[str] = []
            for item in parsed:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    result.append(text)
            return result
        return [stripped]
    if isinstance(value, (list, tuple, set, frozenset)):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return []


def coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def coerce_timestamp_string(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def coerce_dict_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def coerce_list_payload(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def coerce_json_dict_payload(value: Any, json_safe_func: Any = None) -> dict[str, Any]:
    if json_safe_func:
        normalized = json_safe_func(value or {})
    else:
        normalized = value or {}
    return normalized if isinstance(normalized, dict) else {}


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token for token in re.findall(r"[a-z0-9_.:/\\-]+", text.lower()) if len(token) > 1}

def dedupe_keep_tail(items: list[str], *, limit: int) -> list[str]:
    """Remove duplicates from a list of strings while preserving the order and keeping the tail."""
    deduped: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped[-limit:] if limit > 0 else []
