from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..state import LoopState

PATH_TOKEN_RE = re.compile(r"(?<![\w/])(?:\.{0,2}/|/)[^\s'\";,|&<>)]*")
OPTIMISTIC_STATEMENT_RE = re.compile(
    r"(?<![@\w])(pass(?:ed|es|ing)?|verified|success(?:ful(?:ly)?)?|fixed|resolved)(?![@\w])",
    re.IGNORECASE,
)


def recent_invalidation_events(state: LoopState) -> list[dict[str, Any]]:
    payload = state.scratchpad.get("_context_invalidations")
    if not isinstance(payload, list):
        return []
    return [item for item in payload[-24:] if isinstance(item, dict)]


def durably_stale_ids(state: LoopState, key: str) -> set[str]:
    payload = state.scratchpad.get(key)
    if not isinstance(payload, dict):
        return set()
    ids: set[str] = set()
    for item_id, marker in payload.items():
        normalized_id = str(item_id or "").strip()
        if not normalized_id or not isinstance(marker, dict):
            continue
        if bool(marker.get("stale", False)):
            ids.add(normalized_id)
    return ids


def guard_trip_preserved_ids(state: LoopState, key: str) -> set[str]:
    payload = state.scratchpad.get(key)
    if not isinstance(payload, list):
        return set()
    return {str(item).strip() for item in payload if str(item).strip()}


def path_matches_any(target: str, changed_paths: list[str]) -> bool:
    normalized_target = Path(str(target or "").strip()).as_posix().lower()
    if not normalized_target:
        return False
    for changed in changed_paths:
        normalized_changed = Path(str(changed or "").strip()).as_posix().lower()
        if not normalized_changed:
            continue
        if (
            normalized_target == normalized_changed
            or normalized_target.endswith(normalized_changed)
            or normalized_changed.endswith(normalized_target)
        ):
            return True
        changed_name = Path(normalized_changed).name
        if changed_name and changed_name in normalized_target:
            return True
    return False


def path_tokens(text: str) -> list[str]:
    paths: list[str] = []
    for match in PATH_TOKEN_RE.finditer(str(text or "")):
        path = match.group(0).strip().strip("`'\".,:;")
        if path and path not in paths:
            paths.append(path)
    return paths


def is_optimistic_statement(value: str) -> bool:
    return bool(OPTIMISTIC_STATEMENT_RE.search(str(value or "")))


def coerce_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
