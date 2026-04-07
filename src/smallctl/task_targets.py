from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any


_QUOTED_PATH_PATTERNS = (
    re.compile(r"`([^`\n]+)`"),
    re.compile(r'"([^"\n]+)"'),
    re.compile(r"'([^'\n]+)'"),
)
_BARE_PATH_PATTERN = re.compile(
    r"(?<![\w/])(?:\./|\../)?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+(?:\.[A-Za-z0-9._-]+)?"
)


def _normalize_candidate_path(value: str) -> str | None:
    candidate = str(value or "").strip()
    candidate = candidate.lstrip("([{")
    candidate = candidate.rstrip(".,:;)]}")
    if not candidate or "://" in candidate or candidate.startswith("app://"):
        return None
    if " " in candidate:
        return None
    suffix = PurePosixPath(candidate).suffix
    if not suffix:
        return None
    return candidate


def extract_task_target_paths(text: str) -> list[str]:
    if not text:
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    def _remember(value: str) -> None:
        normalized = _normalize_candidate_path(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    for pattern in _QUOTED_PATH_PATTERNS:
        for match in pattern.finditer(text):
            _remember(match.group(1))

    for match in _BARE_PATH_PATTERN.finditer(text):
        _remember(match.group(0))

    return ordered


def task_target_paths_from_harness(harness: Any) -> list[str]:
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) or {}
    scratch_paths = scratchpad.get("_task_target_paths")
    if isinstance(scratch_paths, list):
        cleaned = [str(path).strip() for path in scratch_paths if str(path).strip()]
        if cleaned:
            return cleaned

    candidates: list[str] = []
    run_brief = getattr(state, "run_brief", None)
    if run_brief is not None:
        candidates.append(str(getattr(run_brief, "original_task", "") or ""))
    working_memory = getattr(state, "working_memory", None)
    if working_memory is not None:
        candidates.append(str(getattr(working_memory, "current_goal", "") or ""))
    current_user_task = getattr(harness, "_current_user_task", None)
    if callable(current_user_task):
        try:
            candidates.append(str(current_user_task() or ""))
        except Exception:
            pass

    for text in candidates:
        paths = extract_task_target_paths(text)
        if paths:
            return paths
    return []


def primary_task_target_path(harness: Any) -> str | None:
    paths = task_target_paths_from_harness(harness)
    return paths[0] if paths else None
