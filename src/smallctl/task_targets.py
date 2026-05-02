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
    resolved_followup = scratchpad.get("_resolved_followup")
    blocked_paths: set[str] = set()
    blocked_basenames: set[str] = set()
    if isinstance(resolved_followup, dict) and str(
        resolved_followup.get("target_inheritance") or ""
    ).strip() == "blocked_by_user_constraint":
        for path in resolved_followup.get("blocked_target_paths") or []:
            text = str(path or "").strip().lower()
            if not text:
                continue
            blocked_paths.add(text)
            blocked_basenames.add(PurePosixPath(text).name.lower())

    def _filter_blocked(paths: list[str]) -> list[str]:
        cleaned: list[str] = []
        for path in paths:
            text = str(path or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in blocked_paths or PurePosixPath(text).name.lower() in blocked_basenames:
                continue
            cleaned.append(text)
        return cleaned

    scratch_paths = scratchpad.get("_task_target_paths")
    if isinstance(scratch_paths, list):
        cleaned = _filter_blocked([str(path).strip() for path in scratch_paths if str(path).strip()])
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
        paths = _filter_blocked(paths)
        if paths:
            return paths

    last_handoff = scratchpad.get("_last_task_handoff")
    if isinstance(last_handoff, dict):
        handoff_paths = last_handoff.get("target_paths")
        if isinstance(handoff_paths, list):
            cleaned = _filter_blocked([str(path).strip() for path in handoff_paths if str(path).strip()])
            if cleaned:
                return cleaned
        handoff_text_candidates = [
            str(last_handoff.get("effective_task") or ""),
            str(last_handoff.get("current_goal") or ""),
            str(last_handoff.get("raw_task") or ""),
        ]
        for text in handoff_text_candidates:
            paths = _filter_blocked(extract_task_target_paths(text))
            if paths:
                return paths

    session = getattr(state, "write_session", None)
    session_target = str(getattr(session, "write_target_path", "") or "").strip()
    if session_target:
        if session_target.lower() in blocked_paths or PurePosixPath(session_target).name.lower() in blocked_basenames:
            return []
        return [session_target]
    return []


def primary_task_target_path(harness: Any) -> str | None:
    paths = task_target_paths_from_harness(harness)
    return paths[0] if paths else None
