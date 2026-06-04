from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .task_boundary_constants import _REMOTE_ABSOLUTE_PATH_RE
from .task_boundary_support import normalize_remote_host


def guard_trip_preserves_artifact(artifact: Any) -> bool:
    if artifact is None:
        return False
    tool_name = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip().lower()
    if tool_name in {"web_fetch", "web_search", "artifact_read", "artifact_print"}:
        return False
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if metadata.get("success") is True:
        return True
    exit_code = metadata.get("exit_code")
    if exit_code is not None:
        try:
            return int(exit_code) == 0
        except (TypeError, ValueError):
            return False
    verdict = str(metadata.get("verifier_verdict") or "").strip().lower()
    return verdict == "pass"


def guard_trip_repeated_tool(reason: str) -> str:
    text = str(reason or "")
    match = re.search(r"repeated tool call loop \((?P<tool>[a-zA-Z0-9_]+) repeated", text)
    if match:
        return match.group("tool")
    match = re.search(r"Last repeated action:\s*`?(?P<tool>[a-zA-Z0-9_]+)`?", text)
    if match:
        return match.group("tool")
    return ""


def normalize_target_path(value: Any) -> str:
    text = str(value or "").strip().strip("`")
    if not text:
        return ""
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/").lower()


def target_paths_overlap(left: list[str], right: list[str]) -> bool:
    left_norm = {normalize_target_path(path) for path in left if normalize_target_path(path)}
    right_norm = {normalize_target_path(path) for path in right if normalize_target_path(path)}
    if not left_norm or not right_norm:
        return False
    if left_norm & right_norm:
        return True
    left_names = {Path(path).name.lower() for path in left_norm if path}
    right_names = {Path(path).name.lower() for path in right_norm if path}
    return bool(left_names & right_names)


def remote_target_matches_known_target(
    candidate: dict[str, Any],
    known_targets: list[dict[str, str]],
) -> bool:
    candidate_host = normalize_remote_host(candidate.get("host"))
    candidate_user = str(candidate.get("user") or "").strip().lower()
    if not candidate_host:
        return False
    for known in known_targets:
        known_host = normalize_remote_host(known.get("host"))
        if known_host != candidate_host:
            continue
        known_user = str(known.get("user") or "").strip().lower()
        if candidate_user and known_user and candidate_user != known_user:
            return False
        return True
    return False


def extract_remote_absolute_paths(*texts: Any) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for text_value in texts:
        text = str(text_value or "")
        if not text:
            continue
        for match in _REMOTE_ABSOLUTE_PATH_RE.finditer(text):
            normalized = normalize_target_path(match.group(0))
            if not normalized or not normalized.startswith("/") or normalized in seen:
                continue
            seen.add(normalized)
            collected.append(normalized)
    return collected


def ordinal_followup_index(task: str) -> int | None:
    from .task_boundary_constants import _ORDINAL_FOLLOWUP_RE, _ORDINAL_WORDS, _ORDINAL_WORD_FOLLOWUP_RE

    text = str(task or "").strip().lower()
    if not text:
        return None
    match = _ORDINAL_FOLLOWUP_RE.search(text)
    if match:
        try:
            return int(match.group(1) or match.group(2))
        except (TypeError, ValueError):
            return None
    word_match = _ORDINAL_WORD_FOLLOWUP_RE.search(text)
    if word_match:
        word = str(word_match.group("word") or word_match.group("option_word") or "")
        return _ORDINAL_WORDS.get(word)
    return None


def strip_ordinal_prefix(task: str) -> str:
    from .task_boundary_constants import _GENERIC_EDIT_LEAD_RE, _ORDINAL_PREFIX_RE

    text = str(task or "").strip()
    if not text:
        return ""
    text = _ORDINAL_PREFIX_RE.sub("", text, count=1)
    text = _GENERIC_EDIT_LEAD_RE.sub("", text, count=1)
    return text.strip()
