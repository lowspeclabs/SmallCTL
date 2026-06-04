from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_READ_ONLY_TOOLS = frozenset({"ssh_file_read", "file_read", "web_search", "artifact_read", "artifact_print", "find"})

_MUTATING_COMMAND_RE = re.compile(
    r"\b(start|restart|stop|enable|disable|rm\b|mv\b|cp\b|sed\s+-i|tee\b|>>>?\s*/|cat\s*>\s*/|install\s+-m|truncate\b|chmod\s+\d|chown\s+\S+:\S+)\b",
    re.IGNORECASE,
)
_PATH_TOKEN_RE = re.compile(r"(?<![\w/])(?:\.{0,2}/|/)[^\s'\";,|&<>)]*")
_OPTIMISTIC_STATEMENT_RE = re.compile(
    r"(?<![@\w])(pass(?:ed|es|ing)?|verified|success(?:ful(?:ly)?)?|fixed|resolved)(?![@\w])",
    re.IGNORECASE,
)


def is_read_only_artifact(artifact: Any) -> bool:
    if artifact is None:
        return False
    tool_name = str(getattr(artifact, "tool_name", "") or "").strip().lower()
    if tool_name in _READ_ONLY_TOOLS:
        return True
    if tool_name in {"ssh_exec", "shell_exec"}:
        metadata = getattr(artifact, "metadata", {}) or {}
        if isinstance(metadata, dict):
            command = str(metadata.get("command") or "").strip()
            if command and not _MUTATING_COMMAND_RE.search(command):
                return True
    return False


def extract_path_tokens(text: str) -> list[str]:
    paths: list[str] = []
    for match in _PATH_TOKEN_RE.finditer(str(text or "")):
        path = match.group(0).strip().strip("`'\".,:;")
        if path and path not in paths:
            paths.append(path)
    return paths


def normalize_paths(paths: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for path in paths or []:
        text = str(path or "").strip()
        if not text:
            continue
        normalized.append(Path(text).as_posix().lower())
    deduped: list[str] = []
    for path in normalized:
        if path not in deduped:
            deduped.append(path)
    return deduped


def text_matches_any_path(text: str, paths: list[str]) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    for path in paths:
        if not path:
            continue
        if path in lowered or lowered.endswith(path):
            return True
        basename = Path(path).name
        if basename and basename in lowered:
            return True
    return False
