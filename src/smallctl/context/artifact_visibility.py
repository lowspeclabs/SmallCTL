from __future__ import annotations

import re
from typing import Any


def is_superseded_artifact(artifact: Any) -> bool:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    superseded_by = metadata.get("superseded_by")
    return isinstance(superseded_by, str) and bool(superseded_by.strip())


def is_prompt_visible_artifact(artifact: Any) -> bool:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        return True
    return metadata.get("model_visible", True) is not False


def is_retrieval_visible_artifact(artifact: Any) -> bool:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        return True
    if metadata.get("model_visible", True) is not False:
        return True
    return bool(str(metadata.get("verifier_verdict") or "").strip())


def is_read_only_artifact(artifact: Any) -> bool:
    if artifact is None:
        return False
    tool_name = str(getattr(artifact, "tool_name", "") or "").strip().lower()
    read_only_tools = frozenset({"ssh_file_read", "file_read", "web_search", "artifact_read", "artifact_print", "find"})
    if tool_name in read_only_tools:
        return True
    if tool_name in {"ssh_exec", "shell_exec"}:
        metadata = getattr(artifact, "metadata", {}) or {}
        if isinstance(metadata, dict):
            command = str(metadata.get("command") or "").strip()
            if command:
                mutating_pattern = re.compile(
                    r"\b(start|restart|stop|enable|disable|rm\b|mv\b|cp\b|sed\s+-i|tee\b|>>>?\s*/|cat\s*>\s*/|install\s+-m|truncate\b|chmod\s+\d|chown\s+\S+:\S+)\b",
                    re.IGNORECASE,
                )
                if not mutating_pattern.search(command):
                    return True
    return False


def artifact_path_candidates(artifact: Any, metadata: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    source = str(getattr(artifact, "source", "") or "").strip()
    if source:
        candidates.append(source)
    for key in ("path", "target_path", "write_target_path"):
        value = str(metadata.get(key) or "").strip()
        if value:
            candidates.append(value)
    path_tags = getattr(artifact, "path_tags", [])
    if isinstance(path_tags, list):
        candidates.extend(str(item).strip() for item in path_tags if str(item).strip())
    return candidates
