from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import PendingToolCall
from .tool_loop_guards import _tool_attempt_history
from .tool_artifact_recovery import (
    _artifact_read_recovery_hint,
    _artifact_read_synthesis_hint,
    _clear_artifact_read_guard_state,
    _fallback_repeated_artifact_read,
    _fallback_repeated_file_read,
    _find_full_file_artifact_for_path,
    _read_artifact_text,
    _resolve_artifact_record,
    _should_suppress_resolved_plan_artifact_read,
    _choose_artifact_grep_query,
)


def _extract_artifact_id_from_args(args: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return None

    for key in ("artifact_id", "path", "id"):
        value = args.get(key)
        if not isinstance(value, str):
            continue
        candidate = Path(value.strip()).stem.strip()
        if candidate:
            return candidate
    return None


def _resolve_artifact_record(harness: Any, artifact_id: str) -> Any | None:
    artifact = harness.state.artifacts.get(artifact_id)
    if artifact is not None:
        return artifact

    if not artifact_id.startswith("A"):
        return None

    try:
        numeric_val = int(artifact_id[1:])
    except ValueError:
        return None

    for aid, record in harness.state.artifacts.items():
        if not isinstance(aid, str) or not aid.startswith("A"):
            continue
        try:
            if int(aid[1:]) == numeric_val:
                return record
        except ValueError:
            continue
    return None


def _read_artifact_text(artifact: Any) -> str:
    content_path = getattr(artifact, "content_path", None)
    if isinstance(content_path, str) and content_path.strip():
        path = Path(content_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass

    inline_content = getattr(artifact, "inline_content", None)
    if isinstance(inline_content, str) and inline_content:
        return inline_content
    return ""


def _choose_artifact_grep_query(content: str) -> str | None:
    lowered = content.lower()
    if not any(marker in lowered for marker in ("nmap scan report", "/tcp", "/udp", "host is up")):
        return None
    for query in ("open", "port", "service", "banner", "nmap scan report", "host is up"):
        if query in lowered:
            return query
    return None
