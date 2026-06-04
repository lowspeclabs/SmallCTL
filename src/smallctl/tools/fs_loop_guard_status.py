from __future__ import annotations

from typing import Any


def active_loop_guard_paths(root: dict[str, Any] | None) -> list[dict[str, Any]]:
    active_paths: list[dict[str, Any]] = []
    if root is None:
        return active_paths
    paths = root.get("paths", {})
    if not isinstance(paths, dict):
        return active_paths
    for path, payload in paths.items():
        if not isinstance(payload, dict):
            continue
        if (
            not payload.get("pending_read_before_write")
            and int(payload.get("escalation_level", 0) or 0) <= 0
            and not payload.get("recent_writes")
        ):
            continue
        active_paths.append(
            {
                "path": str(path),
                "escalation_level": int(payload.get("escalation_level", 0) or 0),
                "pending_read_before_write": bool(payload.get("pending_read_before_write")),
                "blocked_attempts": int(payload.get("blocked_attempts", 0) or 0),
                "last_score": int(payload.get("last_score", 0) or 0),
                "last_section_name": str(payload.get("last_section_name", "") or ""),
                "last_next_section_name": str(payload.get("last_next_section_name", "") or ""),
                "outline_required": bool(payload.get("outline_required")),
                "section_checkpoints": list(payload.get("section_checkpoints", []) or []),
            }
        )
    return active_paths


def recent_complete_reads(state: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    history = scratchpad.get("_progress_read_history", []) if isinstance(scratchpad, dict) else []
    if not isinstance(history, list):
        return []

    recent: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reversed(history):
        if not isinstance(item, dict) or not bool(item.get("complete_file")):
            continue
        if bool(item.get("file_content_truncated")):
            continue
        key = str(item.get("path") or item.get("artifact_id") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        recent.append(
            {
                "tool_name": str(item.get("tool_name") or ""),
                "path": str(item.get("path") or ""),
                "artifact_id": str(item.get("artifact_id") or ""),
                "total_lines": item.get("total_lines"),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "note": "Full content is already covered; do not reread only because chat preview was truncated.",
            }
        )
        if len(recent) >= limit:
            break
    return recent


def loop_guard_status_payload(root: dict[str, Any] | None, state: Any) -> dict[str, Any]:
    if root is None:
        return {"active_paths": [], "recent_events": [], "recent_complete_reads": []}
    return {
        "active_paths": active_loop_guard_paths(root),
        "recent_events": list(root.get("events", []) or [])[-10:],
        "recent_complete_reads": recent_complete_reads(state),
    }
