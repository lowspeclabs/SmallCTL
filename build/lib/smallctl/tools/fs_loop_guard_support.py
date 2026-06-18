from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..state import LoopState
from .common import fail
from .fs_loop_guard_actions import forced_escape_action, next_required_read, outline_handoff_question


@dataclass
class LoopGuardDecision:
    action: str
    message: str
    error_kind: str
    trigger_kind: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)


_LOOP_GUARD_STATE_KEY = "_chunk_write_loop_guard"
_LOOP_GUARD_CONFIG_KEY = "_chunk_write_loop_guard_config"
_DEFAULT_LOOP_GUARD_CONFIG: dict[str, Any] = {
    "enabled": True,
    "stagnation_threshold": 3,
    "level2_threshold": 5,
    "recent_writes_limit": 5,
    "tail_lines": 50,
    "similarity_threshold": 0.9,
    "append_overlap_threshold": 0.75,
    "cumulative_write_gate": True,
    "checkpoint_gate": True,
    "diff_gate": True,
}


def _emit_block(
    state: LoopState | None,
    path_state: dict[str, Any],
    decision: LoopGuardDecision,
    *,
    resolved_path: str,
    session_id: str,
    section_name: str,
    next_section_name: str,
    score: int,
    signals: dict[str, Any],
    tail_excerpt: str,
    level: int = 1,
    schedule_read: bool = True,
    outline_required: bool = False,
    event_name: str | None = None,
) -> dict[str, Any]:
    path_state["pending_read_before_write"] = schedule_read
    path_state["escalation_level"] = max(
        int(path_state.get("escalation_level", 0) or 0), level
    )
    path_state["blocked_attempts"] = int(path_state.get("blocked_attempts", 0) or 0) + 1
    path_state["last_error_kind"] = decision.error_kind
    path_state["outline_required"] = outline_required
    _record_guard_event(
        state,
        event=event_name or f"loop_guard_{decision.error_kind}",
        path=resolved_path,
        session_id=session_id,
        payload={
            "escalation_level": level,
            "stagnation_score": score,
            "section_name": section_name,
            "next_section_name": next_section_name,
            "signals": signals,
        },
    )
    metadata: dict[str, Any] = {
        "path": resolved_path,
        "write_session_id": session_id,
        "section_name": section_name,
        "next_section_name": next_section_name,
        "error_kind": decision.error_kind,
        "loop_guard_score": score,
        "loop_guard_escalation_level": level,
        "loop_guard_signals": signals,
        "loop_guard_schedule_read": schedule_read,
        "loop_guard_tail_excerpt": tail_excerpt,
        "next_required_tool": next_required_read(resolved_path),
    }
    if outline_required:
        metadata["loop_guard_outline_required"] = True
        metadata["loop_guard_outline_question"] = outline_handoff_question(resolved_path)
    metadata.update(decision.extra_metadata)
    return fail(decision.message, metadata=metadata)


def _hard_abort_chunked_write(
    *,
    state: LoopState | None,
    path_state: dict[str, Any],
    resolved_path: str,
    session_id: str,
    section_name: str,
    next_section_name: str,
    score: int,
    signals: dict[str, Any],
    tail_excerpt: str,
    trigger_kind: str,
) -> dict[str, Any]:
    attempts = _loop_guard_attempt_count(path_state)
    normalized_section = str(section_name or "").strip() or "unknown"
    postmortem = (
        f"Model stuck in write loop on section '{normalized_section}' "
        f"for path '{resolved_path}' after {attempts} attempts."
    )
    path_state["pending_read_before_write"] = False
    path_state["outline_required"] = False
    path_state["escalation_level"] = 4
    path_state["blocked_attempts"] = attempts
    path_state["last_error_kind"] = "chunked_write_loop_guard_hard_abort"
    _record_guard_event(
        state,
        event="loop_guard_hard_abort",
        path=resolved_path,
        session_id=session_id,
        payload={
            "trigger_kind": trigger_kind,
            "stagnation_score": score,
            "section_name": section_name,
            "next_section_name": next_section_name,
            "attempts": attempts,
            "signals": signals,
        },
    )
    return fail(
        postmortem,
        metadata={
            "path": resolved_path,
            "write_session_id": session_id,
            "section_name": section_name,
            "next_section_name": next_section_name,
            "error_kind": "chunked_write_loop_guard_hard_abort",
            "loop_guard_score": score,
            "loop_guard_escalation_level": 4,
            "loop_guard_signals": signals,
            "loop_guard_tail_excerpt": tail_excerpt,
            "loop_guard_hard_abort": True,
            "loop_guard_postmortem": postmortem,
            "loop_guard_trigger_kind": trigger_kind,
            "loop_guard_attempts": attempts,
            "next_required_action": forced_escape_action(resolved_path, session_id),
        },
    )


def _loop_guard_root(state: LoopState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    root = scratchpad.setdefault(
        _LOOP_GUARD_STATE_KEY,
        {
            "version": 1,
            "paths": {},
            "events": [],
        },
    )
    if not isinstance(root, dict):
        return None
    root.setdefault("version", 1)
    root.setdefault("paths", {})
    root.setdefault("events", [])
    if not isinstance(root["paths"], dict):
        root["paths"] = {}
    if not isinstance(root["events"], list):
        root["events"] = []
    return root


def _path_state(state: LoopState | None, path: str) -> dict[str, Any] | None:
    root = _loop_guard_root(state)
    if root is None:
        return None
    paths = root.setdefault("paths", {})
    if not isinstance(paths, dict):
        return None
    entry = paths.setdefault(
        path,
        {
            "recent_writes": [],
            "section_checkpoints": [],
            "escalation_level": 0,
            "pending_read_before_write": False,
            "blocked_attempts": 0,
            "writes_since_last_read": 0,
            "last_read_at": 0.0,
            "last_score": 0,
            "last_section_name": "",
            "last_next_section_name": "",
            "last_error_kind": "",
            "session_id": "",
            "outline_required": False,
            "last_projected_content": "",
        },
    )
    if not isinstance(entry, dict):
        return None
    entry.setdefault("recent_writes", [])
    entry.setdefault("section_checkpoints", [])
    entry.setdefault("escalation_level", 0)
    entry.setdefault("pending_read_before_write", False)
    entry.setdefault("blocked_attempts", 0)
    entry.setdefault("writes_since_last_read", 0)
    entry.setdefault("last_read_at", 0.0)
    entry.setdefault("last_score", 0)
    entry.setdefault("last_section_name", "")
    entry.setdefault("last_next_section_name", "")
    entry.setdefault("last_error_kind", "")
    entry.setdefault("session_id", "")
    entry.setdefault("outline_required", False)
    entry.setdefault("last_projected_content", "")
    if not isinstance(entry["recent_writes"], list):
        entry["recent_writes"] = []
    if not isinstance(entry["section_checkpoints"], list):
        entry["section_checkpoints"] = []
    return entry


def loop_guard_config(state: LoopState | None) -> dict[str, Any]:
    from .type_coerce import as_bool, as_float, as_int
    config = dict(_DEFAULT_LOOP_GUARD_CONFIG)
    if state is None:
        return config
    scratchpad = getattr(state, "scratchpad", None)
    raw = scratchpad.get(_LOOP_GUARD_CONFIG_KEY, {}) if isinstance(scratchpad, dict) else {}
    if not isinstance(raw, dict):
        return config

    config["enabled"] = as_bool(raw.get("enabled"), bool(config["enabled"]))
    config["stagnation_threshold"] = as_int(raw.get("stagnation_threshold"), int(config["stagnation_threshold"]))
    config["level2_threshold"] = as_int(raw.get("level2_threshold"), int(config["level2_threshold"]))
    config["recent_writes_limit"] = as_int(raw.get("recent_writes_limit"), int(config["recent_writes_limit"]))
    config["tail_lines"] = as_int(raw.get("tail_lines"), int(config["tail_lines"]))
    config["similarity_threshold"] = as_float(
        raw.get("similarity_threshold"),
        float(config["similarity_threshold"]),
    )
    config["append_overlap_threshold"] = as_float(
        raw.get("append_overlap_threshold"),
        float(config["append_overlap_threshold"]),
    )
    config["cumulative_write_gate"] = as_bool(
        raw.get("cumulative_write_gate"),
        bool(config["cumulative_write_gate"]),
    )
    config["checkpoint_gate"] = as_bool(raw.get("checkpoint_gate"), bool(config["checkpoint_gate"]))
    config["diff_gate"] = as_bool(raw.get("diff_gate"), bool(config["diff_gate"]))
    if config["level2_threshold"] < config["stagnation_threshold"]:
        config["level2_threshold"] = config["stagnation_threshold"]
    return config


def _record_guard_event(
    state: LoopState | None,
    *,
    event: str,
    path: str,
    session_id: str,
    payload: dict[str, Any],
) -> None:
    root = _loop_guard_root(state)
    if root is None:
        return
    events = root.setdefault("events", [])
    if not isinstance(events, list):
        return
    entry = {
        "event": str(event or "").strip() or "unknown",
        "at": time.time(),
        "path": path,
        "session_id": session_id,
        **payload,
    }
    # Deduplicate: skip if identical to last event (same event, path, session, payload)
    if events:
        last = events[-1]
        if (
            last.get("event") == entry["event"]
            and last.get("path") == entry["path"]
            and last.get("session_id") == entry["session_id"]
            and {k: v for k, v in last.items() if k not in ("event", "path", "session_id", "at")}
            == {k: v for k, v in entry.items() if k not in ("event", "path", "session_id", "at")}
        ):
            return
    events.append(entry)
    if len(events) > 40:
        del events[: len(events) - 40]
    logger = getattr(state, "log", None)
    if logger is not None:
        try:
            logger.info("loop_guard %s", json.dumps(entry, sort_keys=True, ensure_ascii=True))
        except Exception:
            pass


def _clear_loop_guard_read_schedule(state: LoopState | None) -> None:
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    scratchpad.pop("_chunk_write_loop_guard_read_scheduled", None)


def _append_recent_write(path_state: dict[str, Any], record: dict[str, Any], *, limit: int) -> None:
    recent = path_state.setdefault("recent_writes", [])
    if not isinstance(recent, list):
        recent = []
        path_state["recent_writes"] = recent
    recent.append(record)
    if len(recent) > limit:
        del recent[: len(recent) - limit]


def _loop_guard_attempt_count(path_state: dict[str, Any]) -> int:
    recent = path_state.get("recent_writes", [])
    recent_count = len(recent) if isinstance(recent, list) else 0
    blocked_attempts = int(path_state.get("blocked_attempts", 0) or 0)
    return max(1, recent_count, blocked_attempts + 1)
