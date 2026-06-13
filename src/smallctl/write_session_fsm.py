from __future__ import annotations

import time
from typing import Any

from .state import WriteSession

WRITE_SESSION_ALLOWED_MODES = {
    "single_write",
    "chunked_author",
    "local_repair",
    "stub_and_fill",
}
WRITE_SESSION_ALLOWED_STATUSES = {
    "open",
    "local_repair",
    "fallback",
    "complete",
}
WRITE_SESSION_ALLOWED_TRANSITIONS = {
    "open": {"open", "local_repair", "fallback", "complete"},
    "local_repair": {"local_repair", "open", "fallback", "complete"},
    "fallback": {"fallback", "local_repair", "open", "complete"},
    "complete": {"complete"},
}
_WRITE_SESSION_EVENTS_KEY = "_write_session_events"
WRITE_SESSION_TERMINAL_STATUSES = {"complete"}
_ARCHIVED_WRITE_SESSIONS_KEY = "_archived_write_sessions"


def is_terminal_write_session(session: Any) -> bool:
    status = str(getattr(session, "status", "") or "").strip().lower()
    return status in WRITE_SESSION_TERMINAL_STATUSES


def new_write_session(
    *,
    session_id: str,
    target_path: str,
    intent: str,
    mode: str = "chunked_author",
    suggested_sections: list[str] | None = None,
    next_section: str = "",
) -> WriteSession:
    normalized_mode = str(mode or "chunked_author").strip().lower()
    if normalized_mode not in WRITE_SESSION_ALLOWED_MODES:
        normalized_mode = "chunked_author"
    return WriteSession(
        write_session_id=str(session_id or "").strip(),
        write_target_path=str(target_path or "").strip(),
        write_session_intent=str(intent or "replace_file").strip() or "replace_file",
        write_session_mode=normalized_mode,
        write_session_started_at=time.time(),
        suggested_sections=list(suggested_sections or []),
        write_next_section=str(next_section or "").strip(),
        status="open",
    )


def transition_write_session(
    session: WriteSession,
    *,
    next_status: str | None = None,
    next_mode: str | None = None,
    pending_finalize: bool | None = None,
    current_section: str | None = None,
    next_section: str | None = None,
) -> bool:
    changed = False
    current_status = str(session.status or "open").strip().lower() or "open"
    desired_status = current_status if next_status is None else str(next_status).strip().lower()
    if desired_status not in WRITE_SESSION_ALLOWED_STATUSES:
        desired_status = current_status
    allowed = WRITE_SESSION_ALLOWED_TRANSITIONS.get(current_status, {current_status})
    if desired_status not in allowed:
        desired_status = current_status
    if desired_status != current_status:
        session.status = desired_status
        changed = True

    if next_mode is not None:
        desired_mode = str(next_mode).strip().lower()
        if desired_mode in WRITE_SESSION_ALLOWED_MODES and desired_mode != str(session.write_session_mode or "").strip().lower():
            session.write_session_mode = desired_mode
            changed = True

    if pending_finalize is not None and bool(session.write_pending_finalize) != bool(pending_finalize):
        session.write_pending_finalize = bool(pending_finalize)
        changed = True

    if current_section is not None and str(current_section).strip() != str(session.write_current_section or "").strip():
        session.write_current_section = str(current_section).strip()
        changed = True

    if next_section is not None and str(next_section).strip() != str(session.write_next_section or "").strip():
        session.write_next_section = str(next_section).strip()
        changed = True

    return changed


def record_write_session_event(
    state: Any,
    *,
    event: str,
    session: WriteSession | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if state is None:
        return
    payload = {
        "event": str(event or "").strip() or "unknown",
        "at": time.time(),
        "session_id": str(getattr(session, "write_session_id", "") or ""),
        "target_path": str(getattr(session, "write_target_path", "") or ""),
        "mode": str(getattr(session, "write_session_mode", "") or ""),
        "status": str(getattr(session, "status", "") or ""),
        "details": dict(details or {}),
    }
    events = state.scratchpad.setdefault(_WRITE_SESSION_EVENTS_KEY, [])
    if not isinstance(events, list):
        events = []
    events.append(payload)
    state.scratchpad[_WRITE_SESSION_EVENTS_KEY] = events[-40:]


def _write_session_archive_payload(session: WriteSession, *, reason: str) -> dict[str, Any]:
    return {
        "write_session_id": str(getattr(session, "write_session_id", "") or ""),
        "write_target_path": str(getattr(session, "write_target_path", "") or ""),
        "write_session_intent": str(getattr(session, "write_session_intent", "") or ""),
        "write_session_mode": str(getattr(session, "write_session_mode", "") or ""),
        "status": str(getattr(session, "status", "") or ""),
        "write_staging_path": str(getattr(session, "write_staging_path", "") or ""),
        "write_original_snapshot_path": str(getattr(session, "write_original_snapshot_path", "") or ""),
        "write_target_existed_at_start": bool(getattr(session, "write_target_existed_at_start", False)),
        "write_section_ranges": dict(getattr(session, "write_section_ranges", {}) or {}),
        "write_last_attempt_snapshot_path": str(getattr(session, "write_last_attempt_snapshot_path", "") or ""),
        "write_last_attempt_sections": list(getattr(session, "write_last_attempt_sections", []) or []),
        "write_last_attempt_ranges": dict(getattr(session, "write_last_attempt_ranges", {}) or {}),
        "write_last_staged_hash": str(getattr(session, "write_last_staged_hash", "") or ""),
        "write_sections_completed": list(getattr(session, "write_sections_completed", []) or []),
        "write_current_section": str(getattr(session, "write_current_section", "") or ""),
        "write_next_section": str(getattr(session, "write_next_section", "") or ""),
        "write_pending_finalize": bool(getattr(session, "write_pending_finalize", False)),
        "suggested_sections": list(getattr(session, "suggested_sections", []) or []),
        "archived_at": time.time(),
        "reason": str(reason or "").strip(),
    }


def _append_archived_write_session(state: Any, payload: dict[str, Any]) -> dict[str, Any]:
    archived = state.scratchpad.setdefault(_ARCHIVED_WRITE_SESSIONS_KEY, [])
    if not isinstance(archived, list):
        archived = []
    archived.append(payload)
    state.scratchpad[_ARCHIVED_WRITE_SESSIONS_KEY] = archived[-12:]
    return payload


def archive_terminal_write_session(
    state: Any,
    *,
    reason: str,
) -> dict[str, Any] | None:
    if state is None:
        return None
    # Archive all terminal sessions from the path map, not just the alias.
    active_map = getattr(state, "active_write_sessions_by_path", None)
    terminal_sessions: list[Any] = []
    if isinstance(active_map, dict):
        for session in list(active_map.values()):
            if session is not None and is_terminal_write_session(session):
                terminal_sessions.append(session)
    alias_session = getattr(state, "write_session", None)
    if alias_session is not None and is_terminal_write_session(alias_session):
        if alias_session not in terminal_sessions:
            terminal_sessions.append(alias_session)
    if not terminal_sessions:
        return None
    payloads: list[dict[str, Any]] = []
    for session in terminal_sessions:
        payload = _append_archived_write_session(
            state,
            _write_session_archive_payload(session, reason=reason),
        )
        payloads.append(payload)
        _record_dead_write_session_id(state, payload.get("write_session_id"))
    state.active_write_sessions_by_path = {
        key: session
        for key, session in (active_map or {}).items()
        if session is not None and not is_terminal_write_session(session)
    }
    _refresh_write_session_alias(state)
    if payloads:
        record_write_session_event(
            state,
            event="terminal_write_session_cleared_on_continue",
            session=terminal_sessions[-1],
            details={"reason": payloads[-1]["reason"], "count": len(payloads)},
        )
        return payloads[-1]
    return None


def _refresh_write_session_alias(state: Any) -> None:
    """Point state.write_session at the most recently started non-terminal session, if any."""
    if state is None:
        return
    candidates = [
        session
        for session in (getattr(state, "active_write_sessions_by_path", {}) or {}).values()
        if session is not None and not is_terminal_write_session(session)
    ]
    if candidates:
        candidates.sort(key=lambda s: float(getattr(s, "write_session_started_at", 0.0) or 0.0), reverse=True)
        state.write_session = candidates[0]
    else:
        state.write_session = None


def archive_interrupted_write_session(
    state: Any,
    *,
    reason: str,
) -> dict[str, Any] | None:
    if state is None:
        return None
    active_map = getattr(state, "active_write_sessions_by_path", None)
    interrupted_sessions: list[Any] = []
    if isinstance(active_map, dict):
        for session in list(active_map.values()):
            if session is not None and not is_terminal_write_session(session):
                interrupted_sessions.append(session)
    alias_session = getattr(state, "write_session", None)
    if alias_session is not None and not is_terminal_write_session(alias_session):
        if alias_session not in interrupted_sessions:
            interrupted_sessions.append(alias_session)
    if not interrupted_sessions:
        return None
    payloads: list[dict[str, Any]] = []
    for session in interrupted_sessions:
        payload = _append_archived_write_session(
            state,
            _write_session_archive_payload(session, reason=reason),
        )
        payloads.append(payload)
        _record_dead_write_session_id(state, payload.get("write_session_id"))
    state.active_write_sessions_by_path = {}
    if payloads:
        # Preserve the legacy alias pointing at the last archived session for
        # callers that expect it, while the map is now empty.
        state.write_session = interrupted_sessions[-1]
        record_write_session_event(
            state,
            event="interrupted_write_session_archived",
            session=interrupted_sessions[-1],
            details={"reason": payloads[-1]["reason"], "count": len(payloads)},
        )
        return payloads[-1]
    state.write_session = None
    return None


def _record_dead_write_session_id(state: Any, session_id: str | None) -> None:
    if state is None or not session_id:
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    key = "_dead_write_session_ids"
    dead = scratchpad.get(key)
    if not isinstance(dead, list):
        dead = []
    sid = str(session_id).strip()
    if sid and sid not in dead:
        dead.append(sid)
    scratchpad[key] = dead[-24:]


def recent_write_session_events(state: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    if state is None:
        return []
    events = state.scratchpad.get(_WRITE_SESSION_EVENTS_KEY)
    if not isinstance(events, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in events[-max(1, int(limit)):]:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized
