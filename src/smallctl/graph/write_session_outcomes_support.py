from __future__ import annotations

import hashlib
import time
from typing import Any

from ..recovery_metrics import record_failure_event_metric
from ..recovery_schema import FailureEvent
from ..write_session_fsm import record_write_session_event as record_write_session_event_alias, transition_write_session


def _increment_stranded_replay_count(
    harness: Any,
    *,
    session_id: str,
    operation_id: str,
) -> int:
    key = f"{session_id}:{operation_id}"
    counts = harness.state.scratchpad.setdefault("_write_session_replay_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    count = int(counts.get(key, 0) or 0) + 1
    counts[key] = count
    harness.state.scratchpad["_write_session_replay_counts"] = counts
    return count


def _preserve_unverified_section(harness: Any, session: Any, record: Any) -> None:
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    current_section = str(record.result.metadata.get("write_current_section") or session.write_current_section or "").strip()
    record_write_session_event_alias(
        harness.state,
        event="failed_stage_preserved",
        session=session,
        details={
            "section": current_section,
            "tool_name": record.tool_name,
            "staging_path": staging_path,
        },
    )
    harness._runlog(
        "write_session_failed_stage_preserved",
        "preserved staged content after verifier failure",
        session_id=session.write_session_id,
        detail=staging_path,
    )


def _record_write_session_recovery_failure(
    harness: Any,
    session: Any,
    *,
    failure_class: str,
    message: str,
    evidence: list[str] | None = None,
    next_safe_action: str | None = None,
    severity: str = "recoverable",
    operation_id: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FailureEvent | None:
    state = getattr(harness, "state", None)
    if state is None:
        return None

    normalized_class = str(failure_class or "").strip() or "write_session_stall"
    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    evidence_items = [str(item).strip()[:500] for item in (evidence or []) if str(item).strip()]
    raw = "|".join(
        [
            normalized_class,
            "write_session",
            session_id,
            str(operation_id or ""),
            str(tool_name or ""),
            str(message or ""),
            "|".join(evidence_items),
        ]
    )
    subtask_id = _infer_active_subtask_id(harness, state)
    event = FailureEvent(
        event_id="write-session-" + hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16],
        timestamp=time.time(),
        failure_class=normalized_class,
        severity=severity if severity in {"info", "warning", "recoverable", "hard"} else "recoverable",
        source="write_session",
        message=f"{normalized_class}: {str(message or '').strip()[:240]}",
        evidence=evidence_items,
        fama_kind="write_session_stall",
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        operation_id=operation_id,
        subtask_id=subtask_id,
        suggested_next_action=next_safe_action,
        metadata={
            "session_id": session_id,
            "target_path": str(getattr(session, "write_target_path", "") or ""),
            "mode": str(getattr(session, "write_session_mode", "") or ""),
            "status": str(getattr(session, "status", "") or ""),
            **(metadata or {}),
        },
    )

    events = getattr(state, "failure_events", None)
    if not isinstance(events, list):
        events = []
        try:
            state.failure_events = events
        except Exception:
            return None
    if any(isinstance(item, FailureEvent) and item.event_id == event.event_id for item in events[-8:]):
        return None
    events.append(event)
    del events[:-20]
    record_failure_event_metric(state, event)

    try:
        state.last_failure_class = normalized_class
    except Exception:
        pass
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad["_last_failure_class"] = normalized_class
        if str(normalized_class or "").strip() == "verifier_failed":
            scratchpad["_verifier_failure_step"] = int(getattr(state, "step_count", 0))

    if not _attach_write_session_failure_via_service(harness, event):
        _attach_write_session_failure_direct(state, event)
    _maybe_create_write_session_reflection(harness, state, event)
    _mirror_write_session_working_memory_failure(state, event)

    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "recovery_failure_event_recorded",
            "Write-session recovery failure event recorded",
            event_id=event.event_id,
            failure_class=event.failure_class,
            source=event.source,
            session_id=session_id,
            tool_name=tool_name,
        )
    return event


def _infer_active_subtask_id(harness: Any, state: Any) -> str | None:
    service = getattr(harness, "subtask_ledger", None)
    infer = getattr(service, "infer_or_create_active_subtask", None)
    if callable(infer) and bool(getattr(getattr(harness, "config", None), "subtask_ledger_enabled", True)):
        try:
            active = infer()
            subtask_id = str(getattr(active, "subtask_id", "") or "").strip()
            if subtask_id:
                return subtask_id
        except Exception:
            pass
    ledger = getattr(state, "subtask_ledger", None)
    active = ledger.active() if ledger is not None and callable(getattr(ledger, "active", None)) else None
    subtask_id = str(getattr(active, "subtask_id", "") or "").strip()
    return subtask_id or None


def _attach_write_session_failure_via_service(harness: Any, event: FailureEvent) -> bool:
    service = getattr(harness, "subtask_ledger", None)
    attach_failure = getattr(service, "attach_failure", None)
    if not event.subtask_id or not callable(attach_failure):
        return False
    try:
        attach_failure(event.subtask_id, event)
        return True
    except Exception:
        return False


def _attach_write_session_failure_direct(state: Any, event: FailureEvent) -> None:
    ledger = getattr(state, "subtask_ledger", None)
    active = ledger.active() if ledger is not None and callable(getattr(ledger, "active", None)) else None
    if active is None:
        return
    active.attempts = int(getattr(active, "attempts", 0) or 0) + 1
    if event.failure_class not in active.failure_classes:
        active.failure_classes.append(event.failure_class)
    if event.message and event.message not in active.blockers:
        active.blockers.append(event.message)
        active.blockers = active.blockers[-5:]
    if event.suggested_next_action:
        active.next_action = event.suggested_next_action
    active.updated_at = event.timestamp


def _maybe_create_write_session_reflection(harness: Any, state: Any, event: FailureEvent) -> None:
    service = getattr(harness, "reflexion", None)
    maybe_create = getattr(service, "maybe_create_reflection", None)
    if not callable(maybe_create):
        return
    try:
        maybe_create(event, getattr(state, "subtask_ledger", None))
    except Exception:
        return


def _mirror_write_session_working_memory_failure(state: Any, event: FailureEvent) -> None:
    working_memory = getattr(state, "working_memory", None)
    failures = getattr(working_memory, "failures", None)
    if not isinstance(failures, list):
        return
    line = event.message
    if event.suggested_next_action:
        line = f"{line}; next={event.suggested_next_action}"
    if line not in failures:
        failures.append(line)
        del failures[:-5]


def _maybe_trigger_write_session_fallback(harness: Any, session: Any) -> bool:
    config = getattr(harness, "config", None)
    limit = config.failed_local_patch_limit if config else 3
    abort_limit = max(limit + 2, 5)
    if session.write_failed_local_patches < limit:
        return False

    if session.write_failed_local_patches >= abort_limit:
        _abort_write_session(harness, session)
        return True

    transition_write_session(
        session,
        next_status="fallback",
        next_mode="local_full_rewrite",
    )
    _record_write_session_recovery_failure(
        harness,
        session,
        failure_class="write_session_stall",
        message=f"Falling back to local_full_rewrite after {session.write_failed_local_patches} local patch failures",
        next_safe_action="Retry the write with a simpler approach or escalate to a larger model.",
        metadata={"recovery_kind": "write_session_fallback"},
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "write_session_fallback_triggered",
            "Falling back to local_full_rewrite after too many local patch failures",
            session_id=str(getattr(session, "write_session_id", "") or "").strip(),
            failed_local_patches=session.write_failed_local_patches,
        )
    return True


def _abort_write_session(harness: Any, session: Any) -> None:
    from ..write_session_fsm import archive_interrupted_write_session

    archive_interrupted_write_session(
        harness.state,
        reason=f"write_session_aborted_after_{session.write_failed_local_patches}_failed_patches",
    )
    _record_write_session_recovery_failure(
        harness,
        session,
        failure_class="write_session_stall",
        message=f"Aborting write session `{session.write_session_id}` after {session.write_failed_local_patches} consecutive failures. Session archived. Use a single atomic file_write to start fresh.",
        next_safe_action="Use file_write with replace_strategy='overwrite' to write the complete file in one shot.",
        severity="hard",
        metadata={"recovery_kind": "write_session_abort"},
    )
    harness.state.write_session = None
    harness.state.active_intent = "write_file"
    harness.state.secondary_intents = []
    scratchpad = getattr(harness.state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad["_last_failure_class"] = "write_session_stall"
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "write_session_aborted",
            "Aborted write session after repeated stall failures",
            session_id=str(getattr(session, "write_session_id", "") or "").strip(),
            failed_local_patches=session.write_failed_local_patches,
        )
