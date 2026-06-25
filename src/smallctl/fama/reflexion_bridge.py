from __future__ import annotations

import hashlib
from time import time
from typing import Any

from ..recovery_metrics import record_failure_event_metric
from ..recovery_schema import FailureEvent, FailureSeverity
from ..logging_utils import runlog
from .signals import DEFAULT_FAILURE_CLASS_BY_KIND, FamaSignal

_FAILURE_EVENT_LIMIT = 40


def record_fama_failure_event(
    harness: Any,
    *,
    state: Any,
    signal: FamaSignal,
) -> FailureEvent | None:
    failure_class = str(
        signal.failure_class
        or DEFAULT_FAILURE_CLASS_BY_KIND.get(signal.kind)
        or signal.kind.value
    ).strip()
    if not failure_class:
        return None

    event = FailureEvent(
        event_id=_event_id(signal, failure_class=failure_class),
        timestamp=time(),
        failure_class=failure_class,
        severity=_severity_from_signal(signal),
        source=str(signal.source or "fama"),
        message=_message_from_signal(signal, failure_class=failure_class),
        evidence=[str(signal.evidence or "").strip()] if str(signal.evidence or "").strip() else [],
        fama_kind=signal.kind.value,
        tool_name=signal.tool_name,
        operation_id=signal.operation_id,
        subtask_id=_active_subtask_id(state),
        suggested_next_action=signal.next_safe_action,
        metadata={"step": signal.step},
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
    del events[:-_FAILURE_EVENT_LIMIT]
    record_failure_event_metric(state, event)
    _set_last_failure_class(state, failure_class)
    if not _service_attach_failure(harness, state, event):
        _attach_to_active_subtask(state, event)
    _maybe_create_reflection(harness, state, event)
    _mirror_working_memory_failure(state, event)
    runlog(
        harness,
        "recovery_failure_event_recorded",
        "Recovery failure event recorded",
        event_id=event.event_id,
        failure_class=event.failure_class,
        source=event.source,
        fama_kind=event.fama_kind,
        tool_name=event.tool_name,
    )
    return event


def _event_id(signal: FamaSignal, *, failure_class: str) -> str:
    raw = "|".join(
        [
            failure_class,
            signal.kind.value,
            str(signal.source or ""),
            str(signal.tool_name or ""),
            str(signal.operation_id or ""),
            str(signal.evidence or ""),
        ]
    )
    return "fama-" + hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _severity_from_signal(signal: FamaSignal) -> FailureSeverity:
    try:
        severity = int(signal.severity)
    except (TypeError, ValueError):
        severity = 2
    if severity <= 1:
        return "info"
    if severity >= 3:
        return "recoverable"
    return "warning"


def _message_from_signal(signal: FamaSignal, *, failure_class: str) -> str:
    evidence = str(signal.evidence or "").strip()
    if evidence:
        return f"{failure_class}: {evidence[:240]}"
    return f"{failure_class}: {signal.kind.value}"


def _active_subtask_id(state: Any) -> str | None:
    ledger = getattr(state, "subtask_ledger", None)
    active = ledger.active() if ledger is not None and callable(getattr(ledger, "active", None)) else None
    subtask_id = str(getattr(active, "subtask_id", "") or "").strip()
    return subtask_id or None


def _set_last_failure_class(state: Any, failure_class: str) -> None:
    try:
        state.last_failure_class = failure_class
    except Exception:
        pass
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad["_last_failure_class"] = failure_class


def _attach_to_active_subtask(state: Any, event: FailureEvent) -> None:
    ledger = getattr(state, "subtask_ledger", None)
    active = ledger.active() if ledger is not None and callable(getattr(ledger, "active", None)) else None
    if active is None:
        return
    active.attempts = int(getattr(active, "attempts", 0) or 0) + 1
    if event.failure_class not in active.failure_classes:
        active.failure_classes.append(event.failure_class)
    blocker = event.message
    if blocker and blocker not in active.blockers:
        active.blockers.append(blocker)
        active.blockers = active.blockers[-5:]
    if event.suggested_next_action:
        active.next_action = event.suggested_next_action
    active.updated_at = event.timestamp


def _service_attach_failure(harness: Any, state: Any, event: FailureEvent) -> bool:
    subtask_id = event.subtask_id or _active_subtask_id(state)
    service = getattr(harness, "subtask_ledger", None)
    attach_failure = getattr(service, "attach_failure", None)
    if not subtask_id or not callable(attach_failure):
        return False
    try:
        attach_failure(subtask_id, event)
        return True
    except Exception:
        return False


def _maybe_create_reflection(harness: Any, state: Any, event: FailureEvent) -> None:
    service = getattr(harness, "reflexion", None)
    maybe_create = getattr(service, "maybe_create_reflection", None)
    if not callable(maybe_create):
        return
    try:
        maybe_create(event, getattr(state, "subtask_ledger", None))
    except Exception:
        return


def _mirror_working_memory_failure(state: Any, event: FailureEvent) -> None:
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
