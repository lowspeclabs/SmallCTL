from __future__ import annotations

from typing import Any

from .recovery_schema import FailureEvent, ReflectionMemory, Subtask, SubtaskLedger
from .state_support import (
    _coerce_float,
    _coerce_int,
    _coerce_list_payload,
    _coerce_string_list,
    _filter_dataclass_payload,
    json_safe_value,
)

_FAILURE_SEVERITIES = {"info", "warning", "recoverable", "hard"}
_SUBTASK_STATUSES = {"pending", "active", "blocked", "done", "failed", "abandoned"}


def _coerce_failure_event(value: Any) -> FailureEvent | None:
    if isinstance(value, FailureEvent):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    event_id = str(payload.get("event_id") or "").strip()
    failure_class = str(payload.get("failure_class") or "").strip()
    if not event_id or not failure_class:
        return None
    payload["event_id"] = event_id
    payload["failure_class"] = failure_class
    payload["timestamp"] = _coerce_float(payload.get("timestamp"), default=0.0)
    severity = str(payload.get("severity") or "warning").strip()
    payload["severity"] = severity if severity in _FAILURE_SEVERITIES else "warning"
    payload["source"] = str(payload.get("source") or "")
    payload["message"] = str(payload.get("message") or "")
    payload["evidence"] = _coerce_string_list(payload.get("evidence"))
    for key in ("fama_kind", "tool_name", "tool_call_id", "operation_id", "subtask_id", "suggested_next_action"):
        payload[key] = _optional_str(payload.get(key))
    metadata = json_safe_value(payload.get("metadata") or {})
    payload["metadata"] = metadata if isinstance(metadata, dict) else {}
    return FailureEvent(**_filter_dataclass_payload(FailureEvent, payload))


def _coerce_reflection_memory(value: Any) -> ReflectionMemory | None:
    if isinstance(value, ReflectionMemory):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    reflection_id = str(payload.get("reflection_id") or "").strip()
    failure_class = str(payload.get("failure_class") or "").strip()
    if not reflection_id or not failure_class:
        return None
    payload["reflection_id"] = reflection_id
    payload["failure_class"] = failure_class
    payload["timestamp"] = _coerce_float(payload.get("timestamp"), default=0.0)
    payload["task_id"] = _optional_str(payload.get("task_id"))
    payload["subtask_id"] = _optional_str(payload.get("subtask_id"))
    for key in ("lesson", "avoid", "next_safe_action", "evidence_summary"):
        payload[key] = str(payload.get(key) or "")
    payload["score"] = _coerce_float(payload.get("score"), default=1.0)
    payload["used_count"] = max(0, _coerce_int(payload.get("used_count"), default=0))
    metadata = json_safe_value(payload.get("metadata") or {})
    payload["metadata"] = metadata if isinstance(metadata, dict) else {}
    return ReflectionMemory(**_filter_dataclass_payload(ReflectionMemory, payload))


def _coerce_subtask(value: Any) -> Subtask | None:
    if isinstance(value, Subtask):
        return value
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    subtask_id = str(payload.get("subtask_id") or "").strip()
    title = str(payload.get("title") or "").strip()
    goal = str(payload.get("goal") or "").strip()
    if not subtask_id or not title:
        return None
    payload["subtask_id"] = subtask_id
    payload["title"] = title
    payload["goal"] = goal
    status = str(payload.get("status") or "pending").strip()
    payload["status"] = status if status in _SUBTASK_STATUSES else "pending"
    payload["acceptance"] = _coerce_string_list(payload.get("acceptance"))
    payload["evidence"] = _coerce_string_list(payload.get("evidence"))
    payload["blockers"] = _coerce_string_list(payload.get("blockers"))
    payload["next_action"] = _optional_str(payload.get("next_action"))
    payload["created_at"] = _coerce_float(payload.get("created_at"), default=0.0)
    payload["updated_at"] = _coerce_float(payload.get("updated_at"), default=0.0)
    payload["attempts"] = max(0, _coerce_int(payload.get("attempts"), default=0))
    payload["failure_classes"] = _coerce_string_list(payload.get("failure_classes"))
    return Subtask(**_filter_dataclass_payload(Subtask, payload))


def _coerce_subtask_ledger(value: Any) -> SubtaskLedger | None:
    if isinstance(value, SubtaskLedger):
        return value
    if not isinstance(value, dict):
        return None
    subtasks = [
        task
        for item in _coerce_list_payload(value.get("subtasks"))
        if (task := _coerce_subtask(item)) is not None
    ]
    active_subtask_id = _optional_str(value.get("active_subtask_id"))
    if active_subtask_id and not any(task.subtask_id == active_subtask_id for task in subtasks):
        active_subtask_id = None
    return SubtaskLedger(
        task_id=_optional_str(value.get("task_id")),
        subtasks=subtasks,
        active_subtask_id=active_subtask_id,
    )


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
