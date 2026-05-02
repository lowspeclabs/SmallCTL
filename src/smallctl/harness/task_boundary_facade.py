from __future__ import annotations

from typing import Any

from .task_boundary import TaskBoundaryService


def _task_boundary_service_for(harness: Any) -> TaskBoundaryService:
    service = getattr(harness, "_task_boundary_service", None)
    if service is None:
        service = TaskBoundaryService(harness)
        try:
            setattr(harness, "_task_boundary_service", service)
        except Exception:
            pass
    return service


def _active_task_scope_payload(self: Any) -> dict[str, Any] | None:
    return _task_boundary_service_for(self)._active_task_scope_payload()


def _clip_task_summary_text(self: Any, value: Any, *, limit: int = 240) -> str:
    return _task_boundary_service_for(self)._clip_task_summary_text(value, limit=limit)


def _extract_task_terminal_message(self: Any, result: dict[str, Any] | None) -> str:
    return _task_boundary_service_for(self)._extract_task_terminal_message(result)


def _task_duration_seconds(self: Any, started_at: str, finished_at: str) -> float:
    return _task_boundary_service_for(self)._task_duration_seconds(started_at, finished_at)


def _write_task_summary(self: Any, payload: dict[str, Any]) -> str:
    return _task_boundary_service_for(self)._write_task_summary(payload)


def _begin_task_scope(self: Any, *, raw_task: str, effective_task: str) -> dict[str, Any]:
    return _task_boundary_service_for(self).begin_task_scope(raw_task=raw_task, effective_task=effective_task)


def _finalize_task_scope(
    self: Any,
    *,
    terminal_event: str,
    status: str,
    reason: str = "",
    result: dict[str, Any] | None = None,
    replacement_task: str = "",
) -> dict[str, Any] | None:
    return _task_boundary_service_for(self).finalize_task_scope(
        terminal_event=terminal_event,
        status=status,
        reason=reason,
        result=result,
        replacement_task=replacement_task,
    )


def _reset_task_boundary_state(
    self: Any,
    *,
    reason: str,
    new_task: str = "",
    previous_task: str = "",
    preserve_memory: bool = False,
    preserve_summaries: bool = False,
    preserve_recent_tail: bool = False,
    preserve_guard_context: bool = False,
) -> None:
    _task_boundary_service_for(self).reset_task_boundary_state(
        reason=reason,
        new_task=new_task,
        previous_task=previous_task,
        preserve_memory=preserve_memory,
        preserve_summaries=preserve_summaries,
        preserve_recent_tail=preserve_recent_tail,
        preserve_guard_context=preserve_guard_context,
    )


def _maybe_reset_for_new_task(self: Any, task: str, *, raw_task: str | None = None) -> None:
    _task_boundary_service_for(self).maybe_reset_for_new_task(task, raw_task=raw_task)


def _has_task_local_context(self: Any) -> bool:
    return _task_boundary_service_for(self).has_task_local_context()


def _has_resettable_context(self: Any) -> bool:
    return _task_boundary_service_for(self).has_resettable_context()


def _has_durable_context(self: Any) -> bool:
    return _task_boundary_service_for(self).has_durable_context()


def _last_task_handoff(self: Any) -> dict[str, Any]:
    return _task_boundary_service_for(self).last_task_handoff()


def _is_continue_like_followup(self: Any, task: str) -> bool:
    return _task_boundary_service_for(self)._is_continue_like_followup(task)


def _is_contextual_followup(self: Any, task: str) -> bool:
    return _task_boundary_service_for(self)._is_contextual_followup(task)


def _resolve_followup_task(self: Any, task: str) -> str:
    return _task_boundary_service_for(self).resolve_followup_task(task)


def _store_task_handoff(self: Any, *, raw_task: str, effective_task: str) -> None:
    _task_boundary_service_for(self).store_task_handoff(raw_task=raw_task, effective_task=effective_task)


def _refresh_task_handoff_action_options(self: Any, assistant_text: str) -> None:
    _task_boundary_service_for(self).refresh_task_handoff_action_options(assistant_text)


def _initialize_run_brief(self: Any, task: str, *, raw_task: str | None = None) -> None:
    _task_boundary_service_for(self).initialize_run_brief(task, raw_task=raw_task)


def _current_user_task(self: Any) -> str:
    return _task_boundary_service_for(self).current_user_task()


def bind_task_boundary_facade(cls: type[Any]) -> None:
    cls._task_boundary_service_for = staticmethod(_task_boundary_service_for)
    cls._active_task_scope_payload = _active_task_scope_payload
    cls._clip_task_summary_text = _clip_task_summary_text
    cls._extract_task_terminal_message = _extract_task_terminal_message
    cls._task_duration_seconds = _task_duration_seconds
    cls._write_task_summary = _write_task_summary
    cls._begin_task_scope = _begin_task_scope
    cls._finalize_task_scope = _finalize_task_scope
    cls._reset_task_boundary_state = _reset_task_boundary_state
    cls._maybe_reset_for_new_task = _maybe_reset_for_new_task
    cls._has_task_local_context = _has_task_local_context
    cls._has_resettable_context = _has_resettable_context
    cls._has_durable_context = _has_durable_context
    cls._last_task_handoff = _last_task_handoff
    cls._is_continue_like_followup = _is_continue_like_followup
    cls._is_contextual_followup = _is_contextual_followup
    cls._resolve_followup_task = _resolve_followup_task
    cls._store_task_handoff = _store_task_handoff
    cls._refresh_task_handoff_action_options = _refresh_task_handoff_action_options
    cls._initialize_run_brief = _initialize_run_brief
    cls._current_user_task = _current_user_task
