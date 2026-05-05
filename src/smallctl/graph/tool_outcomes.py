from __future__ import annotations

from time import time
from typing import Any

from ..recovery_metrics import record_failure_event_metric
from ..recovery_schema import FailureEvent
from . import task_completion_outcomes as _task_completion_outcomes
from . import write_session_outcomes as _write_session_outcomes
from .deps import GraphRuntimeDeps
from .routing import LoopRoute
from .state import GraphRunState, ToolExecutionRecord
from .chat_progress import _record_chat_progress_outcome
from .planning_outcomes import apply_planning_tool_outcomes
from .shell_outcomes import (
    _shell_human_retry_hint,
    _shell_ssh_retry_hint,
    _shell_workspace_relative_retry_hint,
)
from .tool_execution_recovery import handle_failed_file_write_outcome
from .tool_execution_recovery_helpers import (
    _maybe_emit_repair_recovery_nudge,
    _maybe_schedule_repair_loop_status_autocontinue,
)
from .error_hardening import (
    _maybe_emit_ground_truth_diffusion,
    _maybe_emit_nginx_sites_enabled_nudge,
    _maybe_schedule_web_search_for_repeated_error,
)
from .tool_outcome_resolution import maybe_apply_terminal_tool_outcome
from .progress_guard import _update_progress_tracking
from .tool_execution_support import (
    _conversation_message_from_dict,
    _get_tool_execution_record,
    _has_matching_tool_message,
    _store_tool_execution_record,
    _tool_envelope_from_dict,
)

_maybe_schedule_write_recovery_readback = _write_session_outcomes._maybe_schedule_write_recovery_readback
_auto_update_active_plan_step = _write_session_outcomes._auto_update_active_plan_step
_register_write_session_stage_artifact = _write_session_outcomes._register_write_session_stage_artifact
_maybe_record_write_session_first_chunk_metric = _write_session_outcomes._maybe_record_write_session_first_chunk_metric
_maybe_emit_patch_existing_first_choice_nudge = _write_session_outcomes._maybe_emit_patch_existing_first_choice_nudge
_handle_write_session_outcome = _write_session_outcomes._handle_write_session_outcome
_maybe_schedule_task_complete_remote_mutation_verifier = _task_completion_outcomes._maybe_schedule_task_complete_remote_mutation_verifier
_maybe_schedule_task_complete_verifier_loop_status = _task_completion_outcomes._maybe_schedule_task_complete_verifier_loop_status
_maybe_schedule_task_complete_repair_loop_status = _task_completion_outcomes._maybe_schedule_task_complete_repair_loop_status
_maybe_emit_task_complete_verifier_nudge = _task_completion_outcomes._maybe_emit_task_complete_verifier_nudge


async def apply_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        if record.tool_name == "shell_exec":
            _maybe_emit_repair_recovery_nudge(harness, record, deps)
            _maybe_schedule_repair_loop_status_autocontinue(graph_state, harness, record)
            _maybe_emit_nginx_sites_enabled_nudge(harness, record)
            _maybe_emit_ground_truth_diffusion(harness, record)
            _maybe_schedule_web_search_for_repeated_error(graph_state, harness, record)
        elif record.tool_name == "ssh_exec":
            _maybe_emit_repair_recovery_nudge(harness, record, deps)
            _maybe_schedule_repair_loop_status_autocontinue(graph_state, harness, record)
            _maybe_emit_nginx_sites_enabled_nudge(harness, record)
            _maybe_emit_ground_truth_diffusion(harness, record)
            _maybe_schedule_web_search_for_repeated_error(graph_state, harness, record)

        if await maybe_apply_terminal_tool_outcome(graph_state, deps, record, chat_mode=False):
            return LoopRoute.FINALIZE

        await handle_failed_file_write_outcome(
            graph_state=graph_state,
            harness=harness,
            deps=deps,
            record=record,
        )
        if graph_state.final_result is not None:
            return LoopRoute.FINALIZE

        _write_session_outcomes._maybe_record_write_session_first_chunk_metric(graph_state, harness, record)
        await _write_session_outcomes._handle_write_session_outcome(harness, record)
        _update_subtask_ledger_from_record(harness, record)

    _update_progress_tracking(harness, graph_state)
    task_boundary_service = getattr(harness, "_task_boundary_service", None)
    if task_boundary_service is not None:
        task_boundary_service.maybe_nudge_internal_divergence()
    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


async def apply_chat_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        await handle_failed_file_write_outcome(
            graph_state=graph_state,
            harness=harness,
            deps=deps,
            record=record,
        )
        if graph_state.final_result is not None:
            return LoopRoute.FINALIZE

        _write_session_outcomes._maybe_record_write_session_first_chunk_metric(graph_state, harness, record)
        await _write_session_outcomes._handle_write_session_outcome(harness, record)
        _update_subtask_ledger_from_record(harness, record)

        if await maybe_apply_terminal_tool_outcome(graph_state, deps, record, chat_mode=True):
            return LoopRoute.FINALIZE

    _update_progress_tracking(harness, graph_state)
    _record_chat_progress_outcome(harness, graph_state.last_tool_results)
    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


def _update_subtask_ledger_from_record(harness: Any, record: ToolExecutionRecord) -> None:
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "subtask_ledger_enabled", True)):
        return
    service = getattr(harness, "subtask_ledger", None)
    if service is None:
        return
    try:
        service.import_plan_if_needed()
        active = service.infer_or_create_active_subtask()
        if record.result.success:
            evidence = _ledger_success_evidence(record)
            if evidence:
                service.attach_evidence(active.subtask_id, evidence)
            return
        if _has_failure_event_for_operation(harness, record.operation_id):
            return
        failure = _failure_event_from_record(harness, record, subtask_id=active.subtask_id)
        events = getattr(getattr(harness, "state", None), "failure_events", None)
        if isinstance(events, list):
            events.append(failure)
            del events[:-40]
        record_failure_event_metric(harness.state, failure)
        service.attach_failure(active.subtask_id, failure)
    except Exception:
        return


def _ledger_success_evidence(record: ToolExecutionRecord) -> str:
    tool_name = str(record.tool_name or "").strip()
    if tool_name in {"task_complete", "task_fail"}:
        return ""
    target = _record_target(record)
    if target:
        return f"{tool_name} succeeded for {target}"
    output = record.result.output
    if isinstance(output, dict):
        message = str(output.get("message") or output.get("status") or "").strip()
        if message:
            return f"{tool_name} succeeded: {message[:160]}"
    return f"{tool_name} succeeded"


def _failure_event_from_record(harness: Any, record: ToolExecutionRecord, *, subtask_id: str) -> FailureEvent:
    failure_class = _record_failure_class(harness, record)
    message = f"{failure_class}: {record.tool_name} failed"
    detail = str(record.result.error or "").strip()
    if not detail and isinstance(record.result.output, dict):
        detail = str(record.result.output.get("stderr") or record.result.output.get("message") or "").strip()
    if detail:
        message = f"{message} - {detail[:180]}"
    return FailureEvent(
        event_id=f"tool-{record.operation_id or record.tool_call_id or int(time() * 1000)}",
        timestamp=time(),
        failure_class=failure_class,
        severity="warning",
        source="tool_outcome",
        message=message,
        evidence=[detail[:240]] if detail else [],
        tool_name=record.tool_name,
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
        subtask_id=subtask_id,
        suggested_next_action="Use the tool failure evidence to take the next smallest different action.",
    )


def _record_failure_class(harness: Any, record: ToolExecutionRecord) -> str:
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    for key in ("failure_class", "failure_mode", "reason", "error_type"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    last_failure = str(getattr(getattr(harness, "state", None), "last_failure_class", "") or "").strip()
    return last_failure or "tool_execution_failed"


def _record_target(record: ToolExecutionRecord) -> str:
    for key in ("path", "target", "command", "host"):
        value = str(record.args.get(key) or "").strip()
        if value:
            return value[:160]
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    for key in ("path", "target", "command", "host"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value[:160]
    return ""


def _has_failure_event_for_operation(harness: Any, operation_id: str) -> bool:
    if not operation_id:
        return False
    events = getattr(getattr(harness, "state", None), "failure_events", None)
    if not isinstance(events, list):
        return False
    return any(str(getattr(event, "operation_id", "") or "") == operation_id for event in events[-8:])
