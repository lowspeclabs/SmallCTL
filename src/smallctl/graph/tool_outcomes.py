from __future__ import annotations

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
        elif record.tool_name == "ssh_exec":
            _maybe_emit_repair_recovery_nudge(harness, record, deps)
            _maybe_schedule_repair_loop_status_autocontinue(graph_state, harness, record)

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

        if await maybe_apply_terminal_tool_outcome(graph_state, deps, record, chat_mode=True):
            return LoopRoute.FINALIZE

    _update_progress_tracking(harness, graph_state)
    _record_chat_progress_outcome(harness, graph_state.last_tool_results)
    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP
