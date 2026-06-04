from __future__ import annotations

import re
from pathlib import Path
from time import time
from typing import Any

from ..recovery_metrics import record_failure_event_metric
from ..recovery_schema import FailureEvent
from . import task_completion_outcomes as _task_completion_outcomes
from . import write_session_outcomes as _write_session_outcomes
from .deps import GraphRuntimeDeps
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord
from .chat_progress import _record_chat_progress_outcome, build_file_read_recovery_message
from .planning_outcomes import apply_planning_tool_outcomes
from .shell_outcomes import (
    _shell_human_retry_hint,
    _shell_ssh_retry_hint,
    _shell_workspace_relative_retry_hint,
)
from .tool_execution_recovery import handle_failed_file_write_outcome
from .escalation_triggers import (
    _maybe_auto_trigger_escalation_for_patch_stall,
    _maybe_auto_trigger_escalation_for_completion_block,
    _maybe_auto_trigger_escalation_for_verifier_stall,
    _maybe_auto_trigger_escalation_for_apt_sources_failure,
    _maybe_confirm_apt_sources_tip,
)
from ..harness.tool_visibility import schedule_retry_tool_exposure
from ..models.conversation import ConversationMessage
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


_REMOTE_TASK_HOST_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_REMOTE_TASK_USER_HOST_RE = re.compile(
    r"\b[A-Za-z0-9._-]+@(?:[A-Za-z0-9.-]+|\d{1,3}(?:\.\d{1,3}){3})\b",
    re.IGNORECASE,
)


async def apply_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        _maybe_clear_missing_input_after_remote_readback(harness, record)
        _maybe_clear_missing_input_after_local_alias_read(harness, record)
        _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record)
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

        _maybe_confirm_apt_sources_tip(harness.state, record)

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
        _maybe_auto_complete_plan_step_for_mutation(harness, record)

    await _maybe_auto_trigger_escalation_for_patch_stall(
        harness=harness,
        graph_state=graph_state,
    )
    await _maybe_auto_trigger_escalation_for_completion_block(
        harness=harness,
        graph_state=graph_state,
    )
    await _maybe_auto_trigger_escalation_for_verifier_stall(
        harness=harness,
        graph_state=graph_state,
    )
    await _maybe_auto_trigger_escalation_for_apt_sources_failure(
        harness=harness,
        graph_state=graph_state,
    )
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
        _maybe_clear_missing_input_after_remote_readback(harness, record)
        _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record)
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
        _maybe_auto_complete_plan_step_for_mutation(harness, record)

        if await maybe_apply_terminal_tool_outcome(graph_state, deps, record, chat_mode=True):
            return LoopRoute.FINALIZE

    _update_progress_tracking(harness, graph_state)
    _record_chat_progress_outcome(harness, graph_state.last_tool_results)
    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


def _maybe_emit_missing_requested_output_file_nudge(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "file_read" or record.result.success:
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    raw_path = str(
        record.args.get("path")
        or metadata.get("requested_path")
        or metadata.get("path")
        or ""
    ).strip()
    if not raw_path:
        return False
    error_text = str(record.result.error or record.result.output or "").lower()
    read_result = str(metadata.get("read_result") or "").strip().lower()
    if read_result != "missing" and "does not exist" not in error_text:
        return False

    message = build_file_read_recovery_message(
        harness,
        PendingToolCall(tool_name="file_read", args={"path": raw_path}),
    )
    scratchpad = getattr(harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        harness.state.scratchpad = scratchpad

    if _is_remote_execution_state(harness.state) and "requested output file" in message:
        nudge_key = f"missing-remote-output-readback:{raw_path}"
        if scratchpad.get("_missing_remote_output_readback_nudged") == nudge_key:
            return False
        scratchpad["_missing_remote_output_readback_nudged"] = nudge_key
        retry_scheduled = schedule_retry_tool_exposure(
            harness.state,
            mode=graph_state.run_mode,
            tool_name="ssh_file_read",
            arguments={"path": raw_path},
        )
        remote_message = (
            f"`file_read(path='{raw_path}')` checks the local workspace, but this is a remote SSH task. "
            f"If `{raw_path}` was written on the remote host, verify it with "
            f"`ssh_file_read(path='{raw_path}')` or read the remote command output you already captured. "
            "Do not call `task_complete` until the remote readback or equivalent remote evidence succeeds."
        )
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=remote_message,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "missing_remote_requested_output_file",
                    "recovery_mode": "read_remote_requested_output",
                    "path": raw_path,
                    "tool_name": "file_read",
                    "retry_tool_name": "ssh_file_read",
                    "retry_scheduled": retry_scheduled,
                },
            )
        )
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "missing_remote_requested_output_file_nudge",
                "nudged model to verify requested remote output with ssh_file_read after local file_read missed",
                path=raw_path,
                run_mode=graph_state.run_mode,
                retry_scheduled=retry_scheduled,
            )
        return True

    if "requested output file" not in message:
        if "required input file" not in message:
            return False
        nudge_key = f"missing-input-file:{raw_path}"
        if scratchpad.get("_missing_input_file_nudged") == nudge_key:
            return False
        scratchpad["_missing_input_file_nudged"] = nudge_key
        scratchpad["_unresolved_missing_input_file"] = {
            "path": raw_path,
            "message": message,
            "operation_id": str(record.operation_id or ""),
            "tool_call_id": str(record.tool_call_id or ""),
        }
        metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
        suggested_path = str(metadata.get("suggested_path") or "").strip()
        suggestion_confidence = str(metadata.get("suggestion_confidence") or "").strip().lower()
        if suggested_path and suggestion_confidence == "high":
            scratchpad["_unresolved_missing_input_file"]["suggested_path"] = suggested_path
            scratchpad["_unresolved_missing_input_file"]["suggestion_confidence"] = suggestion_confidence
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=message,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "missing_required_input_file",
                    "path": raw_path,
                },
            )
        )
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "missing_required_input_file_nudge",
                "blocked completion after required input file was missing",
                path=raw_path,
                run_mode=graph_state.run_mode,
            )
        return True

    nudge_key = f"missing-output-file:{raw_path}"
    if scratchpad.get("_missing_output_file_write_nudged") == nudge_key:
        return False
    scratchpad["_missing_output_file_write_nudged"] = nudge_key

    retry_scheduled = schedule_retry_tool_exposure(
        harness.state,
        mode=graph_state.run_mode,
        tool_name="file_write",
        arguments={"path": raw_path},
    )
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "missing_requested_output_file",
                "recovery_mode": "write_requested_output",
                "path": raw_path,
                "tool_name": "file_read",
                "retry_tool_name": "file_write",
                "retry_scheduled": retry_scheduled,
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "missing_requested_output_file_nudge",
            "nudged model to write requested output file after missing file_read",
            path=raw_path,
            run_mode=graph_state.run_mode,
            retry_scheduled=retry_scheduled,
        )
    return True


def _maybe_clear_missing_input_after_local_alias_read(
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "file_read" or not record.result.success:
        return False
    state = getattr(harness, "state", None)
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    blocker = scratchpad.get("_unresolved_missing_input_file")
    if not isinstance(blocker, dict):
        return False
    blocked_path = str(blocker.get("path") or "").strip()
    suggested_path = str(blocker.get("suggested_path") or "").strip()
    if not blocked_path or not suggested_path:
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    read_paths = [
        str(record.args.get("path") or "").strip(),
        str(metadata.get("requested_path") or "").strip(),
        str(metadata.get("path") or "").strip(),
    ]
    if not any(_paths_likely_same_output(suggested_path, path) for path in read_paths if path):
        return False
    alias = {
        "resolved_path": suggested_path,
        "reason": "near_match_file_read_succeeded",
        "operation_id": str(record.operation_id or ""),
        "tool_call_id": str(record.tool_call_id or ""),
    }
    scratchpad.setdefault("_required_input_aliases", {})[blocked_path] = alias
    blocker["resolved_by_alias"] = alias
    scratchpad["_resolved_missing_input_file"] = blocker
    scratchpad.pop("_unresolved_missing_input_file", None)
    scratchpad.pop("_missing_input_file_nudged", None)
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "missing_required_input_file_blocker_cleared_by_local_alias_read",
            "cleared missing-input blocker after reading high-confidence suggested local path",
            blocked_path=blocked_path,
            suggested_path=suggested_path,
        )
    return True


def _maybe_clear_missing_input_after_remote_readback(
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "ssh_file_read" or not record.result.success:
        return False
    state = getattr(harness, "state", None)
    if state is None or not _is_remote_task_state(state):
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    blocker = scratchpad.get("_unresolved_missing_input_file")
    if not isinstance(blocker, dict):
        return False
    blocked_path = str(blocker.get("path") or "").strip()
    if not blocked_path:
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    read_path = str(metadata.get("path") or record.args.get("path") or "").strip()
    if not read_path:
        return False
    if not _paths_likely_same_output(blocked_path, read_path):
        return False
    scratchpad.pop("_unresolved_missing_input_file", None)
    scratchpad.pop("_missing_input_file_nudged", None)
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "missing_required_input_file_blocker_cleared_by_ssh_file_read",
            "cleared stale local missing-input blocker after successful remote readback",
            blocked_path=blocked_path,
            read_path=read_path,
        )
    return True


def _is_remote_task_state(state: Any) -> bool:
    if _is_remote_execution_state(state):
        return True
    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)
    text = " ".join(
        part
        for part in (
            str(getattr(run_brief, "original_task", "") or ""),
            str(getattr(run_brief, "effective_task", "") or ""),
            str(getattr(working_memory, "current_goal", "") or ""),
        )
        if part
    ).lower()
    if not text:
        return False
    if "ssh" in text or "remote host" in text or "remote server" in text:
        return True
    if _REMOTE_TASK_USER_HOST_RE.search(text) is not None:
        return True
    return _REMOTE_TASK_HOST_RE.search(text) is not None and any(
        marker in text for marker in ("host", "server", "remote")
    )


def _is_remote_execution_state(state: Any) -> bool:
    mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    if mode == "remote_execute":
        return True
    intent = str(getattr(state, "active_intent", "") or "").strip().lower()
    if intent == "requested_ssh_exec":
        return True
    return False


def _paths_likely_same_output(left: str, right: str) -> bool:
    left_norm = _normalized_path_for_match(left)
    right_norm = _normalized_path_for_match(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm.endswith(f"/{right_norm}") or right_norm.endswith(f"/{left_norm}"):
        return True
    return Path(left_norm).name == Path(right_norm).name


def _normalized_path_for_match(path: str) -> str:
    raw = str(path or "").strip().strip("'\"")
    if not raw:
        return ""
    if raw.startswith("home/"):
        raw = f"/{raw}"
    raw = raw.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return re.sub(r"/+", "/", raw).rstrip("/").lower()


def _maybe_auto_complete_plan_step_for_mutation(harness: Any, record: ToolExecutionRecord) -> None:
    if not record.result.success:
        return
    if record.tool_name not in {"file_patch", "ast_patch", "file_write", "file_append"}:
        return
    if not getattr(harness.state, "plan_execution_mode", False):
        return
    active_step_id = str(getattr(harness.state, "active_step_id", "") or "").strip()
    if not active_step_id:
        return
    plan = getattr(harness.state, "active_plan", None) or getattr(harness.state, "draft_plan", None)
    if plan is None:
        return
    step = plan.find_step(active_step_id)
    if step is None:
        return
    target_path = str(
        record.args.get("path")
        or record.args.get("target_path")
        or (record.result.metadata.get("path") if isinstance(record.result.metadata, dict) else "")
        or ""
    ).strip()
    if not target_path:
        return
    step_text = f"{step.title} {step.description} {step.task}".lower()
    normalized_path = target_path.replace("\\", "/").lower()
    path_parts = normalized_path.split("/")
    filename = path_parts[-1] if path_parts else normalized_path
    if filename not in step_text and normalized_path not in step_text:
        return
    _auto_update_active_plan_step(
        harness,
        status="completed",
        note=f"Step satisfied by successful {record.tool_name} on `{target_path}`.",
    )


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
