from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ..harness.task_intent import promote_active_intent_for_tool_call
from ..harness.tool_visibility import (
    consume_retry_tool_exposure,
    hidden_tool_reason,
    recent_hidden_tool_recovery_artifact_id,
    resolve_turn_tool_exposure,
    schedule_retry_tool_exposure,
)
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..state import json_safe_value
from ..state_support import safe_scratchpad
from ..tools.dispatcher import normalize_tool_request
from ..tools.planning import _refresh_plan_playbook_artifact  # noqa: F401
from ..tools.fs import infer_write_session_intent, new_write_session_id  # noqa: F401
from ..write_session_fsm import new_write_session, record_write_session_event  # noqa: F401
from ..client.chunk_parser import format_tool_call_text
from .display import format_tool_result_display
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from . import nodes as _nodes
from .autocontinue import clear_durable_autocontinue_for_pending
from .lifecycle_tool_validation import (
    _apply_tool_call_schema_repair,
    _record_tool_call_schema_repair,
    _tool_call_repair_enabled,
    _tool_call_repair_log_only,
)
from .tool_call_parser import (
    _artifact_read_synthesis_hint,  # noqa: F401
    _detect_timeout_recovered_incomplete_tool_call,
    _detect_hallucinated_tool_call,
    _detect_patch_existing_stage_read_contract_violation,
    _detect_repeated_tool_loop,
    _extract_artifact_id_from_args,  # noqa: F401
    _record_tool_attempt,
    _undo_tool_attempt_if_cached,
)
from .tool_execution_recovery import handle_repeated_tool_loop
from .escalation_triggers import _maybe_auto_trigger_escalation_for_same_tool_failures
from .tool_execution_persistence import persist_tool_results  # noqa: F401
from ..fama.runtime import _handle_signal
from ..fama.signals import FamaFailureKind, FamaSignal, current_step
from .shell_outcomes import (
    _shell_human_retry_hint,  # noqa: F401
    _shell_ssh_retry_hint,  # noqa: F401
    _shell_workspace_relative_retry_hint,  # noqa: F401
)
from .tool_execution_support import (
    _conversation_message_from_dict,  # noqa: F401
    _get_tool_execution_record,
    _has_matching_tool_message,  # noqa: F401
    _store_tool_execution_record,
    _tool_envelope_from_dict,
)
from .hidden_tool_helpers import _validation_handoff_hint_for_blocked_tool
from .tool_execution_node_guards import (
    _block_long_running_remote_timeout_write,
    _block_outline_mode_violation,
)


_TOOL_NOT_EXPOSED_REASON_LABELS = {
    "missing_index": "missing index",
    "no_artifacts": "no artifacts yet",
    "no_active_plan": "no active plan",
    "no_background_jobs": "no background jobs",
    "write_session_not_finalizable": "write session not finalizable",
}


async def _emit_tool_result_event(
    harness: Any,
    event_handler: Any,
    pending: PendingToolCall,
    result: ToolEnvelope,
    *,
    replayed: bool = False,
) -> None:
    """Emit a single TOOL_RESULT UI event with normalized display metadata."""
    await harness._emit(
        event_handler,
        UIEvent(
            event_type=UIEventType.TOOL_RESULT,
            content=json.dumps(json_safe_value(result.to_dict()), ensure_ascii=True),
            data={
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
                "success": result.success,
                "replayed": replayed,
                "output": json_safe_value(result.output),
                "error": result.error,
                "display_text": format_tool_result_display(
                    tool_name=pending.tool_name,
                    result=result,
                    request_text=harness.state.run_brief.original_task,
                ),
            },
        ),
    )

async def _block_empty_file_write_before_dispatch(
    graph_state: GraphRunState,
    deps: Any,
    pending: PendingToolCall,
) -> bool:
    """Turn content-less recovered writes into repair nudges before dispatch."""
    if pending.tool_name not in {"file_write", "file_append"}:
        return False

    harness = deps.harness
    missing_args = _nodes._detect_empty_file_write_payload(harness, pending)
    if missing_args is None:
        return False

    err_msg, details = missing_args
    target_path = str(
        details.get("target_path")
        or pending.args.get("path")
        or ""
    ).strip() or None
    repair_decision = _nodes.schema_validation_repair_decision(
        harness,
        pending,
        err_msg,
        details,
        target_path=target_path,
    )
    harness.state.recent_errors.append(err_msg)
    harness._runlog(
        "tool_call_validation_pre_dispatch_block",
        "blocked content-less file_write before tool dispatch",
        **repair_decision.runlog_data,
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content=repair_decision.repair_message,
            data=repair_decision.alert_data,
        ),
    )
    if repair_decision.conversation_message is not None:
        harness.state.append_message(repair_decision.conversation_message)
    else:
        graph_state.final_result = harness._failure(
            err_msg,
            error_type="schema_validation_error",
            details=repair_decision.details,
        )
        graph_state.error = graph_state.final_result["error"]
    graph_state.pending_tool_calls = []
    graph_state.last_tool_results = []
    return True


async def dispatch_tools(graph_state: GraphRunState, deps: Any) -> None:
    harness = deps.harness
    graph_state.recorded_tool_call_ids = []
    graph_state.last_tool_results = []
    dispatch_start = time.perf_counter()
    dispatch_assistant_text = str(
        graph_state.last_assistant_text
        or harness.state.scratchpad.get("_last_text_write_fallback_assistant_text")
        or ""
    )
    if _nodes._apply_small_model_authoring_budget(harness, graph_state):
        graph_state.last_tool_results = []
        graph_state.pending_tool_calls = graph_state.pending_tool_calls[:1]
    for pending in graph_state.pending_tool_calls:
        pending_source = str(getattr(pending, "source", "model") or "model").strip().lower()
        if pending_source != "model" and _tool_call_repair_enabled(harness):
            repair_result = _apply_tool_call_schema_repair(harness, pending)
            if (
                repair_result is not None
                and repair_result.repaired
                and not _tool_call_repair_log_only(harness)
            ):
                _record_tool_call_schema_repair(harness, pending, repair_result)
        registry = getattr(harness, "registry", None)
        if registry is not None:
            normalized_tool_name, normalized_args, intercepted_result, normalization_metadata = normalize_tool_request(
                registry,
                pending.tool_name,
                pending.args,
                phase=getattr(getattr(harness, "dispatcher", None), "phase", None),
                state=harness.state,
            )
        else:
            normalized_tool_name, normalized_args, intercepted_result, normalization_metadata = (
                pending.tool_name,
                pending.args,
                None,
                {},
            )
        if intercepted_result is not None:
            intercepted_metadata = intercepted_result.metadata if isinstance(intercepted_result.metadata, dict) else {}
            if intercepted_metadata.get("reason") in {
                "remote_task_requires_ssh_exec",
                "remote_path_requires_ssh_exec",
                "remote_path_requires_typed_ssh_file_tool",
            }:
                message = str(
                    intercepted_result.error
                    or "This task should continue over SSH. Use `ssh_exec`, not a local tool."
                )
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=message,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "ssh_exec",
                            "recovery_mode": "remote_task_guard",
                            "tool_name": normalized_tool_name,
                            "tool_call_id": pending.tool_call_id,
                            **normalization_metadata,
                        },
                    )
                )
                harness._runlog(
                    "remote_tool_guard_nudge",
                    "blocked local tool for remote SSH task",
                    step=harness.state.step_count,
                    tool_name=pending.tool_name,
                    arguments=json_safe_value(pending.args),
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                return
        if intercepted_result is None:
            pending.tool_name = normalized_tool_name
            pending.args = normalized_args
            if await _block_long_running_remote_timeout_write(graph_state=graph_state, deps=deps, pending=pending):
                return
            _nodes._apply_declared_read_before_write_reroute(
                graph_state,
                harness,
                pending,
                assistant_text=dispatch_assistant_text,
            )
            if await _block_empty_file_write_before_dispatch(graph_state, deps, pending):
                return
            if await _block_outline_mode_violation(graph_state=graph_state, deps=deps, pending=pending):
                return
            contract_violation = _detect_patch_existing_stage_read_contract_violation(harness, pending)
            if contract_violation is not None:
                err_msg, details = contract_violation
                harness.state.recent_errors.append(err_msg)
                harness.state.append_message(
                    ConversationMessage(
                        role="user",
                        content=err_msg,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "schema_validation",
                            "tool_name": pending.tool_name,
                            "tool_call_id": pending.tool_call_id,
                            "target_path": details.get("target_path"),
                        },
                    )
                )
                harness._runlog(
                    "patch_existing_stage_read_contract_violation",
                    "blocked ambiguous same-target write after staged read recovery",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    target_path=details.get("target_path"),
                    session_id=details.get("write_session_id"),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=err_msg,
                        data={
                            "repair_kind": "schema_validation",
                            "tool_name": pending.tool_name,
                            "tool_call_id": pending.tool_call_id,
                            "target_path": details.get("target_path"),
                            "session_id": details.get("write_session_id"),
                        },
                    ),
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                return

        repeat_error = _detect_repeated_tool_loop(harness, pending)
        _record_tool_attempt(harness, pending)

        if repeat_error is not None:
            _handle_signal(
                harness,
                state=harness.state,
                config=getattr(harness, "config", None),
                signal=FamaSignal(
                    kind=FamaFailureKind.LOOPING,
                    severity=2,
                    source="loop_guard",
                    evidence=repeat_error,
                    step=current_step(harness.state),
                    tool_name=pending.tool_name,
                    failure_class="repeated_action",
                    next_safe_action="Stop retrying the identical tool call. Use a different tool or ask for missing information.",
                ),
                dedupe=True,
            )
            maybe_pending = await handle_repeated_tool_loop(
                harness=harness,
                graph_state=graph_state,
                deps=deps,
                pending=pending,
                repeat_error=repeat_error,
            )
            if maybe_pending is None:
                return
            pending = maybe_pending

        timeout_recovery = _detect_timeout_recovered_incomplete_tool_call(harness, pending)
        if timeout_recovery is not None:
            recovery_message, recovery_details = timeout_recovery
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=recovery_message,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "incomplete_tool_call_timeout",
                        "tool_name": pending.tool_name,
                        "tool_call_id": pending.tool_call_id,
                        "required_fields": recovery_details.get("required_fields", []),
                        "present_fields": recovery_details.get("present_fields", []),
                        "missing_required_fields": recovery_details.get("missing_required_fields", []),
                    },
                )
            )
            harness._runlog(
                "tool_call_timeout_recovery",
                "suppressed fake tool failure after timeout-truncated tool call",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                provider_profile=recovery_details.get("provider_profile"),
                present_fields=recovery_details.get("present_fields", []),
                missing_required_fields=recovery_details.get("missing_required_fields", []),
                arguments=recovery_details.get("arguments", {}),
                raw_arguments_preview=recovery_details.get("raw_arguments_preview", ""),
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=recovery_message,
                    data={
                        "repair_kind": "incomplete_tool_call_timeout",
                        "tool_name": pending.tool_name,
                        "tool_call_id": pending.tool_call_id,
                        "provider_profile": recovery_details.get("provider_profile"),
                        "present_fields": recovery_details.get("present_fields", []),
                        "missing_required_fields": recovery_details.get("missing_required_fields", []),
                    },
                ),
            )
            continue

        hallucination_hint = _detect_hallucinated_tool_call(harness, pending)
        if hallucination_hint:
            log_kv(harness.log, logging.WARNING, "harness_hallucinated_tool_call", tool_name=pending.tool_name)
            await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.SYSTEM, content=hallucination_hint))
            fake_result = ToolEnvelope(
                success=False,
                error=hallucination_hint,
                metadata={"hallucinated_tool": pending.tool_name, "hallucination": True, "suppress_failure_persistence": True}
            )
            graph_state.last_tool_results.append(
                ToolExecutionRecord(
                    operation_id=f"hallucination:{pending.tool_name}",
                    tool_name=pending.tool_name,
                    args=pending.args,
                    tool_call_id=pending.tool_call_id,
                    result=fake_result,
                )
            )
            continue

        pending_source = str(getattr(pending, "source", "model") or "model").strip().lower()
        registry = getattr(harness, "registry", None)
        if pending_source == "model" and registry is not None:
            tool_exposure = resolve_turn_tool_exposure(harness, graph_state.run_mode)
            allowed_tools = [
                str(name).strip()
                for name in tool_exposure.get("names", [])
                if str(name).strip()
            ]
            if pending.tool_name not in set(allowed_tools):
                hidden_reason = hidden_tool_reason(
                    pending.tool_name,
                    state=harness.state,
                    mode=graph_state.run_mode,
                )
                scratchpad = safe_scratchpad(harness.state) or {}
                is_smalltalk_terminal_only = (
                    graph_state.run_mode == "chat"
                    and str(scratchpad.get("_chat_tools_suppressed_reason") or "").strip()
                    == "smalltalk_terminal_only"
                )
                retry_scheduled = False
                if hidden_reason is None and not is_smalltalk_terminal_only:
                    retry_scheduled = schedule_retry_tool_exposure(
                        harness.state,
                        mode=graph_state.run_mode,
                        tool_name=pending.tool_name,
                        arguments=pending.args,
                    )
                    if retry_scheduled:
                        pending_args = pending.args if isinstance(pending.args, dict) else {}
                        path = str(pending_args.get("path") or pending_args.get("target_path") or "").strip()
                        retry_message = (
                            f"Registered but unavailable on this turn: `{pending.tool_name}`. "
                            f"Retry on the next turn with `{pending.tool_name}` immediately. "
                            "Do not restart the analysis or re-read the same evidence unless the tool arguments truly need new context."
                        )
                        if path:
                            retry_message = f"{retry_message} Target path: `{path}`."
                        recovery_artifact_id = recent_hidden_tool_recovery_artifact_id(
                            harness.state,
                            tool_name=pending.tool_name,
                        )
                        if recovery_artifact_id:
                            retry_message = (
                                f"{retry_message} Use `artifact_read(artifact_id='{recovery_artifact_id}')` "
                                "for the full fetched body."
                            )
                        harness.state.append_message(
                            ConversationMessage(
                                role="system",
                                content=retry_message,
                                metadata={
                                    "is_recovery_nudge": True,
                                    "recovery_kind": "tool_not_exposed_this_turn",
                                    "retry_tool_name": pending.tool_name,
                                    "tool_name": pending.tool_name,
                                    "tool_call_id": pending.tool_call_id,
                                    "run_mode": graph_state.run_mode,
                                },
                            )
                        )
                hidden_reason_text = _TOOL_NOT_EXPOSED_REASON_LABELS.get(
                    str(hidden_reason or "").strip(),
                )
                names_fn = getattr(registry, "names", None) if registry is not None else None
                try:
                    is_registered = callable(names_fn) and pending.tool_name in names_fn()
                except Exception:
                    is_registered = False
                if is_registered:
                    error_message = f"Tool `{pending.tool_name}` is registered but unavailable on this turn."
                else:
                    error_message = f"Tool `{pending.tool_name}` is not available."
                if hidden_reason_text:
                    error_message = f"{error_message} Reason: {hidden_reason_text}."
                validation_hint = _validation_handoff_hint_for_blocked_tool(
                    pending,
                    mode=graph_state.run_mode,
                )
                if validation_hint:
                    error_message = f"{error_message} {validation_hint}"
                recovery_artifact_id = recent_hidden_tool_recovery_artifact_id(
                    harness.state,
                    tool_name=pending.tool_name,
                )
                if recovery_artifact_id:
                    error_message = (
                        f"{error_message} Use artifact_read(artifact_id='{recovery_artifact_id}') "
                        "for the full fetched body."
                    )
                result = ToolEnvelope.make_error(
                    pending.tool_name,
                    error_message,
                    reason="tool_not_exposed_this_turn",
                    run_mode=graph_state.run_mode,
                    allowed_tools=allowed_tools,
                    hidden_reason=hidden_reason,
                    retry_scheduled=retry_scheduled,
                    recovery_artifact_id=recovery_artifact_id,
                )
                harness._runlog(
                    "tool_blocked_not_exposed",
                    "blocked model tool call outside the current turn exposure",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    run_mode=graph_state.run_mode,
                    allowed_tools=allowed_tools,
                    hidden_reason=hidden_reason,
                    retry_scheduled=retry_scheduled,
                )
                operation_id = build_operation_id(
                    thread_id=graph_state.thread_id,
                    step_count=harness.state.step_count,
                    tool_call_id=pending.tool_call_id,
                    tool_name=pending.tool_name,
                )

                _store_tool_execution_record(
                    harness,
                    operation_id=operation_id,
                    thread_id=graph_state.thread_id,
                    step_count=harness.state.step_count,
                    pending=pending,
                    result=result,
                )
                graph_state.last_tool_results.append(
                    ToolExecutionRecord(
                        operation_id=operation_id,
                        tool_name=pending.tool_name,
                        args=pending.args,
                        tool_call_id=pending.tool_call_id,
                        result=result,
                    )
                )
                continue
            consume_retry_tool_exposure(
                harness.state,
                mode=graph_state.run_mode,
                tool_name=pending.tool_name,
            )

        if promote_active_intent_for_tool_call(harness.state, pending.tool_name):
            harness._runlog(
                "active_intent_promoted_by_tool",
                "promoted active intent from accepted tool call",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                active_intent=getattr(harness.state, "active_intent", ""),
                intent_tags=list(getattr(harness.state, "intent_tags", []) or []),
            )

        safe_args = json_safe_value(pending.args)
        tool_call_display = format_tool_call_text(
            pending.tool_name,
            json.dumps(safe_args, ensure_ascii=True, sort_keys=True),
            safe_args,
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.TOOL_CALL,
                content=pending.tool_name,
                data=_nodes._planner_speaker_data(
                    graph_state,
                    {
                        "tool_name": pending.tool_name,
                        "tool_call_id": pending.tool_call_id,
                        "args": safe_args,
                        "display_text": tool_call_display,
                    },
                ),
            ),
        )
        clear_durable_autocontinue_for_pending(harness, pending)
        operation_id = build_operation_id(
            thread_id=graph_state.thread_id,
            step_count=harness.state.step_count,
            tool_call_id=pending.tool_call_id,
            tool_name=pending.tool_name,
        )
        existing = _get_tool_execution_record(harness, operation_id)
        replayed = isinstance(existing.get("result"), dict)
        log_kv(
            harness.log,
            logging.INFO,
            "harness_tool_dispatch",
            tool_name=pending.tool_name,
            replayed=replayed,
        )
        if replayed:
            harness._runlog(
                "tool_replay_hit",
                "reusing recorded tool result",
                tool_name=pending.tool_name,
                operation_id=operation_id,
            )
            result = _tool_envelope_from_dict(existing["result"])
        else:
            try:
                registry = getattr(harness, "registry", None)
                names_fn = getattr(registry, "names", None) if registry is not None else None
                if callable(names_fn):
                    if pending.tool_name not in names_fn():
                        raise _nodes.ToolNotFoundError(pending.tool_name)

                if intercepted_result is not None:
                    result = intercepted_result
                else:
                    dispatch_fn = getattr(harness, "_dispatch_tool_call", None)
                    if not callable(dispatch_fn):
                        result = ToolEnvelope.make_error(
                            pending.tool_name,
                            "Tool dispatcher is unavailable in the current harness context.",
                            reason="dispatcher_unavailable",
                        )
                        graph_state.last_tool_results.append(
                            ToolExecutionRecord(
                                operation_id=operation_id,
                                tool_name=pending.tool_name,
                                args=pending.args,
                                tool_call_id=pending.tool_call_id,
                                result=result,
                            )
                        )
                        continue
                    harness._active_ui_tool_context = {
                        "tool_name": pending.tool_name,
                        "tool_call_id": pending.tool_call_id,
                    }
                    harness._active_dispatch_task = asyncio.create_task(dispatch_fn(pending.tool_name, pending.args))
                    result = await harness._active_dispatch_task
            except _nodes.ToolNotFoundError:
                if pending.tool_name in _nodes.HALLUCINATION_MAP:
                    mapped_tool = _nodes.HALLUCINATION_MAP[pending.tool_name]
                    pending_args = pending.args if isinstance(pending.args, dict) else {}
                    raw_id = (
                        pending_args.get("path") or
                        pending_args.get("artifact_id") or
                        pending_args.get("pattern") or
                        "A000X"
                    )
                    artifact_id = str(raw_id).split("/")[-1]
                    if not artifact_id.startswith("A") and "A" in artifact_id:
                        idx = artifact_id.find("A")
                        artifact_id = artifact_id[idx:]

                    hint = f"Tool '{pending.tool_name}' is unavailable. Use '{mapped_tool}(artifact_id=\"{artifact_id}\")' instead."
                    result = ToolEnvelope(
                        success=True,
                        output=hint,
                        metadata={"interceptor_hit": True, "hallucinated_tool": pending.tool_name}
                    )
                else:
                    result = ToolEnvelope.make_error(
                        pending.tool_name,
                        f"Unknown tool: {pending.tool_name}",
                    )
            except Exception as exc:
                elapsed_sec = max(0.0, time.perf_counter() - dispatch_start)
                error_result = ToolEnvelope.make_error(
                    pending.tool_name,
                    f"Tool dispatch failed for `{pending.tool_name}`: {exc}",
                    reason="tool_dispatch_exception",
                    tool_call_id=pending.tool_call_id,
                    args=json_safe_value(pending.args),
                    elapsed_sec=round(elapsed_sec, 3),
                    exception_type=exc.__class__.__name__,
                )
                log_kv(
                    harness.log,
                    logging.WARNING,
                    "tool_dispatch_exception",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    exception_type=exc.__class__.__name__,
                    error=str(exc),
                )
                _store_tool_execution_record(
                    harness,
                    operation_id=operation_id,
                    thread_id=graph_state.thread_id,
                    step_count=harness.state.step_count,
                    pending=pending,
                    result=error_result,
                )
                await _emit_tool_result_event(
                    harness,
                    deps.event_handler,
                    pending,
                    error_result,
                    replayed=False,
                )
                graph_state.last_tool_results.append(
                    ToolExecutionRecord(
                        operation_id=operation_id,
                        tool_name=pending.tool_name,
                        args=pending.args,
                        tool_call_id=pending.tool_call_id,
                        result=error_result,
                    )
                )
                harness.state.recent_errors.append(str(error_result.error or f"Tool dispatch failed: {pending.tool_name}"))
                continue
            except asyncio.CancelledError:
                elapsed_sec = max(0.0, time.perf_counter() - dispatch_start)
                cancelled_result = ToolEnvelope.make_error(
                    pending.tool_name,
                    (
                        f"Tool dispatch cancelled while waiting for `{pending.tool_name}` "
                        f"after {elapsed_sec:.1f}s."
                    ),
                    status="cancelled",
                    reason="tool_dispatch_cancelled",
                    tool_call_id=pending.tool_call_id,
                    args=json_safe_value(pending.args),
                    elapsed_sec=round(elapsed_sec, 3),
                    cancellation_source=str(
                        getattr(harness, "_cancel_source", "") or "cancel_requested"
                    ),
                )
                _store_tool_execution_record(
                    harness,
                    operation_id=operation_id,
                    thread_id=graph_state.thread_id,
                    step_count=harness.state.step_count,
                    pending=pending,
                    result=cancelled_result,
                )
                graph_state.last_tool_results.append(
                    ToolExecutionRecord(
                        operation_id=operation_id,
                        tool_name=pending.tool_name,
                        args=pending.args,
                        tool_call_id=pending.tool_call_id,
                        result=cancelled_result,
                    )
                )
                harness.state.recent_errors.append(str(cancelled_result.error or "Tool dispatch cancelled."))
                harness._runlog(
                    "tool_dispatch_cancelled",
                    "tool dispatch cancelled while awaiting tool result",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    elapsed_sec=round(elapsed_sec, 3),
                    arguments=json_safe_value(pending.args),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.SYSTEM,
                        content=f"Tool dispatch cancelled: {pending.tool_name}.",
                        data={
                            "ui_kind": "tool_dispatch_cancelled",
                            "event": "tool_dispatch_cancelled",
                            "tool_name": pending.tool_name,
                            "tool_call_id": pending.tool_call_id,
                            "elapsed_sec": round(elapsed_sec, 3),
                        },
                    ),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
                )
                graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
                return
            finally:
                harness._active_dispatch_task = None
                harness._active_ui_tool_context = None
            if isinstance(result, dict):
                result = _tool_envelope_from_dict(result)
            elif isinstance(result, ToolEnvelope) and not isinstance(result.metadata, dict):
                result.metadata = {}
            # Persist the artifact immediately while the result still holds the
            # full tool output. Graph state serialization between nodes compacts
            # large outputs, so delaying artifact creation until
            # persist_tool_results produces artifacts whose on-disk content
            # disagrees with their metadata (e.g. a "complete" file_read that
            # only contains a 4k preview).
            if not replayed and result.success:
                tool_results_service = getattr(harness, "tool_results", None)
                if tool_results_service is not None and callable(getattr(tool_results_service, "persist_artifact_early", None)):
                    try:
                        tool_results_service.persist_artifact_early(
                            tool_name=pending.tool_name,
                            result=result,
                            tool_call_id=pending.tool_call_id,
                        )
                    except Exception as exc:
                        log_kv(
                            harness.log,
                            logging.WARNING,
                            "early_artifact_persistence_failed",
                            tool_name=pending.tool_name,
                            tool_call_id=pending.tool_call_id,
                            error=str(exc),
                        )
            _store_tool_execution_record(
                harness,
                operation_id=operation_id,
                thread_id=graph_state.thread_id,
                step_count=harness.state.step_count,
                pending=pending,
                result=result,
            )
        _undo_tool_attempt_if_cached(harness, pending, result)
        log_kv(
            harness.log,
            logging.INFO,
            "harness_tool_result",
            tool_name=pending.tool_name,
            success=result.success,
            replayed=replayed,
        )
        await _emit_tool_result_event(
            harness,
            deps.event_handler,
            pending,
            result,
            replayed=replayed,
        )
        graph_state.last_tool_results.append(
            ToolExecutionRecord(
                operation_id=operation_id,
                tool_name=pending.tool_name,
                args=pending.args,
                tool_call_id=pending.tool_call_id,
                result=result,
                replayed=replayed,
            )
        )
        if (getattr(result, "status", None) == "needs_human" or result.metadata.get("status") == "needs_human"):
            graph_state.pending_tool_calls = []
            break

    graph_state.pending_tool_calls = []
    await _maybe_auto_trigger_escalation_for_same_tool_failures(
        harness=harness,
        graph_state=graph_state,
    )
    dispatch_end = time.perf_counter()
    duration = dispatch_end - dispatch_start
    graph_state.latency_metrics["tool_execution_duration_sec"] = round(duration, 3)

    total_execution_sec = 0.0
    total_approval_wait_sec = 0.0
    for record in graph_state.last_tool_results:
        if record.result and isinstance(record.result.metadata, dict):
            total_execution_sec += record.result.metadata.get("execution_sec", 0.0) or 0.0
            total_approval_wait_sec += record.result.metadata.get("approval_wait_sec", 0.0) or 0.0

    if total_execution_sec > 0:
        graph_state.latency_metrics["tool_actual_execution_sec"] = round(total_execution_sec, 3)
    if total_approval_wait_sec > 0:
        graph_state.latency_metrics["tool_approval_wait_sec"] = round(total_approval_wait_sec, 3)

    if duration > 0.05:
        metrics_msg = f"Tool dispatch: {duration:.2f}s"
        if total_execution_sec > 0:
            metrics_msg += f" (execution: {total_execution_sec:.2f}s"
            if total_approval_wait_sec > 0:
                metrics_msg += f", approval wait: {total_approval_wait_sec:.2f}s"
            metrics_msg += ")"
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.METRICS,
                content=metrics_msg,
                data={
                    "duration_sec": duration,
                    "execution_sec": total_execution_sec if total_execution_sec > 0 else None,
                    "approval_wait_sec": total_approval_wait_sec if total_approval_wait_sec > 0 else None,
                }
            ),
        )
