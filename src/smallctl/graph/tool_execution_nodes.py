from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ..harness.tool_visibility import hidden_tool_reason, resolve_turn_tool_exposure
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..state import json_safe_value
from ..tools.dispatcher import normalize_tool_request
from ..tools.planning import _refresh_plan_playbook_artifact
from ..tools.fs import infer_write_session_intent, new_write_session_id
from ..tools.fs_loop_guard import outline_mode_violation
from ..write_session_fsm import new_write_session, record_write_session_event
from .display import format_tool_result_display
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from . import nodes as _nodes
from .tool_call_parser import (
    _artifact_read_synthesis_hint,
    _detect_hallucinated_tool_call,
    _detect_patch_existing_stage_read_contract_violation,
    _detect_repeated_tool_loop,
    _extract_artifact_id_from_args,
    _record_tool_attempt,
)
from .tool_execution_recovery import handle_repeated_tool_loop
from .tool_execution_persistence import persist_tool_results
from .shell_outcomes import (
    _shell_human_retry_hint,
    _shell_ssh_retry_hint,
    _shell_workspace_relative_retry_hint,
)
from .tool_execution_support import (
    _conversation_message_from_dict,
    _get_tool_execution_record,
    _has_matching_tool_message,
    _store_tool_execution_record,
    _tool_envelope_from_dict,
)


_TOOL_NOT_EXPOSED_REASON_LABELS = {
    "missing_index": "missing index",
    "no_artifacts": "no artifacts yet",
    "no_active_plan": "no active plan",
    "no_background_jobs": "no background jobs",
    "write_session_not_finalizable": "write session not finalizable",
}


async def dispatch_tools(graph_state: GraphRunState, deps: Any) -> None:
    harness = deps.harness
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
            if intercepted_metadata.get("reason") == "remote_task_requires_ssh_exec":
                message = str(intercepted_result.error or "This is a remote task. Use `ssh_exec`, not local `shell_exec`.")
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
                    "remote_shell_exec_guard_nudge",
                    "blocked local shell_exec for remote SSH task",
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
            _nodes._apply_declared_read_before_write_reroute(
                graph_state,
                harness,
                pending,
                assistant_text=dispatch_assistant_text,
            )
            outline_violation = outline_mode_violation(
                harness.state,
                tool_name=pending.tool_name,
                args=pending.args,
            )
            if outline_violation is not None:
                message = str(outline_violation.get("message") or "").strip()
                harness.state.recent_errors.append(message)
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=message,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "chunked_write_loop_guard_outline",
                            "target_path": outline_violation.get("path"),
                            "write_session_id": outline_violation.get("write_session_id"),
                            "tool_name": pending.tool_name,
                            "tool_call_id": pending.tool_call_id,
                        },
                    )
                )
                harness._runlog(
                    "chunked_write_loop_guard_outline_violation",
                    "blocked tool while loop guard outline mode was active",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    target_path=outline_violation.get("path"),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=message,
                        data={
                            "repair_kind": "chunked_write_loop_guard_outline",
                            "tool_name": pending.tool_name,
                            "tool_call_id": pending.tool_call_id,
                            "target_path": outline_violation.get("path"),
                        },
                    ),
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
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

        hallucination_hint = _detect_hallucinated_tool_call(harness, pending)
        if hallucination_hint:
            log_kv(harness.log, logging.WARNING, "harness_hallucinated_tool_call", tool_name=pending.tool_name)
            await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.SYSTEM, content=hallucination_hint))
            fake_result = ToolEnvelope(
                success=False,
                error=hallucination_hint,
                metadata={"hallucinated_tool": pending.tool_name}
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
                hidden_reason_text = _TOOL_NOT_EXPOSED_REASON_LABELS.get(
                    str(hidden_reason or "").strip(),
                )
                error_message = f"Tool `{pending.tool_name}` is registered but unavailable on this turn."
                if hidden_reason_text:
                    error_message = f"{error_message} Reason: {hidden_reason_text}."
                result = ToolEnvelope(
                    success=False,
                    error=error_message,
                    metadata={
                        "reason": "tool_not_exposed_this_turn",
                        "tool_name": pending.tool_name,
                        "run_mode": graph_state.run_mode,
                        "allowed_tools": allowed_tools,
                        "hidden_reason": hidden_reason,
                    },
                )
                harness._runlog(
                    "tool_blocked_not_exposed",
                    "blocked model tool call outside the current turn exposure",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    run_mode=graph_state.run_mode,
                    allowed_tools=allowed_tools,
                    hidden_reason=hidden_reason,
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

        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.SYSTEM, content=f"Invoking {pending.tool_name}..."),
        )
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
                        result = ToolEnvelope(
                            success=False,
                            error="Tool dispatcher is unavailable in the current harness context.",
                            metadata={"tool_name": pending.tool_name, "reason": "dispatcher_unavailable"},
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
                    harness._active_dispatch_task = asyncio.create_task(
                        dispatch_fn(pending.tool_name, pending.args)
                    )
                    result = await harness._active_dispatch_task
            except _nodes.ToolNotFoundError:
                if pending.tool_name in _nodes.HALLUCINATION_MAP:
                    mapped_tool = _nodes.HALLUCINATION_MAP[pending.tool_name]
                    raw_id = (
                        pending.args.get("path") or
                        pending.args.get("artifact_id") or
                        pending.args.get("pattern") or
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
                    result = ToolEnvelope(
                        success=False,
                        error=f"Unknown tool: {pending.tool_name}",
                        metadata={"tool_name": pending.tool_name}
                    )
            except asyncio.CancelledError:
                await harness._emit(
                    deps.event_handler,
                    UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
                )
                graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
                return
            finally:
                harness._active_dispatch_task = None
            _store_tool_execution_record(
                harness,
                operation_id=operation_id,
                thread_id=graph_state.thread_id,
                step_count=harness.state.step_count,
                pending=pending,
                result=result,
            )
        log_kv(
            harness.log,
            logging.INFO,
            "harness_tool_result",
            tool_name=pending.tool_name,
            success=result.success,
            replayed=replayed,
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.TOOL_RESULT,
                content=json.dumps(json_safe_value(result.to_dict()), ensure_ascii=True),
                data={
                    "tool_name": pending.tool_name,
                    "tool_call_id": pending.tool_call_id,
                    "success": result.success,
                    "replayed": replayed,
                        "display_text": format_tool_result_display(
                            tool_name=pending.tool_name,
                            result=result,
                            request_text=harness.state.run_brief.original_task,
                    ),
                },
            ),
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
    dispatch_end = time.perf_counter()
    duration = dispatch_end - dispatch_start
    graph_state.latency_metrics["tool_execution_duration_sec"] = round(duration, 3)

    if duration > 0.05:
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.METRICS,
                content=f"Tool execution: {duration:.2f}s",
                data={
                    "duration_sec": duration,
                }
            ),
        )
