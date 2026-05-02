from __future__ import annotations

import inspect
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..plans import write_plan_file
from .deps import GraphRuntimeDeps
from .interrupts import pause_for_plan_approval
from .routing import LoopRoute
from .state import GraphRunState, ToolExecutionRecord
from . import write_session_outcomes as _write_session_outcomes

_is_plan_export_validation_error = _write_session_outcomes._is_plan_export_validation_error
_build_plan_export_recovery_message = _write_session_outcomes._build_plan_export_recovery_message
_auto_update_active_plan_step = _write_session_outcomes._auto_update_active_plan_step


async def _emit_ui_event(harness: Any, event_handler: Any, event: UIEvent) -> None:
    emit = getattr(harness, "_emit", None)
    if not callable(emit):
        return
    maybe_awaitable = emit(event_handler, event)
    if inspect.isawaitable(maybe_awaitable):
        await maybe_awaitable


async def apply_planning_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    has_explicit_plan_request = any(
        record.tool_name == "plan_request_execution" and record.result.success
        for record in graph_state.last_tool_results
    )
    for record in graph_state.last_tool_results:
        _write_session_outcomes._maybe_schedule_write_recovery_readback(graph_state, harness, record)

        if not record.result.success and _is_plan_export_validation_error(record.result.error):
            repair_attempts = int(harness.state.scratchpad.get("_plan_export_recovery_nudges", 0))
            if repair_attempts < 1:
                repair_message = _build_plan_export_recovery_message(record)
                harness.state.scratchpad["_plan_export_recovery_nudges"] = repair_attempts + 1
                harness.state.recent_errors.append(str(record.result.error or repair_message))
                harness.state.append_message(
                    ConversationMessage(
                        role="user",
                        content=repair_message,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "plan_export_validation",
                            "tool_name": record.tool_name,
                            "tool_call_id": record.tool_call_id,
                        },
                    )
                )
                harness._runlog(
                    "plan_export_repair",
                    "injected plan export repair nudge",
                    tool_name=record.tool_name,
                    tool_call_id=record.tool_call_id,
                    retry_count=repair_attempts + 1,
                    error=str(record.result.error or ""),
                )
                await _emit_ui_event(
                    harness,
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=repair_message,
                        data={
                            "repair_kind": "plan_export_validation",
                            "tool_name": record.tool_name,
                            "tool_call_id": record.tool_call_id,
                            "retry_count": repair_attempts + 1,
                        },
                    ),
                )
                graph_state.last_tool_results = []
                graph_state.last_assistant_text = ""
                graph_state.last_thinking_text = ""
                return LoopRoute.NEXT_STEP

        if record.tool_name == "plan_set" and record.result.success:
            plan = harness.state.draft_plan or harness.state.active_plan
            if plan is not None:
                plan.status = "draft"
                plan.touch()
                harness.state.draft_plan = plan
                harness.state.sync_plan_mirror()
                await _emit_ui_event(
                    harness,
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content="Draft plan created.",
                        data={"status_activity": "draft plan created"},
                    ),
                )
                export_warning = str(record.result.metadata.get("export_warning", "") or "").strip()
                if export_warning:
                    await _emit_ui_event(
                        harness,
                        deps.event_handler,
                        UIEvent(
                            event_type=UIEventType.ALERT,
                            content=f"Draft plan created; skipped invalid export hint: {export_warning}",
                            data={
                                "status_activity": "draft plan created",
                                "warning_type": "plan_export_validation",
                                "rejected_output_path": record.result.metadata.get("rejected_output_path", ""),
                                "suggested_output_path": record.result.metadata.get("suggested_output_path", ""),
                            },
                        ),
                    )
                if plan.requested_output_path:
                    try:
                        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                    except ValueError as exc:
                        harness.log.warning("skipping invalid plan export after plan_set: %s", exc)
                if not has_explicit_plan_request:
                    await pause_for_plan_approval(graph_state, deps)
                    return LoopRoute.FINALIZE
            continue
        if record.tool_name == "plan_step_update" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                plan.touch()
                harness.state.sync_plan_mirror()
                active_step = plan.find_step(str(record.args.get("step_id", "")).strip())
                step_label = active_step.step_id if active_step is not None else str(record.args.get("step_id", "")).strip()
                await _emit_ui_event(
                    harness,
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=f"Plan step updated: {step_label}",
                        data={"status_activity": f"step {step_label} updated"},
                    ),
                )
                if plan.requested_output_path:
                    try:
                        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                    except ValueError as exc:
                        harness.log.warning("skipping invalid plan export after step update: %s", exc)
            continue
        if record.tool_name == "plan_export" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                await _emit_ui_event(
                    harness,
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content="Plan file exported.",
                        data={"status_activity": "plan file exported"},
                    ),
                )
            continue
        if record.tool_name == "plan_request_execution" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            payload = {
                "kind": "plan_execute_approval",
                "question": record.result.metadata.get("question", "Plan ready. Execute it now?"),
                "plan_id": plan.plan_id if plan is not None else "",
                "approved": False,
                "response_mode": "yes/no/revise",
                "current_phase": harness.state.current_phase,
                "thread_id": graph_state.thread_id,
            }
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await _emit_ui_event(
                harness,
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=payload["question"],
                    data={"status_activity": "awaiting plan approval...", "interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": payload["question"],
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE
        if record.tool_name == "task_complete" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None and plan.approved:
                return LoopRoute.FINALIZE

    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP
