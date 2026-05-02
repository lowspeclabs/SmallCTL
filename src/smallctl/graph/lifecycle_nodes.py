from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..guards import check_guards
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..phases import PHASES, filter_phase_blocked_tools, is_phase_contract_active, phase_contract, normalize_phase
from ..state import ExecutionPlan, PlanStep, json_safe_value
from ..normalization import coerce_int as _coerce_int_value
from ..task_targets import primary_task_target_path
from ..tools.dispatcher import normalize_tool_request
from ..tools.fs_loop_guard import clear_loop_guard_outline_requirement
from ..tools.planning import _refresh_plan_playbook_artifact
from ..write_session_fsm import new_write_session, record_write_session_event
from ..tools.fs import infer_write_session_intent, new_write_session_id
from . import node_support as _nodes
from .deps import GraphRuntimeDeps
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from .chat_progress import _chat_progress_guard_failure
from .recovery_context import build_goal_recap
from .progress_guard import _check_completion_confabulation, _check_progress_stagnation
from .write_session_outcomes import (
    maybe_finalize_stranded_write_session,
    maybe_replay_stranded_write_session_record,
)
from .tool_call_parser import (
    _artifact_read_recovery_hint,
    _artifact_read_synthesis_hint,
    _build_schema_repair_message,
    _clear_artifact_read_guard_state,
    _clear_tool_attempt_history,
    _detect_empty_file_write_payload,
    _detect_hallucinated_tool_call,
    _detect_missing_required_tool_arguments,
    _detect_oversize_write_payload,
    _detect_patch_existing_stage_read_contract_violation,
    _detect_placeholder_tool_call,
    _detect_repeated_tool_loop,
    _ensure_chunk_write_session,
    _extract_artifact_id_from_args,
    _fallback_repeated_artifact_read,
    _fallback_repeated_file_read,
    _recover_declared_read_before_write,
    _record_tool_attempt,
    _repair_active_write_session_args,
    _salvage_active_write_session_append,
    _should_enter_chunk_mode,
    _should_suppress_resolved_plan_artifact_read,
    _suggested_chunk_sections,
)
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
from .tool_outcomes import (
    _register_write_session_stage_artifact,
    apply_chat_tool_outcomes,
    apply_planning_tool_outcomes,
    apply_tool_outcomes,
)
from .write_recovery import (
    build_synthetic_write_args,
    can_safely_synthesize,
    recover_write_intent,
    write_recovery_kind,
    write_recovery_metadata,
)
from .lifecycle_prompt import (
    _available_tool_names,
    load_index_manifest,
    prepare_chat_prompt,
    prepare_indexer_prompt,
    prepare_planning_prompt,
    prepare_prompt,
    prepare_staged_prompt,
    select_chat_tools,
    select_indexer_tools,
    select_loop_tools,
    select_planning_tools,
    select_staged_tools,
)


async def initialize_loop_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    task: str,
) -> None:
    harness = deps.harness
    if harness._cancel_requested:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
        )
        graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
        return
    harness.state.pending_interrupt = None
    if graph_state.run_mode == "chat":
        harness.state.scratchpad["_chat_rounds"] = 0
        harness.state.scratchpad.pop("_chat_progress_guard", None)
    else:
        harness.state.scratchpad.pop("_chat_rounds", None)
        harness.state.scratchpad.pop("_chat_progress_guard", None)
    resolved_task = task
    resolve_followup = getattr(harness, "_resolve_followup_task", None)
    if callable(resolve_followup):
        candidate = str(resolve_followup(task) or "").strip()
        if candidate:
            resolved_task = candidate
    is_continue_check = getattr(harness, "_is_continue_like_followup", None)
    is_continue_task = callable(is_continue_check) and is_continue_check(task)
    if is_continue_task:
        # "continue" means continue the current task — preserve all state
        # (recent_messages, working_memory, tool results) so the model sees
        # prior tool call context and can decide what to do next.
        harness._runlog(
            "task_continue",
            "continuing current task, skipping state reset",
            raw_task=task,
            resolved_task=resolved_task[:80] if resolved_task else "",
            old_step_count=harness.state.step_count,
        )
        harness.state.step_count = 0
        harness.state.inactive_steps = 0
        # Prevent stale stagnation counters from tripping the guard
        # immediately on the first step of a continued task.
        harness.state.stagnation_counters.pop("no_actionable_progress", None)
        harness.state.scratchpad.pop("_progress_read_history", None)
        harness.state.scratchpad.pop("_progress_ssh_observation_history", None)
        harness.state.scratchpad.pop("_progress_prior_verdict", None)
        harness.state.scratchpad.pop("_progress_prior_plan_step", None)
        harness.state.scratchpad.pop("_ssh_auth_recovery_state", None)
    else:
        maybe_reset = getattr(harness, "_maybe_reset_for_new_task", None)
        if callable(maybe_reset):
            maybe_reset(resolved_task, raw_task=task)
    begin_task_scope = getattr(harness, "_begin_task_scope", None)
    task_scope: dict[str, object] = {}
    if callable(begin_task_scope):
        started_scope = begin_task_scope(raw_task=task, effective_task=resolved_task)
        if isinstance(started_scope, dict):
            task_scope = started_scope
    harness._runlog(
        "task_start",
        "task received",
        task=task,
        effective_task=resolved_task if resolved_task != task else "",
        task_id=str(task_scope.get("task_id") or ""),
        task_summary_path=str(task_scope.get("summary_path") or ""),
    )
    await harness._ensure_context_limit()
    harness._initialize_run_brief(resolved_task, raw_task=task)
    harness._activate_tool_profiles(resolved_task)
    harness.state.append_message(ConversationMessage(role="user", content=task))
    harness._log_conversation_state("user_message")
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.USER, content=task),
    )

    # Bug patch: recover stranded write sessions whose final chunk was never
    # processed because last_tool_results was dropped across a run restart.
    maybe_replay_stranded_write_session_record(harness, graph_state)


async def resume_loop_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    human_input: str,
) -> None:
    harness = deps.harness
    pending = harness.state.pending_interrupt
    if not isinstance(pending, dict) or not pending:
        graph_state.final_result = harness._failure(
            "No pending interrupt to resume.",
            error_type="interrupt",
        )
        graph_state.error = graph_state.final_result["error"]
        return
    harness._runlog(
        "interrupt_resume",
        "resuming loop from interrupt",
        thread_id=graph_state.thread_id,
        interrupt=pending,
    )
    continue_like = human_input.strip().lower() in ("continue", "keep going", "proceed")
    is_continue_like = getattr(harness, "_is_continue_like_followup", None)
    if callable(is_continue_like):
        continue_like = bool(is_continue_like(human_input))
    interrupted_pending: PendingToolCall | None = None
    interrupted_guidance = ""
    outline_resume_hint = ""
    outline_resume_path = ""
    if continue_like and str(pending.get("kind") or "").strip() == "repeated_tool_loop_resume":
        interrupted_tool_name = str(pending.get("tool_name") or "").strip()
        interrupted_args = pending.get("arguments")
        if interrupted_tool_name and isinstance(interrupted_args, dict):
            interrupted_pending = PendingToolCall(
                tool_name=interrupted_tool_name,
                args=dict(interrupted_args),
            )
            interrupted_guidance = str(pending.get("guidance") or "").strip()
    if continue_like and str(pending.get("kind") or "").strip() == "chunked_write_loop_guard_outline":
        outline_resume_path = str(pending.get("path") or "").strip()
        clear_loop_guard_outline_requirement(
            harness.state,
            path=outline_resume_path or None,
            write_session_id=str(pending.get("write_session_id") or "").strip() or None,
            cwd=harness.state.cwd,
        )
        outline_resume_hint = (
            f"LoopGuard outline confirmed for `{outline_resume_path}`. "
            "Resume the active write session and advance to the next unwritten section. "
            "Do not restart the file from memory."
        )
    if continue_like:
        harness._runlog("step_count_reset", "resetting step count for continuation", old_count=harness.state.step_count)
        harness.state.step_count = 0
        harness.state.inactive_steps = 0
        harness.state.stagnation_counters.pop("no_actionable_progress", None)
        harness.state.scratchpad.pop("_progress_read_history", None)
        harness.state.scratchpad.pop("_progress_ssh_observation_history", None)
        harness.state.scratchpad.pop("_progress_prior_verdict", None)
        harness.state.scratchpad.pop("_progress_prior_plan_step", None)
        harness.state.scratchpad.pop("_ssh_auth_recovery_state", None)

    _clear_tool_attempt_history(harness)
    if hasattr(harness.state, "tool_history") and isinstance(harness.state.tool_history, list):
        harness.state.tool_history.clear()

    harness.state.pending_interrupt = None
    graph_state.pending_interrupt = None
    graph_state.interrupt_payload = None
    graph_state.pending_tool_calls = []

    for key in (
        "_ask_human",
        "_ask_human_question",
        "_file_read_recovery_nudged",
        "_shell_human_retry_nudged",
        "_artifact_read_recovery_nudged",
        "_artifact_read_synthesis_nudged",
        "_artifact_summary_exit_nudged",
        "_schema_validation_nudges",
        "_consecutive_idle",
        "_confabulation_nudged",
    ):
        harness.state.scratchpad.pop(key, None)
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=human_input,
            metadata={
                "resumed_from_interrupt": True,
                "interrupt_kind": pending.get("kind", "ask_human"),
            },
        )
    )
    if interrupted_pending is not None:
        _record_tool_attempt(harness, interrupted_pending)
        goal_recap = build_goal_recap(harness)
        resume_hint = (
            f"RESUME CONTRACT: You are resuming after a loop guard pause. "
            f"YOUR TASK (do not abandon): {goal_recap or 'Continue the current task'}. "
            f"Do not call `{interrupted_pending.tool_name}` again with the same arguments. "
            "Do not read files already in context. "
            "Do NOT switch tasks, projects, or goals. "
            "Choose a concrete next action: write, patch, run a command, or finish."
        )
        if interrupted_guidance:
            resume_hint = f"{resume_hint} {interrupted_guidance}"
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=resume_hint,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "repeated_tool_loop_resume",
                    "guard": "repeated_tool_loop",
                    "tool_name": interrupted_pending.tool_name,
                    "arguments": json_safe_value(interrupted_pending.args),
                },
            )
        )
        harness._runlog(
            "repeated_tool_loop_resume_guard",
            "reseeded interrupted tool guard on continue resume",
            tool_name=interrupted_pending.tool_name,
            arguments=json_safe_value(interrupted_pending.args),
        )
    elif outline_resume_hint:
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=outline_resume_hint,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "chunked_write_loop_guard_outline_resume",
                    "path": outline_resume_path,
                },
            )
        )
        harness._runlog(
            "chunked_write_loop_guard_outline_resume",
            "cleared loop guard outline mode after human confirmation",
            path=outline_resume_path,
        )
    harness._log_conversation_state("resume_user_message")
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.USER, content=human_input),
    )


async def initialize_planning_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    task: str,
) -> None:
    harness = deps.harness
    await initialize_loop_run(graph_state, deps, task=task)
    harness.state.planning_mode_enabled = True
    harness.state.planner_resume_target_mode = "loop"
    harness.state.run_brief.current_phase_objective = f"planning: {task}" if task else "planning"
    if not harness.state.working_memory.current_goal:
        harness.state.working_memory.current_goal = task or harness.state.run_brief.original_task
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content="Planning mode enabled.",
            data={"status_activity": "planning mode active"},
        ),
    )


async def resume_planning_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    human_input: str,
) -> None:
    harness = deps.harness
    pending = harness.state.pending_interrupt
    if not isinstance(pending, dict) or not pending:
        graph_state.final_result = harness._failure(
            "No pending planning interrupt to resume.",
            error_type="interrupt",
        )
        graph_state.error = graph_state.final_result["error"]
        return

    harness.state.pending_interrupt = None
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=human_input,
            metadata={
                "resumed_from_interrupt": True,
                "interrupt_kind": pending.get("kind", "plan_execute_approval"),
            },
        )
    )
    harness._log_conversation_state("resume_planning_message")
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.USER, content=human_input),
    )

    lowered = human_input.strip().lower()
    if lowered in {"yes", "y", "approve", "approved", "execute", "go ahead", "run it"}:
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is not None:
            plan.approved = True
            plan.status = "approved"
            plan.touch()
            harness.state.active_plan = plan
            harness.state.draft_plan = plan
            harness.state.planning_mode_enabled = False
            harness.state.current_phase = "execute"
            harness.state.planner_resume_target_mode = "loop"
            harness.state.sync_plan_mirror()
            _persist_planning_playbook(harness, plan)
            harness.state.touch()
            if plan.requested_output_path:
                try:
                    write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                except ValueError as exc:
                    harness.log.warning("skipping invalid plan export during approval: %s", exc)
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Plan approved.",
                data={"status_activity": "plan approved"},
            ),
        )
        graph_state.final_result = {
            "status": "plan_approved",
            "message": "Plan approved.",
            "approved": True,
            "plan": json_safe_value(harness.state.active_plan or harness.state.draft_plan),
        }
        return

    harness.state.planning_mode_enabled = True
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content="Plan revision requested.",
            data={"status_activity": "awaiting plan revision..."},
        ),
    )


async def prepare_loop_step(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> None:
    harness = deps.harness
    start_time = time.perf_counter()
    if harness._cancel_requested:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
        )
        graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
        return
    if graph_state.run_mode == "chat":
        chat_rounds = _coerce_int_value(harness.state.scratchpad.get("_chat_rounds")) + 1
        harness.state.scratchpad["_chat_rounds"] = chat_rounds
        progress_guard = _chat_progress_guard_failure(harness)
        if progress_guard is not None:
            graph_state.final_result = harness._failure(
                progress_guard["message"],
                error_type="guard",
                details=progress_guard["details"],
            )
            graph_state.error = graph_state.final_result["error"]
            return

    # Safety-net: directly finalize any session that is still stranded.
    if graph_state.final_result is None:
        await maybe_finalize_stranded_write_session(harness, graph_state)

    harness.state.step_count += 1
    harness.state.decay_experiences()
    harness.dispatcher.phase = normalize_phase(harness.state.contract_phase())
    prior_phase = harness.state.current_phase
    harness.state.current_phase = harness.dispatcher.phase

    # Refresh stale phase prefix in current_phase_objective after contract_phase
    # stabilizes the real phase (e.g. resetting from explore back to execute).
    if prior_phase != harness.state.current_phase:
        phase_objective = str(harness.state.run_brief.current_phase_objective or "").strip()
        if phase_objective:
            for phase in PHASES:
                prefix = f"{phase}: "
                if phase_objective.startswith(prefix) and phase != harness.state.current_phase:
                    harness.state.run_brief.current_phase_objective = (
                        f"{harness.state.current_phase}: {phase_objective[len(prefix):]}"
                    )
                    break

    if not graph_state.pending_tool_calls and getattr(harness.state, "write_session", None) is None:
        target_path = primary_task_target_path(harness)
        if target_path:
            _ensure_chunk_write_session(harness, target_path)

    if graph_state.pending_tool_calls:
        suppressed_plan_reads: list[str] = []
        remaining_calls: list[PendingToolCall] = []
        for pending in graph_state.pending_tool_calls:
            if _should_suppress_resolved_plan_artifact_read(harness, pending):
                artifact_id = _extract_artifact_id_from_args(pending.args)
                if artifact_id:
                    suppressed_plan_reads.append(artifact_id)
                continue
            remaining_calls.append(pending)
        if suppressed_plan_reads:
            graph_state.pending_tool_calls = remaining_calls
            for artifact_id in suppressed_plan_reads:
                if harness.state.scratchpad.get("_plan_artifact_read_suppressed") == artifact_id:
                    continue
                harness.state.scratchpad["_plan_artifact_read_suppressed"] = artifact_id
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=(
                            f"Plan artifact {artifact_id} is already loaded in Working Memory. "
                            "Reuse the mirrored plan summary instead of rereading it."
                        ),
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "artifact_read",
                            "artifact_id": artifact_id,
                            "recovery_mode": "plan_mirror",
                        },
                    )
                )
                harness._runlog(
                    "artifact_read_suppressed",
                    "suppressed repeated plan artifact read",
                    artifact_id=artifact_id,
                )

    _check_completion_confabulation(harness, graph_state)
    progress_guard = _check_progress_stagnation(harness, graph_state)
    if progress_guard:
        guard_error = progress_guard
    else:
        guard_error = check_guards(harness.state, harness.guards)
    recovery_hint: tuple[str, str] | None = None
    if guard_error:
        if "stagnation limit" in guard_error or "loop detected" in guard_error or "repeated tool call loop" in guard_error:
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=(
                        f"System: {guard_error}. "
                        "You are stuck in a loop. Try a different tool, check permissions, or rethink your approach instead of repeating the same action."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "stagnation",
                        "guard_error": guard_error,
                    },
                )
            )
            harness._runlog(
                "stagnation_recovery",
                "injected strategy pivot nudge",
                step=harness.state.step_count,
            )
            if "no_progress" in harness.state.stagnation_counters:
                harness.state.stagnation_counters["no_progress"] = 0
            if "repeat_command" in harness.state.stagnation_counters:
                harness.state.stagnation_counters["repeat_command"] = 0
            if "repeat_patch" in harness.state.stagnation_counters:
                harness.state.stagnation_counters["repeat_patch"] = 0
            harness.state.tool_history.clear()
            guard_error = None

        if guard_error:
            recovery_hint = _artifact_read_recovery_hint(harness, guard_error)
            if recovery_hint is not None and graph_state.run_mode != "chat":
                artifact_id, query = recovery_hint
                _clear_artifact_read_guard_state(harness, artifact_id)
                graph_state.pending_tool_calls = [
                    PendingToolCall(
                        tool_name="artifact_grep",
                        args={
                            "artifact_id": artifact_id,
                            "query": query,
                        },
                        source="system",
                    )
                ]
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=(
                            f"Auto-advancing repeated `artifact_read` on artifact {artifact_id} "
                            f"to `artifact_grep` with query `{query}`."
                        ),
                        metadata={
                            "recovery_kind": "artifact_read",
                            "artifact_id": artifact_id,
                            "query": query,
                            "recovery_mode": "direct_dispatch",
                        },
                    )
                )
                harness._runlog(
                    "artifact_read_recovery",
                    "scheduled recovery dispatch",
                    step=harness.state.step_count,
                    artifact_id=artifact_id,
                    query=query,
                    guard_error=guard_error,
                )
                guard_error = None
        elif recovery_hint is not None and graph_state.run_mode == "chat":
            recovery_armed = harness.state.scratchpad.get("_artifact_read_recovery_nudged")
            if recovery_armed != recovery_hint[0]:
                artifact_id, query = recovery_hint
                msg = (
                    f"You are repeating `artifact_read` on artifact {artifact_id}. "
                    f"Use `artifact_grep` with query `{query}` instead of reading the same artifact again."
                )
                harness.state.scratchpad["_artifact_read_recovery_nudged"] = artifact_id
                harness.state.scratchpad["_artifact_read_recovery_query"] = query
                harness.state.append_message(
                    ConversationMessage(
                        role="user",
                        content=msg,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "artifact_read",
                            "artifact_id": artifact_id,
                            "query": query,
                        },
                    )
                )
                harness._runlog(
                    "artifact_read_recovery",
                    "injected recovery nudge",
                    step=harness.state.step_count,
                    artifact_id=artifact_id,
                    query=query,
                    guard_error=guard_error,
                )
                guard_error = None

    if graph_state.pending_tool_calls and not guard_error:
        for pending in graph_state.pending_tool_calls:
            if _should_enter_chunk_mode(harness, pending):
                target_path = str(pending.args.get("path") or "")
                content = str(pending.args.get("content") or "")

                suggestions = _get_suggested_sections(target_path)
                session_id = new_write_session_id()

                harness.state.write_session = new_write_session(
                    session_id=session_id,
                    target_path=target_path,
                    intent=infer_write_session_intent(
                        target_path,
                        getattr(harness.state, "cwd", None),
                    ),
                    mode="chunked_author",
                    suggested_sections=suggestions,
                    next_section=suggestions[0] if suggestions else "",
                )
                _register_write_session_stage_artifact(harness, harness.state.write_session)
                record_write_session_event(
                    harness.state,
                    event="session_opened",
                    session=harness.state.write_session,
                    details={"source": "chunk_mode_trigger", "size": len(content)},
                )

                msg = (
                    f"Writing `{target_path}` requires chunked authoring for this model/task ({len(content)} chars in the attempted write). "
                    f"I have initialized a Write Session `{session_id}`. "
                    "Please break this file into logical sections (e.g. imports, classes, functions) and write them one by one. "
                    f"Use `file_write` with `write_session_id='{session_id}'`, `section_name='...'`, and `next_section_name='...'`. "
                    "When finished, call `file_write` for the last chunk without `next_section_name` to finalize."
                )
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=msg,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "chunk_mode_trigger",
                            "session_id": session_id,
                            "target_path": target_path,
                        },
                    )
                )
                harness._runlog(
                    "chunk_mode_triggered",
                    "intercepted large write, suggesting chunk mode",
                    session_id=session_id,
                    target_path=target_path,
                    size=len(content),
                )
                graph_state.pending_tool_calls = []
                break

            oversize = _detect_oversize_write_payload(harness, pending)
            if oversize:
                err_msg, details = oversize
                harness.state.recent_errors.append(err_msg)
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=err_msg,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "oversize_write",
                            **details,
                        },
                    )
                )
                harness._runlog(
                    "oversize_write_intercepted",
                    "rejected one-shot write exceeding hard threshold",
                    **details,
                )
                graph_state.pending_tool_calls = []
                break

    if guard_error:
        harness.state.recent_errors.append(guard_error)
        log_kv(
            harness.log,
            logging.WARNING,
            "harness_guard_triggered",
            step=harness.state.step_count,
            guard_error=guard_error,
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=guard_error),
        )
        graph_state.final_result = harness._failure(guard_error, error_type="guard")
        graph_state.error = graph_state.final_result["error"]

    graph_state.latency_metrics["overhead_preparation_duration_sec"] = round(time.perf_counter() - start_time, 3)
