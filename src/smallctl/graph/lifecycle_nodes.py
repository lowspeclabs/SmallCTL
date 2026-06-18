from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..guards import check_guards
from ..interrupt_replies import interrupt_response_action
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..phases import PHASES, filter_phase_blocked_tools, is_phase_contract_active, phase_contract, normalize_phase
from ..plans import write_plan_file
from ..state import ExecutionPlan, PlanStep, json_safe_value
from ..normalization import coerce_int as _coerce_int_value
from ..runtime_error_repair import maybe_record_reported_runtime_error
from ..task_targets import primary_task_target_path
from ..tools.dispatcher import normalize_tool_request
from ..tools.fs_loop_guard import clear_loop_guard_outline_requirement
from ..tools.planning import _refresh_plan_playbook_artifact
from ..write_session_fsm import new_write_session, record_write_session_event
from ..tools.fs import infer_write_session_intent, new_write_session_id
from ..graph.tool_write_session_policy import _ensure_chunk_write_session
from . import node_support as _nodes
from .deps import GraphRuntimeDeps
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from .autocontinue import drain_durable_autocontinue
from .chat_progress import _chat_progress_guard_failure
from .recovery_context import build_goal_recap
from .progress_guard import _check_completion_confabulation, _check_progress_stagnation
from ..harness.task_transactions import recovery_context_lines, transaction_from_scratchpad
from .write_session_outcomes import (
    maybe_finalize_stranded_write_session,
    maybe_replay_stranded_write_session_record,
)
from .lifecycle_guard_recovery import (
    _dispatch_artifact_read_recovery,
    _dispatch_stagnation_recovery,
    _inject_artifact_read_recovery_nudge,
)
from .lifecycle_tool_validation import _validate_pending_tool_calls
from .lifecycle_nodes_support import (
    _apply_continue_task_state_reset,
    _apply_small_model_remote_constraints,
    _handle_cancel_requested,
    _initialize_chat_mode_scratchpad,
    _resolve_followup_task,
)
from .tool_call_parser import (
    _artifact_read_recovery_hint,
    _artifact_read_synthesis_hint,
    _clear_artifact_read_guard_state,
    _clear_tool_attempt_history,
    _detect_hallucinated_tool_call,
    _detect_repeated_tool_loop,
    _extract_artifact_id_from_args,
    _fallback_repeated_artifact_read,
    _fallback_repeated_file_read,
    _recover_declared_read_before_write,
    _record_tool_attempt,
    _salvage_active_write_session_append,
    _should_suppress_resolved_plan_artifact_read,
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
    if await _handle_cancel_requested(graph_state, deps):
        return
    _initialize_chat_mode_scratchpad(harness, graph_state.run_mode)
    resolved_task, is_continue_task = _resolve_followup_task(harness, task)
    if is_continue_task:
        _apply_continue_task_state_reset(harness, task=task, resolved_task=resolved_task)
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
    harness.state.pending_interrupt = None
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

    _apply_small_model_remote_constraints(harness, resolved_task)

    harness.state.append_message(ConversationMessage(role="user", content=task))
    maybe_record_reported_runtime_error(harness.state, task)
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
    get_pending = getattr(harness, "get_pending_interrupt", None)
    if callable(get_pending):
        pending = get_pending() or {}
    else:
        pending = harness.state.pending_interrupt or {}
    if not isinstance(pending, dict) or not pending:
        graph_state.final_result = harness._failure(
            "No pending interrupt to resume.",
            error_type="interrupt",
        )
        graph_state.error = graph_state.final_result["error"]
        return
    created_at = pending.get("created_at")
    if isinstance(created_at, (int, float)):
        elapsed = time.time() - created_at
        timeout = getattr(getattr(harness, "config", None), "needs_human_timeout_sec", 600)
        if elapsed > timeout:
            postmortem = f"Task timed out after {int(elapsed)}s in paused state awaiting user input."
            graph_state.final_result = harness._failure(
                postmortem,
                error_type="guard",
                details={
                    "interrupt_kind": pending.get("kind", "ask_human"),
                    "elapsed_sec": elapsed,
                    "timeout_sec": timeout,
                },
            )
            graph_state.error = graph_state.final_result["error"]
            harness.state.pending_interrupt = None
            graph_state.pending_interrupt = None
            graph_state.interrupt_payload = None
            harness._runlog(
                "interrupt_resume_timeout",
                "auto-failed task because interrupt exceeded needs_human_timeout",
                elapsed_sec=elapsed,
                timeout_sec=timeout,
                interrupt_kind=pending.get("kind", "ask_human"),
            )
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
    validation_resume_hint = ""
    if str(pending.get("kind") or "").strip() == "validation_execution_request":
        validation_command = str(pending.get("command") or "").strip()
        validation_reason = str(pending.get("reason") or "").strip()
        if validation_command:
            validation_resume_hint = (
                "VALIDATION EXECUTION APPROVED: Run the requested validation command now using "
                f"`shell_exec`: `{validation_command}`. "
                "After it finishes, use the stdout/stderr/exit code as phase-validation evidence before proceeding. "
                "Do not call a tool named `run`; the local command execution tool is `shell_exec`."
            )
            if validation_reason:
                validation_resume_hint += f" Validation reason: {validation_reason}."
    apt_validator_resume_hint = ""
    apt_validator_pending: PendingToolCall | None = None
    if str(pending.get("kind") or "").strip() == "apt_deb822_validator_approval":
        action = interrupt_response_action(pending, human_input)
        if action == "approve":
            tool_name = str(pending.get("tool_name") or "ssh_exec").strip() or "ssh_exec"
            arguments = pending.get("arguments")
            if isinstance(arguments, dict) and arguments:
                apt_validator_pending = PendingToolCall(
                    tool_name=tool_name,
                    args=dict(arguments),
                    source="system",
                )
                apt_validator_resume_hint = (
                    "APT deb822 validation approved. Run the exact validator command now before retrying apt. "
                    "Preserve the current remote host/user targeting."
                )
        elif action == "reject":
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=(
                        "APT remains blocked until the deb822 validator runs. Ask for a different recovery step or "
                        "provide new instructions."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "apt_deb822_validator_rejected",
                    },
                )
            )
            harness._runlog(
                "apt_deb822_validator_rejected",
                "deb822 validator approval rejected; keeping apt blocked visible",
                tool_name=str(pending.get("tool_name") or "ssh_exec"),
                host=str(pending.get("host") or ""),
                user=str(pending.get("user") or ""),
            )
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
        # Inject staged file content into the resume hint so the model has
        # the legal context to append the next section without violating
        # the read-before-write gate.
        session = getattr(harness.state, "write_session", None)
        staged_content = ""
        next_section = ""
        if session is not None:
            from pathlib import Path
            staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
            if staging_path:
                try:
                    staged_content = Path(staging_path).read_text(encoding="utf-8")
                except Exception:
                    staged_content = ""
            next_section = str(getattr(session, "write_next_section", "") or "").strip()
        staged_hint = ""
        if staged_content:
            staged_hint = (
                f"\n\nCurrent staged content for `{outline_resume_path}`:\n"
                f"```python\n{staged_content[:2000]}\n```"
            )
        next_section_hint = ""
        if next_section:
            next_section_hint = f" Next section to write: `{next_section}`."
        outline_resume_hint = (
            f"LoopGuard outline confirmed for `{outline_resume_path}`. "
            "Resume the active write session and advance to the next unwritten section. "
            "Do not restart the file from memory."
            f"{next_section_hint}{staged_hint}"
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
    maybe_record_reported_runtime_error(harness.state, human_input)
    if apt_validator_pending is not None:
        graph_state.pending_tool_calls = [apt_validator_pending]
        if apt_validator_resume_hint:
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=apt_validator_resume_hint,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "apt_deb822_validator_approved",
                        "tool_name": apt_validator_pending.tool_name,
                        "arguments": json_safe_value(apt_validator_pending.args),
                    },
                )
            )
        harness._runlog(
            "apt_deb822_validator_approved",
            "scheduled approved deb822 validator after interrupt resume",
            tool_name=apt_validator_pending.tool_name,
            arguments=json_safe_value(apt_validator_pending.args),
        )
    elif interrupted_pending is not None:
        _record_tool_attempt(harness, interrupted_pending)
        goal_recap = build_goal_recap(harness)
        state = getattr(harness, "state", None)
        scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
        transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
        tx_lines = recovery_context_lines(transaction)
        tx_note = (" " + " ".join(tx_lines)) if tx_lines else ""
        suppressed_tool = str(scratchpad.get("_repeated_tool_loop_suppressed_tool") or "").strip()
        suppression_note = ""
        if suppressed_tool and suppressed_tool == interrupted_pending.tool_name:
            suppression_note = (
                f"IMPORTANT: `{suppressed_tool}` has been REMOVED from your available tools. "
                "You physically cannot call it again. Use a different tool or make a state-changing action. "
            )
        resume_hint = (
            f"RESUME CONTRACT: You are resuming after a loop guard pause. "
            f"YOUR TASK (do not abandon): {goal_recap or 'Continue the current task'}. "
            f"Do not call `{interrupted_pending.tool_name}` again with the same arguments. "
            f"{suppression_note}"
            "Do not read files already in context. "
            "Do NOT switch tasks, projects, or goals. "
            f"{tx_note} "
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
    elif validation_resume_hint:
        harness.state.planning_mode_enabled = False
        harness.state.current_phase = "execute"
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=validation_resume_hint,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "validation_execution_request_resume",
                    "command": str(pending.get("command") or "").strip(),
                },
            )
        )
        harness._runlog(
            "validation_execution_request_resume",
            "approved validation execution handoff to loop runtime",
            command=str(pending.get("command") or "").strip(),
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
    get_pending = getattr(harness, "get_pending_interrupt", None)
    if callable(get_pending):
        pending = get_pending() or {}
    else:
        pending = harness.state.pending_interrupt or {}
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

    if interrupt_response_action(pending, human_input) == "approve":
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
            _nodes._persist_planning_playbook(harness, plan)
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

    drain_durable_autocontinue(graph_state, harness)
    deferred_schema_repairs = harness.state.scratchpad.pop("_deferred_schema_validation_repair_messages", None)
    if isinstance(deferred_schema_repairs, list):
        for item in deferred_schema_repairs:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            role = str(item.get("role") or "system").strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                role = "system"
            harness.state.append_message(
                ConversationMessage(
                    role=role,
                    content=content,
                    metadata=metadata,
                )
            )
            harness._runlog(
                "tool_call_repair_deferred_delivered",
                "delivered deferred schema repair nudge",
                tool_name=metadata.get("tool_name"),
                tool_call_id=metadata.get("tool_call_id"),
                required_fields=metadata.get("required_fields", []),
            )

    # Safety-net: directly finalize any session that is still stranded.
    if graph_state.final_result is None:
        await maybe_finalize_stranded_write_session(harness, graph_state)

    harness.state.step_count += 1

    from .lifecycle_step_budget import STEP_BUDGET_NUDGE_THRESHOLD, _maybe_inject_step_budget_nudge

    # Hard step-budget safety net for small models: if we've burned past the
    # threshold without convergence, force a synthesize-and-exit directive.
    if harness.state.step_count > STEP_BUDGET_NUDGE_THRESHOLD:
        _maybe_inject_step_budget_nudge(harness, graph_state)

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
            # Fix 2 (RCA 8b79ca76): Don't pre-arm a write session if the target
            # file already exists and is non-empty. The file was likely written
            # by a prior completed session; creating a new orphan session traps
            # the model in a task_complete deadlock.
            from pathlib import Path
            from ..tools.fs import _resolve
            try:
                resolved = _resolve(target_path, getattr(harness.state, "cwd", None))
                file_already_exists = resolved.exists() and resolved.is_file() and resolved.stat().st_size > 0
            except Exception:
                file_already_exists = False
            if not file_already_exists:
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
        if (
            "stagnation limit" in guard_error
            or "loop detected" in guard_error
            or "repeated tool call loop" in guard_error
            or "Progress stagnation guard tripped" in guard_error
        ):
            _dispatch_stagnation_recovery(harness, guard_error)
            guard_error = None

        if guard_error:
            recovery_hint = _artifact_read_recovery_hint(harness, guard_error)
            if recovery_hint is not None and graph_state.run_mode != "chat":
                _dispatch_artifact_read_recovery(harness, graph_state, recovery_hint)
                guard_error = None
        elif recovery_hint is not None and graph_state.run_mode == "chat":
            _inject_artifact_read_recovery_nudge(harness, recovery_hint)
            guard_error = None

    if graph_state.pending_tool_calls and not guard_error:
        should_return = await _validate_pending_tool_calls(harness, graph_state, deps)
        if should_return:
            return

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
