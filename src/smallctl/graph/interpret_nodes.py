from __future__ import annotations

import json
import re
from typing import Any

from ..memory.taxonomy import (
    PREMATURE_TASK_COMPLETE,
    TOOL_NOT_CALLED,
)
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..phases import filter_phase_blocked_tools, is_phase_contract_active, phase_contract
from ..state import normalize_intent_label
from ..harness.task_classifier import (
    classify_runtime_intent,
    looks_like_numbered_implementation_followup,
    looks_like_readonly_chat_request,
    runtime_policy_for_intent,
)
from ..prompts import build_planning_prompt, build_system_prompt
from ..runtime_error_repair import current_reported_runtime_error
from ..state import ExecutionPlan, PlanStep, clip_text_value, json_safe_value
from ..task_targets import primary_task_target_path
from ..harness.tool_visibility import (
    hidden_tool_reason,
    resolve_turn_tool_exposure,
    schedule_retry_tool_exposure,
)
from .recovery_context import build_goal_recap
from ..write_session_fsm import new_write_session, record_write_session_event
from ..tools.dispatcher import normalize_tool_request
from ..tools.planning import _refresh_plan_playbook_artifact
from . import nodes as _nodes
from .chat_progress import build_repeated_reasoning_loop_message
from .deps import GraphRuntimeDeps
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall
from .interrupts import pause_for_plan_approval
from .write_recovery import (
    build_synthetic_write_args,
    can_safely_synthesize,
    recover_write_intent,
    write_recovery_kind,
    write_recovery_metadata,
    _force_finalize_if_complete_file,
)
from .terminal_completion import (
    extract_completion_message as _extract_completion_message,
    latest_verifier_allows_terminal_recovery as _latest_verifier_allows_terminal_recovery,
    maybe_promote_raw_terminal_json_task_complete as _maybe_promote_raw_terminal_json_task_complete,
    maybe_promote_terminal_prose_task_complete as _maybe_promote_terminal_prose_task_complete,
    raw_terminal_json_completion_message as _raw_terminal_json_completion_message,
    readonly_answer_can_complete as _readonly_answer_can_complete,
    terminal_prose_completion_message as _terminal_prose_completion_message,
    working_memory_signals_completion as _working_memory_signals_completion,
)


from .declared_file_read import (
    _consume_reasoning_fallback_flag,
    _maybe_synthesize_declared_file_read,
)
from .lifecycle_tool_validation import (
    _apply_tool_call_schema_repair,
    _record_tool_call_schema_repair,
    _schema_validation_repair_failure,
    _tool_call_repair_enabled,
    _tool_call_repair_log_only,
)
from .hidden_tool_helpers import (
    _build_hidden_tool_block_message,
    _format_allowed_tool_summary,
    _hidden_tool_retry_message,
    _rerouteable_hidden_tool_call,
    _strip_hidden_chat_terminal_completion_calls,
    _validation_handoff_hint_for_blocked_tool,
)
from .tool_model_rules import _model_is_gemma_4


def _is_readonly_lookup_intent(harness: Any) -> bool:
    """Return True when the current task is best treated as answer-only research."""
    task = str(harness.state.run_brief.original_task or "").strip()
    active_intent = normalize_intent_label(getattr(harness.state, "active_intent", "") or "")
    if active_intent == "readonly_lookup":
        return True
    if active_intent and active_intent != "general_task":
        return False
    return looks_like_readonly_chat_request(task)


def _readonly_answer_looks_complete(text: str) -> bool:
    """Heuristic: does the assistant text look like a finished research answer?"""
    lowered = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if len(lowered) < 80:
        return False
    # Forward-looking phrasing means the model is not done yet.
    if any(marker in lowered for marker in ("let me ", "i'll ", "i will ", "i need to ", "next i ", "can inspect", "going to ", "gonna ")):
        return False
    # Structural answer signals: markdown list, numbered steps, fenced code.
    if re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+\S", text):
        return True
    if "```" in text:
        return True
    # Conclusive language.
    if any(marker in lowered for marker in ("to install", "steps:", "summary", "conclusion", "in summary", "recommended")):
        return True
    return False


_NON_ACTIONABLE_FORWARD_MARKERS = (
    "i'll ",
    "i will ",
    "let me ",
    "i need to ",
    "next i ",
    "i'm going to ",
    "im going to ",
    "gonna ",
    "going to ",
)


def _looks_like_non_actionable_prose(text: str) -> bool:
    """Detect prose that announces intent without concrete action."""
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not normalized:
        return True
    return any(marker in normalized for marker in _NON_ACTIONABLE_FORWARD_MARKERS)


def _non_actionable_turn_signature(text: str) -> str:
    """Stable signature for a non-actionable assistant turn."""
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return normalized[:120]


async def interpret_model_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    reasoning_fallback_active = _consume_reasoning_fallback_flag(harness)
    assistant_text = str(graph_state.last_assistant_text or "")
    assistant_text_for_guards = "" if reasoning_fallback_active else assistant_text
    strategy = harness.state.strategy
    if not isinstance(strategy, dict):
        strategy = harness.state.scratchpad.get("strategy", {})
    if not isinstance(strategy, dict):
        strategy = {}
    thought_arch = strategy.get("thought_architecture")

    if graph_state.final_result is not None:
        return LoopRoute.FINALIZE

    # Bypass model-output interpretation when the runtime has system-sourced
    # pending tool calls (e.g. recovery-scheduled file_read). Route straight
    # to dispatch so recovery is deterministic.
    if graph_state.pending_tool_calls and any(
        str(getattr(p, "source", "model") or "").strip().lower() == "system"
        for p in graph_state.pending_tool_calls
    ):
        return LoopRoute.DISPATCH_TOOLS

    summarizer_client = getattr(harness, "summarizer_client", None)
    if (
        harness.summarizer
        and graph_state.last_thinking_text
        and len(graph_state.last_thinking_text) > 800
        and summarizer_client
    ):
        if int(harness.state.scratchpad.get("_distill_count", 0)) < 4:
            harness.state.scratchpad["_distill_count"] = int(harness.state.scratchpad.get("_distill_count", 0)) + 1
            insight = await harness.summarizer.distill_thinking_async(
                client=summarizer_client,
                thinking_text=graph_state.last_thinking_text,
                task=harness.state.run_brief.original_task,
            )
            if insight:
                harness._record_experience(
                    tool_name="reasoning",
                    result=ToolEnvelope(success=True, output=insight),
                    source="summarized",
                    notes=insight,
                )
                if harness.state.recent_messages:
                    last_msg = harness.state.recent_messages[-1]
                    if last_msg.role == "assistant":
                        last_msg.metadata["thinking_insight"] = insight

    if is_phase_contract_active(strategy):
        contract_phase_fn = getattr(harness.state, "contract_phase", None)
        current_phase = contract_phase_fn() if callable(contract_phase_fn) else harness.state.current_phase
        if graph_state.pending_tool_calls:
            original_calls = list(graph_state.pending_tool_calls)
            allowed_calls, blocked_tools = filter_phase_blocked_tools(
                original_calls,
                phase=current_phase,
            )
            if blocked_tools:
                graph_state.pending_tool_calls = allowed_calls
                if not allowed_calls:
                    if current_phase == "explore" and all(
                        c.tool_name in {"task_complete", "task_fail"} for c in original_calls
                    ):
                        if harness.state.step_count < harness.config.min_exploration_steps:
                            harness.state.append_message(ConversationMessage(
                                role="system",
                                content=(
                                    f"ANTI-LAZINESS: You are trying to finish at step {harness.state.step_count}, "
                                    f"but this task requires at least {harness.config.min_exploration_steps} discovery steps. "
                                    "Perform more deep-dive exploration before concluding."
                                ),
                                metadata={
                                    "is_recovery_nudge": True,
                                    "recovery_kind": "phase_contract_min_exploration_steps",
                                },
                            ))
                            return LoopRoute.NEXT_STEP
                        harness.state.current_phase = "verify"
                        harness._runlog(
                            "phase_transition",
                            "auto-transition to VERIFICATION via premature completion attempt",
                            level="debug",
                            subsystem="graph",
                            old_phase=current_phase,
                            new_phase="verify",
                            trigger="premature_completion_attempt",
                            blocked_tools=blocked_tools,
                            allowed_tools=[c.tool_name for c in allowed_calls],
                            contract_inferred=is_phase_contract_active(strategy),
                        )
                    else:
                        phase_bits = phase_contract(current_phase)
                        harness.state.append_message(ConversationMessage(
                            role="system",
                            content=(
                                f"You are in the DISCOVERY phase ({current_phase.upper()}). "
                                f"Phase contract focus: {phase_bits.focus}. "
                                f"Blocked tools: {', '.join(sorted(set(blocked_tools)))}. "
                                "Use the phase handoff artifacts before trying a different kind of action."
                            ),
                            metadata={
                                "is_recovery_nudge": True,
                                "recovery_kind": "phase_contract_all_tools_blocked",
                            },
                        ))
                        harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                        return LoopRoute.NEXT_STEP
                else:
                    phase_bits = phase_contract(current_phase)
                    harness.state.append_message(ConversationMessage(
                        role="system",
                        content=(
                            f"You are in the {current_phase.upper()} phase. "
                            f"Phase contract focus: {phase_bits.focus}. "
                            f"Some requested tools were blocked: {', '.join(sorted(set(blocked_tools)))}. "
                            "Continue with the allowed tools and the current handoff artifacts."
                        ),
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "phase_contract_partial_tools_blocked",
                        },
                    ))
                    harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                    return LoopRoute.NEXT_STEP

            if current_phase == "verify":
                blocked = ["task_complete", "task_fail", "file_write", "file_patch", "ast_patch", "long_context_lookup", "summarize_report", "artifact_read", "grep"]
                if any(c.tool_name in blocked for c in graph_state.pending_tool_calls):
                    graph_state.pending_tool_calls = [c for c in graph_state.pending_tool_calls if c.tool_name not in blocked]
                    harness.state.append_message(ConversationMessage(
                        role="system",
                        content=(
                            "You are in VERIFICATION. Review the verifier evidence and acceptance criteria before "
                            "moving to execution or repair."
                        ),
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "verify_phase_tools_blocked",
                        },
                    ))
                    harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                    return LoopRoute.NEXT_STEP

    # Anti-premature-fail guard: block task_fail/task_complete within 2 steps
    # of a verifier failure to force at least one repair attempt.
    if graph_state.pending_tool_calls:
        has_terminal = any(c.tool_name in {"task_complete", "task_fail"} for c in graph_state.pending_tool_calls)
        if has_terminal:
            failure_step = harness.state.scratchpad.get("_verifier_failure_step")
            if isinstance(failure_step, int) and failure_step > 0:
                if harness.state.step_count - failure_step < 2:
                    graph_state.pending_tool_calls = [
                        c for c in graph_state.pending_tool_calls
                        if c.tool_name not in {"task_complete", "task_fail"}
                    ]
                    if graph_state.pending_tool_calls:
                        harness.state.append_message(ConversationMessage(
                            role="system",
                            content=(
                                "VERIFIER RECOVERY: The latest verifier check failed. "
                                "Attempt a repair (file_patch, file_write, or shell_exec) before "
                                "calling task_fail or task_complete."
                            ),
                            metadata={
                                "is_recovery_nudge": True,
                                "recovery_kind": "verifier_failure_repair_required",
                            },
                        ))
                        harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                        return LoopRoute.NEXT_STEP

    if graph_state.pending_tool_calls:
        tool_exposure = resolve_turn_tool_exposure(harness, graph_state.run_mode)
        allowed_tool_names = {
            str(name).strip()
            for name in tool_exposure.get("names", [])
            if str(name).strip()
        }
        registered_tool_names: set[str] = set()
        registry = getattr(harness, "registry", None)
        names_fn = getattr(registry, "names", None) if registry is not None else None
        if callable(names_fn):
            try:
                registered_tool_names = {
                    str(name).strip()
                    for name in names_fn()
                    if str(name).strip()
                }
            except Exception:
                registered_tool_names = set()

        incomplete_payload = harness.state.scratchpad.get("_last_incomplete_tool_call")
        fallback_assistant_text = str(
            harness.state.scratchpad.get("_last_text_write_fallback_assistant_text") or ""
        )
        recovery_assistant_text = graph_state.last_assistant_text or fallback_assistant_text
        partial_tool_calls = []
        hidden_tool_calls: list[PendingToolCall] = []
        allowed_pending_calls: list[PendingToolCall] = []
        schema_repair_failures: list[tuple[PendingToolCall, str, dict[str, Any], str | None]] = []
        if isinstance(incomplete_payload, dict):
            raw_partial_calls = incomplete_payload.get("partial_tool_calls_raw")
            if isinstance(raw_partial_calls, list):
                partial_tool_calls = raw_partial_calls
        for pending in graph_state.pending_tool_calls:
            _nodes._repair_active_write_session_args(
                harness,
                pending,
                assistant_text=recovery_assistant_text,
            )
            _nodes._repair_empty_target_file_patch_to_file_write(harness, pending)
            if _nodes._apply_declared_read_before_write_reroute(
                graph_state,
                harness,
                pending,
                assistant_text=recovery_assistant_text,
            ):
                pass
            if pending.tool_name in {"file_write", "file_append"}:
                intent = recover_write_intent(
                    harness=harness,
                    pending=pending,
                    assistant_text=recovery_assistant_text,
                    partial_tool_calls=partial_tool_calls,
                )
                if intent is not None:
                    _nodes._increment_run_metric(graph_state, "write_recovery_attempt_count")
                    harness.state.scratchpad["_last_write_recovery"] = write_recovery_metadata(
                        intent,
                        status="attempt",
                    )
                    harness._runlog(
                        "write_recovery_attempt",
                        "attempting write-intent recovery",
                        tool_name=pending.tool_name,
                        tool_call_id=pending.tool_call_id,
                        path=intent.path,
                        confidence=intent.confidence,
                        evidence=intent.evidence,
                        recovery_kind=write_recovery_kind(intent),
                        source=intent.source,
                    )
                    if can_safely_synthesize(intent, harness=harness):
                        if _force_finalize_if_complete_file(intent):
                            harness._runlog(
                                "write_recovery_complete_file_guard",
                                "cleared continuation intent for complete-file payload",
                                tool_call_id=pending.tool_call_id,
                                path=intent.path,
                            )
                        pending.tool_name = "file_write"
                        pending.args = build_synthetic_write_args(intent)
                        pending.raw_arguments = json.dumps(pending.args, ensure_ascii=True, sort_keys=True)
                        _nodes._increment_run_metric(graph_state, "write_recovery_success_count")
                        if "assistant_fenced_code" in intent.evidence or "assistant_inline_tool_block" in intent.evidence:
                            _nodes._increment_run_metric(graph_state, "write_recovery_from_assistant_code_count")
                        harness.state.scratchpad["_last_write_recovery"] = write_recovery_metadata(
                            intent,
                            status="synthesized",
                        )
                        harness._runlog(
                            "write_recovery_synthesized",
                            "synthesized file_write from recovered intent",
                            tool_call_id=pending.tool_call_id,
                            path=intent.path,
                            confidence=intent.confidence,
                            evidence=intent.evidence,
                            recovery_kind=write_recovery_kind(intent),
                        )
                    else:
                        _nodes._increment_run_metric(graph_state, "write_recovery_declined_count")
                        if str(intent.confidence).strip().lower() == "low":
                            _nodes._increment_run_metric(graph_state, "write_recovery_low_confidence_count")
                        harness.state.scratchpad["_last_write_recovery"] = write_recovery_metadata(
                            intent,
                            status="declined",
                        )
                        harness._runlog(
                            "write_recovery_declined",
                            "declined write-intent recovery",
                            tool_call_id=pending.tool_call_id,
                            path=intent.path,
                            confidence=intent.confidence,
                            evidence=intent.evidence,
                            recovery_kind=write_recovery_kind(intent),
                        )
            repair_result = None
            if _tool_call_repair_enabled(harness):
                repair_result = _apply_tool_call_schema_repair(harness, pending)
                if (
                    repair_result is not None
                    and repair_result.repaired
                    and not _tool_call_repair_log_only(harness)
                ):
                    _record_tool_call_schema_repair(harness, pending, repair_result)
            missing_args = _schema_validation_repair_failure(pending, repair_result)
            if missing_args is None:
                missing_args = _nodes._detect_placeholder_tool_call(harness, pending)
            if missing_args is None:
                missing_args = _nodes._detect_empty_file_write_payload(harness, pending)
            if missing_args is not None and pending.tool_name in {"file_write", "file_append"}:
                retry_count = _nodes._record_empty_write_retry_metric(graph_state, harness, pending)
                salvaged = _nodes._salvage_active_write_session_append(harness, pending)
                if salvaged is not None:
                    session = getattr(harness.state, "write_session", None)
                    pending.tool_name = salvaged.tool_name
                    pending.args = salvaged.args
                    pending.raw_arguments = salvaged.raw_arguments
                    if session is not None:
                        session.write_salvage_count += 1
                        graph_state.latency_metrics["write_session_salvage_count"] = session.write_salvage_count
                        harness._runlog(
                            "write_session_append_salvaged",
                            "salvaged malformed file_append into file_write for active session",
                            session_id=session.write_session_id,
                            tool_call_id=pending.tool_call_id,
                            retry_count=retry_count,
                            salvage_count=session.write_salvage_count,
                            section_name=str(pending.args.get("section_name") or ""),
                        )
                    harness.state.scratchpad.pop("_last_incomplete_tool_call", None)
                    continue
            if missing_args is None:
                timeout_recovery = _nodes._detect_timeout_recovered_incomplete_tool_call(harness, pending)
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
                        "injected timeout-specific tool call recovery nudge",
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
                    graph_state.pending_tool_calls = []
                    graph_state.last_assistant_text = ""
                    graph_state.last_thinking_text = ""
                    return LoopRoute.NEXT_STEP
                missing_args = _nodes._detect_missing_required_tool_arguments(harness, pending)
                if missing_args is None:
                    missing_args = _nodes._detect_patch_existing_stage_read_contract_violation(harness, pending)
            if missing_args is None:
                from .tool_loop_guards import _detect_unknown_tool_call
                missing_args = _detect_unknown_tool_call(harness, pending)
            if missing_args is None:
                if (
                    str(getattr(pending, "source", "model") or "model").strip().lower() == "model"
                    and pending.tool_name in registered_tool_names
                    and pending.tool_name not in allowed_tool_names
                ):
                    hidden_tool_calls.append(pending)
                    continue
                allowed_pending_calls.append(pending)
                continue
            err_msg, details = missing_args
            target_path = None
            if pending.tool_name in {"file_write", "file_patch", "ast_patch"}:
                target_path = primary_task_target_path(harness)
                if pending.tool_name == "file_write" and target_path:
                    _nodes._ensure_chunk_write_session(harness, target_path)
                if target_path:
                    details = dict(details)
                    details["target_path"] = target_path
            schema_repair_failures.append((pending, err_msg, details, target_path))
            continue

        if schema_repair_failures:
            pending, err_msg, details, target_path = schema_repair_failures[0]
            repair_decision = _nodes.schema_validation_repair_decision(
                harness,
                pending,
                err_msg,
                details,
                target_path=target_path,
            )
            if repair_decision.status == "fail":
                harness.state.recent_errors.append(err_msg)
                harness._runlog(
                    "tool_call_validation_error",
                    "tool call missing required arguments",
                    **repair_decision.runlog_data,
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ERROR,
                        content=err_msg,
                        data=repair_decision.alert_data,
                    ),
                )
                graph_state.pending_tool_calls = []
                graph_state.final_result = harness._failure(
                    err_msg,
                    error_type="schema_validation_error",
                    details=repair_decision.details,
                )
                graph_state.error = graph_state.final_result["error"]
                return LoopRoute.FINALIZE

            harness.state.recent_errors.append(err_msg)
            harness._runlog(
                "tool_call_repair",
                "injected schema repair nudge",
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
            repair_conversation_message = repair_decision.conversation_message
            if repair_conversation_message is None:
                return LoopRoute.FINALIZE
            if allowed_pending_calls:
                _nodes.defer_schema_validation_repair_message(harness, repair_conversation_message)
                harness._runlog(
                    "tool_call_repair_deferred",
                    "deferred schema repair nudge while dispatching valid sibling tool calls",
                    tool_name=repair_decision.tool_name,
                    tool_call_id=repair_decision.tool_call_id,
                    valid_sibling_count=len(allowed_pending_calls),
                    retry_count=repair_decision.retry_count,
                )
            else:
                harness.state.append_message(repair_conversation_message)
                graph_state.pending_tool_calls = []
                graph_state.last_assistant_text = ""
                graph_state.last_thinking_text = ""
                return LoopRoute.NEXT_STEP

        graph_state.pending_tool_calls = allowed_pending_calls
        if hidden_tool_calls:
            allowed_tool_list = [
                str(name).strip()
                for name in tool_exposure.get("names", [])
                if str(name).strip()
            ]
            blocked_tool_names = [pending.tool_name for pending in hidden_tool_calls]
            hidden_reasons = {
                pending.tool_name: hidden_tool_reason(
                    pending.tool_name,
                    state=harness.state,
                    mode=graph_state.run_mode,
                )
                for pending in hidden_tool_calls
            }
            retry_pending = _rerouteable_hidden_tool_call(
                hidden_tool_calls,
                hidden_reasons=hidden_reasons,
                mode=graph_state.run_mode,
            )

            # ── Escalation cap: force-finalize after 2 consecutive exposures ───
            # If task_complete has been blocked at the exposure layer 2+ turns in
            # a row AND working memory explicitly signals the goal is done,
            # treat the session as complete rather than looping indefinitely.
            if any(p.tool_name == "task_complete" for p in hidden_tool_calls):
                block_count = int(harness.state.scratchpad.get("_task_complete_blocked_count", 0)) + 1
                harness.state.scratchpad["_task_complete_blocked_count"] = block_count
                if (
                    block_count >= 2
                    and _working_memory_signals_completion(harness)
                    and current_reported_runtime_error(harness.state) is None
                ):
                    completion_message = _extract_completion_message(harness, hidden_tool_calls)
                    harness._runlog(
                        "task_complete_blocked_force_finalize",
                        "force-finalizing after repeated task_complete exposure blocks with completion signal in working memory",
                        block_count=block_count,
                        completion_message=completion_message[:200] if completion_message else "",
                    )
                    harness.state.scratchpad["_task_complete"] = True
                    harness.state.scratchpad["_task_complete_message"] = completion_message
                    harness.state.touch()
                    graph_state.final_result = {
                        "status": "completed",
                        "message": {
                            "status": "complete",
                            "message": completion_message,
                        },
                        "assistant": graph_state.last_assistant_text,
                        "thinking": graph_state.last_thinking_text,
                        "usage": graph_state.last_usage,
                    }
                    return LoopRoute.FINALIZE
            else:
                # Reset the counter when a different tool is in hidden_tool_calls
                harness.state.scratchpad["_task_complete_blocked_count"] = 0
            # ─────────────────────────────────────────────────────────────────

            recovery_message = _build_hidden_tool_block_message(
                hidden_tool_calls,
                allowed_names=allowed_tool_list,
                harness=harness,
                mode=graph_state.run_mode,
            )
            if retry_pending is not None and schedule_retry_tool_exposure(
                harness.state,
                mode=graph_state.run_mode,
                tool_name=retry_pending.tool_name,
                arguments=retry_pending.args,
            ):
                retry_message = _hidden_tool_retry_message(retry_pending)
                if retry_message:
                    recovery_message = f"{recovery_message} {retry_message}"
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=recovery_message,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "tool_not_exposed_this_turn",
                        "blocked_tools": blocked_tool_names,
                        "allowed_tools": allowed_tool_list,
                        "hidden_reasons": hidden_reasons,
                        "retry_tool_name": str(retry_pending.tool_name or "").strip() if retry_pending is not None else "",
                    },
                )
            )
            harness._runlog(
                "tool_blocked_not_exposed",
                "blocked registered tool call that was hidden on this turn",
                blocked_tools=blocked_tool_names,
                run_mode=graph_state.run_mode,
                allowed_tools=allowed_tool_list,
            )
            if not graph_state.pending_tool_calls:
                return LoopRoute.NEXT_STEP

        return LoopRoute.DISPATCH_TOOLS

    nudges = int(harness.state.scratchpad.get("_no_tool_nudges", 0))
    assistant_text = graph_state.last_assistant_text or ""

    if (
        graph_state.run_mode == "planning"
        or harness.state.planning_mode_enabled
    ):
        synthesized_plan = _nodes._synthesize_plan_from_text(harness, assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            _nodes._persist_planning_playbook(harness, synthesized_plan)
            harness.state.touch()
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is not None and plan.status != "approved":
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

    planning_request = harness._extract_planning_request(harness.state.run_brief.original_task)
    if (
        graph_state.pending_tool_calls == []
        and planning_request is not None
        and _nodes._planning_response_looks_like_plan(assistant_text)
    ):
        synthesized_plan = _nodes._synthesize_plan_from_text(harness, assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            _nodes._persist_planning_playbook(harness, synthesized_plan)
            harness.state.touch()
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is not None:
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

    if assistant_text:
        low_assistant_text = assistant_text.lower()
        has_facts = bool(harness.state.working_memory.known_facts)
        has_tool_evidence = any(
            message.role == "tool"
            for message in harness.state.recent_messages[-6:]
        )
        task_mode = str(getattr(harness.state, "task_mode", "") or "").strip().lower()
        chat_only_auto_finalize = graph_state.run_mode == "chat" and task_mode == "chat"
        has_recent_errors = bool(getattr(harness.state, "recent_errors", []))
        looks_like_future_action = any(
            marker in low_assistant_text
            for marker in (
                "i'll ",
                "i will ",
                "let me ",
                "going to ",
                "gonna ",
                "i can try",
                "i'll try",
                "i will try",
                "next i",
            )
        )
        # For read-only / research tasks (e.g. "do a websearch and respond"),
        # a substantive prose answer after tool evidence is the deliverable.
        # Auto-promote it to task_complete so the loop doesn't stall with
        # action_stall / completion_confabulation / consecutive_idle nudges.
        readonly_lookup = _is_readonly_lookup_intent(harness)
        if (
            nudges == 0
            and has_tool_evidence
            and len(assistant_text) > 120
            and not has_recent_errors
            and not looks_like_future_action
            and (
                chat_only_auto_finalize
                or (
                    readonly_lookup
                    and _readonly_answer_looks_complete(assistant_text)
                )
            )
            and harness.state.current_phase != "repair"
        ):
            harness._runlog(
                "auto_finalize",
                "prose answer with tool evidence; skipping nudge",
                text_len=len(assistant_text),
                readonly_lookup=readonly_lookup,
            )
            graph_state.final_result = {
                "status": "completed",
                "message": {
                    "status": "complete",
                    "message": assistant_text[:500],
                },
                "assistant": assistant_text,
            }
            return LoopRoute.FINALIZE

    _ACTION_KEYWORDS = ["call", "run", "execute", "use", "using", "invok", "command", "tool"]
    _HTML_TOOL_TAGS = ["<tool_call>", "<function=", "<parameter="]
    _FUNC_SYNTAX = [f"{t}(" for t in ["shell_exec", "artifact_read", "file_read", "dir_list", "task_complete", "bash_exec"]]

    low_text = assistant_text.lower()
    thinking_looks_like_action = any(kw in graph_state.last_thinking_text.lower() for kw in _ACTION_KEYWORDS)
    text_looks_like_action_list = any(kw in low_text for kw in _ACTION_KEYWORDS)
    text_has_tool_tags = any(tag in low_text for tag in _HTML_TOOL_TAGS)
    text_has_func_calls = any(fn in low_text for fn in _FUNC_SYNTAX)

    if _maybe_synthesize_declared_file_read(graph_state, harness, assistant_text):
        return LoopRoute.DISPATCH_TOOLS

    if (
        assistant_text_for_guards
        and not graph_state.pending_tool_calls
        and "task_complete" in low_text
        and _maybe_promote_terminal_prose_task_complete(
            graph_state,
            harness,
            nudge_count=nudges,
        )
    ):
        return LoopRoute.DISPATCH_TOOLS

    if (
        not reasoning_fallback_active
        and not graph_state.pending_tool_calls
        and (thinking_looks_like_action or text_looks_like_action_list or text_has_tool_tags or text_has_func_calls)
    ):
        if graph_state.run_mode == "planning" or harness.state.planning_mode_enabled:
            synthesized_plan = _nodes._synthesize_plan_from_text(harness, assistant_text)
            if synthesized_plan is not None:
                harness.state.draft_plan = synthesized_plan
                harness.state.active_plan = synthesized_plan
                harness.state.planning_mode_enabled = True
                harness.state.sync_plan_mirror()
                _nodes._persist_planning_playbook(harness, synthesized_plan)
                harness.state.touch()
                await pause_for_plan_approval(graph_state, deps)
                return LoopRoute.FINALIZE
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                await pause_for_plan_approval(graph_state, deps)
                return LoopRoute.FINALIZE
        # Read-only research tasks can mention verbs like "install" in the
        # user's question and end up here. If the assistant has already
        # produced a substantive answer after gathering evidence, don't treat
        # it as a stalled action.
        if _is_readonly_lookup_intent(harness) and _readonly_answer_looks_complete(assistant_text):
            harness._runlog(
                "readonly_lookup_skip_action_stall",
                "substantive answer for read-only task; skipping action_stall",
            )
        else:
            stalls = int(harness.state.scratchpad.get("_action_stalls", 0))
            if stalls < 1:
                harness.state.scratchpad["_action_stalls"] = stalls + 1
                msg = "### SYSTEM ALERT: You identified or described a tool action, but you did not emit the JSON tool call."
                if text_has_tool_tags or text_has_func_calls:
                    msg = "### FORMAT ERROR: You used text-based tool tags or functional syntax (e.g. <tool_call> or shell_exec()). This is FORBIDDEN. You MUST use the JSON block format."

                model_name = ""
                client = getattr(harness, "client", None)
                if client is not None:
                    model_name = str(getattr(client, "model", "") or "").strip()
                if not model_name:
                    scratchpad = getattr(harness.state, "scratchpad", {}) or {}
                    if isinstance(scratchpad, dict):
                        model_name = str(scratchpad.get("_model_name") or "").strip()
                # Import here to avoid circular imports at module load time.
                from .tool_model_rules_model_detection import _model_is_exact_small_gemma_4_it
                if _model_is_exact_small_gemma_4_it(model_name):
                    msg += (
                        "\n\nGEMMA 4 e2b/e4b EXAMPLE: After </think>, output exactly one line like: "
                        '`{"name":"ssh_exec","arguments":{"host":"192.168.1.89","user":"root","password":"secret","command":"docker ps"}}`. '
                        "Replace the example values with the ones from the task. Do not add prose before or after the JSON."
                    )

                harness.state.append_message(ConversationMessage(
                    role="user",
                    content=f"{msg}\n\nDO NOT repeat your earlier findings or analysis. Just generate the JSON block immediately after your reasoning. Do not describe what you are going to do; just DO it.",
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "action_stall",
                        "retry_count": stalls + 1,
                        "format_error": bool(text_has_tool_tags or text_has_func_calls),
                    },
                ))
                harness._record_experience(
                    tool_name="reasoning",
                    result=ToolEnvelope(success=False, error=msg),
                    source="guarded_stall",
                    notes=f"Model described action but missed JSON format. Failure mode: {TOOL_NOT_CALLED}",
                )
                harness._runlog("action_stall", "improper tool format or description", stalls=stalls+1, has_tags=text_has_tool_tags)
                return LoopRoute.NEXT_STEP

    if not graph_state.pending_tool_calls and "hello" in low_text and ("task" in low_text or "complete" in low_text):
        if any(v in harness.state.run_brief.original_task.lower() for v in ["ping", "list", "read", "run"]):
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content="### MISSION CHECK: You mention 'hello' or completing the greeting, but a real task is still pending. DO NOT finish yet. Proceed with the primary mission.",
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "mission_check_hello_completion",
                    },
                )
            )
            harness._record_experience(
                tool_name="task_complete",
                result=ToolEnvelope(success=False, error="Blocked hello completion attempt"),
                source="guarded_completion",
                notes=f"Model attempted 'hello' completion while mission pending. Failure mode: {PREMATURE_TASK_COMPLETE}",
            )
        harness._runlog("premature_completion_blocked", "blocked hello completion", task=harness.state.run_brief.original_task)
        return LoopRoute.NEXT_STEP

    stream_halted = bool(harness.state.scratchpad.get("_last_stream_halted_without_done"))

    if (
        not reasoning_fallback_active
        and graph_state.run_mode not in {"chat", "planning"}
        and not harness.state.planning_mode_enabled
        and not graph_state.pending_tool_calls
        and not stream_halted
        and _looks_like_non_actionable_prose(assistant_text_for_guards)
        and not _readonly_answer_looks_complete(assistant_text_for_guards)
    ):
        signature = _non_actionable_turn_signature(assistant_text_for_guards)
        counts = harness.state.scratchpad.setdefault("_non_actionable_prose_counts", {})
        if not isinstance(counts, dict):
            counts = {}
            harness.state.scratchpad["_non_actionable_prose_counts"] = counts
        count = int(counts.get(signature, 0)) + 1
        counts[signature] = count
        harness.state.scratchpad["_non_actionable_prose_last_signature"] = signature

        registry = getattr(harness, "registry", None)
        has_escalate = False
        if registry is not None:
            names_fn = getattr(registry, "names", None)
            if callable(names_fn):
                try:
                    has_escalate = "escalate_to_bigger_model" in {str(n).strip() for n in names_fn()}
                except Exception:
                    has_escalate = False

        if count >= 3:
            if has_escalate:
                msg = (
                    "Your last response did not include a concrete tool call. "
                    "You appear to be stuck repeating intent without acting. "
                    "Call `escalate_to_bigger_model(reason='repeated non-actionable turns')` now, or call `task_fail` if blocked."
                )
            else:
                msg = (
                    "Your last response did not include a concrete tool call. "
                    "You appear to be stuck repeating intent without acting. "
                    "Call `task_fail(message='...')` now to avoid an endless loop."
                )
        else:
            msg = (
                "Your last response did not include a tool call. "
                "Emit ONE concrete next action as a tool call now, or call `task_fail` if blocked."
            )

        harness.state.append_message(ConversationMessage(
            role="user",
            content=msg,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "non_actionable_prose",
                "signature": signature,
                "count": count,
                "has_escalate": has_escalate,
            },
        ))
        harness._runlog(
            "non_actionable_prose_recovery",
            "injected recovery nudge for non-actionable assistant turn",
            signature=signature,
            count=count,
            has_escalate=has_escalate,
        )
        return LoopRoute.NEXT_STEP

    if not assistant_text_for_guards.strip() and not graph_state.pending_tool_calls and not stream_halted:
        blank_nudges = int(harness.state.scratchpad.get("_blank_message_nudges", 0))
        if blank_nudges < 2:
            harness.state.scratchpad["_blank_message_nudges"] = blank_nudges + 1
            msg = _nodes._build_blank_message_nudge(harness, repeated=blank_nudges >= 1)
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=msg,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "blank_message",
                        "retry_count": blank_nudges + 1,
                    },
                )
            )
            harness._runlog(
                "blank_message_recovery",
                "injected recovery nudge for an empty assistant turn",
                retry_count=blank_nudges + 1,
            )
            return LoopRoute.NEXT_STEP

    # Guard against repeated reasoning-only turns that never call a tool.
    if (
        assistant_text_for_guards
        and not graph_state.pending_tool_calls
        and not stream_halted
        and _nodes._looks_like_freeze_or_hang(harness, assistant_text_for_guards)
    ):
        reasoning_loop_nudges = int(harness.state.scratchpad.get("_repeated_reasoning_loop_nudges", 0))
        if reasoning_loop_nudges < 2:
            harness.state.scratchpad["_repeated_reasoning_loop_nudges"] = reasoning_loop_nudges + 1
            msg = build_repeated_reasoning_loop_message(harness, graph_state)
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=msg,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "repeated_reasoning_loop",
                        "retry_count": reasoning_loop_nudges + 1,
                    },
                )
            )
            harness._runlog(
                "repeated_reasoning_loop_recovery",
                "injected recovery nudge for repeated reasoning without action",
                retry_count=reasoning_loop_nudges + 1,
            )
            return LoopRoute.NEXT_STEP

    if assistant_text_for_guards and not graph_state.pending_tool_calls and not stream_halted:
        if _maybe_promote_terminal_prose_task_complete(
            graph_state,
            harness,
            nudge_count=nudges,
        ):
            return LoopRoute.DISPATCH_TOOLS

    if nudges < 4 and assistant_text_for_guards and not stream_halted:
        harness.state.scratchpad["_no_tool_nudges"] = nudges + 1
        msg = (
            "You reached a conclusion but did not call `task_complete`. "
            "If you are finished, you MUST call `task_complete(message='...')` with your final answer in this same turn. "
            "Do not repeat your earlier analysis; simply emit the tool call."
        )
        if nudges >= 2:
            msg = "REPEAT WARNING: You are stuck in a loop. You MUST call `task_complete` NOW to save your progress."

        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=msg,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "missing_task_complete",
                },
            )
        )
        harness._runlog("no_tool_recovery", "injected recovery nudge", nudge_count=nudges+1)
        return LoopRoute.NEXT_STEP

    if stream_halted or _nodes._looks_like_freeze_or_hang(harness, assistant_text):
        model_name = _nodes._harness_model_name(harness)
        gemma_stream_halt = stream_halted and _model_is_gemma_4(model_name)
        freeze_nudges = int(harness.state.scratchpad.get("_small_model_continue_nudges", 0))
        max_freeze_nudges = 5 if gemma_stream_halt else 2
        if freeze_nudges < max_freeze_nudges:
            harness.state.scratchpad["_small_model_continue_nudges"] = freeze_nudges + 1
            halt_reason = str(harness.state.scratchpad.get("_last_stream_halt_reason", "") or "")
            msg = _nodes._build_small_model_continue_message(
                harness,
                assistant_text,
                stream_halt_reason=halt_reason,
            )
            if halt_reason == "stream_ended_without_done" and not assistant_text.strip():
                msg = (
                    "The response stream ended before a clean completion signal. "
                    f"{build_goal_recap(harness)} Continue from the last concrete step within that objective. "
                    "Do not restart the task; either call the next tool or emit the next JSON tool call immediately."
                )
            if gemma_stream_halt and halt_reason == "reasoning_only_stream_stall":
                msg = (
                    "Gemma stream auto-continue: the prior response stayed in reasoning without emitting a tool call. "
                    f"{build_goal_recap(harness)} Continue from the exact last state now. "
                    "Do not summarize or restart. Emit exactly one available tool call, or call `task_complete(message='...')` "
                    "only if the objective is fully verified."
                )
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=msg,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "model_halt" if stream_halted or not _nodes._is_small_model(harness) else "small_model_freeze",
                        "recovery_mode": "gemma_stream_autocontinue" if gemma_stream_halt else "stream_continue_nudge",
                        "retry_count": freeze_nudges + 1,
                        "max_retry_count": max_freeze_nudges,
                    },
                )
            )
            harness._runlog(
                "small_model_freeze_recovery",
                "injected continuation nudge for a stalled model",
                retry_count=freeze_nudges + 1,
                max_retry_count=max_freeze_nudges,
                model_name=_nodes._harness_model_name(harness),
                stream_halted=stream_halted,
                gemma_stream_autocontinue=gemma_stream_halt,
            )
            harness.state.scratchpad.pop("_last_stream_halted_without_done", None)
            harness.state.scratchpad.pop("_last_stream_halt_reason", None)
            harness.state.scratchpad.pop("_last_stream_halt_details", None)
            return LoopRoute.NEXT_STEP

    if stream_halted and not graph_state.pending_tool_calls:
        halt_reason = str(harness.state.scratchpad.get("_last_stream_halt_reason", "") or "model_stream_stall")
        halt_details = harness.state.scratchpad.get("_last_stream_halt_details")
        details = halt_details if isinstance(halt_details, dict) else {}
        message = "Model stream halted repeatedly without a tool call or final actionable answer."
        if halt_reason:
            message = f"{message} Halt reason: {halt_reason}."
        harness._runlog(
            "model_stream_halt_exhausted",
            "exhausted stream-halt recovery before no-tool finalization",
            halt_reason=halt_reason,
            details=details,
            assistant_preview=str(graph_state.last_assistant_text or "")[:200],
        )
        graph_state.final_result = harness._failure(
            message,
            error_type="model_stream_stall",
            details={"halt_reason": halt_reason, **details},
        )
        graph_state.error = graph_state.final_result.get("error")
        return LoopRoute.FINALIZE

    if nudges >= 4 and graph_state.last_assistant_text and (
        harness.state.current_phase == "execute"
        or any(message.role == "tool" for message in harness.state.recent_messages[-6:])
        or bool(harness.state.working_memory.known_facts)
    ):
        harness._runlog("recovery", "finalizing after multiple no-tool nudges")
        harness.state.scratchpad["_task_complete"] = True
        harness.state.scratchpad["_task_complete_message"] = graph_state.last_assistant_text[:500]
        harness.state.touch()
        graph_state.final_result = {
            "status": "completed",
            "message": {
                "status": "complete",
                "message": graph_state.last_assistant_text[:500],
            },
            "assistant": graph_state.last_assistant_text,
        }
        return LoopRoute.FINALIZE

    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.SYSTEM, content="No tool calls returned; stopping loop."),
    )
    graph_state.final_result = {
        "status": "stopped",
        "reason": "no_tool_calls",
        "assistant": graph_state.last_assistant_text,
        "thinking": graph_state.last_thinking_text,
    }
    return LoopRoute.FINALIZE


_CHAT_SHELL_COMMAND_RE = re.compile(
    r"(?:^|\s|`)(?:docker|systemctl|journalctl|ssh|scp|ls|cat|grep|find|ps|top|kill|curl|wget|python3?|bash|sh|apt|yum|dnf|npm|pip|git|make|cmake|go|rustc|cargo|java|node|kubectl)\b",
    re.IGNORECASE,
)


def _assistant_text_looks_like_shell_command(text: str) -> bool:
    """Detect assistant prose that is actually a bare shell/command line."""
    stripped = str(text or "").strip().strip("`'")
    if not stripped:
        return False
    if _CHAT_SHELL_COMMAND_RE.search(stripped):
        return True
    # A single line that looks like a tool-name invocation, e.g. "ssh_exec(...)"
    if "(" in stripped and stripped.endswith(")"):
        first_token = stripped.split("(", 1)[0].strip()
        if first_token and not first_token[0].isdigit() and " " not in first_token:
            return True
    return False


async def interpret_chat_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    scratchpad = harness.state.scratchpad
    reasoning_fallback_active = _consume_reasoning_fallback_flag(harness)
    signature = _nodes._chat_turn_signature(graph_state)
    prior_signature = str(scratchpad.get("_chat_last_turn_signature") or "")
    if signature:
        scratchpad["_chat_last_turn_signature"] = signature

    if graph_state.pending_tool_calls and not _strip_hidden_chat_terminal_completion_calls(graph_state, harness):
        recovery_assistant_text = str(graph_state.last_assistant_text or "")
        for pending in graph_state.pending_tool_calls:
            _nodes._repair_active_write_session_args(
                harness,
                pending,
                assistant_text=recovery_assistant_text,
            )
        return LoopRoute.DISPATCH_TOOLS

    assistant_text = str(graph_state.last_assistant_text or "")
    assistant_text_for_guards = "" if reasoning_fallback_active else assistant_text
    low_text = assistant_text.lower()
    action_keywords = ["call", "run", "execute", "use", "using", "invok", "command", "tool"]
    html_tool_tags = ["<tool_call>", "<function=", "<parameter="]
    func_syntax = [f"{tool_name}(" for tool_name in ["shell_exec", "ssh_exec", "artifact_read", "file_read", "dir_list", "task_complete", "bash_exec"]]
    thinking_looks_like_action = any(keyword in str(graph_state.last_thinking_text or "").lower() for keyword in action_keywords)
    text_looks_like_action = any(keyword in low_text for keyword in action_keywords)
    text_has_tool_tags = any(tag in low_text for tag in html_tool_tags)
    text_has_func_calls = any(token in low_text for token in func_syntax)
    current_task_fn = getattr(harness, "_current_user_task", None)
    try:
        current_task = current_task_fn() if callable(current_task_fn) else ""
    except Exception:
        current_task = ""
    current_task = str(current_task or getattr(harness.state.run_brief, "original_task", "") or "")
    recent_tool_block = any(
        "registered but unavailable on this turn" in str(error or "").lower()
        for error in list(getattr(harness.state, "recent_errors", []) or [])[-4:]
    )
    actionable_chat_task = looks_like_numbered_implementation_followup(current_task)

    # Smalltalk bypass: pure chat tasks with no actionable work should
    # finalize directly after a natural-language response, not nudge.
    task_mode = str(getattr(harness.state, "task_mode", "") or "").strip().lower()
    if (
        graph_state.run_mode == "chat"
        and assistant_text_for_guards
        and not graph_state.pending_tool_calls
        and _maybe_promote_raw_terminal_json_task_complete(graph_state, harness)
    ):
        return LoopRoute.DISPATCH_TOOLS

    if (
        graph_state.run_mode == "chat"
        and task_mode == "chat"
        and assistant_text_for_guards
        and not graph_state.pending_tool_calls
    ):
        has_active_write = (
            getattr(harness.state, "write_session", None) is not None
            and str(getattr(harness.state.write_session, "status", "") or "").strip().lower() != "complete"
        )
        has_active_plan = bool(
            harness.state.planning_mode_enabled
            or harness.state.active_plan is not None
            or harness.state.draft_plan is not None
        )
        if (
            not has_active_write
            and not has_active_plan
            and not recent_tool_block
            and not actionable_chat_task
            and not text_looks_like_action
            and not text_has_tool_tags
            and not text_has_func_calls
            and not _assistant_text_looks_like_shell_command(assistant_text)
        ):
            graph_state.final_result = {
                "status": "chat_completed",
                "assistant": assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
            return LoopRoute.FINALIZE

    if not reasoning_fallback_active and (thinking_looks_like_action or text_looks_like_action or text_has_tool_tags or text_has_func_calls):
        stalls = int(scratchpad.get("_chat_action_stalls", 0))
        if stalls < 2:
            scratchpad["_chat_action_stalls"] = stalls + 1
            message = (
                "### SYSTEM ALERT: You described a tool action in chat mode, but no usable tool call was dispatched."
            )
            if text_has_tool_tags or text_has_func_calls:
                message = (
                    "### FORMAT ERROR: You used text-based tool tags or raw function syntax "
                    "(for example `<tool_call>` or `ssh_exec(...)`). Emit a usable tool call instead."
                )
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=(
                        f"{message}\n\n"
                        "Do not repeat the analysis. Continue from the last concrete step and emit the next tool call immediately."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "chat_action_stall",
                        "thread_id": graph_state.thread_id,
                        "retry_count": stalls + 1,
                    },
                )
            )
            harness._runlog(
                "chat_action_stall_nudge",
                "nudged chat runtime after action-like prose without a dispatched tool call",
                thread_id=graph_state.thread_id,
                retry_count=stalls + 1,
                has_tags=text_has_tool_tags,
                has_function_syntax=text_has_func_calls,
            )
            return LoopRoute.NEXT_STEP

    if signature and signature == prior_signature:
        nudge_key = f"{graph_state.thread_id}:{signature}"
        if scratchpad.get("_chat_repeated_thinking_nudged") != nudge_key:
            scratchpad["_chat_repeated_thinking_nudged"] = nudge_key
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=_nodes._build_repeated_chat_thinking_message(harness, graph_state),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "repeated_thinking",
                        "thread_id": graph_state.thread_id,
                    },
                )
            )
            harness._runlog(
                "chat_repeated_thinking_nudge",
                "nudged model after repeated no-tool thinking",
                thread_id=graph_state.thread_id,
                signature=signature,
            )
        return LoopRoute.NEXT_STEP

    completion_guard = _nodes._chat_completion_recovery_guard(harness)
    if completion_guard is not None:
        nudge_key = "|".join(
            [
                graph_state.thread_id,
                _nodes._chat_turn_signature(graph_state),
                str(completion_guard.get("signature") or ""),
            ]
        )
        if scratchpad.get("_chat_completion_guard_nudged") != nudge_key:
            scratchpad["_chat_completion_guard_nudged"] = nudge_key
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=str(completion_guard.get("message") or "").strip(),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": str(completion_guard.get("kind") or "chat_completion_guard"),
                        "thread_id": graph_state.thread_id,
                    },
                )
            )
            harness._runlog(
                "chat_completion_guard_nudge",
                "blocked plain-text chat completion while recovery blockers remain",
                thread_id=graph_state.thread_id,
                recovery_kind=str(completion_guard.get("kind") or ""),
            )
        return LoopRoute.NEXT_STEP

    if recent_tool_block or actionable_chat_task:
        graph_state.final_result = {
            "status": "stopped",
            "reason": "chat_action_blocked",
            "assistant": assistant_text,
            "thinking": graph_state.last_thinking_text,
            "usage": graph_state.last_usage,
        }
        return LoopRoute.FINALIZE

    graph_state.final_result = {
        "status": "chat_completed",
        "assistant": assistant_text,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
    }
    return LoopRoute.FINALIZE


async def interpret_planning_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    if graph_state.final_result is not None:
        return LoopRoute.FINALIZE
    if graph_state.pending_tool_calls:
        return LoopRoute.DISPATCH_TOOLS

    plan = harness.state.active_plan or harness.state.draft_plan
    if plan is None:
        synthesized_plan = _nodes._synthesize_plan_from_text(harness, graph_state.last_assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            _nodes._persist_planning_playbook(harness, synthesized_plan)
            harness.state.touch()
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

        planning_tool_names: set[str] = set()
        try:
            exposure = resolve_turn_tool_exposure(harness, "planning")
            names = exposure.get("names") if isinstance(exposure, dict) else []
            planning_tool_names = {str(name).strip() for name in names or [] if str(name).strip()}
        except Exception:
            planning_tool_names = set()
        if "plan_set" in planning_tool_names:
            nudge_content = "Planning mode is active. Create a structured plan with `plan_set` before trying to execute anything."
        else:
            nudge_content = (
                "Planning mode is active. Provide a concise structured plan in plain text before trying to execute anything. "
                "Do not mention unavailable planning tools."
            )

        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=nudge_content,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "planning_mode_requires_plan_set",
                    "planner_nudge": True,
                    "plan_set_available": "plan_set" in planning_tool_names,
                },
            )
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Planning nudge issued.",
                data={"status_activity": "gathering facts..."},
            ),
        )
        return LoopRoute.NEXT_STEP

    if plan.status != "approved":
        await pause_for_plan_approval(graph_state, deps)
        return LoopRoute.FINALIZE

    graph_state.final_result = {
        "status": "plan_ready",
        "assistant": graph_state.last_assistant_text,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
    }
    return LoopRoute.FINALIZE
