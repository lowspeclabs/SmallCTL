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
from ..prompts import build_planning_prompt, build_system_prompt
from ..state import ExecutionPlan, PlanStep, clip_text_value, json_safe_value
from ..task_targets import primary_task_target_path
from ..harness.tool_visibility import (
    hidden_tool_reason,
    recent_hidden_tool_recovery_artifact_id,
    resolve_turn_tool_exposure,
    schedule_retry_tool_exposure,
)
from .recovery_context import build_goal_recap
from ..write_session_fsm import new_write_session, record_write_session_event
from ..tools.dispatcher import normalize_tool_request
from ..tools.planning import _refresh_plan_playbook_artifact
from . import nodes as _nodes
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


_HIDDEN_TOOL_REASON_LABELS = {
    "missing_index": "missing index",
    "no_artifacts": "no artifacts yet",
    "no_active_plan": "no active plan",
    "no_background_jobs": "no background jobs",
    "write_session_not_finalizable": "write session not finalizable",
}
_CHAT_SYNTHETIC_TERMINAL_TOOLS = {"task_complete", "task_fail"}

# Strings in known_facts that indicate the goal was already met.
_COMPLETION_FACT_MARKERS = ("[COMPLETED]", "[DONE]", "[SUCCESS]", "task complete", "successfully removed", "successfully uninstalled")


def _working_memory_signals_completion(harness: Any) -> bool:
    """Return True when working memory or scratchpad indicates the goal is met.

    Checked in order of confidence:
    1. Scratchpad ``_task_complete`` flag (set by prior auto-finalize paths).
    2. ``[COMPLETED]``-style tags in ``known_facts``.
    3. Last assistant text explicitly stating completion (weak signal, only used
       if the two stronger signals are absent).
    """
    # 1. Explicit scratchpad completion flag.
    if harness.state.scratchpad.get("_task_complete"):
        return True

    # 2. Known-facts completion markers.
    facts = getattr(getattr(harness.state, "working_memory", None), "known_facts", None)
    if isinstance(facts, list):
        facts_text = " ".join(str(f) for f in facts).lower()
        if any(m.lower() in facts_text for m in _COMPLETION_FACT_MARKERS):
            return True

    return False


def _extract_completion_message(harness: Any, hidden_tool_calls: list[PendingToolCall]) -> str:
    """Extract the best completion message for a force-finalize event.

    Prefers the ``message`` argument from the pending task_complete call;
    falls back to the last assistant text, then to a generic message derived
    from known_facts.
    """
    for pending in hidden_tool_calls:
        if pending.tool_name == "task_complete":
            args = pending.args if isinstance(pending.args, dict) else {}
            msg = str(args.get("message") or "").strip()
            if msg:
                return msg

    # Fall back to last assistant text (truncated).
    last_text = str(getattr(harness.state, "_last_assistant_text", "") or "").strip()
    if not last_text:
        # Try known_facts summary.
        facts = getattr(getattr(harness.state, "working_memory", None), "known_facts", None)
        if isinstance(facts, list) and facts:
            last_text = str(facts[-1]).strip()
    return last_text[:500] if last_text else "Task completed (force-finalized after repeated task_complete blocks)."


def _format_allowed_tool_summary(names: list[str], *, limit: int = 8) -> str:
    visible_names = [str(name).strip() for name in names if str(name).strip()]
    if not visible_names:
        return "No tools are available on this turn."
    shown = visible_names[:limit]
    summary = ", ".join(shown)
    if len(visible_names) > limit:
        summary = f"{summary}, ..."
    return f"Available now: {summary}"


def _build_hidden_tool_block_message(
    blocked_calls: list[PendingToolCall],
    *,
    allowed_names: list[str],
    harness: Any,
    mode: str,
) -> str:
    blocked_bits: list[str] = []
    recovery_hints: list[str] = []
    for pending in blocked_calls:
        label = f"`{pending.tool_name}`"
        reason_code = hidden_tool_reason(
            pending.tool_name,
            state=harness.state,
            mode=mode,
        )
        reason_text = _HIDDEN_TOOL_REASON_LABELS.get(str(reason_code or "").strip())
        if reason_text:
            label = f"{label} ({reason_text})"
        blocked_bits.append(label)
        artifact_id = recent_hidden_tool_recovery_artifact_id(
            harness.state,
            tool_name=pending.tool_name,
        )
        if artifact_id:
            recovery_hints.append(f"Use `artifact_read(artifact_id='{artifact_id}')` for the full fetched body.")
    blocked_summary = ", ".join(blocked_bits)
    message = (
        f"Registered but unavailable on this turn: {blocked_summary}. "
        f"{_format_allowed_tool_summary(allowed_names)}"
    )
    if recovery_hints:
        message = f"{message} {' '.join(recovery_hints)}"
    return message


def _rerouteable_hidden_tool_call(
    blocked_calls: list[PendingToolCall],
    *,
    hidden_reasons: dict[str, str | None],
    mode: str,
) -> PendingToolCall | None:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"chat", "planning", "loop"}:
        return None
    for pending in blocked_calls:
        if hidden_reasons.get(pending.tool_name):
            continue
        if normalized_mode == "loop":
            if pending.tool_name == "web_fetch":
                return pending
            continue
        if pending.tool_name in {
            "shell_exec",
            "ssh_exec",
            "file_patch",
            "ast_patch",
            "file_write",
            "file_append",
            "finalize_write_session",
            "web_fetch",
        }:
            return pending
    return None


def _hidden_tool_retry_message(pending: PendingToolCall) -> str:
    tool_name = str(pending.tool_name or "").strip()
    if not tool_name:
        return ""
    details: list[str] = [
        f"Retry on the next turn with `{tool_name}` immediately.",
        "Do not restart the analysis or re-read the same evidence unless the patch/tool arguments truly need new context.",
    ]
    path = str(pending.args.get("path") or pending.args.get("target_path") or "").strip()
    if path:
        details.append(f"Target path: `{path}`.")
    command = str(pending.args.get("command") or "").strip()
    if command:
        details.append(f"Reuse this command: `{command}`.")
    return " ".join(details)


def _strip_hidden_chat_terminal_completion_calls(
    graph_state: GraphRunState,
    harness: Any,
) -> bool:
    if graph_state.run_mode != "chat" or not graph_state.pending_tool_calls:
        return False

    assistant_text = str(graph_state.last_assistant_text or "").strip()
    if not assistant_text:
        return False

    suppressed_reason = str(
        getattr(harness.state, "scratchpad", {}).get("_chat_tools_suppressed_reason") or ""
    ).strip()
    if suppressed_reason != "non_lookup_chat_terminal_only":
        return False

    model_calls = [
        pending
        for pending in graph_state.pending_tool_calls
        if str(getattr(pending, "source", "model") or "model").strip().lower() == "model"
    ]
    if len(model_calls) != len(graph_state.pending_tool_calls):
        return False
    if any(pending.tool_name not in _CHAT_SYNTHETIC_TERMINAL_TOOLS for pending in model_calls):
        return False

    tool_exposure = resolve_turn_tool_exposure(harness, graph_state.run_mode)
    allowed_tool_names = {
        str(name).strip()
        for name in tool_exposure.get("names", [])
        if str(name).strip()
    }
    if any(pending.tool_name in allowed_tool_names for pending in model_calls):
        return False

    blocked_tools = [pending.tool_name for pending in model_calls]
    graph_state.pending_tool_calls = []
    harness._runlog(
        "chat_hidden_terminal_tool_ignored",
        "ignored hidden chat terminal tool and finalized from assistant prose",
        blocked_tools=blocked_tools,
        suppressed_reason=suppressed_reason,
    )
    return True


async def interpret_model_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    strategy = harness.state.strategy
    if not isinstance(strategy, dict):
        strategy = harness.state.scratchpad.get("strategy", {})
    if not isinstance(strategy, dict):
        strategy = {}
    thought_arch = strategy.get("thought_architecture")

    if graph_state.final_result is not None:
        return LoopRoute.FINALIZE

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
                                role="user",
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
                        harness._runlog("phase_transition", "auto-transition to VERIFICATION via premature completion attempt")
                    else:
                        phase_bits = phase_contract(current_phase)
                        harness.state.append_message(ConversationMessage(
                            role="user",
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
                        role="user",
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
                        role="user",
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
            repair_attempts = int(harness.state.scratchpad.get("_schema_validation_nudges", 0))
            _nodes._remember_write_session_schema_failure(
                harness,
                pending,
                details,
                error_message=err_msg,
                nudge_count=repair_attempts + 1,
            )
            if repair_attempts >= 1:
                harness.state.recent_errors.append(err_msg)
                harness._runlog(
                    "tool_call_validation_error",
                    "tool call missing required arguments",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    required_fields=details.get("required_fields", []),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ERROR,
                        content=err_msg,
                        data={"error_type": "schema_validation_error", **details},
                    ),
                )
                graph_state.pending_tool_calls = []
                graph_state.final_result = harness._failure(
                    err_msg,
                    error_type="schema_validation_error",
                    details=details,
                )
                graph_state.error = graph_state.final_result["error"]
                return LoopRoute.FINALIZE

            harness.state.scratchpad["_schema_validation_nudges"] = repair_attempts + 1
            harness.state.recent_errors.append(err_msg)
            repair_message = err_msg
            if not details.get("offending_field"):
                repair_message = _nodes._build_schema_repair_message(
                    harness,
                    pending,
                    details.get("required_fields", []),
                )
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=repair_message,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "schema_validation",
                        "tool_name": pending.tool_name,
                        "required_fields": details.get("required_fields", []),
                        "tool_call_id": pending.tool_call_id,
                        "target_path": target_path,
                    },
                )
            )
            harness._runlog(
                "tool_call_repair",
                "injected schema repair nudge",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                required_fields=details.get("required_fields", []),
                retry_count=repair_attempts + 1,
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=repair_message,
                    data={
                        "repair_kind": "schema_validation",
                        "tool_name": pending.tool_name,
                        "tool_call_id": pending.tool_call_id,
                        "required_fields": details.get("required_fields", []),
                        "retry_count": repair_attempts + 1,
                        "target_path": target_path,
                    },
                ),
            )
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
                if block_count >= 2 and _working_memory_signals_completion(harness):
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
                    role="user",
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
        if (
            nudges == 0
            and has_facts
            and has_tool_evidence
            and len(assistant_text) > 120
            and chat_only_auto_finalize
            and harness.state.current_phase != "repair"
            and not has_recent_errors
            and not looks_like_future_action
        ):
            harness._runlog(
                "auto_finalize",
                "prose answer with tool evidence; skipping nudge",
                text_len=len(assistant_text),
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

    if not graph_state.pending_tool_calls and (thinking_looks_like_action or text_looks_like_action_list or text_has_tool_tags or text_has_func_calls):
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
        stalls = int(harness.state.scratchpad.get("_action_stalls", 0))
        if stalls < 1:
            harness.state.scratchpad["_action_stalls"] = stalls + 1
            msg = "### SYSTEM ALERT: You identified or described a tool action, but you did not emit the JSON tool call."
            if text_has_tool_tags or text_has_func_calls:
                msg = "### FORMAT ERROR: You used text-based tool tags or functional syntax (e.g. <tool_call> or shell_exec()). This is FORBIDDEN. You MUST use the JSON block format."

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

    if not assistant_text.strip() and not graph_state.pending_tool_calls and not stream_halted:
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

    if nudges < 4 and assistant_text and not stream_halted:
        harness.state.scratchpad["_no_tool_nudges"] = nudges + 1
        msg = (
            "You reached a conclusion but did not call `task_complete`. "
            "If you are finished, you MUST call `task_complete(message='...')` with your final answer. "
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
        freeze_nudges = int(harness.state.scratchpad.get("_small_model_continue_nudges", 0))
        if freeze_nudges < 2:
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
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=msg,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "model_halt" if stream_halted or not _nodes._is_small_model(harness) else "small_model_freeze",
                        "retry_count": freeze_nudges + 1,
                    },
                )
            )
            harness._runlog(
                "small_model_freeze_recovery",
                "injected continuation nudge for a stalled model",
                retry_count=freeze_nudges + 1,
                model_name=_nodes._harness_model_name(harness),
                stream_halted=stream_halted,
            )
            harness.state.scratchpad.pop("_last_stream_halted_without_done", None)
            harness.state.scratchpad.pop("_last_stream_halt_reason", None)
            harness.state.scratchpad.pop("_last_stream_halt_details", None)
            return LoopRoute.NEXT_STEP

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


async def interpret_chat_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    scratchpad = harness.state.scratchpad
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
    low_text = assistant_text.lower()
    action_keywords = ["call", "run", "execute", "use", "using", "invok", "command", "tool"]
    html_tool_tags = ["<tool_call>", "<function=", "<parameter="]
    func_syntax = [f"{tool_name}(" for tool_name in ["shell_exec", "ssh_exec", "artifact_read", "file_read", "dir_list", "task_complete", "bash_exec"]]
    thinking_looks_like_action = any(keyword in str(graph_state.last_thinking_text or "").lower() for keyword in action_keywords)
    text_looks_like_action = any(keyword in low_text for keyword in action_keywords)
    text_has_tool_tags = any(tag in low_text for tag in html_tool_tags)
    text_has_func_calls = any(token in low_text for token in func_syntax)

    if thinking_looks_like_action or text_looks_like_action or text_has_tool_tags or text_has_func_calls:
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

        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=(
                    "Planning mode is active. Create a structured plan with `plan_set` before trying to execute anything."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "planning_mode_requires_plan_set",
                    "planner_nudge": True,
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
