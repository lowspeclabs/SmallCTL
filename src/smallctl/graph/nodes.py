from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
import time
from typing import Any

from ..client import OpenAICompatClient
from ..guards import check_guards, is_small_model_name
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..phases import normalize_phase
from ..prompts import build_planning_prompt, build_system_prompt
from ..plans import write_plan_file
from ..normalization import coerce_int as _coerce_int_value, coerce_datetime as _coerce_datetime
from ..state import clip_text_value, json_safe_value, ExecutionPlan, PlanStep
from ..write_session_fsm import new_write_session, record_write_session_event
from ..task_targets import primary_task_target_path
from ..tools.dispatcher import normalize_tool_request
from ..tools.planning import _refresh_plan_playbook_artifact
from .deps import GraphRuntimeDeps
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from ..memory.taxonomy import (
    TOOL_NOT_CALLED,
    PREMATURE_TASK_COMPLETE,
)
from .display import format_tool_result_display
from .interrupts import pause_for_plan_approval
from .model_stream import process_model_stream
from .recovery_context import build_goal_recap
from .tool_call_parser import (
    parse_tool_calls,
    _detect_empty_file_write_payload,
    _detect_missing_required_tool_arguments,
    _detect_placeholder_tool_call,
    _build_schema_repair_message,
    _ensure_chunk_write_session,
    _suggested_chunk_sections,
    _detect_repeated_tool_loop,
    _fallback_repeated_file_read,
    _fallback_repeated_artifact_read,
    _detect_hallucinated_tool_call,
    _record_tool_attempt,
    _clear_tool_attempt_history,
    _should_suppress_resolved_plan_artifact_read,
    _extract_artifact_id_from_args,
    _artifact_read_recovery_hint,
    _artifact_read_synthesis_hint,
    _clear_artifact_read_guard_state,
    _active_write_session_for_target,
    _should_enter_chunk_mode,
    _detect_oversize_write_payload,
    _repair_active_write_session_args,
    _salvage_active_write_session_append,
)
from .tool_outcomes import (
    apply_tool_outcomes,
    apply_chat_tool_outcomes,
    apply_planning_tool_outcomes,
    _chat_progress_guard_failure,
    _get_tool_execution_record,
    _shell_human_retry_hint,
    _shell_ssh_retry_hint,
    _shell_workspace_relative_retry_hint,
    _store_tool_execution_record,
    _has_matching_tool_message,
    _conversation_message_from_dict,
    _tool_envelope_from_dict,
)
from .write_recovery import (
    build_synthetic_write_args,
    can_safely_synthesize,
    recover_write_intent,
    write_recovery_kind,
    write_recovery_metadata,
)


def _get_suggested_sections(path: str) -> list[str]:
    return _suggested_chunk_sections(path)


def _increment_run_metric(
    graph_state: GraphRunState,
    name: str,
    *,
    delta: int = 1,
) -> int:
    current = int(graph_state.latency_metrics.get(name, 0) or 0)
    updated = current + delta
    graph_state.latency_metrics[name] = updated
    return updated


def _record_empty_write_retry_metric(
    graph_state: GraphRunState,
    harness: Any,
    pending: PendingToolCall,
) -> int:
    count = _increment_run_metric(graph_state, "empty_write_retry_count")
    session = getattr(harness.state, "write_session", None)
    pending_args = getattr(pending, "args", {}) or {}
    pending_path = str(pending_args.get("path") or "").strip()
    same_target = not pending_path or str(getattr(session, "write_target_path", "") or "").strip() == pending_path
    if session is not None and same_target and pending.tool_name in {"file_write", "file_append"}:
        session.write_empty_payload_retries += 1
        harness._runlog(
            "write_session_empty_payload_retry",
            "empty write payload detected during active session",
            session_id=session.write_session_id,
            tool_name=pending.tool_name,
            retry_count=session.write_empty_payload_retries,
            run_count=count,
        )
    return count


class ToolNotFoundError(Exception):
    """Raised when a tool is requested but not found in the registry."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name}")


HALLUCINATION_MAP = {
    "file_read": "artifact_read",
    "grep": "summarize_report",
    "ls": "dir_list",
}


_WRITE_SESSION_SCHEMA_FAILURE_KEY = "_last_write_session_schema_failure"


def _matching_write_session_for_pending(harness: Any, pending: PendingToolCall) -> WriteSession | None:
    if pending.tool_name not in {"file_write", "file_append"}:
        return None
    session = getattr(harness.state, "write_session", None)
    if session is None or str(getattr(session, "status", "")).strip().lower() == "complete":
        return None

    pending_args = getattr(pending, "args", {}) or {}
    session_id = str(pending_args.get("write_session_id") or "").strip()
    if session_id:
        return session if session_id == session.write_session_id else None

    target_path = str(
        pending_args.get("path")
        or primary_task_target_path(harness)
        or ""
    ).strip()
    if not target_path:
        return None
    return _active_write_session_for_target(harness, target_path)


def _remember_write_session_schema_failure(
    harness: Any,
    pending: PendingToolCall,
    details: dict[str, Any],
    *,
    error_message: str,
    nudge_count: int,
) -> None:
    session = _matching_write_session_for_pending(harness, pending)
    if session is None:
        return

    pending_args = dict(getattr(pending, "args", {}) or {})
    section_name = str(
        pending_args.get("section_name")
        or pending_args.get("section_id")
        or session.write_next_section
        or session.write_current_section
        or "imports"
    ).strip() or "imports"
    required_fields = [
        str(field)
        for field in (details.get("required_fields") or [])
        if str(field).strip()
    ]
    payload = {
        "tool_name": pending.tool_name,
        "tool_call_id": str(pending.tool_call_id or ""),
        "error_message": error_message,
        "required_fields": required_fields,
        "attempted_arg_keys": sorted(str(key) for key in pending_args.keys()),
        "target_path": str(
            pending_args.get("path")
            or details.get("target_path")
            or session.write_target_path
            or ""
        ).strip(),
        "write_session_id": session.write_session_id,
        "recommended_section_name": section_name,
        "nudge_count": int(nudge_count),
        "status": str(session.status or ""),
    }
    harness.state.scratchpad[_WRITE_SESSION_SCHEMA_FAILURE_KEY] = payload


def _planning_response_looks_like_plan(text: str) -> bool:
    normalized = (text or "").strip()
    if len(normalized) < 40:
        return False
    lowered = normalized.lower()
    markers = (
        "plan",
        "goal",
        "success criteria",
        "substep",
        "expected artifact",
        "ready for confirmation",
        "ready to proceed",
        "ready for approval",
    )
    if any(marker in lowered for marker in markers):
        return True
    return bool(re.search(r"^\|\s*\d+\s*\|", normalized, flags=re.MULTILINE))


def _planner_speaker_data(graph_state: GraphRunState, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(extra or {})
    if graph_state.run_mode == "planning" or getattr(graph_state.loop_state, "planning_mode_enabled", False):
        payload.setdefault("speaker", "planner")
    return payload


def _harness_model_name(harness: Any) -> str:
    model_name = getattr(getattr(harness, "client", None), "model", None)
    if model_name:
        return str(model_name)
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {})
    if isinstance(scratchpad, dict):
        return str(scratchpad.get("_model_name", ""))
    return ""


def _is_small_model(harness: Any) -> bool:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {})
    if isinstance(scratchpad, dict) and "_model_is_small" in scratchpad:
        return bool(scratchpad.get("_model_is_small"))
    return is_small_model_name(_harness_model_name(harness))


def _model_uses_gpt_oss_commentary_rules(harness: Any) -> bool:
    model_name = _harness_model_name(harness).strip().lower()
    return bool(
        model_name
        and any(marker in model_name for marker in ("openai/gpt-oss-20b", "gpt-oss-20b", "openai/gpt-oss"))
    )


def _recent_assistant_texts(harness: Any, *, limit: int = 2) -> list[str]:
    texts: list[str] = []
    for message in reversed(getattr(harness.state, "recent_messages", [])):
        if getattr(message, "role", "") != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        texts.append(content)
        if len(texts) >= limit:
            break
    return texts


def _looks_like_freeze_or_hang(harness: Any, assistant_text: str) -> bool:
    text = str(assistant_text or "").strip()
    if not text:
        return False
    recent = _recent_assistant_texts(harness, limit=3)
    if not recent:
        return False
    if text in recent:
        return True
    if len(recent) >= 2 and recent[0] == recent[1]:
        return True
    if len(recent) >= 3 and recent[0] == recent[1] == recent[2]:
        return True
    return False


def _build_blank_message_nudge(harness: Any, *, repeated: bool) -> str:
    goal_recap = build_goal_recap(harness)
    goal_note = f" {goal_recap}" if goal_recap else ""
    if repeated:
        return (
            "Blank Message Nudge: the last assistant turn had no text and no tool calls."
            f"{goal_note} Provide a concrete next step, emit the JSON tool call, or call `task_complete(message='...')` if finished."
        )
    return (
        "The assistant turn was empty."
        f"{goal_note} Please respond with a concrete thought or tool call; if you are finished, call `task_complete(message='...')`."
    )


def _build_small_model_continue_message(
    harness: Any,
    assistant_text: str,
    *,
    stream_halt_reason: str = "",
) -> str:
    model_name = _harness_model_name(harness)
    clipped_text, clipped = clip_text_value(str(assistant_text or "").strip(), limit=180)
    lead = "You may be frozen or hanging."
    if stream_halt_reason == "stream_ended_without_done":
        lead = "The response stream ended before a clean completion signal."
    if clipped_text:
        if stream_halt_reason == "stream_ended_without_done":
            lead = f"The response stream ended after: {clipped_text}."
        else:
            lead = f"You may be frozen or hanging after: {clipped_text}."
    if clipped:
        lead = f"{lead} [truncated]"
    model_note = f" Model: {model_name}." if model_name else ""
    goal_recap = build_goal_recap(harness)
    goal_note = f" {goal_recap}" if goal_recap else ""
    return (
        f"{lead}{model_note}{goal_note} Continue from the last concrete step within that objective. "
        "Do not restart the task; either call the next tool or emit the next JSON tool call immediately."
    )


def _task_prefers_summary_synthesis(harness: Any) -> bool:
    texts = [
        str(getattr(getattr(harness, "state", None), "run_brief", None).original_task or "")
        if getattr(getattr(harness, "state", None), "run_brief", None) is not None
        else "",
        str(getattr(getattr(harness, "state", None), "working_memory", None).current_goal or "")
        if getattr(getattr(harness, "state", None), "working_memory", None) is not None
        else "",
    ]
    current_user_task = getattr(harness, "_current_user_task", None)
    if callable(current_user_task):
        texts.append(str(current_user_task() or ""))
    merged = " ".join(text.strip().lower() for text in texts if text and text.strip())
    if not merged:
        return False
    asks_for_summary = any(keyword in merged for keyword in ("table", "summary", "summarize", "report", "overview", "present"))
    asks_about_listing = any(keyword in merged for keyword in ("list", "listing", "files", "directories", "artifact", "results", "output", "current env"))
    return asks_for_summary and asks_about_listing


def _chat_turn_signature(graph_state: GraphRunState) -> str:
    thinking_text = re.sub(r"\s+", " ", str(graph_state.last_thinking_text or "").strip())
    if thinking_text:
        return thinking_text
    assistant_text = re.sub(r"\s+", " ", str(graph_state.last_assistant_text or "").strip())
    if not assistant_text:
        return ""
    return assistant_text


def _build_repeated_chat_thinking_message(harness: Any, graph_state: GraphRunState) -> str:
    thinking_text = str(graph_state.last_thinking_text or "").strip()
    clipped_thinking, was_clipped = clip_text_value(thinking_text, limit=240)
    goal_recap = build_goal_recap(harness)
    goal_note = f" {goal_recap}" if goal_recap else ""
    repeat_note = f" Previous thinking: {clipped_thinking}{' [truncated]' if was_clipped else ''}." if clipped_thinking else ""
    return (
        "You repeated the same reasoning without making forward progress."
        f"{repeat_note}{goal_note} "
        "Do not restate the same thoughts. Continue from the last concrete step and either call the next tool "
        "or call `task_complete(message='...')` if you are actually finished."
    )


def _build_artifact_summary_exit_message(harness: Any, *, artifact_id: str = "") -> str:
    objective = str(getattr(getattr(harness, "state", None), "run_brief", None).original_task or "").strip()
    artifact_note = f" from artifact {artifact_id}" if artifact_id else ""
    objective_note = f" for `{objective}`" if objective else ""
    return (
        f"You already have enough evidence{artifact_note}{objective_note}. "
        "Produce the requested table or summary now with `task_complete(message='...')` "
        "instead of rereading or printing the same artifact again."
    )


def _should_pause_repeated_tool_loop(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name in {"dir_list", "artifact_read", "artifact_print", "artifact_grep", "file_read"}:
        return True
    return _task_prefers_summary_synthesis(harness)


def _build_repeated_tool_loop_interrupt_payload(
    *,
    harness: Any,
    graph_state: GraphRunState,
    pending: PendingToolCall,
    repeat_error: str,
) -> dict[str, Any]:
    question = (
        "I hit a repeated tool loop while working on the current task. "
        "Reply `continue` to resume from the current evidence, or send a more specific next instruction."
    )
    guidance = (
        "Repeated tool loop detected. Continue from the evidence already in context. "
        "Do not retry the same read/list/print command unless you are paging to unseen lines."
    )
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if pending.tool_name in {"artifact_read", "artifact_print"} and _task_prefers_summary_synthesis(harness):
        guidance = _build_artifact_summary_exit_message(harness, artifact_id=artifact_id)
    return {
        "kind": "repeated_tool_loop_resume",
        "question": question,
        "current_phase": harness.state.current_phase,
        "active_profiles": list(harness.state.active_tool_profiles),
        "thread_id": graph_state.thread_id,
        "tool_name": pending.tool_name,
        "arguments": json_safe_value(pending.args),
        "guard": "repeated_tool_loop",
        "guard_error": repeat_error,
        "guidance": guidance,
    }


def _build_authoring_budget_message(harness: Any, phase: str, pending_tool_name: str) -> str:
    phase_label = phase or "author"
    return (
        f"Authoring budget applied in {phase_label} phase. "
        f"Focus on one concrete action at a time, starting with `{pending_tool_name}`. "
        "Finish that action, inspect the result, then decide the next step on the next turn."
    )


def _build_file_read_recovery_message(harness: Any, pending: PendingToolCall) -> str:
    raw_path = str(pending.args.get("path", "") or "").strip()
    if not raw_path:
        return (
            "You already read this file. Do not reread it; use the evidence you already have "
            "to patch the file, run the focused test, or move on."
        )

    path = Path(raw_path)
    cwd = getattr(harness.state, "cwd", None)
    if not path.is_absolute():
        base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
        try:
            path = (base / path).resolve()
        except Exception:
            path = base / path
    else:
        try:
            path = path.resolve()
        except Exception:
            pass

    return (
        f"You already read `{path}`. Do not reread the same file; use the evidence you already "
        "have to patch it, run the focused test, or move on."
    )


def _apply_small_model_authoring_budget(harness: Any, graph_state: GraphRunState) -> bool:
    if not _is_small_model(harness):
        return False
    contract_phase = harness.state.contract_phase()
    if contract_phase not in {"author", "repair"}:
        return False
    if len(graph_state.pending_tool_calls) <= 1:
        return False

    config = getattr(harness, "config", None)
    allow_multi = config.allow_multi_section_turns_for_small_edits if config else True
    
    if allow_multi:
        # Check if all pending calls are file_write in the same session OR metadata updates
        session = getattr(harness.state, "write_session", None)
        if session and all(
            (p.tool_name == "file_write" and p.args.get("write_session_id") == session.write_session_id)
            or p.tool_name in {"loop_status", "plan_set", "context_briefs_list", "artifact_list"}
            for p in graph_state.pending_tool_calls
        ):
            return False

    first_pending = graph_state.pending_tool_calls[0]
    remaining = graph_state.pending_tool_calls[1:]
    harness.state.scratchpad["_authoring_action_budget_nudges"] = int(
        harness.state.scratchpad.get("_authoring_action_budget_nudges", 0)
    ) + 1
    harness.state.scratchpad["_authoring_action_budget_phase"] = contract_phase
    harness.state.scratchpad["_authoring_action_budget_tool"] = first_pending.tool_name
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=_build_authoring_budget_message(harness, contract_phase, first_pending.tool_name),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "authoring_budget",
                "budget_phase": contract_phase,
                "tool_name": first_pending.tool_name,
                "suppressed_tool_count": len(remaining),
            },
        )
    )
    harness._runlog(
        "authoring_action_budget",
        "restricted multi-call authoring turn to a single action",
        phase=contract_phase,
        first_tool=first_pending.tool_name,
        suppressed=len(remaining),
    )
    graph_state.pending_tool_calls = [first_pending]
    return True


def _extract_plan_steps_from_text(text: str) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        table_match = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", stripped)
        if table_match:
            step_id = f"P{table_match.group(1)}"
            title = table_match.group(2).strip()
            description = table_match.group(3).strip()
            steps.append(PlanStep(step_id=step_id, title=title, description=description))
            continue
        numbered_match = re.match(r"^\d+[.)]\s*(.+)$", stripped)
        if numbered_match:
            step_id = f"P{len(steps) + 1}"
            steps.append(PlanStep(step_id=step_id, title=numbered_match.group(1).strip()))
    return steps


def _synthesize_plan_from_text(harness: Any, text: str) -> ExecutionPlan | None:
    assistant_text = (text or "").strip()
    if not assistant_text or not _planning_response_looks_like_plan(assistant_text):
        return None
    goal = str(harness.state.run_brief.original_task or "").strip() or assistant_text.splitlines()[0].strip()
    steps = _extract_plan_steps_from_text(assistant_text)
    if not steps:
        steps = [PlanStep(step_id="P1", title="Review proposed plan")]
    return ExecutionPlan(
        plan_id=f"plan-{uuid.uuid4().hex[:8]}",
        goal=goal,
        summary=assistant_text,
        steps=steps[:6],
        status="draft",
        approved=False,
    )


def _persist_planning_playbook(harness: Any, plan: ExecutionPlan) -> None:
    try:
        _refresh_plan_playbook_artifact(state=harness.state, harness=harness, plan=plan)
    except Exception as exc:
        harness.log.warning("failed to persist plan playbook artifact: %s", exc)


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
    harness.state.scratchpad.pop("_tool_attempt_history", None)
    resolved_task = task
    resolve_followup = getattr(harness, "_resolve_followup_task", None)
    if callable(resolve_followup):
        candidate = str(resolve_followup(task) or "").strip()
        if candidate:
            resolved_task = candidate
    maybe_reset = getattr(harness, "_maybe_reset_for_new_task", None)
    if callable(maybe_reset):
        maybe_reset(resolved_task)
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
    # Reset step counter if user explicitly asks to continue after a guard trip
    continue_like = human_input.strip().lower() in ("continue", "keep going", "proceed")
    is_continue_like = getattr(harness, "_is_continue_like_followup", None)
    if callable(is_continue_like):
        continue_like = bool(is_continue_like(human_input))
    if continue_like:
        harness._runlog("step_count_reset", "resetting step count for continuation", old_count=harness.state.step_count)
        harness.state.step_count = 0
        harness.state.inactive_steps = 0

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
    harness.state.step_count += 1
    harness.state.decay_experiences()
    harness.dispatcher.phase = normalize_phase(harness.state.current_phase)
    harness.state.current_phase = harness.dispatcher.phase

    # Pre-arm chunked authoring before the first write attempt on risky write-first tasks.
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
            if (
                recovery_hint is not None
                and graph_state.run_mode != "chat"
            ):
                artifact_id, query = recovery_hint
                _clear_artifact_read_guard_state(harness, artifact_id)
                graph_state.pending_tool_calls = [
                    PendingToolCall(
                        tool_name="artifact_grep",
                        args={
                            "artifact_id": artifact_id,
                            "query": query,
                        },
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
        elif (
            recovery_hint is not None
            and graph_state.run_mode == "chat"
        ):
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

    # Risk-Based Chunk Mode: Intercept large writes from small models and suggest a session
    if graph_state.pending_tool_calls and not guard_error:
        for pending in graph_state.pending_tool_calls:
            if _should_enter_chunk_mode(harness, pending):
                target_path = str(pending.args.get("path") or "")
                content = str(pending.args.get("content") or "")
                
                suggestions = _get_suggested_sections(target_path)
                from ..tools.fs import infer_write_session_intent, new_write_session_id

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
                from .tool_outcomes import _register_write_session_stage_artifact
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
            
            # Oversize Write Nudge: Reject one-shot writes that are way too big
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


async def prepare_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = build_system_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=_available_tool_names(harness, mode="loop"),
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
        indexer_mode=bool(getattr(harness, "_indexer", False)),
    )
    try:
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except RuntimeError as exc:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
        )
        graph_state.final_result = harness._failure(str(exc), error_type="prompt_budget")
        graph_state.error = graph_state.final_result["error"]
        return None


async def prepare_chat_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    chat_tool_names = _available_tool_names(harness, mode="chat")
    system_prompt = build_system_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=chat_tool_names,
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
        indexer_mode=bool(getattr(harness, "_indexer", False)),
    )
    system_prompt = f"{system_prompt} You may use available tools when needed to answer accurately."
    if "shell_exec" in chat_tool_names:
        system_prompt = (
            f"{system_prompt} "
            "SHELL: `shell_exec` is available for command execution, but it requires user approval before execution."
        )
    try:
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except RuntimeError as exc:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
        )
        graph_state.final_result = harness._failure(str(exc), error_type="prompt_budget")
        graph_state.error = graph_state.final_result["error"]
        return None


async def prepare_planning_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = build_planning_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=_available_tool_names(harness, mode="planning"),
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
        indexer_mode=bool(getattr(harness, "_indexer", False)),
    )
    try:
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Gathering planning facts...",
                data={"status_activity": "gathering facts..."},
            ),
        )
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except RuntimeError as exc:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
        )
        graph_state.final_result = harness._failure(str(exc), error_type="prompt_budget")
        graph_state.error = graph_state.final_result["error"]
        return None


def load_index_manifest(cwd: str) -> dict[str, Any] | None:
    path = Path(cwd) / ".smallctl" / "index_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def select_loop_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    profiles = set(harness.state.active_tool_profiles)
    phase = harness.state.current_phase
    tools = harness.registry.export_openai_tools(
        phase=phase,
        mode="loop",
        profiles=profiles,
    )
    harness.log.info("select_loop_tools: phase=%s profiles=%s count=%d", phase, profiles, len(tools))
    return tools


def select_chat_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return deps.harness._chat_mode_tools()


def select_indexer_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    # Only allow core read tools and indexer write tools
    indexer_profiles = {"indexer", "core", "support"}
    return harness.registry.export_openai_tools(
        mode="indexer",
        profiles=indexer_profiles,
    )


def select_planning_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    return harness.registry.export_openai_tools(
        phase=harness.state.current_phase,
        mode="planning",
        profiles=set(harness.state.active_tool_profiles),
    )


async def prepare_indexer_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = (
        "You are a high-performance codebase indexer. Your goal is to systematically extract and record every symbol, reference, and import. "
        "EFFICIENCY IS CRITICAL: Use `index_batch_write` to submit ALL symbols, imports, and references for a file segment in a single tool call. "
        "Do not call individual write tools if you have multiple records to submit. "
        "PIPELINE: 1. List files. 2. Read file segments. 3. Extract all relevant metadata. 4. Call `index_batch_write` with the full collection. 5. Move to the next segment or file. "
        "Once all relevant files are indexed, call `index_finalize()`."
    )
    try:
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except Exception as exc:
        graph_state.error = str(exc)
        return None


def _available_tool_names(harness: Any, *, mode: str) -> list[str]:
    if mode == "chat":
        tools = harness._chat_mode_tools()
    else:
        tools = harness.registry.export_openai_tools(
            phase=harness.state.current_phase,
            mode=mode,
            profiles=set(harness.state.active_tool_profiles),
        )
    return [
        str(entry["function"]["name"])
        for entry in tools
        if isinstance(entry, dict)
        and "function" in entry
        and isinstance(entry["function"], dict)
        and "name" in entry["function"]
    ]


async def model_call(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    harness = deps.harness
    result = await process_model_stream(graph_state, deps, messages=messages, tools=tools)
    if graph_state.final_result is not None:
        return

    halt_detected = bool(getattr(result, "halted", False))
    halt_reason = str(getattr(result, "halt_reason", "") or "").strip()
    halt_details = json_safe_value(getattr(result, "halt_details", {}) or {})
    if halt_detected:
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        if halt_reason:
            harness.state.scratchpad["_last_stream_halt_reason"] = halt_reason
        if isinstance(halt_details, dict):
            harness.state.scratchpad["_last_stream_halt_details"] = halt_details
    else:
        harness.state.scratchpad.pop("_last_stream_halted_without_done", None)
        harness.state.scratchpad.pop("_last_stream_halt_reason", None)
        harness.state.scratchpad.pop("_last_stream_halt_details", None)

    usage_payload = json_safe_value(result.usage)
    if not isinstance(usage_payload, dict):
        usage_payload = {}
    if usage_payload:
        harness._apply_usage(usage_payload)

    graph_state.last_usage = usage_payload
    graph_state.last_assistant_text = result.stream.assistant_text
    graph_state.last_thinking_text = result.stream.thinking_text

    duration = result.duration
    ttft = result.ttft

    graph_state.latency_metrics["model_call_duration_sec"] = round(duration, 3)
    graph_state.latency_metrics["ttft_sec"] = round(ttft, 3)

    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.METRICS,
            content=f"Model call: {duration:.2f}s (TTFT: {ttft:.2f}s)",
            data={
                "duration_sec": duration,
                "ttft_sec": ttft,
                "usage": usage_payload,
            }
        ),
    )

    parse_result = parse_tool_calls(
        result.stream,
        result.timeline,
        graph_state,
        deps,
        model_name=getattr(harness.client, "model", None),
    )
    graph_state.pending_tool_calls = parse_result.pending_tool_calls
    graph_state.last_assistant_text = parse_result.final_assistant_text

    if parse_result.final_assistant_text.strip() != result.stream.assistant_text.strip():
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ASSISTANT,
                content=parse_result.final_assistant_text.strip(),
                data=_planner_speaker_data(
                    graph_state,
                    {"kind": "replace"},
                ),
            ),
        )

    if not graph_state.pending_tool_calls:
        harness.state.inactive_steps += 1
        harness.state.scratchpad["_consecutive_idle"] = int(harness.state.scratchpad.get("_consecutive_idle", 0)) + 1
    else:
        harness.state.scratchpad["_consecutive_idle"] = 0

    if int(harness.state.scratchpad.get("_consecutive_idle", 0)) >= 2:
        nudge = (
            "\n[SYSTEM NUDGE]: You have provided 2 consecutive turns without any tool actions. "
            "Please focus on making concrete progress towards the goal (explore/execute) "
            "rather than providing high-level summaries or explanation. "
            "If you are finished, use the task_complete tool."
        )
        harness.state.append_message(ConversationMessage(role="system", content=nudge))
        harness.state.scratchpad["_consecutive_idle"] = 1

    if parse_result.final_assistant_text:
        harness._runlog(
            "model_output",
            "assistant output complete",
            assistant_text=parse_result.final_assistant_text,
        )
    if parse_result.final_assistant_text or result.stream.tool_calls:
        harness._record_assistant_message(
            assistant_text=parse_result.final_assistant_text,
            tool_calls=result.stream.tool_calls,
            speaker="planner" if graph_state.run_mode == "planning" or harness.state.planning_mode_enabled else None,
            hidden_from_prompt=_model_uses_gpt_oss_commentary_rules(harness),
        )
        harness._log_conversation_state("assistant_message")
    if result.stream.thinking_text:
        harness._runlog(
            "model_thinking",
            "thinking output complete",
            thinking_text=result.stream.thinking_text,
        )
    for entry in result.timeline:
        if entry.kind == "tool_call":
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.TOOL_CALL,
                    content=entry.content,
                    data=_planner_speaker_data(graph_state, entry.data),
                ),
            )


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

        # Throttling: only distill if we haven't done it too much in this run
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
                # Attach for pruning in history
                if harness.state.recent_messages:
                    last_msg = harness.state.recent_messages[-1]
                    if last_msg.role == "assistant":
                        last_msg.metadata["thinking_insight"] = insight

    if thought_arch == "multi_phase_discovery":
        current_phase = harness.state.current_phase
        if graph_state.pending_tool_calls:
            # If in explore, only allow gathering tools. Block task_complete/file_write.
            if current_phase == "explore":
                blocked = ["task_complete", "task_fail", "file_write"]
                original_calls = list(graph_state.pending_tool_calls)
                graph_state.pending_tool_calls = [c for c in original_calls if c.tool_name not in blocked]

                # If the model ONLY called blocked completion tools, it's effectively a 'no tool' completion attempt.
                if original_calls and not graph_state.pending_tool_calls:
                    if all(c.tool_name in ["task_complete", "task_fail"] for c in original_calls):
                        # Depth Lever: Rejection if too early
                        if harness.state.step_count < harness.config.min_exploration_steps:
                             harness.state.append_message(ConversationMessage(
                                 role="user",
                                 content=f"ANTI-LAZINESS: You are trying to finish at step {harness.state.step_count}, but this task requires at least {harness.config.min_exploration_steps} discovery steps. Perform more deep-dive exploration before concluding."
                             ))
                             return LoopRoute.NEXT_STEP

                        # Relax: If it's trying to finish, it probably has the info.
                        # Allow transition to verify if it hasn't happened yet.
                        if current_phase == "explore":
                             harness.state.current_phase = "verify"
                             harness._runlog("phase_transition", "auto-transition to VERIFICATION via premature completion attempt")
                             # Fall through to transition logic below
                if len(graph_state.pending_tool_calls) < len(original_calls):
                    harness.state.append_message(ConversationMessage(
                        role="user",
                        content="You are still in the DISCOVERY phase. Gathering complete? Call `memory_update(section='known_facts', content='...')` to transition to VERIFICATION."
                    ))
                    harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                    return LoopRoute.NEXT_STEP

            # If in verify, only allow memory_update
            if current_phase == "verify":
                 blocked = ["task_complete", "task_fail", "file_write", "long_context_lookup", "summarize_report", "artifact_read", "grep"]
                 if any(c.tool_name in blocked for c in graph_state.pending_tool_calls):
                     graph_state.pending_tool_calls = [c for c in graph_state.pending_tool_calls if c.tool_name not in blocked]
                     harness.state.append_message(ConversationMessage(
                        role="user",
                        content="You are in VERIFICATION. List all required constants via `memory_update` then proceed to SYNTHESIS."
                    ))
                     harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                     return LoopRoute.NEXT_STEP

        # Transition logic based on results
        for record in graph_state.last_tool_results:
            if record.tool_name == "memory_update" and record.result.success:
                if current_phase == "explore":
                    harness.state.current_phase = "verify"
                    harness._runlog("phase_transition", "transition to VERIFICATION")
                    harness.state.append_message(ConversationMessage(
                        role="user",
                        content="Transitioning to VERIFICATION phase. Please list/verify all constants."
                    ))
                    return LoopRoute.NEXT_STEP
                elif current_phase == "verify":
                    harness.state.current_phase = "execute"
                    harness._runlog("phase_transition", "transition to SYNTHESIS")
                    harness.state.append_message(ConversationMessage(
                        role="user",
                        content="Transitioning to SYNTHESIS phase. You may now implement the final answer and call task_complete."
                   ))
                    return LoopRoute.NEXT_STEP

    if graph_state.pending_tool_calls:
        incomplete_payload = harness.state.scratchpad.get("_last_incomplete_tool_call")
        fallback_assistant_text = str(
            harness.state.scratchpad.get("_last_text_write_fallback_assistant_text") or ""
        )
        recovery_assistant_text = graph_state.last_assistant_text or fallback_assistant_text
        partial_tool_calls = []
        if isinstance(incomplete_payload, dict):
            raw_partial_calls = incomplete_payload.get("partial_tool_calls_raw")
            if isinstance(raw_partial_calls, list):
                partial_tool_calls = raw_partial_calls
        for pending in graph_state.pending_tool_calls:
            _repair_active_write_session_args(
                harness,
                pending,
                assistant_text=recovery_assistant_text,
            )
            if pending.tool_name in {"file_write", "file_append"}:
                intent = recover_write_intent(
                    harness=harness,
                    pending=pending,
                    assistant_text=recovery_assistant_text,
                    partial_tool_calls=partial_tool_calls,
                )
                if intent is not None:
                    _increment_run_metric(graph_state, "write_recovery_attempt_count")
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
                        pending.tool_name = "file_write"
                        pending.args = build_synthetic_write_args(intent)
                        pending.raw_arguments = json.dumps(pending.args, ensure_ascii=True, sort_keys=True)
                        _increment_run_metric(graph_state, "write_recovery_success_count")
                        if "assistant_fenced_code" in intent.evidence or "assistant_inline_tool_block" in intent.evidence:
                            _increment_run_metric(graph_state, "write_recovery_from_assistant_code_count")
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
                        _increment_run_metric(graph_state, "write_recovery_declined_count")
                        if str(intent.confidence).strip().lower() == "low":
                            _increment_run_metric(graph_state, "write_recovery_low_confidence_count")
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
            missing_args = _detect_placeholder_tool_call(harness, pending)
            if missing_args is None:
                missing_args = _detect_empty_file_write_payload(harness, pending)
            if missing_args is not None and pending.tool_name in {"file_write", "file_append"}:
                retry_count = _record_empty_write_retry_metric(graph_state, harness, pending)
                salvaged = _salvage_active_write_session_append(harness, pending)
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
                missing_args = _detect_missing_required_tool_arguments(harness, pending)
            if missing_args is None:
                continue
            err_msg, details = missing_args
            target_path = None
            if pending.tool_name == "file_write":
                target_path = primary_task_target_path(harness)
                if target_path:
                    _ensure_chunk_write_session(harness, target_path)
                if target_path:
                    details = dict(details)
                    details["target_path"] = target_path
            repair_attempts = int(harness.state.scratchpad.get("_schema_validation_nudges", 0))
            _remember_write_session_schema_failure(
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
                repair_message = _build_schema_repair_message(
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

        return LoopRoute.DISPATCH_TOOLS

    nudges = int(harness.state.scratchpad.get("_no_tool_nudges", 0))
    assistant_text = graph_state.last_assistant_text or ""

    if (
        graph_state.run_mode == "planning"
        or harness.state.planning_mode_enabled
    ):
        synthesized_plan = _synthesize_plan_from_text(harness, assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            _persist_planning_playbook(harness, synthesized_plan)
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
        and _planning_response_looks_like_plan(assistant_text)
    ):
        synthesized_plan = _synthesize_plan_from_text(harness, assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            _persist_planning_playbook(harness, synthesized_plan)
            harness.state.touch()
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is not None:
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

    if assistant_text:
        # Let a substantial, tool-backed prose answer finalize before action-format heuristics can
        # mistake its reasoning text for a missing tool call.
        has_facts = bool(harness.state.working_memory.known_facts)
        has_tool_evidence = any(
            message.role == "tool"
            for message in harness.state.recent_messages[-6:]
        )
        if (
            nudges == 0
            and has_facts
            and has_tool_evidence
            and len(assistant_text) > 120
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

    # Action Stall Guard: Model justified an action but didn't emit the JSON.
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
            synthesized_plan = _synthesize_plan_from_text(harness, assistant_text)
            if synthesized_plan is not None:
                harness.state.draft_plan = synthesized_plan
                harness.state.active_plan = synthesized_plan
                harness.state.planning_mode_enabled = True
                harness.state.sync_plan_mirror()
                _persist_planning_playbook(harness, synthesized_plan)
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
                 content=f"{msg}\n\nDO NOT repeat your earlier findings or analysis. Just generate the JSON block immediately after your reasoning. Do not describe what you are going to do; just DO it."
             ))
             harness._record_experience(
                 tool_name="reasoning",
                 result=ToolEnvelope(success=False, error=msg),
                 source="guarded_stall",
                 notes=f"Model described action but missed JSON format. Failure mode: {TOOL_NOT_CALLED}",
             )
             harness._runlog("action_stall", "improper tool format or description", stalls=stalls+1, has_tags=text_has_tool_tags)
             return LoopRoute.NEXT_STEP

    # Guard against premature success for "hello" when real work was requested
    if not graph_state.pending_tool_calls and "hello" in low_text and ("task" in low_text or "complete" in low_text):
        if any(v in harness.state.run_brief.original_task.lower() for v in ["ping", "list", "read", "run"]):
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content="### MISSION CHECK: You mention 'hello' or completing the greeting, but a real task is still pending. DO NOT finish yet. Proceed with the primary mission.",
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
            msg = _build_blank_message_nudge(harness, repeated=blank_nudges >= 1)
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
                metadata={"is_recovery_nudge": True}
            )
        )
        harness._runlog("no_tool_recovery", "injected recovery nudge", nudge_count=nudges+1)
        return LoopRoute.NEXT_STEP

    if stream_halted or _looks_like_freeze_or_hang(harness, assistant_text):
        freeze_nudges = int(harness.state.scratchpad.get("_small_model_continue_nudges", 0))
        if freeze_nudges < 2:
            harness.state.scratchpad["_small_model_continue_nudges"] = freeze_nudges + 1
            halt_reason = str(harness.state.scratchpad.get("_last_stream_halt_reason", "") or "")
            msg = _build_small_model_continue_message(
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
                        "recovery_kind": "model_halt" if stream_halted or not _is_small_model(harness) else "small_model_freeze",
                        "retry_count": freeze_nudges + 1,
                    },
                )
            )
            harness._runlog(
                "small_model_freeze_recovery",
                "injected continuation nudge for a stalled model",
                retry_count=freeze_nudges + 1,
                model_name=_harness_model_name(harness),
                stream_halted=stream_halted,
            )
            harness.state.scratchpad.pop("_last_stream_halted_without_done", None)
            harness.state.scratchpad.pop("_last_stream_halt_reason", None)
            harness.state.scratchpad.pop("_last_stream_halt_details", None)
            return LoopRoute.NEXT_STEP

    # If we are stuck but already have tool evidence, finalize with the current answer.
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
    signature = _chat_turn_signature(graph_state)
    prior_signature = str(scratchpad.get("_chat_last_turn_signature") or "")
    if signature:
        scratchpad["_chat_last_turn_signature"] = signature

    if graph_state.pending_tool_calls:
        return LoopRoute.DISPATCH_TOOLS

    if signature and signature == prior_signature:
        nudge_key = f"{graph_state.thread_id}:{signature}"
        if scratchpad.get("_chat_repeated_thinking_nudged") != nudge_key:
            scratchpad["_chat_repeated_thinking_nudged"] = nudge_key
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=_build_repeated_chat_thinking_message(harness, graph_state),
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

    graph_state.final_result = {
        "status": "chat_completed",
        "assistant": graph_state.last_assistant_text,
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
        synthesized_plan = _synthesize_plan_from_text(harness, graph_state.last_assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            _persist_planning_playbook(harness, synthesized_plan)
            harness.state.touch()
            await pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=(
                    "Planning mode is active. Create a structured plan with `plan_set` before trying to execute anything."
                ),
                metadata={"is_recovery_nudge": True, "planner_nudge": True},
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


async def dispatch_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> None:
    harness = deps.harness
    graph_state.last_tool_results = []
    dispatch_start = time.perf_counter()
    if _apply_small_model_authoring_budget(harness, graph_state):
        graph_state.last_tool_results = []
        graph_state.pending_tool_calls = graph_state.pending_tool_calls[:1]
    for pending in graph_state.pending_tool_calls:
        registry = getattr(harness, "registry", None)
        if registry is not None:
            normalized_tool_name, normalized_args, intercepted_result, _ = normalize_tool_request(
                registry,
                pending.tool_name,
                pending.args,
                phase=getattr(getattr(harness, "dispatcher", None), "phase", None),
            )
        else:
            normalized_tool_name, normalized_args, intercepted_result = (
                pending.tool_name,
                pending.args,
                None,
            )
        if intercepted_result is None:
            pending.tool_name = normalized_tool_name
            pending.args = normalized_args

        repeat_error = _detect_repeated_tool_loop(harness, pending)

        # Record raw intent AFTER loop check (so check doesn't see self)
        # but BEFORE any fallbacks mutate 'pending'
        _record_tool_attempt(harness, pending)

        if repeat_error is not None:
            shell_human_hint = _shell_human_retry_hint(harness, pending)
            if shell_human_hint is not None:
                if harness.state.scratchpad.get("_shell_human_retry_nudged") != shell_human_hint:
                    harness.state.scratchpad["_shell_human_retry_nudged"] = shell_human_hint
                    harness.state.append_message(
                        ConversationMessage(
                            role="system",
                            content=shell_human_hint,
                            metadata={
                                "is_recovery_nudge": True,
                                "recovery_kind": "shell_exec",
                                "recovery_mode": "human_input",
                            },
                        )
                    )
                    harness._runlog(
                        "shell_exec_human_retry_nudge",
                        "nudged model away from retrying a human-gated shell command",
                        step=harness.state.step_count,
                        tool_name=pending.tool_name,
                        arguments=json_safe_value(pending.args),
                        guard_error=repeat_error,
                    )
                    graph_state.pending_tool_calls = []
                    graph_state.last_tool_results = []
                    return

            ssh_shell_hint = _shell_ssh_retry_hint(harness, pending)
            if ssh_shell_hint is not None:
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=ssh_shell_hint,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "ssh_exec",
                            "recovery_mode": "routing",
                        },
                    )
                )
                harness._runlog(
                    "shell_exec_ssh_routing_nudge",
                    "nudged model to use ssh_exec for SSH commands",
                    step=harness.state.step_count,
                    tool_name=pending.tool_name,
                    arguments=json_safe_value(pending.args),
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                return

            workspace_relative_shell_hint = _shell_workspace_relative_retry_hint(harness, pending)
            if workspace_relative_shell_hint is not None:
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=workspace_relative_shell_hint,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "shell_exec",
                            "recovery_mode": "workspace_relative_path",
                        },
                    )
                )
                harness._runlog(
                    "shell_exec_workspace_relative_nudge",
                    "nudged model away from retrying a root-level /temp path",
                    step=harness.state.step_count,
                    tool_name=pending.tool_name,
                    arguments=json_safe_value(pending.args),
                    guard_error=repeat_error,
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                return

            if pending.tool_name == "file_read":
                recovered = _fallback_repeated_file_read(harness, pending)
                if recovered is not None:
                    log_kv(
                        harness.log,
                        logging.INFO,
                        "harness_repeated_tool_loop_recovered",
                        step=harness.state.step_count,
                        original_tool_name=pending.tool_name,
                        recovered_tool_name=recovered.tool_name,
                        recovered_args=recovered.args,
                    )
                    pending = recovered
                    repeat_error = None
                else:
                    file_read_fingerprint = json.dumps(json_safe_value(pending.args), sort_keys=True)
                    if harness.state.scratchpad.get("_file_read_recovery_nudged") != file_read_fingerprint:
                        harness.state.scratchpad["_file_read_recovery_nudged"] = file_read_fingerprint
                        file_read_hint = _build_file_read_recovery_message(harness, pending)
                        harness.state.append_message(
                            ConversationMessage(
                                role="system",
                                content=file_read_hint,
                                metadata={
                                    "is_recovery_nudge": True,
                                    "recovery_kind": "file_read",
                                    "recovery_mode": "read_once_then_act",
                                    "path": str(pending.args.get("path", "") or ""),
                                },
                            )
                        )
                        harness._runlog(
                            "file_read_recovery_nudge",
                            "nudged model away from rereading the same file",
                            step=harness.state.step_count,
                            tool_name=pending.tool_name,
                            arguments=json_safe_value(pending.args),
                            guard_error=repeat_error,
                        )
                        graph_state.pending_tool_calls = []
                        graph_state.last_tool_results = []
                        return

            if repeat_error is not None:
                summary_exit_artifact_id = _extract_artifact_id_from_args(pending.args)
                if (
                    pending.tool_name in {"artifact_read", "artifact_print"}
                    and _task_prefers_summary_synthesis(harness)
                ):
                    nudge_key = f"{pending.tool_name}:{summary_exit_artifact_id or 'summary_exit'}"
                    if harness.state.scratchpad.get("_artifact_summary_exit_nudged") != nudge_key:
                        harness.state.scratchpad["_artifact_summary_exit_nudged"] = nudge_key
                        harness.state.append_message(
                            ConversationMessage(
                                role="system",
                                content=_build_artifact_summary_exit_message(
                                    harness,
                                    artifact_id=summary_exit_artifact_id,
                                ),
                                metadata={
                                    "is_recovery_nudge": True,
                                    "recovery_kind": "artifact_summary_exit",
                                    "artifact_id": summary_exit_artifact_id,
                                    "recovery_mode": "synthesis",
                                },
                            )
                        )
                        harness._runlog(
                            "artifact_summary_exit_nudge",
                            "nudged model to answer from current artifact evidence",
                            step=harness.state.step_count,
                            artifact_id=summary_exit_artifact_id,
                            guard_error=repeat_error,
                        )
                        graph_state.pending_tool_calls = []
                        graph_state.last_tool_results = []
                        return

                synthesis_artifact_id = _artifact_read_synthesis_hint(harness, repeat_error)
                if synthesis_artifact_id is not None and pending.tool_name == "artifact_read":
                    if harness.state.scratchpad.get("_artifact_read_synthesis_nudged") != synthesis_artifact_id:
                        harness.state.scratchpad["_artifact_read_synthesis_nudged"] = synthesis_artifact_id
                        synth_msg = (
                            f"You already tried `artifact_read` and `artifact_grep` on artifact {synthesis_artifact_id}. "
                            "Synthesize the answer from the evidence you already have instead of reading the same artifact again."
                        )
                        harness.state.append_message(
                            ConversationMessage(
                                role="system",
                                content=synth_msg,
                                metadata={
                                    "is_recovery_nudge": True,
                                    "recovery_kind": "artifact_read",
                                    "artifact_id": synthesis_artifact_id,
                                    "recovery_mode": "synthesis",
                                },
                            )
                        )
                        harness._runlog(
                            "artifact_read_synthesis_nudge",
                            "nudged model to synthesize from existing artifact evidence",
                            step=harness.state.step_count,
                            artifact_id=synthesis_artifact_id,
                            guard_error=repeat_error,
                        )
                        graph_state.pending_tool_calls = []
                        graph_state.last_tool_results = []
                        return

                recovered = _fallback_repeated_artifact_read(harness, pending)
                if recovered is not None:
                    log_kv(
                        harness.log,
                        logging.INFO,
                        "harness_repeated_tool_loop_recovered",
                        step=harness.state.step_count,
                        original_tool_name=pending.tool_name,
                        recovered_tool_name=recovered.tool_name,
                        recovered_args=recovered.args,
                    )
                    pending = recovered
                else:
                    if _should_pause_repeated_tool_loop(harness, pending):
                        payload = _build_repeated_tool_loop_interrupt_payload(
                            harness=harness,
                            graph_state=graph_state,
                            pending=pending,
                            repeat_error=repeat_error,
                        )
                        guidance = str(payload.get("guidance", "") or "").strip()
                        if guidance:
                            harness.state.append_message(
                                ConversationMessage(
                                    role="system",
                                    content=guidance,
                                    metadata={
                                        "is_recovery_nudge": True,
                                        "recovery_kind": "repeated_tool_loop",
                                        "guard": "repeated_tool_loop",
                                        "tool_name": pending.tool_name,
                                    },
                                )
                            )
                        harness.state.pending_interrupt = payload
                        graph_state.interrupt_payload = payload
                        harness._runlog(
                            "repeated_tool_loop_interrupt",
                            "paused repeated tool loop for human-guided resume",
                            step=harness.state.step_count,
                            tool_name=pending.tool_name,
                            arguments=json_safe_value(pending.args),
                            error=repeat_error,
                        )
                        await harness._emit(
                            deps.event_handler,
                            UIEvent(
                                event_type=UIEventType.ALERT,
                                content=payload["question"],
                                data={"interrupt": payload},
                            ),
                        )
                        graph_state.pending_tool_calls = []
                        return
                    harness.state.recent_errors.append(repeat_error)
                    log_kv(
                        harness.log,
                        logging.WARNING,
                        "harness_repeated_tool_loop",
                        step=harness.state.step_count,
                        tool_name=pending.tool_name,
                        arguments=pending.args,
                        error=repeat_error,
                    )
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(event_type=UIEventType.ERROR, content=repeat_error),
                    )
                    graph_state.pending_tool_calls = []
                    graph_state.final_result = harness._failure(
                        repeat_error,
                        error_type="guard",
                        details={
                            "tool_name": pending.tool_name,
                            "arguments": json_safe_value(pending.args),
                            "guard": "repeated_tool_loop",
                        },
                    )
                    graph_state.error = graph_state.final_result["error"]
                    return

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
                # Check for registry presence to satisfy the 'catch ToolNotFoundError' requirement
                if pending.tool_name not in harness.registry.names():
                    raise ToolNotFoundError(pending.tool_name)

                if intercepted_result is not None:
                    result = intercepted_result
                else:
                    harness._active_dispatch_task = asyncio.create_task(
                        harness._dispatch_tool_call(pending.tool_name, pending.args)
                    )
                    result = await harness._active_dispatch_task
            except ToolNotFoundError:
                if pending.tool_name in HALLUCINATION_MAP:
                    mapped_tool = HALLUCINATION_MAP[pending.tool_name]
                    # Attempt to extract an ID-like string from arguments to give a better hint
                    raw_id = (
                        pending.args.get("path") or
                        pending.args.get("artifact_id") or
                        pending.args.get("pattern") or
                        "A000X"
                    )
                    # Leniency: if they passed a path like 'A0001', use it as the ID
                    artifact_id = str(raw_id).split("/")[-1]
                    if not artifact_id.startswith("A") and "A" in artifact_id:
                        # try to find the A prefix
                        idx = artifact_id.find("A")
                        artifact_id = artifact_id[idx:]

                    hint = f"Tool '{pending.tool_name}' is unavailable. Use '{mapped_tool}(artifact_id=\"{artifact_id}\")' instead."
                    result = ToolEnvelope(
                        success=True,
                        output=hint,
                        metadata={"interceptor_hit": True, "hallucinated_tool": pending.tool_name}
                    )
                else:
                    # Not in map, fall back to standard failure
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
                data=_planner_speaker_data(
                    graph_state,
                    {
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

        # STOP execution of remaining tools in this turn if this tool needs human input
        if (getattr(result, "status", None) == "needs_human" or
            result.metadata.get("status") == "needs_human"):
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


async def persist_tool_results(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> None:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        stored = _get_tool_execution_record(harness, record.operation_id)
        serialized_message = stored.get("tool_message")
        if isinstance(serialized_message, dict):
            message = _conversation_message_from_dict(serialized_message)
        else:
            message = await harness._record_tool_result(
                tool_name=record.tool_name,
                tool_call_id=record.tool_call_id,
                result=record.result,
                arguments=record.args,
            )
            stored["tool_message"] = message.to_dict()
            artifact_id = message.metadata.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                stored["artifact_id"] = artifact_id
            harness.state.tool_execution_records[record.operation_id] = stored
            harness._record_experience(
                tool_name=record.tool_name,
                result=record.result,
            )

        if _has_matching_tool_message(harness, message):
            continue
        harness.state.append_message(message)
        harness._log_conversation_state("tool_message")


# Re-export private helpers for backward compatibility (tests)
from .tool_call_parser import (
    _tool_call_fingerprint,
    _fallback_repeated_file_read,
    _fallback_repeated_artifact_read,
)
