from __future__ import annotations

import json
from typing import Any

from ..guards import is_small_model_name
from ..memory.taxonomy import (
    PREMATURE_TASK_COMPLETE,
    TOOL_NOT_CALLED,
)
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..plans import write_plan_file
from ..prompts import build_planning_prompt, build_system_prompt
from ..state import clip_text_value, json_safe_value
from ..task_targets import primary_task_target_path
from ..tools.dispatcher import normalize_tool_request
from ..write_session_fsm import new_write_session, record_write_session_event
from .deps import GraphRuntimeDeps
from .chat_progress import (
    build_artifact_summary_exit_message,
    build_blank_message_nudge,
    build_file_read_recovery_message,
    build_repeated_chat_thinking_message,
    build_repeated_tool_loop_interrupt_payload,
    build_small_model_continue_message,
    chat_completion_recovery_guard,
    chat_turn_signature,
    looks_like_freeze_or_hang,
    recent_assistant_texts,
    should_pause_repeated_tool_loop,
    task_prefers_summary_synthesis,
)
from .planning_support import (
    extract_plan_steps_from_text,
    persist_planning_playbook,
    planning_response_looks_like_plan,
    synthesize_plan_from_text,
)
from .shell_outcomes import (
    _shell_human_retry_hint,
    _shell_ssh_retry_hint,
    _shell_workspace_relative_retry_hint,
)
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from .tool_call_parser import (
    _active_write_session_for_target,
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
    parse_tool_calls,
)
from .chat_progress import _chat_progress_guard_failure
from .tool_execution_support import (
    _conversation_message_from_dict,
    _get_tool_execution_record,
    _has_matching_tool_message,
    _store_tool_execution_record,
    _tool_envelope_from_dict,
)
from .tool_outcomes import apply_chat_tool_outcomes, apply_planning_tool_outcomes, apply_tool_outcomes
from .write_recovery import (
    build_synthetic_write_args,
    can_safely_synthesize,
    recover_write_intent,
    write_recovery_kind,
    write_recovery_metadata,
)


def get_suggested_sections(path: str) -> list[str]:
    return _suggested_chunk_sections(path)


def increment_run_metric(
    graph_state: GraphRunState,
    name: str,
    *,
    delta: int = 1,
) -> int:
    current = int(graph_state.latency_metrics.get(name, 0) or 0)
    updated = current + delta
    graph_state.latency_metrics[name] = updated
    return updated


def record_empty_write_retry_metric(
    graph_state: GraphRunState,
    harness: Any,
    pending: PendingToolCall,
) -> int:
    count = increment_run_metric(graph_state, "empty_write_retry_count")
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


def apply_declared_read_before_write_reroute(
    graph_state: GraphRunState,
    harness: Any,
    pending: PendingToolCall,
    *,
    assistant_text: str = "",
) -> bool:
    declared_read_recovery = _recover_declared_read_before_write(
        harness,
        pending,
        assistant_text=assistant_text,
    )
    if declared_read_recovery is None:
        return False
    redirected_pending, redirect_reason = declared_read_recovery

    original_tool_name = pending.tool_name
    original_args = json_safe_value(pending.args)
    pending.tool_name = redirected_pending.tool_name
    pending.args = redirected_pending.args
    pending.raw_arguments = redirected_pending.raw_arguments
    pending.source = redirected_pending.source
    increment_run_metric(graph_state, "declared_read_before_write_reroute_count")
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"You declared a read-first recovery step, so the pending `{original_tool_name}` call was "
                f"rerouted to `file_read(path='{pending.args.get('path', '')}')`. "
                "Inspect the staged/current content first, then choose one narrow follow-up write."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "declared_read_before_write",
                "original_tool_name": original_tool_name,
                "rerouted_tool_name": pending.tool_name,
                "path": str(pending.args.get("path", "") or ""),
                "redirect_reason": json_safe_value(redirect_reason),
            },
        )
    )
    harness._runlog(
        "intent_mismatch_detected_redirection_activated",
        "intent mismatch detected redirection activated",
        tool_call_id=pending.tool_call_id,
        mismatch_kind="declared_read_before_write",
        original_tool_name=original_tool_name,
        redirected_tool_name=pending.tool_name,
        original_args=original_args,
        rerouted_args=json_safe_value(pending.args),
        reason=json_safe_value(redirect_reason),
    )
    return True


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


def matching_write_session_for_pending(harness: Any, pending: PendingToolCall) -> Any | None:
    if pending.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch"}:
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


def remember_write_session_schema_failure(
    harness: Any,
    pending: PendingToolCall,
    details: dict[str, Any],
    *,
    error_message: str,
    nudge_count: int,
) -> None:
    session = matching_write_session_for_pending(harness, pending)
    if session is None:
        return

    pending_args = dict(getattr(pending, "args", {}) or {})
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
        "nudge_count": int(nudge_count),
        "status": str(session.status or ""),
    }
    if pending.tool_name in {"file_write", "file_append"}:
        section_name = str(
            pending_args.get("section_name")
            or pending_args.get("section_id")
            or session.write_next_section
            or session.write_current_section
            or "imports"
        ).strip() or "imports"
        payload["recommended_section_name"] = section_name
    harness.state.scratchpad[_WRITE_SESSION_SCHEMA_FAILURE_KEY] = payload


def planner_speaker_data(graph_state: GraphRunState, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(extra or {})
    if graph_state.run_mode == "planning" or getattr(graph_state.loop_state, "planning_mode_enabled", False):
        payload.setdefault("speaker", "planner")
    return payload


def harness_model_name(harness: Any) -> str:
    model_name = getattr(getattr(harness, "client", None), "model", None)
    if model_name:
        return str(model_name)
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {})
    if isinstance(scratchpad, dict):
        return str(scratchpad.get("_model_name", ""))
    return ""


def is_small_model(harness: Any) -> bool:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {})
    if isinstance(scratchpad, dict) and "_model_is_small" in scratchpad:
        return bool(scratchpad.get("_model_is_small"))
    return is_small_model_name(harness_model_name(harness))


def model_uses_gpt_oss_commentary_rules(harness: Any) -> bool:
    model_name = harness_model_name(harness).strip().lower()
    return bool(
        model_name
        and any(marker in model_name for marker in ("openai/gpt-oss-20b", "gpt-oss-20b", "openai/gpt-oss"))
    )


def recent_assistant_texts(harness: Any, *, limit: int = 2) -> list[str]:
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


def looks_like_freeze_or_hang(harness: Any, assistant_text: str) -> bool:
    text = str(assistant_text or "").strip()
    if not text:
        return False
    recent = recent_assistant_texts(harness, limit=3)
    if not recent:
        return False
    if text in recent:
        return True
    if len(recent) >= 2 and recent[0] == recent[1]:
        return True
    if len(recent) >= 3 and recent[0] == recent[1] == recent[2]:
        return True
    return False


def build_authoring_budget_message(harness: Any, phase: str, pending_tool_name: str) -> str:
    phase_label = phase or "author"
    return (
        f"Authoring budget applied in {phase_label} phase. "
        f"Focus on one concrete action at a time, starting with `{pending_tool_name}`. "
        "Finish that action, inspect the result, then decide the next step on the next turn."
    )


def apply_small_model_authoring_budget(harness: Any, graph_state: GraphRunState) -> bool:
    if not is_small_model(harness):
        return False
    contract_phase = harness.state.contract_phase()
    if contract_phase not in {"author", "repair"}:
        return False
    if len(graph_state.pending_tool_calls) <= 1:
        return False

    config = getattr(harness, "config", None)
    allow_multi = config.allow_multi_section_turns_for_small_edits if config else True

    if allow_multi:
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
            content=build_authoring_budget_message(harness, contract_phase, first_pending.tool_name),
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


_get_suggested_sections = get_suggested_sections
_increment_run_metric = increment_run_metric
_record_empty_write_retry_metric = record_empty_write_retry_metric
_apply_declared_read_before_write_reroute = apply_declared_read_before_write_reroute
_planner_speaker_data = planner_speaker_data
_harness_model_name = harness_model_name
_is_small_model = is_small_model
_model_uses_gpt_oss_commentary_rules = model_uses_gpt_oss_commentary_rules
_build_blank_message_nudge = build_blank_message_nudge
_build_small_model_continue_message = build_small_model_continue_message
_task_prefers_summary_synthesis = task_prefers_summary_synthesis
_chat_turn_signature = chat_turn_signature
_build_repeated_chat_thinking_message = build_repeated_chat_thinking_message
_chat_completion_recovery_guard = chat_completion_recovery_guard
_build_artifact_summary_exit_message = build_artifact_summary_exit_message
_should_pause_repeated_tool_loop = should_pause_repeated_tool_loop
_build_repeated_tool_loop_interrupt_payload = build_repeated_tool_loop_interrupt_payload
_build_authoring_budget_message = build_authoring_budget_message
_build_file_read_recovery_message = build_file_read_recovery_message
_apply_small_model_authoring_budget = apply_small_model_authoring_budget
_extract_plan_steps_from_text = extract_plan_steps_from_text
_synthesize_plan_from_text = synthesize_plan_from_text
_persist_planning_playbook = persist_planning_playbook
