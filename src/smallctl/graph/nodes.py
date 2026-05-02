from __future__ import annotations

from .interpret_nodes import (
    interpret_chat_output,
    interpret_model_output,
    interpret_planning_output,
)
from .lifecycle_nodes import (
    initialize_loop_run,
    initialize_planning_run,
    load_index_manifest,
    prepare_chat_prompt,
    prepare_indexer_prompt,
    prepare_loop_step,
    prepare_planning_prompt,
    prepare_prompt,
    prepare_staged_prompt,
    resume_loop_run,
    resume_planning_run,
    select_chat_tools,
    select_indexer_tools,
    select_loop_tools,
    select_planning_tools,
    select_staged_tools,
    _available_tool_names,
)
from .model_call_nodes import model_call
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from .tool_call_parser import (
    _build_schema_repair_message,
    _detect_empty_file_write_payload,
    _detect_missing_required_tool_arguments,
    _detect_patch_existing_stage_read_contract_violation,
    _detect_placeholder_tool_call,
    _detect_timeout_recovered_incomplete_tool_call,
    _ensure_chunk_write_session,
    _fallback_repeated_artifact_read,
    _fallback_repeated_file_read,
    _repair_active_write_session_args,
    _salvage_active_write_session_append,
    _tool_call_fingerprint,
)
from .tool_outcomes import apply_chat_tool_outcomes, apply_planning_tool_outcomes, apply_tool_outcomes
from .tool_execution_nodes import dispatch_tools, persist_tool_results
from .node_support import (
    HALLUCINATION_MAP,
    ToolNotFoundError,
    _WRITE_SESSION_SCHEMA_FAILURE_KEY,
    apply_declared_read_before_write_reroute as _apply_declared_read_before_write_reroute,
    apply_small_model_authoring_budget as _apply_small_model_authoring_budget,
    build_artifact_summary_exit_message as _build_artifact_summary_exit_message,
    build_authoring_budget_message as _build_authoring_budget_message,
    build_blank_message_nudge as _build_blank_message_nudge,
    build_file_read_recovery_message as _build_file_read_recovery_message,
    build_repeated_chat_thinking_message as _build_repeated_chat_thinking_message,
    build_repeated_tool_loop_interrupt_payload as _build_repeated_tool_loop_interrupt_payload,
    build_small_model_continue_message as _build_small_model_continue_message,
    chat_completion_recovery_guard as _chat_completion_recovery_guard,
    chat_turn_signature as _chat_turn_signature,
    get_suggested_sections as _get_suggested_sections,
    harness_model_name as _harness_model_name,
    increment_run_metric as _increment_run_metric,
    is_small_model as _is_small_model,
    looks_like_freeze_or_hang as _looks_like_freeze_or_hang,
    matching_write_session_for_pending as _matching_write_session_for_pending,
    model_uses_gpt_oss_commentary_rules as _model_uses_gpt_oss_commentary_rules,
    planner_speaker_data as _planner_speaker_data,
    recent_assistant_texts as _recent_assistant_texts,
    record_empty_write_retry_metric as _record_empty_write_retry_metric,
    remember_write_session_schema_failure as _remember_write_session_schema_failure,
    should_pause_repeated_tool_loop as _should_pause_repeated_tool_loop,
    task_prefers_summary_synthesis as _task_prefers_summary_synthesis,
)
from .planning_support import (
    extract_plan_steps_from_text as _extract_plan_steps_from_text,
    persist_planning_playbook as _persist_planning_playbook,
    planning_response_looks_like_plan as _planning_response_looks_like_plan,
    synthesize_plan_from_text as _synthesize_plan_from_text,
)
