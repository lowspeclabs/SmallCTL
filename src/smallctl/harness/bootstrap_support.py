from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..client import OpenAICompatClient
from ..context import ContextPolicy
from ..guards import is_small_model_name
from ..context import ArtifactStore
from ..provider_profiles import resolve_provider_profile as resolve_provider_profile_config
from ..tools import ToolDispatcher
from ..state import LoopState, json_safe_value
from ..phases import normalize_phase
from .approvals import ApprovalService
from .backend_recovery import BackendRecoveryService
from .compaction import CompactionService
from .factory import SubtaskService
from .memory import MemoryService
from .prompt_builder import PromptBuilderService
from .reflexion_service import ReflexionService
from .run_mode import ModeDecisionService
from .subtask_ledger_service import SubtaskLedgerService
from .task_boundary import TaskBoundaryService
from .tool_results import ToolResultService


def build_initial_state(*, phase: str, planning_mode: bool, strategy: dict[str, Any] | None) -> LoopState:
    normalized_phase = normalize_phase(phase)
    state = LoopState(
        current_phase=normalized_phase,
        strategy=strategy,
        planning_mode_enabled=bool(planning_mode),
    )
    if isinstance(strategy, dict):
        state.scratchpad["strategy"] = json_safe_value(strategy)
    return state


def resolve_provider_profile(endpoint: str, model: str, provider_profile: str) -> str:
    resolved, _ = resolve_provider_profile_config(
        endpoint,
        model,
        provider_profile,
    )
    return resolved


def build_client(
    *,
    endpoint: str,
    model: str,
    api_key: str | None,
    chat_endpoint: str,
    provider_profile: str,
    first_token_timeout_sec: int | None,
    runtime_context_probe: bool,
    run_logger: Any,
    backend_recovery_handler: Any = None,
) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url=endpoint,
        model=model,
        api_key=api_key,
        chat_endpoint=chat_endpoint,
        provider_profile=provider_profile,
        first_token_timeout_sec=first_token_timeout_sec,
        runtime_context_probe=runtime_context_probe,
        run_logger=run_logger,
        backend_recovery_handler=backend_recovery_handler,
    )


def build_context_policy(
    *,
    policy: ContextPolicy | None,
    effective_max_prompt_tokens: int | None,
    reserve_completion_tokens: int,
    reserve_tool_tokens: int,
    summarize_at_ratio: float,
    recent_message_limit: int,
    max_summary_items: int,
    max_artifact_snippets: int,
    artifact_snippet_token_limit: int,
    artifact_summarization_threshold: int,
    multi_file_artifact_snippet_limit: int,
    multi_file_primary_file_limit: int,
    remote_task_artifact_snippet_limit: int,
    remote_task_primary_file_limit: int,
    tool_result_inline_token_limit: int,
    artifact_read_inline_token_limit: int,
) -> ContextPolicy:
    if policy is None:
        return ContextPolicy(
            max_prompt_tokens=effective_max_prompt_tokens,
            reserve_completion_tokens=reserve_completion_tokens,
            reserve_tool_tokens=reserve_tool_tokens,
            summarize_at_ratio=summarize_at_ratio,
            recent_message_limit=recent_message_limit,
            max_summary_items=max_summary_items,
            max_artifact_snippets=max_artifact_snippets,
            artifact_snippet_token_limit=artifact_snippet_token_limit,
            artifact_summarization_threshold=artifact_summarization_threshold,
            multi_file_artifact_snippet_limit=multi_file_artifact_snippet_limit,
            multi_file_primary_file_limit=multi_file_primary_file_limit,
            remote_task_artifact_snippet_limit=remote_task_artifact_snippet_limit,
            remote_task_primary_file_limit=remote_task_primary_file_limit,
            tool_result_inline_token_limit=tool_result_inline_token_limit,
            artifact_read_inline_token_limit=artifact_read_inline_token_limit,
        )
    if effective_max_prompt_tokens is not None:
        policy.max_prompt_tokens = effective_max_prompt_tokens
    return policy


def build_harness_kwargs(
    *,
    endpoint: str,
    model: str,
    phase: str,
    provider_profile: str,
    api_key: str | None,
    tool_profiles: list[str] | None,
    reasoning_mode: str,
    thinking_visibility: bool,
    thinking_start_tag: str,
    thinking_end_tag: str,
    chat_endpoint: str,
    runtime_context_probe: bool,
    graph_checkpointer: str,
    graph_checkpoint_path: str | None,
    graph_node_timeout_sec: int,
    graph_model_call_timeout_sec: int,
    graph_dispatch_tools_timeout_sec: int,
    graph_idle_watchdog_sec: int,
    graph_recursion_limit: int,
    graph_coding_recursion_limit: int,
    needs_human_timeout_sec: int,
    fresh_run: bool,
    fresh_run_turns: int,
    planning_mode: bool,
    contract_flow_ui: bool,
    staged_execution_enabled: bool,
    staged_step_prompt_tokens: int,
    context_limit: int | None,
    max_prompt_tokens: int | None,
    max_prompt_tokens_explicit: bool,
    reserve_completion_tokens: int,
    reserve_tool_tokens: int,
    first_token_timeout_sec: int | None,
    healthcheck_url: str | None,
    restart_command: str | None,
    startup_grace_period_sec: int,
    max_restarts_per_hour: int,
    backend_healthcheck_url: str | None,
    backend_restart_command: str | None,
    backend_unload_command: str | None,
    backend_healthcheck_timeout_sec: int,
    backend_restart_grace_sec: int,
    summarize_at_ratio: float,
    recent_message_limit: int,
    max_summary_items: int,
    max_artifact_snippets: int,
    artifact_snippet_token_limit: int,
    artifact_summarization_threshold: int,
    multi_file_artifact_snippet_limit: int,
    multi_file_primary_file_limit: int,
    remote_task_artifact_snippet_limit: int,
    remote_task_primary_file_limit: int,
    run_logger: Any,
    artifact_start_index: int | None,
    tool_result_inline_token_limit: int,
    indexer: bool,
    allow_interactive_shell_approval: bool,
    shell_approval_session_default: bool,
    provider_profile_resolved: str,
    run_mode: str,
    tool_plan_runtime_enabled: bool,
    tool_plan_auto_select: bool,
    tool_plan_readonly_only: bool,
    tool_plan_max_steps: int,
    tool_plan_max_repair_attempts: int,
    schema_validation_max_repair_attempts: int,
    tool_plan_observation_token_limit: int,
    tool_plan_max_observation_chars_per_step: int,
    tool_plan_solver_fresh_output_limit: int,
    tool_plan_allow_web: bool,
    tool_plan_allow_artifact_read: bool,
    tool_plan_allow_git: bool,
    tool_plan_fallback_to_loop_on_invalid_plan: bool,
    tool_dag_enabled: bool,
    tool_dag_max_parallel: int,
    tool_dag_timeout_sec: int,
    tool_dag_preserve_result_order: bool,
    min_exploration_steps: int,
    chunk_mode_min_bytes: int,
    chunk_mode_new_file_only: bool,
    chunk_mode_supported_models: list[str],
    small_model_soft_write_chars: int,
    small_model_hard_write_chars: int,
    new_file_chunk_mode_line_estimate: int,
    allow_multi_section_turns_for_small_edits: bool,
    failed_local_patch_limit: int,
    enable_write_intent_recovery: bool,
    enable_assistant_code_write_recovery: bool,
    write_recovery_min_confidence: str,
    write_recovery_allow_raw_text_targets: bool,
    solver_refine_enabled: bool,
    solver_refine_max_passes: int,
    solver_refine_on_final_answer: bool,
    solver_refine_on_patch_plan: bool,
    solver_refine_on_task_complete: bool,
    solver_refine_token_budget: int,
    rewoo_lane_frames_enabled: bool,
    rewoo_planner_frame_enabled: bool,
    rewoo_solver_frame_enabled: bool,
    rewoo_refiner_frame_enabled: bool,
    rewoo_frame_token_budget: int,
    test_time_scaling_enabled: bool,
    test_time_scaling_runtimes: list[str],
    test_time_scaling_trigger: str,
    test_time_scaling_max_candidates: int,
    test_time_scaling_min_candidates: int,
    test_time_scaling_policy: str,
    test_time_scaling_strategy: str,
    test_time_scaling_score_threshold: float,
    test_time_scaling_parallel_max: int,
    test_time_scaling_timeout_sec: int,
    test_time_scaling_mutating_parallel_enabled: bool,
    test_time_scaling_all_fail_action: str,
    escalation_enabled: bool,
    escalation_expose_tool: bool,
    escalation_auto_trigger: bool,
    escalation_endpoint: str | None,
    escalation_model: str | None,
    escalation_provider_profile: str,
    escalation_api_key: str | None,
    escalation_api_key_env: str | None,
    escalation_chat_endpoint: str,
    escalation_max_prompt_chars: int,
    escalation_max_response_tokens: int,
    escalation_temperature: float,
    escalation_timeout_sec: int,
    escalation_max_per_task: int,
    escalation_cooldown_turns: int,
    escalation_repeated_failure_threshold: int,
    escalation_require_tool_plan_evidence: bool,
    escalation_redact_secrets: bool,
    fama_enabled: bool,
    fama_mode: str,
    fama_default_ttl_steps: int,
    fama_max_active_mitigations: int,
    fama_signal_window: int,
    fama_done_gate_on_failure: bool,
    fama_capsule_token_budget: int,
    fama_llm_judge_enabled: bool,
    fama_llm_judge_min_severity: int,
) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "model": model,
        "phase": phase,
        "provider_profile": provider_profile_resolved,
        "api_key": api_key,
        "tool_profiles": tool_profiles,
        "reasoning_mode": reasoning_mode,
        "thinking_visibility": thinking_visibility,
        "thinking_start_tag": thinking_start_tag,
        "thinking_end_tag": thinking_end_tag,
        "chat_endpoint": chat_endpoint,
        "runtime_context_probe": runtime_context_probe,
        "checkpoint_on_exit": False,
        "checkpoint_path": None,
        "graph_checkpointer": graph_checkpointer,
        "graph_checkpoint_path": graph_checkpoint_path,
        "graph_node_timeout_sec": graph_node_timeout_sec,
        "graph_model_call_timeout_sec": graph_model_call_timeout_sec,
        "graph_dispatch_tools_timeout_sec": graph_dispatch_tools_timeout_sec,
        "graph_idle_watchdog_sec": graph_idle_watchdog_sec,
        "graph_recursion_limit": graph_recursion_limit,
        "graph_coding_recursion_limit": graph_coding_recursion_limit,
        "needs_human_timeout_sec": needs_human_timeout_sec,
        "fresh_run": fresh_run,
        "fresh_run_turns": fresh_run_turns,
        "planning_mode": planning_mode,
        "contract_flow_ui": contract_flow_ui,
        "staged_execution_enabled": staged_execution_enabled,
        "staged_step_prompt_tokens": staged_step_prompt_tokens,
        "context_limit": context_limit,
        "max_prompt_tokens": max_prompt_tokens,
        "max_prompt_tokens_explicit": max_prompt_tokens_explicit,
        "reserve_completion_tokens": reserve_completion_tokens,
        "reserve_tool_tokens": reserve_tool_tokens,
        "first_token_timeout_sec": first_token_timeout_sec,
        "healthcheck_url": healthcheck_url,
        "restart_command": restart_command,
        "startup_grace_period_sec": startup_grace_period_sec,
        "max_restarts_per_hour": max_restarts_per_hour,
        "backend_healthcheck_url": backend_healthcheck_url,
        "backend_restart_command": backend_restart_command,
        "backend_unload_command": backend_unload_command,
        "backend_healthcheck_timeout_sec": backend_healthcheck_timeout_sec,
        "backend_restart_grace_sec": backend_restart_grace_sec,
        "summarize_at_ratio": summarize_at_ratio,
        "recent_message_limit": recent_message_limit,
        "max_summary_items": max_summary_items,
        "max_artifact_snippets": max_artifact_snippets,
        "artifact_snippet_token_limit": artifact_snippet_token_limit,
        "artifact_summarization_threshold": artifact_summarization_threshold,
        "multi_file_artifact_snippet_limit": multi_file_artifact_snippet_limit,
        "multi_file_primary_file_limit": multi_file_primary_file_limit,
        "remote_task_artifact_snippet_limit": remote_task_artifact_snippet_limit,
        "remote_task_primary_file_limit": remote_task_primary_file_limit,
        "run_logger": run_logger,
        "artifact_start_index": artifact_start_index,
        "tool_result_inline_token_limit": tool_result_inline_token_limit,
        "indexer": indexer,
        "allow_interactive_shell_approval": allow_interactive_shell_approval,
        "shell_approval_session_default": shell_approval_session_default,
        "run_mode": run_mode,
        "tool_plan_runtime_enabled": tool_plan_runtime_enabled,
        "tool_plan_auto_select": tool_plan_auto_select,
        "tool_plan_readonly_only": tool_plan_readonly_only,
        "tool_plan_max_steps": tool_plan_max_steps,
        "tool_plan_max_repair_attempts": tool_plan_max_repair_attempts,
        "schema_validation_max_repair_attempts": schema_validation_max_repair_attempts,
        "tool_plan_observation_token_limit": tool_plan_observation_token_limit,
        "tool_plan_max_observation_chars_per_step": tool_plan_max_observation_chars_per_step,
        "tool_plan_solver_fresh_output_limit": tool_plan_solver_fresh_output_limit,
        "tool_plan_allow_web": tool_plan_allow_web,
        "tool_plan_allow_artifact_read": tool_plan_allow_artifact_read,
        "tool_plan_allow_git": tool_plan_allow_git,
        "tool_plan_fallback_to_loop_on_invalid_plan": tool_plan_fallback_to_loop_on_invalid_plan,
        "tool_dag_enabled": tool_dag_enabled,
        "tool_dag_max_parallel": tool_dag_max_parallel,
        "tool_dag_timeout_sec": tool_dag_timeout_sec,
        "tool_dag_preserve_result_order": tool_dag_preserve_result_order,
        "min_exploration_steps": min_exploration_steps,
        "chunk_mode_min_bytes": chunk_mode_min_bytes,
        "chunk_mode_new_file_only": chunk_mode_new_file_only,
        "chunk_mode_supported_models": chunk_mode_supported_models,
        "small_model_soft_write_chars": small_model_soft_write_chars,
        "small_model_hard_write_chars": small_model_hard_write_chars,
        "new_file_chunk_mode_line_estimate": new_file_chunk_mode_line_estimate,
        "allow_multi_section_turns_for_small_edits": allow_multi_section_turns_for_small_edits,
        "failed_local_patch_limit": failed_local_patch_limit,
        "enable_write_intent_recovery": enable_write_intent_recovery,
        "enable_assistant_code_write_recovery": enable_assistant_code_write_recovery,
        "write_recovery_min_confidence": write_recovery_min_confidence,
        "write_recovery_allow_raw_text_targets": write_recovery_allow_raw_text_targets,
        "solver_refine_enabled": solver_refine_enabled,
        "solver_refine_max_passes": solver_refine_max_passes,
        "solver_refine_on_final_answer": solver_refine_on_final_answer,
        "solver_refine_on_patch_plan": solver_refine_on_patch_plan,
        "solver_refine_on_task_complete": solver_refine_on_task_complete,
        "solver_refine_token_budget": solver_refine_token_budget,
        "rewoo_lane_frames_enabled": rewoo_lane_frames_enabled,
        "rewoo_planner_frame_enabled": rewoo_planner_frame_enabled,
        "rewoo_solver_frame_enabled": rewoo_solver_frame_enabled,
        "rewoo_refiner_frame_enabled": rewoo_refiner_frame_enabled,
        "rewoo_frame_token_budget": rewoo_frame_token_budget,
        "test_time_scaling_enabled": test_time_scaling_enabled,
        "test_time_scaling_runtimes": test_time_scaling_runtimes,
        "test_time_scaling_trigger": test_time_scaling_trigger,
        "test_time_scaling_max_candidates": test_time_scaling_max_candidates,
        "test_time_scaling_min_candidates": test_time_scaling_min_candidates,
        "test_time_scaling_policy": test_time_scaling_policy,
        "test_time_scaling_strategy": test_time_scaling_strategy,
        "test_time_scaling_score_threshold": test_time_scaling_score_threshold,
        "test_time_scaling_parallel_max": test_time_scaling_parallel_max,
        "test_time_scaling_timeout_sec": test_time_scaling_timeout_sec,
        "test_time_scaling_mutating_parallel_enabled": test_time_scaling_mutating_parallel_enabled,
        "test_time_scaling_all_fail_action": test_time_scaling_all_fail_action,
        "escalation_enabled": escalation_enabled,
        "escalation_expose_tool": escalation_expose_tool,
        "escalation_auto_trigger": escalation_auto_trigger,
        "escalation_endpoint": escalation_endpoint,
        "escalation_model": escalation_model,
        "escalation_provider_profile": escalation_provider_profile,
        "escalation_api_key": escalation_api_key,
        "escalation_api_key_env": escalation_api_key_env,
        "escalation_chat_endpoint": escalation_chat_endpoint,
        "escalation_max_prompt_chars": escalation_max_prompt_chars,
        "escalation_max_response_tokens": escalation_max_response_tokens,
        "escalation_temperature": escalation_temperature,
        "escalation_timeout_sec": escalation_timeout_sec,
        "escalation_max_per_task": escalation_max_per_task,
        "escalation_cooldown_turns": escalation_cooldown_turns,
        "escalation_repeated_failure_threshold": escalation_repeated_failure_threshold,
        "escalation_require_tool_plan_evidence": escalation_require_tool_plan_evidence,
        "escalation_redact_secrets": escalation_redact_secrets,
        "fama_enabled": fama_enabled,
        "fama_mode": fama_mode,
        "fama_default_ttl_steps": fama_default_ttl_steps,
        "fama_max_active_mitigations": fama_max_active_mitigations,
        "fama_signal_window": fama_signal_window,
        "fama_done_gate_on_failure": fama_done_gate_on_failure,
        "fama_capsule_token_budget": fama_capsule_token_budget,
        "fama_llm_judge_enabled": fama_llm_judge_enabled,
        "fama_llm_judge_min_severity": fama_llm_judge_min_severity,
    }


def finalize_harness_bootstrap(
    *,
    self: Any,
    known_server_context_limit: int | None,
    artifact_start_index: int | None,
    provider_profile: str,
    context_policy: ContextPolicy,
    scaling_context: int | None,
) -> None:
    self.discovered_server_context_limit = known_server_context_limit
    self.server_context_limit = known_server_context_limit
    if scaling_context is not None:
        context_policy.recalculate_quotas(
            scaling_context,
            backend_profile=provider_profile,
        )
        self.state.recent_message_limit = context_policy.recent_message_limit
    self.conversation_id = uuid.uuid4().hex[:8]
    if not self.state.thread_id:
        self.state.thread_id = self.conversation_id
    self._sync_run_logger_session_id()
    artifact_base_dir = Path(self.state.cwd).resolve() / ".smallctl" / "artifacts"
    self.artifact_store = ArtifactStore(
        artifact_base_dir,
        self.conversation_id,
        session_id=self.state.thread_id,
        artifact_start_index=artifact_start_index,
    )
    memory_base_dir = Path(self.state.cwd).resolve() / ".smallctl" / "memory"
    from ..memory_store import ExperienceStore

    self.warm_memory_store = ExperienceStore(memory_base_dir / "warm-experiences.jsonl")
    self.cold_memory_store = ExperienceStore(memory_base_dir / "cold-experiences.jsonl")
    self._cancel_requested = False
    self._active_dispatch_task = None
    self._active_task_scope = None
    self._task_sequence = 0
    self._pending_task_shutdown_reason = ""

    self.mode_decision = ModeDecisionService(self)
    self.prompt_builder = PromptBuilderService(self)
    self.tool_results = ToolResultService(self)
    self.compaction = CompactionService(self)
    self.memory = MemoryService(self)
    self.approvals = ApprovalService(self)
    self.subtasks = SubtaskService(self)
    self.subtask_ledger = SubtaskLedgerService(self)
    self.reflexion = ReflexionService(self)
    self._backend_recovery_service = BackendRecoveryService(self)
    self._task_boundary_service = TaskBoundaryService(self)
