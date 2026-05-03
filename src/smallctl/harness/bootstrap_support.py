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
from .run_mode import ModeDecisionService
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
    fresh_run: bool,
    fresh_run_turns: int,
    planning_mode: bool,
    contract_flow_ui: bool,
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
        "fresh_run": fresh_run,
        "fresh_run_turns": fresh_run_turns,
        "planning_mode": planning_mode,
        "contract_flow_ui": contract_flow_ui,
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
    self._backend_recovery_service = BackendRecoveryService(self)
    self._task_boundary_service = TaskBoundaryService(self)
