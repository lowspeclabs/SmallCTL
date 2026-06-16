from __future__ import annotations

import asyncio
import asyncio.subprocess
import logging
from typing import Any

logger = logging.getLogger(__name__)

from ..context import (
    ArtifactStore,
    ContextPolicy,
    ContextSummarizer,
    LexicalRetriever,
    PromptAssembler,
    SubtaskRunner,
)
from ..guards import GuardConfig
from ..phases import normalize_phase
from ..tools import ToolDispatcher, build_registry
from .bootstrap_support import (
    build_client,
    build_context_policy,
    build_initial_state,
    finalize_harness_bootstrap,
    resolve_provider_profile,
)
from .approvals import ApprovalService
from .backend_recovery import BackendRecoveryService
from .compaction import CompactionService
from .config import HarnessConfig
from .factory import SubtaskService
from .memory import MemoryService
from .prompt_builder import PromptBuilderService
from .reflexion_service import ReflexionService
from .run_mode import ModeDecisionService
from .subtask_ledger_service import SubtaskLedgerService
from .task_boundary import TaskBoundaryService
from .tool_results import ToolResultService


def initialize_harness(self: Any, config: HarnessConfig) -> None:
    normalized_phase = normalize_phase(config.phase)
    self._initial_phase = normalized_phase
    self.state = build_initial_state(
        phase=normalized_phase,
        planning_mode=config.planning_mode,
        strategy=config.strategy,
    )

    self.backend_healthcheck_url = str(config.healthcheck_url or config.backend_healthcheck_url or "").strip() or None
    self.backend_restart_command = str(config.restart_command or config.backend_restart_command or "").strip() or None
    self.backend_unload_command = str(config.backend_unload_command or "").strip() or None
    self.backend_healthcheck_timeout_sec = max(1, int(config.backend_healthcheck_timeout_sec))
    self.backend_restart_grace_sec = max(
        1,
        int(config.startup_grace_period_sec if config.startup_grace_period_sec is not None else config.backend_restart_grace_sec),
    )
    self.backend_max_restarts_per_hour = max(0, int(config.max_restarts_per_hour))

    resolved_provider_profile = resolve_provider_profile(config.endpoint, config.model, config.provider_profile)
    self.client = build_client(
        endpoint=config.endpoint,
        model=config.model,
        api_key=config.api_key,
        chat_endpoint=config.chat_endpoint,
        provider_profile=resolved_provider_profile,
        first_token_timeout_sec=config.first_token_timeout_sec,
        runtime_context_probe=config.runtime_context_probe,
        run_logger=config.run_logger,
        backend_recovery_handler=self.recover_backend_wedge,
    )
    self.summarizer_client = None
    if config.summarizer_endpoint:
        self.summarizer_client = build_client(
            endpoint=config.summarizer_endpoint,
            model=config.summarizer_model or config.model,
            api_key=config.summarizer_api_key or config.api_key,
            chat_endpoint=config.chat_endpoint,
            provider_profile=resolve_provider_profile(
                config.summarizer_endpoint,
                config.summarizer_model or config.model,
                "auto",
            ),
            first_token_timeout_sec=config.first_token_timeout_sec,
            runtime_context_probe=False,
            run_logger=config.run_logger,
        )

    self.reasoning_mode = config.reasoning_mode
    self.thinking_visibility = config.thinking_visibility
    self.thinking_start_tag = config.thinking_start_tag
    self.thinking_end_tag = config.thinking_end_tag
    self.state.scratchpad["_model_name"] = config.model
    self.state.scratchpad["_model_is_small"] = self._is_small_model_name(config.model)
    self.state.scratchpad["_max_repair_steps"] = int(getattr(config, "max_repair_steps", 3) or 3)
    self._backend_recovery_service = BackendRecoveryService(self)
    self._task_boundary_service = TaskBoundaryService(self)
    self._active_processes: set[asyncio.subprocess.Process] = set()
    self._background_persistence_tasks: set[asyncio.Task[Any]] = set()
    self._teardown_task: asyncio.Task[None] | None = None
    self.strategy_prompt = config.strategy_prompt
    self.event_handler = None
    self.allow_interactive_shell_approval = bool(config.allow_interactive_shell_approval)
    self.shell_approval_session_default = bool(config.shell_approval_session_default)
    self.sudo_password = config.sudo_password
    self._configured_tool_profiles = list(config.tool_profiles) if config.tool_profiles else None
    self._strategy_prompt = config.strategy_prompt
    self._indexer = config.indexer
    self.provider_profile = self.client.provider_profile

    # Sync resolved values back so config stays the source of truth
    config.provider_profile = self.provider_profile
    config.backend_healthcheck_url = self.backend_healthcheck_url
    config.backend_restart_command = self.backend_restart_command
    config.backend_unload_command = self.backend_unload_command
    config.backend_healthcheck_timeout_sec = self.backend_healthcheck_timeout_sec
    config.backend_restart_grace_sec = self.backend_restart_grace_sec
    config.startup_grace_period_sec = self.backend_restart_grace_sec
    config.max_restarts_per_hour = self.backend_max_restarts_per_hour
    if not config.max_prompt_tokens_explicit:
        config.max_prompt_tokens_explicit = config.max_prompt_tokens is not None

    self.config = config

    # Loop-mode guard: FAMA must stay enabled in loop mode unless explicitly disabled via --fama-disabled
    if str(config.run_mode or "").strip().lower() == "loop" and not config.fama_enabled and not config.fama_disabled:
        logger.warning(
            "FAMA was disabled in config but run_mode is 'loop'. Auto-enabling FAMA to prevent "
            "retry loops and tool misuse. Pass --fama-disabled to override this guard."
        )
        config.fama_enabled = True

    self.state.scratchpad["_fama_config"] = {
        "enabled": bool(config.fama_enabled),
        "mode": str(config.fama_mode or "lite"),
        "capsule_token_budget": int(config.fama_capsule_token_budget),
        "llm_judge_enabled": bool(config.fama_llm_judge_enabled),
        "llm_judge_min_severity": int(config.fama_llm_judge_min_severity),
    }
    if not config.fama_enabled:
        logger.warning(
            "FAMA (failure-aware mitigation) is DISABLED. "
            "Retry loops, verifier failures, and tool misuse will not be auto-detected. "
            "Run with --fama-enabled or remove fama_enabled: false from config to enable."
        )
        self.state.scratchpad["_fama_config"]["_disable_warning_emitted"] = True
    self.state.scratchpad["_recovery_config"] = {
        "reflexion_enabled": bool(config.reflexion_enabled),
        "reflexion_inject_top_k": int(config.reflexion_inject_top_k),
        "subtask_ledger_enabled": bool(config.subtask_ledger_enabled),
        "subtask_inject_completed_limit": int(config.subtask_inject_completed_limit),
    }
    self.state.scratchpad["_chunk_write_loop_guard_config"] = {
        "enabled": config.loop_guard_enabled,
        "stagnation_threshold": config.loop_guard_stagnation_threshold,
        "level2_threshold": config.loop_guard_level2_threshold,
        "recent_writes_limit": config.loop_guard_recent_writes_limit,
        "tail_lines": config.loop_guard_tail_lines,
        "similarity_threshold": config.loop_guard_similarity_threshold,
        "cumulative_write_gate": config.loop_guard_cumulative_write_gate,
        "checkpoint_gate": config.loop_guard_checkpoint_gate,
        "diff_gate": config.loop_guard_diff_gate,
    }

    self._configured_planning_mode = bool(config.planning_mode)
    self.checkpoint_on_exit = config.checkpoint_on_exit
    self.checkpoint_path = config.checkpoint_path
    self.graph_checkpointer = str(config.graph_checkpointer or "memory").strip().lower()
    self.graph_checkpoint_path = config.graph_checkpoint_path
    self.fresh_run = config.fresh_run
    self.fresh_run_turns = max(0, int(config.fresh_run_turns))
    self._fresh_run_turns_remaining = self.fresh_run_turns if self.fresh_run else 0
    self.registry = build_registry(self)
    self.dispatcher = ToolDispatcher(
        registry=self.registry,
        state=self.state,
        phase=normalized_phase,
        run_logger=config.run_logger,
    )
    self.configured_max_prompt_tokens: int | None = config.max_prompt_tokens
    self.configured_max_prompt_tokens_explicit = bool(config.max_prompt_tokens_explicit)

    known_server_context_limit = config.context_limit
    if (
        config.runtime_context_probe
        and config.max_prompt_tokens is not None
        and config.context_limit is not None
        and int(config.context_limit) == int(config.max_prompt_tokens)
    ):
        config.context_limit = None
        known_server_context_limit = None

    # Context scaling reads these before the bootstrap finalizer reassigns them
    self.discovered_server_context_limit = known_server_context_limit
    self.server_context_limit = known_server_context_limit

    effective_max_prompt_tokens = self._resolve_effective_prompt_budget(
        configured_max_prompt_tokens=self.configured_max_prompt_tokens,
        configured_max_prompt_tokens_explicit=self.configured_max_prompt_tokens_explicit,
        server_context_limit=known_server_context_limit,
        provider_profile=self.provider_profile,
        model_name=config.model,
    )
    if effective_max_prompt_tokens is None and self.configured_max_prompt_tokens is not None:
        effective_max_prompt_tokens = max(64, int(self.configured_max_prompt_tokens))

    configured_prompt_budget = self.configured_max_prompt_tokens
    if configured_prompt_budget is None and config.policy is not None:
        configured_prompt_budget = config.policy.max_prompt_tokens
    self.configured_max_prompt_tokens = configured_prompt_budget
    self.context_policy = build_context_policy(
        policy=config.policy,
        effective_max_prompt_tokens=effective_max_prompt_tokens,
        reserve_completion_tokens=config.reserve_completion_tokens,
        reserve_tool_tokens=config.reserve_tool_tokens,
        summarize_at_ratio=config.summarize_at_ratio,
        recent_message_limit=config.recent_message_limit,
        max_summary_items=config.max_summary_items,
        max_artifact_snippets=config.max_artifact_snippets,
        artifact_snippet_token_limit=config.artifact_snippet_token_limit,
        artifact_summarization_threshold=config.artifact_summarization_threshold,
        multi_file_artifact_snippet_limit=config.multi_file_artifact_snippet_limit,
        multi_file_primary_file_limit=config.multi_file_primary_file_limit,
        remote_task_artifact_snippet_limit=config.remote_task_artifact_snippet_limit,
        remote_task_primary_file_limit=config.remote_task_primary_file_limit,
        tool_result_inline_token_limit=config.tool_result_inline_token_limit,
        artifact_read_inline_token_limit=config.artifact_read_inline_token_limit,
    )
    self.context_policy.apply_backend_profile(self.provider_profile)
    self.prompt_assembler = PromptAssembler(self.context_policy)
    self.retriever = LexicalRetriever(self.context_policy)
    self.summarizer = ContextSummarizer(self.context_policy)
    self.subtask_runner = SubtaskRunner(max_child_depth=1)
    self.guards = GuardConfig()
    scaling_context = self.context_policy.max_prompt_tokens or self.server_context_limit
    finalize_harness_bootstrap(
        self=self,
        known_server_context_limit=known_server_context_limit,
        artifact_start_index=config.artifact_start_index,
        provider_profile=self.provider_profile,
        context_policy=self.context_policy,
        scaling_context=scaling_context,
    )
    self.context_policy.apply_model_profile(config.model)
    if scaling_context is not None:
        self.context_policy.recalculate_quotas(scaling_context)
        self.state.recent_message_limit = self.context_policy.recent_message_limit

    # Lifecycle telemetry: harness fully initialized (Fix 1)
    logger.info(
        "harness_initialized session_id=%s model=%s phase=%s",
        self.state.thread_id,
        config.model,
        self._initial_phase,
    )
