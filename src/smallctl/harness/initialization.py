from __future__ import annotations

import asyncio
import asyncio.subprocess
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    build_harness_kwargs,
    build_initial_state,
    finalize_harness_bootstrap,
    resolve_provider_profile,
)
from .approvals import ApprovalService
from .backend_recovery import BackendRecoveryService
from .compaction import CompactionService
from .factory import SubtaskService
from .memory import MemoryService
from .prompt_builder import PromptBuilderService
from .run_mode import ModeDecisionService
from .task_boundary import TaskBoundaryService
from .tool_results import ToolResultService


def initialize_harness(self: Any, **params: Any) -> None:
    endpoint = params["endpoint"]
    model = params["model"]
    phase = params.get("phase", "explore")
    provider_profile = params.get("provider_profile", "generic")
    api_key = params.get("api_key")
    tool_profiles = params.get("tool_profiles")
    reasoning_mode = params.get("reasoning_mode", "auto")
    thinking_visibility = params.get("thinking_visibility", True)
    thinking_start_tag = params.get("thinking_start_tag", "<think>")
    thinking_end_tag = params.get("thinking_end_tag", "</think>")
    chat_endpoint = params.get("chat_endpoint", "/chat/completions")
    runtime_context_probe = params.get("runtime_context_probe", True)
    context_limit = params.get("context_limit")
    max_prompt_tokens = params.get("max_prompt_tokens")
    max_prompt_tokens_explicit = params.get("max_prompt_tokens_explicit", max_prompt_tokens is not None)
    reserve_completion_tokens = params.get("reserve_completion_tokens", 1024)
    reserve_tool_tokens = params.get("reserve_tool_tokens", 512)
    first_token_timeout_sec = params.get("first_token_timeout_sec")
    healthcheck_url = params.get("healthcheck_url")
    restart_command = params.get("restart_command")
    startup_grace_period_sec = params.get("startup_grace_period_sec", 20)
    max_restarts_per_hour = params.get("max_restarts_per_hour", 2)
    backend_healthcheck_url = params.get("backend_healthcheck_url")
    backend_restart_command = params.get("backend_restart_command")
    backend_unload_command = params.get("backend_unload_command")
    backend_healthcheck_timeout_sec = params.get("backend_healthcheck_timeout_sec", 5)
    backend_restart_grace_sec = params.get("backend_restart_grace_sec", 20)
    summarize_at_ratio = params.get("summarize_at_ratio", 0.8)
    recent_message_limit = params.get("recent_message_limit", 24)
    max_summary_items = params.get("max_summary_items", 3)
    max_artifact_snippets = params.get("max_artifact_snippets", 4)
    artifact_snippet_token_limit = params.get("artifact_snippet_token_limit", 400)
    multi_file_artifact_snippet_limit = params.get("multi_file_artifact_snippet_limit", 8)
    multi_file_primary_file_limit = params.get("multi_file_primary_file_limit", 3)
    remote_task_artifact_snippet_limit = params.get("remote_task_artifact_snippet_limit", 8)
    remote_task_primary_file_limit = params.get("remote_task_primary_file_limit", 2)
    checkpoint_on_exit = params.get("checkpoint_on_exit", False)
    checkpoint_path = params.get("checkpoint_path")
    graph_checkpointer = params.get("graph_checkpointer", "memory")
    graph_checkpoint_path = params.get("graph_checkpoint_path")
    fresh_run = params.get("fresh_run", False)
    fresh_run_turns = params.get("fresh_run_turns", 1)
    planning_mode = params.get("planning_mode", False)
    contract_flow_ui = params.get("contract_flow_ui", False)
    summarizer_endpoint = params.get("summarizer_endpoint")
    summarizer_model = params.get("summarizer_model")
    summarizer_api_key = params.get("summarizer_api_key")
    run_logger = params.get("run_logger")
    artifact_start_index = params.get("artifact_start_index")
    tool_result_inline_token_limit = params.get("tool_result_inline_token_limit", 250)
    artifact_read_inline_token_limit = params.get("artifact_read_inline_token_limit", 1024)
    strategy_prompt = params.get("strategy_prompt")
    strategy = params.get("strategy")
    indexer = params.get("indexer", False)
    policy = params.get("policy")
    allow_interactive_shell_approval = params.get("allow_interactive_shell_approval", False)
    shell_approval_session_default = params.get("shell_approval_session_default", False)
    loop_guard_enabled = params.get("loop_guard_enabled", True)
    loop_guard_stagnation_threshold = params.get("loop_guard_stagnation_threshold", 3)
    loop_guard_level2_threshold = params.get("loop_guard_level2_threshold", 5)
    loop_guard_recent_writes_limit = params.get("loop_guard_recent_writes_limit", 5)
    loop_guard_tail_lines = params.get("loop_guard_tail_lines", 50)
    loop_guard_similarity_threshold = params.get("loop_guard_similarity_threshold", 0.9)
    loop_guard_cumulative_write_gate = params.get("loop_guard_cumulative_write_gate", True)
    loop_guard_checkpoint_gate = params.get("loop_guard_checkpoint_gate", True)
    loop_guard_diff_gate = params.get("loop_guard_diff_gate", True)

    normalized_phase = normalize_phase(phase)
    self._initial_phase = normalized_phase
    self.state = build_initial_state(
        phase=normalized_phase,
        planning_mode=planning_mode,
        strategy=strategy,
    )
    self.backend_healthcheck_url = str(
        healthcheck_url or backend_healthcheck_url or ""
    ).strip() or None
    self.backend_restart_command = str(
        restart_command or backend_restart_command or ""
    ).strip() or None
    self.backend_unload_command = str(backend_unload_command or "").strip() or None
    self.backend_healthcheck_timeout_sec = max(1, int(backend_healthcheck_timeout_sec))
    self.backend_restart_grace_sec = max(
        1,
        int(startup_grace_period_sec if startup_grace_period_sec is not None else backend_restart_grace_sec),
    )
    self.backend_max_restarts_per_hour = max(0, int(max_restarts_per_hour))
    resolved_provider_profile = resolve_provider_profile(endpoint, model, provider_profile)
    self.client = build_client(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        chat_endpoint=chat_endpoint,
        provider_profile=resolved_provider_profile,
        first_token_timeout_sec=first_token_timeout_sec,
        runtime_context_probe=runtime_context_probe,
        run_logger=run_logger,
        backend_recovery_handler=self.recover_backend_wedge,
    )
    self.summarizer_client = None
    if summarizer_endpoint:
        self.summarizer_client = build_client(
            endpoint=summarizer_endpoint,
            model=summarizer_model or model,
            api_key=summarizer_api_key or api_key,
            chat_endpoint=chat_endpoint,
            provider_profile=resolve_provider_profile(
                summarizer_endpoint,
                summarizer_model or model,
                "auto",
            ),
            first_token_timeout_sec=first_token_timeout_sec,
            runtime_context_probe=False,
            run_logger=run_logger,
        )
    self.reasoning_mode = reasoning_mode
    self.thinking_visibility = thinking_visibility
    self.thinking_start_tag = thinking_start_tag
    self.thinking_end_tag = thinking_end_tag
    self.state.scratchpad["_model_name"] = model
    self.state.scratchpad["_model_is_small"] = self._is_small_model_name(model)
    self._backend_recovery_service = BackendRecoveryService(self)
    self._task_boundary_service = TaskBoundaryService(self)
    self._active_processes: set[asyncio.subprocess.Process] = set()
    self._background_persistence_tasks: set[asyncio.Task[Any]] = set()
    self._teardown_task: asyncio.Task[None] | None = None
    self.strategy_prompt = strategy_prompt
    self.event_handler = None
    self.allow_interactive_shell_approval = bool(allow_interactive_shell_approval)
    self.shell_approval_session_default = bool(shell_approval_session_default)
    self._configured_tool_profiles = list(tool_profiles) if tool_profiles else None
    self._strategy_prompt = strategy_prompt
    self._indexer = indexer
    self.provider_profile = self.client.provider_profile
    self._harness_kwargs = build_harness_kwargs(
        endpoint=endpoint,
        model=model,
        phase=normalized_phase,
        provider_profile=self.provider_profile,
        api_key=api_key,
        tool_profiles=tool_profiles,
        reasoning_mode=reasoning_mode,
        thinking_visibility=thinking_visibility,
        thinking_start_tag=thinking_start_tag,
        thinking_end_tag=thinking_end_tag,
        chat_endpoint=chat_endpoint,
        runtime_context_probe=runtime_context_probe,
        graph_checkpointer=graph_checkpointer,
        graph_checkpoint_path=graph_checkpoint_path,
        fresh_run=fresh_run,
        fresh_run_turns=fresh_run_turns,
        planning_mode=planning_mode,
        contract_flow_ui=contract_flow_ui,
        context_limit=context_limit,
        max_prompt_tokens=max_prompt_tokens,
        max_prompt_tokens_explicit=bool(max_prompt_tokens_explicit),
        reserve_completion_tokens=reserve_completion_tokens,
        reserve_tool_tokens=reserve_tool_tokens,
        first_token_timeout_sec=first_token_timeout_sec,
        healthcheck_url=self.backend_healthcheck_url,
        restart_command=self.backend_restart_command,
        startup_grace_period_sec=self.backend_restart_grace_sec,
        max_restarts_per_hour=self.backend_max_restarts_per_hour,
        backend_healthcheck_url=self.backend_healthcheck_url,
        backend_restart_command=self.backend_restart_command,
        backend_unload_command=self.backend_unload_command,
        backend_healthcheck_timeout_sec=self.backend_healthcheck_timeout_sec,
        backend_restart_grace_sec=self.backend_restart_grace_sec,
        summarize_at_ratio=summarize_at_ratio,
        recent_message_limit=recent_message_limit,
        max_summary_items=max_summary_items,
        max_artifact_snippets=max_artifact_snippets,
        artifact_snippet_token_limit=artifact_snippet_token_limit,
        multi_file_artifact_snippet_limit=multi_file_artifact_snippet_limit,
        multi_file_primary_file_limit=multi_file_primary_file_limit,
        remote_task_artifact_snippet_limit=remote_task_artifact_snippet_limit,
        remote_task_primary_file_limit=remote_task_primary_file_limit,
        run_logger=run_logger,
        artifact_start_index=artifact_start_index,
        tool_result_inline_token_limit=tool_result_inline_token_limit,
        indexer=indexer,
        allow_interactive_shell_approval=self.allow_interactive_shell_approval,
        shell_approval_session_default=self.shell_approval_session_default,
        provider_profile_resolved=self.provider_profile,
    )
    self._harness_kwargs.update(
        {
            "loop_guard_enabled": bool(loop_guard_enabled),
            "loop_guard_stagnation_threshold": int(loop_guard_stagnation_threshold),
            "loop_guard_level2_threshold": int(loop_guard_level2_threshold),
            "loop_guard_recent_writes_limit": int(loop_guard_recent_writes_limit),
            "loop_guard_tail_lines": int(loop_guard_tail_lines),
            "loop_guard_similarity_threshold": float(loop_guard_similarity_threshold),
            "loop_guard_cumulative_write_gate": bool(loop_guard_cumulative_write_gate),
            "loop_guard_checkpoint_gate": bool(loop_guard_checkpoint_gate),
            "loop_guard_diff_gate": bool(loop_guard_diff_gate),
        }
    )
    self.config = SimpleNamespace(**self._harness_kwargs)
    self.state.scratchpad["_chunk_write_loop_guard_config"] = {
        "enabled": self.config.loop_guard_enabled,
        "stagnation_threshold": self.config.loop_guard_stagnation_threshold,
        "level2_threshold": self.config.loop_guard_level2_threshold,
        "recent_writes_limit": self.config.loop_guard_recent_writes_limit,
        "tail_lines": self.config.loop_guard_tail_lines,
        "similarity_threshold": self.config.loop_guard_similarity_threshold,
        "cumulative_write_gate": self.config.loop_guard_cumulative_write_gate,
        "checkpoint_gate": self.config.loop_guard_checkpoint_gate,
        "diff_gate": self.config.loop_guard_diff_gate,
    }
    self._configured_planning_mode = bool(planning_mode)
    self.checkpoint_on_exit = checkpoint_on_exit
    self.checkpoint_path = checkpoint_path
    self.graph_checkpointer = str(graph_checkpointer or "memory").strip().lower()
    self.graph_checkpoint_path = graph_checkpoint_path
    self.fresh_run = fresh_run
    self.fresh_run_turns = max(0, int(fresh_run_turns))
    self._fresh_run_turns_remaining = self.fresh_run_turns if self.fresh_run else 0
    self.registry = build_registry(self)
    self.dispatcher = ToolDispatcher(
        registry=self.registry,
        state=self.state,
        phase=normalized_phase,
        run_logger=run_logger,
    )
    self.configured_max_prompt_tokens: int | None = max_prompt_tokens
    self.configured_max_prompt_tokens_explicit = bool(max_prompt_tokens_explicit)

    known_server_context_limit = context_limit
    if (
        runtime_context_probe
        and max_prompt_tokens is not None
        and context_limit is not None
        and int(context_limit) == int(max_prompt_tokens)
    ):
        self._harness_kwargs["context_limit"] = None
        known_server_context_limit = None

    # Context scaling reads these attributes before the later bootstrap finalizer
    # reassigns them, so initialize them as soon as the known server limit is settled.
    self.discovered_server_context_limit = known_server_context_limit
    self.server_context_limit = known_server_context_limit

    if self.configured_max_prompt_tokens is not None:
        effective_max_prompt_tokens = max(64, int(self.configured_max_prompt_tokens))
    else:
        effective_max_prompt_tokens = self._resolve_effective_prompt_budget(
            configured_max_prompt_tokens=self.configured_max_prompt_tokens,
            configured_max_prompt_tokens_explicit=self.configured_max_prompt_tokens_explicit,
            server_context_limit=known_server_context_limit,
        )

    configured_prompt_budget = self.configured_max_prompt_tokens
    if configured_prompt_budget is None and policy is not None:
        configured_prompt_budget = policy.max_prompt_tokens
    self.configured_max_prompt_tokens = configured_prompt_budget
    self.context_policy = build_context_policy(
        policy=policy,
        effective_max_prompt_tokens=effective_max_prompt_tokens,
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
        artifact_start_index=artifact_start_index,
        provider_profile=self.provider_profile,
        context_policy=self.context_policy,
        scaling_context=scaling_context,
    )
    self.context_policy.apply_model_profile(model)
    if scaling_context is not None:
        self.context_policy.recalculate_quotas(scaling_context)
        self.state.recent_message_limit = self.context_policy.recent_message_limit
