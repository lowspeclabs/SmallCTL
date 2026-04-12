from __future__ import annotations

import json
import logging
import asyncio
import asyncio.subprocess
import re
import shlex
import uuid
import sys
import ast
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from ..client import OpenAICompatClient
from ..client.usage import detect_provider_profile
from ..context import (
    ArtifactStore,
    ChildRunRequest,
    ChildRunResult,
    ContextPolicy,
    ContextSummarizer,
    MessageTierManager,
    LexicalRetriever,
    PromptAssembler,
    SubtaskRunner,
    build_retrieval_query,
    format_reused_artifact_message,
)
from ..guards import GuardConfig, is_small_model_name
from ..logging_utils import RunLogger, log_kv
from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..models.events import UIEvent, UIEventType
from ..phases import normalize_phase
from ..state import (
    LOOP_STATE_SCHEMA_VERSION,
    ExperienceMemory,
    LoopState,
    PromptBudgetSnapshot,
    RunBrief,
    WorkingMemory,
    _coerce_experience_memory,
    align_memory_entries,
    clip_string_list,
    clip_text_value,
    json_safe_value,
)
from ..task_targets import extract_task_target_paths
from ..normalization import dedupe_keep_tail
from ..plans import write_plan_file
from ..tools import ToolDispatcher, build_registry
from ..tools.ansible import AnsibleRunnerAdapter, SessionInventory
from ..tools.profiles import classify_tool_profiles
from ..memory.taxonomy import (
    PHASE_MISMATCH,
    PREMATURE_TASK_COMPLETE,
    SCHEMA_VALIDATION_ERROR,
    TOOL_NOT_CALLED,
    ZERO_ARG_TOOL_ARG_LEAK,
    REPEATED_TOOL_LOOP,
    WRONG_TOOL_CALLED,
    UNKNOWN_FAILURE,
)
from .run_mode import ModeDecisionService, should_enable_complex_write_chat_draft
from .prompt_builder import PromptBuilderService
from .tool_results import ToolResultService
from .compaction import CompactionService
from .memory import MemoryService
from .approvals import ApprovalService
from .factory import SubtaskService


class Harness:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        phase: str = "explore",
        provider_profile: str = "generic",
        api_key: str | None = None,
        tool_profiles: list[str] | None = None,
        use_ansible: bool = True,
        inventory_path: str | None = None,
        ansible_check_mode_in_plan: bool = True,
        reasoning_mode: str = "auto",
        thinking_visibility: bool = True,
        thinking_start_tag: str = "<think>",
        thinking_end_tag: str = "</think>",
        chat_endpoint: str = "/chat/completions",
        runtime_context_probe: bool = True,
        context_limit: int | None = None,
        max_prompt_tokens: int | None = None,
        reserve_completion_tokens: int = 1024,
        reserve_tool_tokens: int = 512,
        first_token_timeout_sec: int | None = None,
        healthcheck_url: str | None = None,
        restart_command: str | None = None,
        startup_grace_period_sec: int = 20,
        max_restarts_per_hour: int = 2,
        backend_healthcheck_url: str | None = None,
        backend_restart_command: str | None = None,
        backend_unload_command: str | None = None,
        backend_healthcheck_timeout_sec: int = 5,
        backend_restart_grace_sec: int = 20,
        summarize_at_ratio: float = 0.8,
        recent_message_limit: int = 24,
        max_summary_items: int = 3,
        max_artifact_snippets: int = 4,
        artifact_snippet_token_limit: int = 400,
        checkpoint_on_exit: bool = False,
        checkpoint_path: str | None = None,
        graph_checkpointer: str = "memory",
        graph_checkpoint_path: str | None = None,
        fresh_run: bool = False,
        fresh_run_turns: int = 1,
        planning_mode: bool = False,
        contract_flow_ui: bool = False,
        summarizer_endpoint: str | None = None,
        summarizer_model: str | None = None,
        summarizer_api_key: str | None = None,
        run_logger: RunLogger | None = None,
        artifact_start_index: int | None = None,
        tool_result_inline_token_limit: int = 250,
        artifact_read_inline_token_limit: int = 1024,
        strategy_prompt: str | None = None,
        strategy: dict[str, Any] | None = None,
        indexer: bool = False,
        policy: ContextPolicy | None = None,
        allow_interactive_shell_approval: bool = False,
        shell_approval_session_default: bool = False,
    ) -> None:
        self.log = logging.getLogger("smallctl.harness")
        self.run_logger = run_logger
        normalized_phase = normalize_phase(phase)
        self._initial_phase = normalized_phase
        self.state = LoopState(
            current_phase=normalized_phase,
            strategy=strategy,
            planning_mode_enabled=bool(planning_mode),
        )
        if isinstance(strategy, dict):
            self.state.scratchpad["strategy"] = json_safe_value(strategy)
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
        self.client = OpenAICompatClient(
            base_url=endpoint,
            model=model,
            api_key=api_key,
            chat_endpoint=chat_endpoint,
            provider_profile=self._resolve_provider_profile(provider_profile, endpoint=endpoint, model=model),
            first_token_timeout_sec=first_token_timeout_sec,
            runtime_context_probe=runtime_context_probe,
            run_logger=run_logger,
            backend_recovery_handler=self.recover_backend_wedge,
        )
        self.summarizer_client = None
        if summarizer_endpoint:
            summarizer_provider = detect_provider_profile(
                summarizer_endpoint,
                summarizer_model or model,
            )
            self.summarizer_client = OpenAICompatClient(
                base_url=summarizer_endpoint,
                model=summarizer_model or model,
                api_key=summarizer_api_key or api_key,
                chat_endpoint=chat_endpoint,
                provider_profile=summarizer_provider,
                first_token_timeout_sec=first_token_timeout_sec,
                runtime_context_probe=False,
                run_logger=run_logger,
            )
        self.reasoning_mode = reasoning_mode
        self.thinking_visibility = thinking_visibility
        self.thinking_start_tag = thinking_start_tag
        self.thinking_end_tag = thinking_end_tag
        self.state.scratchpad["_model_name"] = model
        self.state.scratchpad["_model_is_small"] = is_small_model_name(model)
        self._active_processes: set[asyncio.subprocess.Process] = set()
        self.strategy_prompt = strategy_prompt
        self.event_handler = None
        self.allow_interactive_shell_approval = bool(allow_interactive_shell_approval)
        self.shell_approval_session_default = bool(shell_approval_session_default)
        self._configured_tool_profiles = list(tool_profiles) if tool_profiles else None
        self._strategy_prompt = strategy_prompt
        self._indexer = indexer
        self._use_ansible = use_ansible
        self.provider_profile = self.client.provider_profile
        self._harness_kwargs = {
            "endpoint": endpoint,
            "model": model,
            "phase": normalized_phase,
            "provider_profile": self.provider_profile,
            "api_key": api_key,
            "tool_profiles": tool_profiles,
            "use_ansible": use_ansible,
            "inventory_path": inventory_path,
            "ansible_check_mode_in_plan": ansible_check_mode_in_plan,
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
            "reserve_completion_tokens": reserve_completion_tokens,
            "reserve_tool_tokens": reserve_tool_tokens,
            "first_token_timeout_sec": first_token_timeout_sec,
            "healthcheck_url": self.backend_healthcheck_url,
            "restart_command": self.backend_restart_command,
            "startup_grace_period_sec": self.backend_restart_grace_sec,
            "max_restarts_per_hour": self.backend_max_restarts_per_hour,
            "backend_healthcheck_url": self.backend_healthcheck_url,
            "backend_restart_command": self.backend_restart_command,
            "backend_unload_command": self.backend_unload_command,
            "backend_healthcheck_timeout_sec": self.backend_healthcheck_timeout_sec,
            "backend_restart_grace_sec": self.backend_restart_grace_sec,
            "summarize_at_ratio": summarize_at_ratio,
            "recent_message_limit": recent_message_limit,
            "max_summary_items": max_summary_items,
            "max_artifact_snippets": max_artifact_snippets,
            "artifact_snippet_token_limit": artifact_snippet_token_limit,
            "run_logger": run_logger,
            "artifact_start_index": artifact_start_index,
            "tool_result_inline_token_limit": tool_result_inline_token_limit,
            "indexer": indexer,
            "allow_interactive_shell_approval": self.allow_interactive_shell_approval,
            "shell_approval_session_default": self.shell_approval_session_default,
        }
        self.config = SimpleNamespace(**self._harness_kwargs)
        self._configured_planning_mode = bool(planning_mode)
        self.checkpoint_on_exit = checkpoint_on_exit
        self.checkpoint_path = checkpoint_path
        self.graph_checkpointer = str(graph_checkpointer or "memory").strip().lower()
        self.graph_checkpoint_path = graph_checkpoint_path
        self.fresh_run = fresh_run
        self.fresh_run_turns = max(0, int(fresh_run_turns))
        self._fresh_run_turns_remaining = self.fresh_run_turns if self.fresh_run else 0
        tier2_adapter = None
        if use_ansible:
            inventory = SessionInventory.localhost_default()
            if inventory_path:
                try:
                    inventory.merge_inventory_file(inventory_path)
                except Exception as exc:
                    self.log.warning("Unable to merge inventory file: %s", exc)
            self.state.inventory_state = inventory.list()
            runner = AnsibleRunnerAdapter(inventory)
            tier2_adapter = runner.dispatch
        self.registry = build_registry(
            self,
            include_ansible=use_ansible,
        )
        self.dispatcher = ToolDispatcher(
            registry=self.registry,
            state=self.state,
            phase=normalized_phase,
            ansible_check_mode_in_plan=ansible_check_mode_in_plan,
            tier2_adapter=tier2_adapter,
            run_logger=run_logger,
        )
        configured_prompt_budget = max_prompt_tokens
        if configured_prompt_budget is None and policy is not None:
            configured_prompt_budget = policy.max_prompt_tokens
        self.configured_max_prompt_tokens: int | None = configured_prompt_budget

        known_server_context_limit = context_limit
        if (
            runtime_context_probe
            and max_prompt_tokens is not None
            and context_limit is not None
            and int(context_limit) == int(max_prompt_tokens)
        ):
            # Legacy config paths can mirror the prompt budget into context_limit.
            # Treat that as "unknown" so we still probe the real runtime window.
            known_server_context_limit = None
            self._harness_kwargs["context_limit"] = None

        effective_max_prompt_tokens = self._resolve_effective_prompt_budget(
            configured_max_prompt_tokens=self.configured_max_prompt_tokens,
            server_context_limit=known_server_context_limit,
        )

        if policy is None:
            self.context_policy = ContextPolicy(
                max_prompt_tokens=effective_max_prompt_tokens,
                reserve_completion_tokens=reserve_completion_tokens,
                reserve_tool_tokens=reserve_tool_tokens,
                summarize_at_ratio=summarize_at_ratio,
                recent_message_limit=recent_message_limit,
                max_summary_items=max_summary_items,
                max_artifact_snippets=max_artifact_snippets,
                artifact_snippet_token_limit=artifact_snippet_token_limit,
                tool_result_inline_token_limit=tool_result_inline_token_limit,
                artifact_read_inline_token_limit=artifact_read_inline_token_limit,
            )
        else:
            self.context_policy = policy
            if effective_max_prompt_tokens is not None:
                self.context_policy.max_prompt_tokens = effective_max_prompt_tokens
        self.context_policy.apply_backend_profile(self.provider_profile)
        self.state.recent_message_limit = self.context_policy.recent_message_limit
        self.prompt_assembler = PromptAssembler(self.context_policy)
        self.retriever = LexicalRetriever(self.context_policy)
        self.summarizer = ContextSummarizer(self.context_policy)
        self.subtask_runner = SubtaskRunner(max_child_depth=1)
        self.guards = GuardConfig()
        self.discovered_server_context_limit: int | None = known_server_context_limit
        self.server_context_limit: int | None = known_server_context_limit
        scaling_context = self.server_context_limit or self.context_policy.max_prompt_tokens
        if scaling_context is not None:
            self.context_policy.recalculate_quotas(
                scaling_context,
                backend_profile=self.provider_profile,
            )
            self.state.recent_message_limit = self.context_policy.recent_message_limit
        self.conversation_id = uuid.uuid4().hex[:8]
        if not self.state.thread_id:
            self.state.thread_id = self.conversation_id
        self._sync_run_logger_session_id()
        artifact_base_dir = Path(self.state.cwd).resolve() / ".smallctl" / "artifacts"
        self.artifact_store = ArtifactStore(
            artifact_base_dir, 
            self.conversation_id,
            artifact_start_index=artifact_start_index
        )
        memory_base_dir = Path(self.state.cwd).resolve() / ".smallctl" / "memory"
        from ..memory_store import ExperienceStore
        self.warm_memory_store = ExperienceStore(memory_base_dir / "warm-experiences.jsonl")
        self.cold_memory_store = ExperienceStore(memory_base_dir / "cold-experiences.jsonl")
        # Fresh harness starts should begin empty. Prior experience is restored
        # only through an explicit graph resume, which hydrates the loop state.

        self._cancel_requested = False
        self._active_dispatch_task: asyncio.Task[Any] | None = None
        self._active_task_scope: dict[str, Any] | None = None
        self._task_sequence = 0
        self._pending_task_shutdown_reason = ""
        self._use_ansible = use_ansible
        self._configured_tool_profiles = list(tool_profiles) if tool_profiles else None

        # Initialize Services
        self.mode_decision = ModeDecisionService(self)
        self.prompt_builder = PromptBuilderService(self)
        self.tool_results = ToolResultService(self)
        self.compaction = CompactionService(self)
        self.memory = MemoryService(self)
        self.approvals = ApprovalService(self)
        self.subtasks = SubtaskService(self)
        
        self._runlog(
            "harness_started",
            "harness initialized",
            endpoint=endpoint,
            model=model,
            phase=normalized_phase,
            provider_profile=self.provider_profile,
        )

    def set_interactive_shell_approval(self, enabled: bool) -> None:
        self.allow_interactive_shell_approval = bool(enabled)
        self._harness_kwargs["allow_interactive_shell_approval"] = self.allow_interactive_shell_approval

    def set_shell_approval_session_default(self, enabled: bool) -> None:
        self.shell_approval_session_default = bool(enabled)
        self._harness_kwargs["shell_approval_session_default"] = self.shell_approval_session_default

    @staticmethod
    def _context_limit_headroom(context_limit: int) -> int:
        return max(1024, int(context_limit) // 4)

    @staticmethod
    def _resolve_provider_profile(provider_profile: str, *, endpoint: str, model: str) -> str:
        profile = str(provider_profile or "auto").strip().lower()
        if profile != "auto":
            return profile
        return detect_provider_profile(endpoint, model)

    @classmethod
    def _derive_prompt_budget_from_context_limit(cls, context_limit: int | None) -> int | None:
        if context_limit is None:
            return None
        limit = max(1, int(context_limit))
        return max(64, limit - cls._context_limit_headroom(limit))

    @classmethod
    def _resolve_effective_prompt_budget(
        cls,
        *,
        configured_max_prompt_tokens: int | None,
        server_context_limit: int | None,
        current_max_prompt_tokens: int | None = None,
        observed_n_keep: int | None = None,
    ) -> int | None:
        candidates: list[int] = []
        if configured_max_prompt_tokens is not None:
            candidates.append(max(64, int(configured_max_prompt_tokens)))

        derived_server_budget = cls._derive_prompt_budget_from_context_limit(server_context_limit)
        if derived_server_budget is not None:
            candidates.append(derived_server_budget)

        if (
            current_max_prompt_tokens is not None
            and observed_n_keep is not None
            and server_context_limit is not None
            and observed_n_keep >= server_context_limit
        ):
            overflow = observed_n_keep - server_context_limit + 128
            candidates.append(max(64, int(current_max_prompt_tokens) - overflow))

        if not candidates:
            return None
        return min(candidates)

    def _apply_server_context_limit(
        self,
        context_limit: int | None,
        *,
        source: str,
        observed_n_keep: int | None = None,
    ) -> int | None:
        if context_limit is None:
            return self.context_policy.max_prompt_tokens

        normalized_limit = max(1, int(context_limit))
        if self.server_context_limit is None or normalized_limit < self.server_context_limit:
            self.discovered_server_context_limit = normalized_limit
            self.server_context_limit = normalized_limit

        effective_max_prompt_tokens = self._resolve_effective_prompt_budget(
            configured_max_prompt_tokens=self.configured_max_prompt_tokens,
            server_context_limit=self.server_context_limit,
            current_max_prompt_tokens=self.context_policy.max_prompt_tokens,
            observed_n_keep=observed_n_keep,
        )
        if effective_max_prompt_tokens is not None:
            self.context_policy.max_prompt_tokens = effective_max_prompt_tokens

        if self.server_context_limit is not None:
            self.context_policy.recalculate_quotas(
                self.server_context_limit,
                backend_profile=self.provider_profile,
            )
            self.state.recent_message_limit = self.context_policy.recent_message_limit
            self._harness_kwargs["context_limit"] = self.server_context_limit

        self._runlog(
            "context_limit",
            "server context limit applied",
            source=source,
            context_limit=self.server_context_limit,
            configured_max_prompt_tokens=self.configured_max_prompt_tokens,
            max_prompt_tokens=self.context_policy.max_prompt_tokens,
            hot_message_limit=self.context_policy.hot_message_limit,
        )
        return self.context_policy.max_prompt_tokens

    async def run_task(self, task: str) -> dict[str, Any]:
        return await self.run_task_with_events(task)

    async def run_subtask(
        self,
        brief: str,
        phase: str = "plan",
        depth: int = 1,
        max_prompt_tokens: int | None = None,
        recent_message_limit: int = 4,
        metadata: dict[str, Any] | None = None,
        harness_factory: Callable[..., "Harness"] | None = None,
        artifact_start_index: int | None = None,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> ChildRunResult:
        return await self.subtasks.run_subtask(
            brief=brief,
            phase=phase,
            depth=depth,
            max_prompt_tokens=max_prompt_tokens,
            recent_message_limit=recent_message_limit,
            metadata=metadata,
            harness_factory=harness_factory,
            artifact_start_index=artifact_start_index,
            event_handler=event_handler,
        )

    async def run_auto(
        self,
        task: str,
        *,
        event_handler: Callable[[UIEvent], Awaitable[None]] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        from ..graph.runtime import (
            ChatGraphRuntime,
            IndexerGraphRuntime,
            LoopGraphRuntime,
        )

        if self._indexer:
            runtime = IndexerGraphRuntime.from_harness(self, event_handler=event_handler)
            return await runtime.run(task)

        return await self.run_auto_with_events(task, event_handler=event_handler)

    async def resume_task(self, human_input: str) -> dict[str, Any]:
        return await self.resume_task_with_events(human_input)

    def restore_graph_state(self, thread_id: str | None = None) -> bool:
        from ..graph.runtime import LoopGraphRuntime

        runtime = LoopGraphRuntime.from_harness(self)
        restored = runtime.restore(thread_id=thread_id)
        if restored:
            self._sync_run_logger_session_id()
        return restored

        return result

    def _sync_run_logger_session_id(self) -> None:
        run_logger = getattr(self, "run_logger", None)
        if run_logger is None or not hasattr(run_logger, "set_session_id"):
            return
        session_id = str(getattr(self.state, "thread_id", "") or self.conversation_id or "").strip()
        if not session_id:
            return
        try:
            run_logger.set_session_id(session_id)
        except Exception:
            self.log.debug("Unable to sync run logger session id", exc_info=True)

    async def decide_run_mode(self, task: str) -> str:
        resolved_task = self._resolve_followup_task(task)
        return await self.mode_decision.decide(resolved_task or task)

    def _set_planning_request(self, *, output_path: str | None = None, output_format: str | None = None) -> None:
        self.mode_decision._set_planning_request(output_path=output_path, output_format=output_format)

    def _extract_planning_request(self, task: str) -> tuple[str | None, str | None] | None:
        return self.mode_decision._extract_planning_request(task)

    async def run_chat_with_events(
        self,
        task: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        from ..graph.runtime import ChatGraphRuntime

        self.event_handler = event_handler
        runtime = ChatGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.run(task)

    async def run_auto_with_events(
        self,
        task: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        from ..graph.runtime import AutoGraphRuntime

        self.event_handler = event_handler
        runtime = AutoGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )

        return await runtime.run(task)

    def cancel(self) -> None:
        self._cancel_requested = True
        self.note_task_shutdown("cancel_requested")
        self.approvals.reject_pending_shell_approvals()
        self.approvals.reject_pending_sudo_password_prompts()
        if self._active_dispatch_task and not self._active_dispatch_task.done():
            self._active_dispatch_task.cancel()
        log_kv(self.log, logging.INFO, "harness_cancel_requested")
        # Direct cleanup on cancel helps avoid abandoned processes
        asyncio.create_task(self.teardown())

    def note_task_shutdown(self, reason: str) -> None:
        self._pending_task_shutdown_reason = str(reason or "").strip()

    async def teardown(self) -> None:
        """Kill and cleanup any remaining active processes efficiently."""
        shutdown_reason = str(getattr(self, "_pending_task_shutdown_reason", "") or "").strip()
        if shutdown_reason:
            self._finalize_task_scope(
                terminal_event="task_interrupted",
                status="interrupted",
                reason=shutdown_reason,
            )
            self._pending_task_shutdown_reason = ""
        self.approvals.reject_pending_shell_approvals()
        self.approvals.reject_pending_sudo_password_prompts()
        if not self._active_processes:
            return
            
        procs = list(self._active_processes)
        self._active_processes.clear()
        
        # 1. Fire termination signals to everyone at once
        for p in procs:
            try:
                if p.returncode is None:
                    p.terminate()
            except (ProcessLookupError, RuntimeError):
                pass
                
        # 2. Give them a collective moment to die
        if procs:
            try:
                # Wait for all to finish with a small collective timeout
                await asyncio.wait(
                    [asyncio.create_task(p.wait()) for p in procs],
                    timeout=0.3
                )
            except Exception:
                pass
                
        # 3. Force kill anyone still lingering
        for p in procs:
            try:
                if p.returncode is None:
                    p.kill()
                    # Ensure the process is fully reaped before we lose the reference
                    # and the loop closes.
                    await p.wait()
            except Exception:
                pass

    async def request_shell_approval(
        self,
        *,
        command: str,
        cwd: str,
        timeout_sec: int,
        proof_bundle: dict[str, Any] | None = None,
    ) -> bool:
        return await self.approvals.request_shell_approval(
            command=command,
            cwd=cwd,
            timeout_sec=timeout_sec,
            proof_bundle=proof_bundle,
        )

    async def request_sudo_password(
        self,
        *,
        command: str,
        prompt_text: str,
    ) -> str | None:
        return await self.approvals.request_sudo_password(command=command, prompt_text=prompt_text)

    def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
        self.approvals.resolve_shell_approval(approval_id, approved)

    def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
        self.approvals.resolve_sudo_password(prompt_id, password)

    def _reject_pending_shell_approvals(self) -> None:
        self.approvals.reject_pending_shell_approvals()

    def _reject_pending_sudo_password_prompts(self) -> None:
        self.approvals.reject_pending_sudo_password_prompts()

    def _reject_shell_approval(self, approval_id: str) -> None:
        # Resolve explicit rejection for the specific approval prompt.
        self.approvals.resolve_shell_approval(approval_id, False)

    async def run_task_with_events(
        self,
        task: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        from ..graph.runtime import LoopGraphRuntime

        self.event_handler = event_handler
        self._reset_task_boundary_state(reason="run_task", new_task=task)
        
        runtime = LoopGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.run(task)

    async def resume_task_with_events(
        self,
        human_input: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        from ..graph.runtime import LoopGraphRuntime, PlanningGraphRuntime

        self.event_handler = event_handler
        interrupt = self.get_pending_interrupt() or {}
        if str(interrupt.get("kind") or "") == "plan_execute_approval":
            runtime = PlanningGraphRuntime.from_harness(
                self,
                event_handler=event_handler,
            )
            return await runtime.resume(human_input)
        runtime = LoopGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )
        return await runtime.resume(human_input)

    def has_pending_interrupt(self) -> bool:
        return isinstance(self.state.pending_interrupt, dict) and bool(self.state.pending_interrupt)

    def get_pending_interrupt(self) -> dict[str, Any] | None:
        if not isinstance(self.state.pending_interrupt, dict):
            return None
        return dict(self.state.pending_interrupt)

    @staticmethod
    async def _emit(
        handler: Callable[[UIEvent], Awaitable[None] | None] | None,
        event: UIEvent,
    ) -> None:
        if handler is None:
            return
        maybe = handler(event)
        if maybe is not None and hasattr(maybe, "__await__"):
            await maybe

    def _finalize(self, result: dict[str, Any]) -> dict[str, Any]:
        status = str((result or {}).get("status") or "").strip().lower()
        task_summary = None
        if status not in {"needs_human", "plan_ready", "plan_approved"}:
            terminal_event = "task_interrupted" if status == "cancelled" else ""
            summary_status = "interrupted" if status == "cancelled" else status
            task_summary = self._finalize_task_scope(
                terminal_event=terminal_event,
                status=summary_status or "stopped",
                reason=str((result or {}).get("reason") or ""),
                result=result,
            )
            self._pending_task_shutdown_reason = ""
        summary_path = str((task_summary or {}).get("summary_path") or "").strip()
        task_id = str((task_summary or {}).get("task_id") or "").strip()
        self._runlog(
            "task_finalize",
            "task finished",
            result=result,
            task_id=task_id,
            task_summary_path=summary_path,
        )
        self._record_terminal_experience(result)
        self._rewrite_active_plan_export()
        if self.checkpoint_on_exit:
            self._persist_checkpoint(result)
            
        # Add runtime metrics for benchmarking/AHO
        result["step_count"] = self.state.step_count
        result["inactive_steps"] = self.state.inactive_steps
        result["token_usage"] = self.state.token_usage

        if getattr(self, "run_logger", None) and hasattr(self.run_logger, "run_dir"):
            try:
                import json
                summary_payload = {
                    "final_task_status": result.get("status", "unknown"),
                    "total_tool_calls": self.state.step_count,
                    "guard_trips": sum(1 for e in (getattr(self.state, "recent_errors", []) or []) if "Guard tripped" in str(e)),
                    "postmortem_summary": result.get("reason") or "No reason provided",
                }
                summary_path = self.run_logger.run_dir / "task_summary.json"
                summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass
        
        self._cancel_requested = False
        self._active_dispatch_task = None
        return result

    def _rewrite_active_plan_export(self) -> None:
        plan = self.state.active_plan or self.state.draft_plan
        if plan is None or not plan.requested_output_path:
            return
        try:
            write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
        except Exception as exc:
            self.log.warning("failed to rewrite active plan export: %s", exc)

    def _create_child_harness(
        self,
        *,
        request: ChildRunRequest,
        harness_factory: Callable[..., "Harness"] | None = None,
        artifact_start_index: int | None = None,
    ) -> "Harness":
        return self.subtasks.create_child_harness(
            request=request,
            harness_factory=harness_factory,
            artifact_start_index=artifact_start_index,
        )

    def _build_subtask_result(
        self,
        *,
        child: "Harness",
        request: ChildRunRequest,
        result: dict[str, Any],
    ) -> ChildRunResult:
        return self.subtasks.build_subtask_result(child=child, request=request, result=result)

    def _persist_checkpoint(self, result: dict[str, Any]) -> None:
        path = (
            Path(self.checkpoint_path).resolve()
            if self.checkpoint_path
            else Path(self.state.cwd).resolve() / ".smallctl-checkpoint.json"
        )
        payload = {
            "checkpoint_schema_version": 1,
            "loop_state_schema_version": LOOP_STATE_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "result": json_safe_value(result),
            "state": self.state.to_dict(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log_kv(self.log, logging.INFO, "harness_checkpoint_saved", path=str(path))
        except Exception:
            self.log.exception("failed to persist checkpoint")

    @staticmethod
    def _failure(
        message: str,
        *,
        error_type: str = "runtime",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "reason": message,
            "error": {
                "type": error_type,
                "message": message,
                "details": details or {},
            },
        }

    def _runlog(self, event: str, message: str, **data: Any) -> None:
        if self.run_logger:
            self.run_logger.log("harness", event, message, **data)
            if event.startswith("model_"):
                self.run_logger.log("model_output", event, message, **data)

    async def recover_backend_wedge(self, payload: dict[str, Any]) -> dict[str, Any]:
        details = dict(payload.get("details") or {})
        health_url = self.backend_healthcheck_url or f"{self.client.base_url}/models"
        timeout_sec = max(1, int(self.backend_healthcheck_timeout_sec))
        health_before = await self._probe_backend_health(health_url, timeout_sec=timeout_sec)
        action = "none"
        status = "unrecovered"
        message = "Backend did not emit a first token before timeout."
        health_after = dict(health_before)
        restart_window: dict[str, Any] | None = None
        attempted_actions: list[str] = []
        unload_available = self.backend_unload_command or self.provider_profile in {"ollama", "lmstudio"}
        if health_before.get("ok") and unload_available:
            action = self._backend_unload_action()
            attempted_actions.append(action)
            command_result = await self._run_backend_unload_command(self.backend_unload_command)
            if command_result.get("ok"):
                health_after = await self._wait_for_backend_health(
                    health_url,
                    timeout_sec=max(timeout_sec, int(self.backend_restart_grace_sec)),
                )
                if health_after.get("ok"):
                    status = "recovered"
                    message = self._backend_unload_message("succeeded and health probe recovered")
                else:
                    message = self._backend_unload_message("ran, but the health probe did not recover")
            else:
                health_after = {"ok": False}
                message = str(command_result.get("message") or "Backend unload command failed.")
            if status != "recovered" and self.backend_restart_command:
                restart_result = await self._attempt_backend_restart_recovery(
                    health_url,
                    timeout_sec=timeout_sec,
                )
                restart_action = str(restart_result.get("action") or "").strip()
                if restart_action == "restart_command":
                    attempted_actions.append(restart_action)
                restart_message = str(restart_result.get("message") or "").strip()
                if restart_message:
                    message = f"{message} {restart_message}".strip()
                action = restart_action or action
                status = str(restart_result.get("status") or status)
                if isinstance(restart_result.get("health_after"), dict):
                    health_after = dict(restart_result["health_after"])
                restart_window = restart_result.get("restart_window")
        elif self.backend_restart_command:
            restart_result = await self._attempt_backend_restart_recovery(
                health_url,
                timeout_sec=timeout_sec,
            )
            action = str(restart_result.get("action") or action)
            status = str(restart_result.get("status") or status)
            message = str(restart_result.get("message") or message)
            if action == "restart_command":
                attempted_actions.append(action)
            if isinstance(restart_result.get("health_after"), dict):
                health_after = dict(restart_result["health_after"])
            restart_window = restart_result.get("restart_window")
        else:
            if health_before.get("ok"):
                message = (
                    "Backend accepted health probes but appears wedged on generation; "
                    "no unload or restart recovery is configured."
                )
            else:
                message = "Backend health probe failed and no restart command is configured."
        result = {
            "status": status,
            "action": action,
            "message": message,
            "health_url": health_url,
            "health_before": health_before,
            "health_after": health_after,
            "reason": str(details.get("reason") or ""),
            "attempted_actions": attempted_actions,
        }
        if restart_window is not None:
            result["restart_window"] = restart_window
        self.state.scratchpad["_last_backend_recovery"] = result
        self._runlog(
            "backend_recovery",
            message,
            provider_profile=self.provider_profile,
            status=status,
            action=action,
            health_url=health_url,
            health_before=health_before,
            health_after=health_after,
            details=details,
            attempted_actions=attempted_actions,
            restart_window=restart_window,
        )
        return result

    def _backend_restart_history(self) -> list[float]:
        history = self.state.scratchpad.setdefault("_backend_restart_history", [])
        if not isinstance(history, list):
            history = []
            self.state.scratchpad["_backend_restart_history"] = history
        return history

    def _check_backend_restart_rate_limit(self) -> dict[str, Any]:
        history = self._backend_restart_history()
        if self.backend_max_restarts_per_hour <= 0:
            return {"allowed": False, "count": len(history), "window_sec": 3600}
        cutoff = time.time() - 3600.0
        recent = [float(ts) for ts in history if float(ts) >= cutoff]
        self.state.scratchpad["_backend_restart_history"] = recent
        return {
            "allowed": len(recent) < self.backend_max_restarts_per_hour,
            "count": len(recent),
            "window_sec": 3600,
        }

    def _record_backend_restart_attempt(self) -> None:
        history = self._backend_restart_history()
        cutoff = time.time() - 3600.0
        recent = [float(ts) for ts in history if float(ts) >= cutoff]
        recent.append(time.time())
        self.state.scratchpad["_backend_restart_history"] = recent

    async def _probe_backend_health(self, health_url: str, *, timeout_sec: int) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:
            return {"ok": False, "error": f"httpx unavailable: {exc}"}
        headers = {"Authorization": f"Bearer {self.client.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=float(timeout_sec)) as probe_client:
                response = await probe_client.get(health_url, headers=headers)
            return {"ok": response.status_code < 500, "status_code": response.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _wait_for_backend_health(self, health_url: str, *, timeout_sec: int) -> dict[str, Any]:
        deadline = time.monotonic() + max(1, int(timeout_sec))
        last_result: dict[str, Any] = {"ok": False, "error": "health probe not started"}
        while time.monotonic() < deadline:
            last_result = await self._probe_backend_health(
                health_url,
                timeout_sec=min(self.backend_healthcheck_timeout_sec, max(1, int(timeout_sec))),
            )
            if last_result.get("ok"):
                return last_result
            await asyncio.sleep(1.0)
        return last_result

    async def _attempt_backend_restart_recovery(
        self,
        health_url: str,
        *,
        timeout_sec: int,
    ) -> dict[str, Any]:
        rate_limit = self._check_backend_restart_rate_limit()
        if not rate_limit.get("allowed", False):
            return {
                "status": "unrecovered",
                "action": "rate_limited",
                "message": (
                    f"Backend restart suppressed by supervisor rate limit "
                    f"({rate_limit.get('count', 0)}/{self.backend_max_restarts_per_hour} in the last hour)."
                ),
                "health_after": {"ok": False},
                "restart_window": rate_limit,
            }

        self._record_backend_restart_attempt()
        command_result = await self._run_backend_restart_command(self.backend_restart_command)
        if command_result.get("ok"):
            health_after = await self._wait_for_backend_health(
                health_url,
                timeout_sec=max(timeout_sec, int(self.backend_restart_grace_sec)),
            )
            if health_after.get("ok"):
                return {
                    "status": "recovered",
                    "action": "restart_command",
                    "message": "Backend restart command succeeded and health probe recovered.",
                    "health_after": health_after,
                }
            return {
                "status": "unrecovered",
                "action": "restart_command",
                "message": "Backend restart command ran, but the health probe did not recover.",
                "health_after": health_after,
            }
        return {
            "status": "unrecovered",
            "action": "restart_command",
            "message": str(command_result.get("message") or "Backend restart command failed."),
            "health_after": {"ok": False},
        }

    async def _run_backend_restart_command(self, command: str) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            return {"ok": False, "message": f"Unable to launch restart command: {exc}"}
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"ok": False, "message": "Restart command timed out after 60s."}
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode == 0:
            return {"ok": True, "stdout": stdout_text, "stderr": stderr_text}
        return {
            "ok": False,
            "message": f"Restart command exited with status {proc.returncode}.",
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

    async def _run_backend_unload_command(self, command: str | None) -> dict[str, Any]:
        if command:
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as exc:
                return {"ok": False, "message": f"Unable to launch unload command: {exc}"}
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"ok": False, "message": "Unload command timed out after 60s."}
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if proc.returncode == 0:
                return {"ok": True, "stdout": stdout_text, "stderr": stderr_text}
            return {
                "ok": False,
                "message": f"Unload command exited with status {proc.returncode}.",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        if self.provider_profile == "lmstudio":
            return await self._run_lmstudio_backend_unload()
        if self.provider_profile != "ollama":
            return {"ok": False, "message": "No backend unload strategy is available for this provider."}
        return await self._run_ollama_backend_unload()

    def _backend_unload_action(self) -> str:
        if self.backend_unload_command:
            return "unload_command"
        if self.provider_profile == "lmstudio":
            return "lmstudio_api_unload"
        if self.provider_profile == "ollama":
            return "ollama_keep_alive_zero"
        return "unload_command"

    def _backend_unload_message(self, outcome: str) -> str:
        if self.backend_unload_command:
            return f"Backend unload command {outcome}."
        if self.provider_profile == "lmstudio":
            return f"LM Studio unload request {outcome}."
        if self.provider_profile == "ollama":
            return f"Ollama unload request {outcome}."
        return f"Backend unload request {outcome}."

    async def _run_ollama_backend_unload(self) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:
            return {"ok": False, "message": f"httpx unavailable: {exc}"}

        base_url = str(self.client.base_url or "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        unload_url = f"{base_url}/api/generate"
        payload = {
            "model": self.client.model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        headers = {
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=float(self.backend_healthcheck_timeout_sec)) as unload_client:
                response = await unload_client.post(unload_url, headers=headers, json=payload)
        except Exception as exc:
            return {"ok": False, "message": f"Ollama unload request failed: {exc}"}
        if response.status_code >= 400:
            return {
                "ok": False,
                "message": f"Ollama unload request failed with status {response.status_code}.",
            }
        body_text = response.text.strip()
        return {
            "ok": True,
            "status_code": response.status_code,
            "body": body_text,
            "message": "Ollama unload request completed.",
        }

    async def _run_lmstudio_backend_unload(self) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:
            return {"ok": False, "message": f"httpx unavailable: {exc}"}

        base_url = str(self.client.base_url or "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        list_url = f"{base_url}/api/v1/models"
        unload_url = f"{base_url}/api/v1/models/unload"
        headers = {"Content-Type": "application/json"}
        api_key = str(self.client.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=float(self.backend_healthcheck_timeout_sec)) as unload_client:
                response = await unload_client.get(list_url, headers=headers)
                if response.status_code >= 400:
                    return {
                        "ok": False,
                        "message": f"LM Studio model list request failed with status {response.status_code}.",
                    }
                payload = response.json()
                instance_ids, loaded_summary = self._find_lmstudio_loaded_instance_ids(payload)
                if not instance_ids:
                    loaded_blob = ", ".join(loaded_summary) if loaded_summary else "none"
                    return {
                        "ok": False,
                        "message": (
                            f"LM Studio model '{self.client.model}' is not currently loaded "
                            f"(loaded instances: {loaded_blob})."
                        ),
                    }
                for instance_id in instance_ids:
                    unload_response = await unload_client.post(
                        unload_url,
                        headers=headers,
                        json={"instance_id": instance_id},
                    )
                    if unload_response.status_code >= 400:
                        return {
                            "ok": False,
                            "message": (
                                f"LM Studio unload request failed for instance '{instance_id}' "
                                f"with status {unload_response.status_code}."
                            ),
                        }
        except Exception as exc:
            return {"ok": False, "message": f"LM Studio unload request failed: {exc}"}
        return {
            "ok": True,
            "instance_ids": instance_ids,
            "message": f"LM Studio unload request completed for {len(instance_ids)} instance(s).",
        }

    def _find_lmstudio_loaded_instance_ids(self, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
        if not isinstance(payload, dict):
            return [], []
        models = payload.get("models")
        if not isinstance(models, list):
            return [], []
        target_model = str(self.client.model or "").strip()
        instance_ids: list[str] = []
        all_loaded_instance_ids: list[str] = []
        seen: set[str] = set()
        seen_all: set[str] = set()
        loaded_summary: list[str] = []
        target_known = False
        for entry in models:
            if not isinstance(entry, dict):
                continue
            model_key = str(entry.get("key") or "").strip()
            if model_key == target_model:
                target_known = True
            loaded_instances = entry.get("loaded_instances")
            if not isinstance(loaded_instances, list):
                continue
            for loaded_entry in loaded_instances:
                if not isinstance(loaded_entry, dict):
                    continue
                instance_id = str(loaded_entry.get("id") or "").strip()
                if not instance_id:
                    continue
                loaded_summary.append(f"{model_key}:{instance_id}" if model_key else instance_id)
                if instance_id not in seen_all:
                    seen_all.add(instance_id)
                    all_loaded_instance_ids.append(instance_id)
                if instance_id in seen:
                    continue
                if instance_id == target_model or model_key == target_model:
                    seen.add(instance_id)
                    instance_ids.append(instance_id)
        if instance_ids:
            return instance_ids, loaded_summary
        if target_known:
            return all_loaded_instance_ids, loaded_summary
        return [], loaded_summary

    @staticmethod
    def _stream_print(text: str) -> None:
        try:
            print(text, end="", flush=True)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe, end="", flush=True)

    async def _ensure_context_limit(self) -> None:
        await self.prompt_builder.ensure_context_limit()

    async def _rebuild_messages_after_context_overflow(
        self,
        *,
        n_ctx: int,
        n_keep: int | None = None,
        error_message: str = "",
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> list[dict[str, Any]] | None:
        new_limit = self._apply_server_context_limit(
            n_ctx,
            source="stream_context_overflow",
            observed_n_keep=n_keep,
        )
        system_prompt = str(self.state.scratchpad.get("_last_system_prompt") or "")
        if not system_prompt:
            return None
        self._runlog(
            "context_limit_rebuild",
            "shrinking prompt budget after upstream context overflow",
            n_ctx=n_ctx,
            n_keep=n_keep,
            error=error_message,
            max_prompt_tokens=new_limit,
        )
        return await self._build_prompt_messages(system_prompt, event_handler=event_handler)

    def _apply_usage(self, usage: dict[str, Any]) -> None:
        normalized_usage = json_safe_value(usage or {})
        if not isinstance(normalized_usage, dict) or not normalized_usage:
            return
        prompt_tokens = _coerce_usage_token_count(normalized_usage.get("prompt_tokens"))
        completion_tokens = _coerce_usage_token_count(normalized_usage.get("completion_tokens"))
        total_tokens = _coerce_usage_token_count(normalized_usage.get("total_tokens"))
        self.state.token_usage += total_tokens
        if prompt_tokens > 0:
            self.state.scratchpad["context_used_tokens"] = prompt_tokens
        elif total_tokens > 0:
            self.state.scratchpad["context_used_tokens"] = total_tokens
        self.state.last_completion_tokens = completion_tokens
        self.state.scratchpad["last_completion_tokens"] = completion_tokens
        self._runlog("usage", "token usage update", usage=normalized_usage)

    async def _build_prompt_messages(
        self,
        system_prompt: str,
        *,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> list[dict[str, Any]]:
        return await self.prompt_builder.build_messages(system_prompt, event_handler=event_handler)

    async def _record_tool_result(
        self,
        tool_name: str,
        tool_call_id: str | None,
        result: ToolEnvelope,
        arguments: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> ConversationMessage:
        return await self.tool_results.record_result(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=result,
            arguments=arguments,
            operation_id=operation_id,
        )
    def _record_assistant_message(
        self,
        *,
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
        speaker: str | None = None,
        hidden_from_prompt: bool = False,
    ) -> None:
        metadata: dict[str, Any] = {}
        normalized_speaker = str(speaker or "").strip().lower()
        if normalized_speaker:
            metadata["speaker"] = normalized_speaker
        if hidden_from_prompt:
            metadata["hidden_from_prompt"] = True
        self.state.append_message(
            ConversationMessage(
                role="assistant",
                content=assistant_text or None,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        )

    def _active_task_scope_payload(self) -> dict[str, Any] | None:
        payload = getattr(self, "_active_task_scope", None)
        if isinstance(payload, dict) and payload:
            return dict(payload)
        state = getattr(self, "state", None)
        scratchpad = getattr(state, "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return None
        stored = scratchpad.get("_active_task_scope")
        if not isinstance(stored, dict) or not stored:
            return None
        restored = dict(stored)
        self._active_task_scope = restored
        sequence = stored.get("sequence")
        try:
            restored_sequence = int(sequence)
        except (TypeError, ValueError):
            restored_sequence = 0
        current_sequence = int(getattr(self, "_task_sequence", 0) or 0)
        if restored_sequence > current_sequence:
            self._task_sequence = restored_sequence
        return restored

    def _clip_task_summary_text(self, value: Any, *, limit: int = 240) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        clipped, truncated = clip_text_value(text, limit=limit)
        return f"{clipped} [truncated]" if truncated else clipped

    def _extract_task_terminal_message(self, result: dict[str, Any] | None) -> str:
        if not isinstance(result, dict) or not result:
            return ""
        message = result.get("message")
        if isinstance(message, dict):
            candidate = (
                message.get("message")
                or message.get("question")
                or message.get("status")
            )
            if candidate:
                return self._clip_task_summary_text(candidate)
        if isinstance(message, str) and message.strip():
            return self._clip_task_summary_text(message)
        reason = str(result.get("reason") or "").strip()
        if reason:
            return self._clip_task_summary_text(reason)
        error = result.get("error")
        if isinstance(error, dict):
            candidate = error.get("message")
            if candidate:
                return self._clip_task_summary_text(candidate)
        return ""

    def _task_duration_seconds(self, started_at: str, finished_at: str) -> float:
        try:
            started = datetime.fromisoformat(str(started_at))
            finished = datetime.fromisoformat(str(finished_at))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, round((finished - started).total_seconds(), 3))

    def _write_task_summary(self, payload: dict[str, Any]) -> str:
        summary_path_text = str(payload.get("summary_path") or "").strip()
        if not summary_path_text:
            return ""
        path = Path(summary_path_text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(json_safe_value(payload), indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            logger = getattr(self, "log", logging.getLogger("smallctl.harness"))
            logger.exception("failed to write task summary")
            return ""

    def _begin_task_scope(self, *, raw_task: str, effective_task: str) -> dict[str, Any]:
        normalized_raw = str(raw_task or "").strip()
        normalized_effective = str(effective_task or normalized_raw).strip()
        current = self._active_task_scope_payload()
        if current:
            current_effective = str(
                current.get("effective_task") or current.get("raw_task") or ""
            ).strip()
            current_raw = str(current.get("raw_task") or "").strip()
            if (
                (current_effective and current_effective == normalized_effective)
                or (current_raw and current_raw == normalized_raw)
            ):
                return current
            self._finalize_task_scope(
                terminal_event="task_aborted",
                status="aborted",
                reason="replaced_by_new_task",
                replacement_task=normalized_effective,
            )

        prior_sequence = getattr(self, "_task_sequence", 0)
        if not prior_sequence:
            state = getattr(self, "state", None)
            scratchpad = getattr(state, "scratchpad", None)
            if isinstance(scratchpad, dict):
                prior_sequence = scratchpad.get("_task_sequence", 0)
        try:
            sequence = int(prior_sequence) + 1
        except (TypeError, ValueError):
            sequence = 1
        self._task_sequence = sequence

        task_id = f"task-{sequence:04d}"
        summary_path = ""
        if self.run_logger is not None:
            summary_path = str(
                (self.run_logger.run_dir / "tasks" / task_id / "task_summary.json").resolve()
            )
        scope = {
            "task_id": task_id,
            "sequence": sequence,
            "raw_task": normalized_raw,
            "effective_task": normalized_effective,
            "target_paths": extract_task_target_paths(normalized_effective),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "start_step_count": int(getattr(self.state, "step_count", 0) or 0),
            "start_token_usage": int(getattr(self.state, "token_usage", 0) or 0),
            "summary_path": summary_path,
        }
        self._active_task_scope = dict(scope)
        self.state.scratchpad["_task_sequence"] = sequence
        self.state.scratchpad["_active_task_scope"] = json_safe_value(scope)
        self.state.scratchpad["_active_task_id"] = task_id
        return dict(scope)

    def _finalize_task_scope(
        self,
        *,
        terminal_event: str,
        status: str,
        reason: str = "",
        result: dict[str, Any] | None = None,
        replacement_task: str = "",
    ) -> dict[str, Any] | None:
        scope = self._active_task_scope_payload()
        if not scope:
            return None

        finished_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        result_status = str((result or {}).get("status") or "").strip().lower()
        summary_terminal_event = terminal_event or "task_finalize"
        summary_text = self._extract_task_terminal_message(result)
        if not summary_text and reason:
            summary_text = self._clip_task_summary_text(reason)

        start_step_count = int(scope.get("start_step_count") or 0)
        start_token_usage = int(scope.get("start_token_usage") or 0)
        current_step_count = int(getattr(self.state, "step_count", 0) or 0)
        current_token_usage = int(getattr(self.state, "token_usage", 0) or 0)

        payload = {
            "task_id": str(scope.get("task_id") or "").strip(),
            "sequence": int(scope.get("sequence") or 0),
            "raw_task": str(scope.get("raw_task") or "").strip(),
            "effective_task": str(scope.get("effective_task") or "").strip(),
            "terminal_event": summary_terminal_event,
            "status": str(status or result_status or "stopped").strip(),
            "result_status": result_status,
            "reason": self._clip_task_summary_text(reason),
            "message": summary_text,
            "started_at": str(scope.get("started_at") or "").strip(),
            "finished_at": finished_at,
            "duration_seconds": self._task_duration_seconds(
                str(scope.get("started_at") or "").strip(),
                finished_at,
            ),
            "step_count": max(0, current_step_count - start_step_count),
            "token_usage": max(0, current_token_usage - start_token_usage),
            "current_phase": str(getattr(self.state, "current_phase", "") or "").strip(),
            "active_tool_profiles": list(getattr(self.state, "active_tool_profiles", []) or []),
            "target_paths": list(scope.get("target_paths") or []),
            "artifact_count": len(getattr(self.state, "artifacts", {}) or {}),
            "recent_error_count": len(getattr(self.state, "recent_errors", []) or []),
            "last_recent_error": self._clip_task_summary_text(
                (getattr(self.state, "recent_errors", []) or [""])[-1]
                if getattr(self.state, "recent_errors", [])
                else "",
                limit=180,
            ),
            "summary_path": str(scope.get("summary_path") or "").strip(),
        }
        if replacement_task:
            payload["replacement_task"] = replacement_task
        error = (result or {}).get("error")
        if isinstance(error, dict):
            payload["error_type"] = str(error.get("type") or "").strip()

        summary_path = self._write_task_summary(payload)
        payload["summary_path"] = summary_path

        if terminal_event:
            self._runlog(
                terminal_event,
                "task ended without normal completion",
                task_id=payload["task_id"],
                status=payload["status"],
                result_status=result_status,
                reason=payload["reason"],
                replacement_task=replacement_task,
                summary_path=summary_path,
                raw_task=payload["raw_task"],
                effective_task=payload["effective_task"],
            )

        self._active_task_scope = None
        self.state.scratchpad.pop("_active_task_scope", None)
        self.state.scratchpad.pop("_active_task_id", None)
        return payload

    def _reset_task_boundary_state(
        self,
        *,
        reason: str,
        new_task: str = "",
        previous_task: str = "",
    ) -> None:
        preserved_previous_task = str(
            previous_task
            or self.state.run_brief.original_task
            or self.state.scratchpad.get("_last_task_text")
            or ""
        ).strip()
        preserved_scratchpad: dict[str, Any] = {}
        for key in (
            "_model_name",
            "_model_is_small",
            "_max_steps",
            "strategy",
            "_session_notepad",
            "_last_task_text",
            "_last_task_handoff",
            "_task_boundary_previous_task",
            "_task_sequence",
        ):
            if key in self.state.scratchpad:
                preserved_scratchpad[key] = self.state.scratchpad[key]
        if preserved_previous_task:
            preserved_scratchpad["_task_boundary_previous_task"] = preserved_previous_task

        background_processes = json_safe_value(self.state.background_processes)
        inventory_state = json_safe_value(self.state.inventory_state)

        # A true task boundary needs a clean task-local slate. Continue-like followups
        # are resolved before we get here, so this path should not carry over prior-task
        # messages, briefs, or working memory into the next objective.
        self.state.current_phase = self._initial_phase
        self.state.step_count = 0
        self.state.inactive_steps = 0
        self.state.latest_verdict = None

        # Preserve only durable cross-task settings and handoff metadata; clear
        # everything else that is scoped to the previous task.
        self.state.scratchpad = preserved_scratchpad
        self.state.recent_messages = []
        self.state.recent_errors = []
        self.state.run_brief = RunBrief()
        self.state.working_memory = WorkingMemory()
        self.state.acceptance_ledger = {}
        self.state.acceptance_waivers = []
        self.state.acceptance_waived = False
        self.state.last_verifier_verdict = None
        self.state.last_failure_class = ""

        self.state.files_changed_this_cycle = []
        self.state.repair_cycle_id = ""
        self.state.stagnation_counters = {}
        self.state.draft_plan = None
        self.state.active_plan = None
        self.state.plan_resolved = False
        self.state.plan_artifact_id = ""
        self.state.planning_mode_enabled = self._configured_planning_mode
        self.state.planner_requested_output_path = ""
        self.state.planner_requested_output_format = ""
        self.state.planner_resume_target_mode = "loop"
        self.state.planner_interrupt = None
        self.state.artifacts = {}
        if self.state.write_session and self.state.write_session.status != "complete":
            from ..graph.tool_outcomes import _register_write_session_stage_artifact
            _register_write_session_stage_artifact(self, self.state.write_session)
        self.state.episodic_summaries = []
        self.state.context_briefs = []
        self.state.prompt_budget = PromptBudgetSnapshot()
        self.state.retrieval_cache = []
        self.state.active_intent = ""
        self.state.secondary_intents = []
        self.state.intent_tags = []
        self.state.retrieved_experience_ids = []
        self.state.tool_execution_records = {}
        self.state.pending_interrupt = None
        self.state.tool_history = []
        self.state.background_processes = background_processes if isinstance(background_processes, dict) else {}
        self.state.inventory_state = inventory_state if isinstance(inventory_state, dict) else {}
        self.state.warm_experiences = []
        self.state.touch()
        self._runlog(
            "task_boundary_reset",
            "reset task-local state for new task",
            reason=reason,
            previous_task=previous_task,
            new_task=new_task,
        )

    def _maybe_reset_for_new_task(self, task: str) -> None:
        previous_task = str(self.state.run_brief.original_task or self.state.scratchpad.get("_last_task_text") or "").strip()
        if not previous_task:
            return
        new_task = str(task or "").strip()
        if not new_task or new_task == previous_task:
            return
        has_task_local_context = self._has_task_local_context()
        if has_task_local_context:
            self._finalize_task_scope(
                terminal_event="task_aborted",
                status="aborted",
                reason="replaced_by_new_task",
                replacement_task=new_task,
            )
            self._reset_task_boundary_state(
                reason="task_switch",
                new_task=new_task,
                previous_task=previous_task,
            )

    def _has_task_local_context(self) -> bool:
        return bool(
            self.state.recent_messages
            or self.state.recent_errors
            or self.state.artifacts
            or self.state.episodic_summaries
            or self.state.context_briefs
            or self.state.run_brief.task_contract
            or self.state.run_brief.current_phase_objective
            or self.state.working_memory.current_goal
            or self.state.working_memory.plan
            or self.state.working_memory.decisions
            or self.state.working_memory.open_questions
            or self.state.working_memory.known_facts
            or self.state.working_memory.failures
            or self.state.working_memory.next_actions
            or self.state.acceptance_ledger
            or self.state.acceptance_waivers
            or self.state.scratchpad.get("_task_complete")
            or self.state.scratchpad.get("_task_failed")
        )

    def _last_task_handoff(self) -> dict[str, Any]:
        payload = self.state.scratchpad.get("_last_task_handoff")
        if not isinstance(payload, dict):
            return {}
        return dict(payload)

    def _is_continue_like_followup(self, task: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(task or "").strip().lower()).strip()
        if not normalized:
            return False
        fillers = {"please", "pls", "now", "again", "just", "then", "more", "further"}
        tokens = [token for token in normalized.split() if token not in fillers]
        collapsed = " ".join(tokens)
        return collapsed in {
            "continue",
            "cntinue",
            "conitnue",
            "continune",
            "cotinue",
            "keep going",
            "resume",
            "proceed",
            "go on",
            "carry on",
        }

    def _resolve_followup_task(self, task: str) -> str:
        raw_task = str(task or "").strip()
        if not raw_task or not self._is_continue_like_followup(raw_task):
            return raw_task

        handoff = self._last_task_handoff()
        candidate = str(
            handoff.get("effective_task")
            or handoff.get("current_goal")
            or self.state.run_brief.original_task
            or self.state.scratchpad.get("_last_task_text")
            or ""
        ).strip()
        if not candidate:
            return raw_task

        if not (self._has_task_local_context() or handoff):
            return raw_task

        return candidate

    def _store_task_handoff(self, *, raw_task: str, effective_task: str) -> None:
        effective = str(effective_task or "").strip()
        if not effective:
            return
        target_paths = extract_task_target_paths(effective)
        self.state.scratchpad["_last_task_handoff"] = {
            "raw_task": str(raw_task or "").strip(),
            "effective_task": effective,
            "current_goal": str(self.state.working_memory.current_goal or effective),
            "target_paths": list(target_paths),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _initialize_run_brief(self, task: str, *, raw_task: str | None = None) -> None:
        effective_task = str(task or "").strip()
        source_task = str(raw_task or effective_task).strip()
        existing_task = str(self.state.run_brief.original_task or "").strip()
        if not existing_task:
            existing_task = str(self.state.scratchpad.pop("_task_boundary_previous_task", "") or "").strip()
        if existing_task and effective_task and effective_task != existing_task:
            # We are handing over the objective. Keep the original intent and append the new follow-up direction.
            merged_task = f"{existing_task}\nFollow-up: {effective_task}"
        else:
            merged_task = effective_task or existing_task

        self.state.run_brief.original_task = merged_task
        self.state.run_brief.task_contract = self._derive_task_contract(merged_task)
        self.state.run_brief.current_phase_objective = f"{self.state.current_phase}: {effective_task}"

        existing_goal = str(self.state.working_memory.current_goal or "").strip()
        if existing_goal and effective_task and effective_task not in existing_goal:
            self.state.working_memory.current_goal = f"{existing_goal}\nFollow-up: {effective_task}"
        else:
            self.state.working_memory.current_goal = merged_task

        self.state.scratchpad["_task_target_paths"] = extract_task_target_paths(effective_task)
        self._store_task_handoff(raw_task=source_task, effective_task=effective_task)
        if hasattr(self.memory, "prime_write_policy"):
            self.memory.prime_write_policy(effective_task)
        self.state.working_memory.next_actions = dedupe_keep_tail(
            self.state.working_memory.next_actions + [self.memory._next_action_for_task(effective_task)],
            limit=6,
        )

    async def _maybe_compact_context(
        self,
        query: str,
        system_prompt: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> None:
        await self.compaction.maybe_compact_context(
            query=query,
            system_prompt=system_prompt,
            event_handler=event_handler,
        )

    def _update_working_memory(self) -> None:
        self.memory.update_working_memory(self.context_policy.recent_message_limit)

    def _refresh_active_intent(self) -> None:
        self.memory._refresh_active_intent()

    def _extract_intent_state(self, task: str) -> tuple[str, list[str], list[str]]:
        text = (task or "").lower()
        secondary: list[str] = []
        tags: list[str] = []
        requested_tool = self._infer_requested_tool_name(task)
        
        # Primary Intent logic from memory-upgrade.md
        primary = "general_task"
        if requested_tool:
            primary = f"requested_{requested_tool}"
            secondary.append("complete_validation_task")
            tags.append(requested_tool)
            if requested_tool in {"scratch_list", "cwd_get", "loop_status"}:
                secondary.append("call_zero_arg_tool")
            if requested_tool in {"task_complete", "task_fail", "ask_human"}:
                secondary.append("control_tool")
        elif any(token in text for token in {"inspect", "read", "grep", "find", "search", "list"}):
            primary = "inspect_repo"
            secondary.append("read_artifacts")
        elif any(token in text for token in {"write", "edit", "patch", "create", "update", "diff"}):
            primary = "write_file"
            secondary.append("mutate_repo")
        elif "contract" in text or "plan" in text:
            primary = "plan_execution"
            secondary.append("complete_validation_task")
        
        if self.state.working_memory.failures:
            secondary.append("recover_from_validation_error")
            
        tags.extend(self._infer_environment_tags())
        tags.extend(self._infer_entity_tags(task))
        tags.extend([t for t in self.state.working_memory.next_actions[-2:] if " " not in t][:2])
        
        return primary, clip_string_list(secondary, limit=3, item_char_limit=48)[0], clip_string_list(tags, limit=6, item_char_limit=64)[0]

    def _infer_environment_tags(self) -> list[str]:
        tags = [self.provider_profile, self.state.current_phase]
        cwd = self.state.cwd.lower()
        if "localhost" in cwd:
            tags.append("localhost")
        if "scripts" in cwd:
            tags.append("scripts")
        return tags

    def _infer_entity_tags(self, task: str) -> list[str]:
        text = (task or "").lower()
        tags = []
        if "ansible" in text:
            tags.append("ansible")
        if "python" in text or ".py" in text:
            tags.append("python")
        if "bash" in text or ".sh" in text:
            tags.append("bash")
        return tags

    def _infer_requested_tool_name(self, task: str) -> str:
        text = (task or "").lower()
        creation_markers = (
            "build",
            "create",
            "make",
            "write",
            "generate",
            "implement",
            "save",
            "produce",
        )
        if (
            "script" in text
            and any(marker in text for marker in creation_markers)
        ) or re.search(r"\b(?:\.py|\.sh|\.bash|\.ps1)\b", text):
            return "write_file"
        if "file_patch" in text or "patch file" in text:
            return "file_patch"
        if "read_file" in text or "cat" in text:
            return "read_file"
        if "write_file" in text:
            return "write_file"
        if "shell" in text or "exec" in text or self._looks_like_shell_request(task):
            return "shell_exec"
        return ""

    def _completion_next_action(self) -> str:
        return "Decide whether the current evidence is sufficient; call task_complete when it is."

    def _next_action_for_task(self, task: str) -> str:
        return self.memory._next_action_for_task(task)

    def _record_experience(
        self,
        *,
        tool_name: str,
        result: ToolEnvelope,
        evidence_refs: list[str] | None = None,
        notes: str = "",
        source: str = "observed",
    ) -> ExperienceMemory:
        return self.memory.record_experience(
            tool_name=tool_name,
            result=result,
            evidence_refs=evidence_refs,
            notes=notes,
            source=source,
        )

    def _normalize_failure_mode(self, error: Any, *, tool_name: str, success: bool) -> str:
        return self.memory._normalize_failure_mode(error, tool_name=tool_name, success=success)

    def _reinforce_retrieved_experiences(self, *, tool_name: str, success: bool) -> None:
        self.memory._reinforce_retrieved_experiences(tool_name=tool_name, success=success)

    def _record_terminal_experience(self, result: dict[str, Any]) -> None:
        self.memory.record_terminal_experience(result)

    def _argument_fingerprint(self, arguments: Any) -> str:
        return self.memory._argument_fingerprint(arguments)

    def _derive_task_contract(self, task: str) -> str:
        return self.memory.derive_task_contract(task)

    def _select_retrieval_query(self) -> str:
        return build_retrieval_query(self.state)

    def _log_conversation_state(self, event: str) -> None:
        if self.run_logger is None:
            return
        self.run_logger.log(
            "chat",
            "conversation_history",
            "conversation snapshot",
            conversation_id=self.conversation_id,
            snapshot_event=event,
            step=self.state.step_count,
            history=[m.to_dict() for m in self.state.conversation_history],
            recent_messages=[m.to_dict() for m in self.state.recent_messages],
            prompt_budget=self.state.prompt_budget.__dict__,
        )

    def _chat_mode_tools(self) -> list[dict[str, Any]]:
        task = self._current_user_task()
        if not self._chat_mode_requires_tools(task):
            self.state.scratchpad["_chat_tools_exposed"] = False
            self.state.scratchpad["_chat_tools_suppressed_reason"] = "non_lookup_chat"
            self._runlog(
                "chat_tool_selection",
                "chat tool exposure suppressed",
                task=task,
                reason="non_lookup_chat",
            )
            return []
        self.state.scratchpad["_chat_tools_exposed"] = True
        self.state.scratchpad.pop("_chat_tools_suppressed_reason", None)
        tools = self.registry.export_openai_tools(
            phase=self.state.current_phase,
            mode="chat",
            profiles=set(self.state.active_tool_profiles),
        )
        shell_spec = self.registry.get("shell_exec")
        active_tool_names = {
            str(entry["function"]["name"])
            for entry in tools
            if isinstance(entry, dict)
            and isinstance(entry.get("function"), dict)
            and "name" in entry["function"]
        }
        if (
            shell_spec is not None
            and shell_spec.profile_allowed(set(self.state.active_tool_profiles))
            and "shell_exec" not in active_tool_names
        ):
            tools.append(shell_spec.openai_schema())
            self._runlog(
                "chat_tool_selection",
                "shell execution exposed in chat mode",
                task=task,
                reason="approval_gated_shell",
            )
        return tools

    def _activate_tool_profiles(self, task: str) -> None:
        if self._configured_tool_profiles:
            profiles = set(self._configured_tool_profiles)
        else:
            profiles = classify_tool_profiles(
                task,
                use_ansible=self._use_ansible,
            )
        self.state.active_tool_profiles = sorted(profiles)
        self.state.scratchpad["_last_task_text"] = task
        self._runlog(
            "tool_profiles",
            "selected tool profiles",
            task=task,
            profiles=self.state.active_tool_profiles,
            source="config" if self._configured_tool_profiles else "dynamic",
        )

    async def _dispatch_tool_call(self, tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
        cached = self._maybe_reuse_file_read(tool_name=tool_name, args=args)
        if cached is not None:
            return cached

        if tool_name not in self.registry.names():
            sanitized = self._attempt_tool_sanitization(tool_name)
            if sanitized:
                self._runlog(
                    "tool_sanitization",
                    "hallucinated tool name detected and split",
                    hallucinated=tool_name,
                    sanitized=sanitized,
                )
                tool_name = sanitized

        return await self.dispatcher.dispatch(tool_name, args)

    def _attempt_tool_sanitization(self, tool_name: str) -> str | None:
        # If the model mashes two known tools like 'toolAtoolB'
        for registered_name in self.registry.names():
            if not registered_name:
                continue
            if tool_name.startswith(registered_name) and tool_name != registered_name:
                remainder = tool_name[len(registered_name):]
                if remainder in self.registry.names():
                    # It's a mashup of two known tools. 
                    # For now, we just return the first one to allow progress.
                    # A more complex fix would execute both.
                    return registered_name
        return None

    def _maybe_reuse_file_read(self, *, tool_name: str, args: dict[str, Any]) -> ToolEnvelope | None:
        if tool_name != "file_read":
            return None
        cache = self.state.scratchpad.get("file_read_cache")
        if not isinstance(cache, dict):
            return None
        cache_key = _file_read_cache_key(self.state.cwd, args)
        if not cache_key:
            return None
        artifact_id = cache.get(cache_key)
        if not isinstance(artifact_id, str) or not artifact_id:
            return None
        artifact = self.state.artifacts.get(artifact_id)
        if artifact is None:
            return None
        self._runlog(
            "tool_cache_hit",
            "reusing prior file_read result",
            tool_name=tool_name,
            artifact_id=artifact_id,
            path=artifact.source,
        )
        return ToolEnvelope(
            success=True,
            output={
                "status": "cached",
                "artifact_id": artifact_id,
                "path": artifact.source,
                "summary": artifact.summary,
            },
            metadata={
                "cache_hit": True,
                "artifact_id": artifact_id,
                "path": artifact.source,
                "tool_name": tool_name,
            },
        )

    def _is_smalltalk(self, task: str) -> bool:
        text = task.strip().lower()
        smalltalk = {
            "hi",
            "hello",
            "hey",
            "yo",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "how are you",
            "what's up",
            "whats up",
        }
        return text in smalltalk

    def _needs_loop_for_content_lookup(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False

        file_markers = (
            "file",
            "log",
            "logs",
            ".log",
            ".jsonl",
            ".txt",
            ".md",
            ".py",
            "/",
            "\\",
            "code",
            "source",
            "src",
        )
        content_queries = (
            "what is",
            "what's",
            "show",
            "read",
            "tell me",
            "summarize",
            "contents",
            "content",
            "line ",
            "lines ",
            "bug",
            "error",
            "issue",
            "debug",
        )
        asks_for_specific_line = bool(re.search(r"\bline(?:s)?\s+\d+\b", text))
        asks_for_range = bool(re.search(r"\b\d+\s*-\s*\d+\b", text))
        asks_for_log_or_file_content = any(marker in text for marker in file_markers) and any(
            query in text for query in content_queries
        )
        asks_for_command_execution = (
            bool(re.search(r"\b(run|execute|exec)\b", text))
            and bool(
                re.search(
                    r"\b(ls|dir|pwd|cd|cat|type|findstr|grep|git\s+status|get-childitem|pytest|python|powershell|pwsh|cmd)\b",
                    text,
                )
            )
        )
        asks_for_directory_contents = (
            any(
                phrase in text
                for phrase in (
                    "what files",
                    "which files",
                    "list files",
                    "show files",
                    "what folders",
                    "which folders",
                    "list folders",
                    "show folders",
                    "list directory",
                    "show directory",
                    "directory contents",
                    "folder contents",
                    "current directory",
                    "this directory",
                    "current folder",
                    "this folder",
                    "what is in this directory",
                    "what is in the current directory",
                    "what is in this folder",
                    "what is in the current folder",
                )
            )
            or bool(
                re.search(
                    r"\b(list|show|what|which)\b.*\b(files?|folders?|directories?|contents?)\b",
                    text,
                )
            )
        )
        asks_where_specific_line_is = asks_for_specific_line and any(
            phrase in text for phrase in ("what is", "what's", "show", "read")
        )
        return (
            asks_for_specific_line
            or asks_for_range
            or asks_for_log_or_file_content
            or asks_where_specific_line_is
            or asks_for_directory_contents
            or asks_for_command_execution
        )

    def _needs_contextual_loop_escalation(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        if not self._looks_like_execution_followup(text):
            return False
        if self._recent_assistant_proposed_command():
            return True
        return self._recent_assistant_referenced_tool_name("shell_exec")

    # Threshold (in tokens) above which a tool-result message is considered
    # "oversized" for the purpose of in-place compaction.  Small enough that we
    # only strip genuinely large payloads (document dumps, etc.) and leave normal
    # results alone.
    _TOOL_MSG_COMPACT_THRESHOLD: int = 150

    def _compact_oversized_tool_messages(self, *, soft_limit: int) -> bool:
        """Replace large tool-result message content with compact artifact references.

        Called as a last-resort shedding pass when all other budget-reduction
        strategies have been exhausted.  Walks ``state.recent_messages`` in
        reverse (most-recent first) and, for any tool message whose content
        exceeds ``_TOOL_MSG_COMPACT_THRESHOLD`` tokens, replaces the content
        with the compact artifact reference produced by
        ``ArtifactStore.compact_tool_message``.

        Returns True if at least one message was compacted.
        """
        from ..context.policy import estimate_text_tokens

        compacted_any = False
        for message in reversed(self.state.recent_messages):
            if message.role != "tool":
                continue
            content = message.content or ""
            if estimate_text_tokens(content) <= self._TOOL_MSG_COMPACT_THRESHOLD:
                continue
            # Try to recover the compact reference from the artifact store
            artifact_id = message.metadata.get("artifact_id") if message.metadata else None
            if not isinstance(artifact_id, str) or not artifact_id:
                # No artifact backing — truncate aggressively so we don't break budget
                char_cap = self._TOOL_MSG_COMPACT_THRESHOLD * 4
                if len(content) > char_cap:
                    message.content = content[:char_cap] + " [truncated]"
                    compacted_any = True
                continue
            artifact = self.state.artifacts.get(artifact_id)
            if artifact is None:
                continue
            # Re-derive the compact reference (tool_name is stored on the artifact)
            from ..models.tool_result import ToolEnvelope
            # Build a minimal ToolEnvelope representative enough for the compact formatter
            dummy_result = ToolEnvelope(
                success=True,
                output=None,
                error=None,
                metadata=dict(artifact.metadata or {}),
            )
            compact = self.artifact_store.compact_tool_message(
                artifact,
                dummy_result,
                request_text=self._current_user_task(),
            )
            if estimate_text_tokens(compact) < estimate_text_tokens(content):
                self._runlog(
                    "budget_policy",
                    "compacted oversized tool message to artifact reference",
                    artifact_id=artifact_id,
                    original_tokens=estimate_text_tokens(content),
                    compacted_tokens=estimate_text_tokens(compact),
                )
                if artifact and str(artifact.tool_name or "").strip() in {"shell_exec", "ssh_exec"}:
                    exit_code = artifact.metadata.get("exit_code")
                    if exit_code is not None:
                        status_tag = "EXIT_CODE=0" if exit_code == 0 else f"EXIT_CODE={exit_code} (FAILED)"
                        compact = f"{status_tag}\n{compact}"
                message.content = compact
                compacted_any = True
        return compacted_any

    def _current_user_task(self) -> str:
        for message in reversed(self.state.recent_messages):
            if message.role == "user" and message.content:
                content = str(message.content or "").strip()
                resolved = self._resolve_followup_task(content)
                return resolved or content
        last_task = self.state.scratchpad.get("_last_task_text")
        if isinstance(last_task, str) and last_task:
            return last_task
        return self.state.run_brief.original_task

    def _chat_mode_requires_tools(self, task: str) -> bool:
        if self._is_smalltalk(task):
            return False
        model_name = getattr(self.client, "model", None)
        if should_enable_complex_write_chat_draft(
            task,
            model_name=model_name,
            cwd=getattr(self.state, "cwd", None),
        ):
            return True
        if self._needs_memory_persistence(task):
            return True
        if self._needs_loop_for_content_lookup(task):
            return True
        return self._looks_like_readonly_chat_request(task) or self._looks_like_action_request(task)

    def _looks_like_action_request(self, task: str) -> bool:
        text = task.strip().lower()
        action_markers = ("run", "exec", "shell", "terminal", "ping", "curl", "wget", "git")
        return any(m in text for m in action_markers)

    def _needs_memory_persistence(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        memory_markers = (
            "save this in memory",
            "save memory",
            "remember this",
            "store this in memory",
            "store this",
            "note this",
            "pin this",
            "persist this",
            "keep this in memory",
            "write this down",
        )
        return any(marker in text for marker in memory_markers)

    def _looks_like_shell_request(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        shell_markers = (
            "bash",
            "shell",
            "terminal",
            "command",
            "command line",
            "run ",
            "execute",
            "exec",
            "script",
            "scan",
            "nmap",
            "ssh",
            "scp",
            "sftp",
            "ping",
            "curl",
            "wget",
            "traceroute",
            "tracepath",
            "netstat",
            "route",
            "ip route",
            "ip addr",
            "tcpdump",
            "netcat",
            "nc ",
            "dig",
            "nslookup",
            "whoami",
            "ps ",
            "top ",
            "lsof",
            "df ",
            "du ",
        )
        if any(marker in text for marker in shell_markers):
            return True
        return bool(
            re.search(
                r"\b(run|execute|exec|launch|invoke|start|inspect|check)\b.*\b(command|shell|terminal|script|scan|nmap|port|ports|ssh|ping|curl|wget)\b",
                text,
            )
        )

    def _looks_like_readonly_chat_request(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        if self._looks_like_execution_followup(text):
            return False
        readonly_markers = (
            "what",
            "which",
            "show",
            "read",
            "find",
            "search",
            "grep",
            "list",
            "current",
            "status",
            "where",
            "how many",
            "inspect",
            "check",
            "look at",
            "can you see",
            "tell me",
            "summarize",
        )
        readonly_targets = (
            "file",
            "files",
            "folder",
            "directory",
            "repo",
            "repository",
            "cwd",
            "working directory",
            "log",
            "logs",
            "artifact",
            "artifacts",
            "process",
            "cpu",
            "ram",
            "memory",
            "host",
            "system",
            "status",
            "code",
            "source",
            "src",
        )
        has_readonly_marker = any(marker in text for marker in readonly_markers)
        has_target = any(target in text for target in readonly_targets)
        return has_readonly_marker and has_target

    def _looks_like_execution_followup(self, text: str) -> bool:
        followup_phrases = (
            "use the command",
            "use that command",
            "run it",
            "run that",
            "execute it",
            "execute that",
            "try again",
            "use the shell command",
            "run the shell command",
            "execute the shell command",
        )
        return any(phrase in text for phrase in followup_phrases)

    def _recent_assistant_proposed_command(self) -> bool:
        recent_assistants = [
            message.content or ""
            for message in reversed(self.state.recent_messages)
            if message.role == "assistant" and (message.content or "").strip()
        ][:2]
        if not recent_assistants:
            return False
        command_pattern = re.compile(
            r"```(?:bash|sh|shell|zsh|pwsh|powershell)?\s*\n.+?```",
            re.IGNORECASE | re.DOTALL,
        )
        shell_tokens = re.compile(
            r"\b(top|ps|ls|pwd|cd|cat|grep|find|git|pytest|python|bash|sh|systemctl|journalctl)\b",
            re.IGNORECASE,
        )
        for content in recent_assistants:
            if command_pattern.search(content):
                return True
            if shell_tokens.search(content):
                return True
        return False

    def _recent_assistant_referenced_tool_name(self, tool_name: str) -> bool:
        target = str(tool_name or "").strip().lower()
        if not target:
            return False
        for message in reversed(self.state.recent_messages):
            if message.role != "assistant" or not message.content:
                continue
            if target in message.content.lower():
                return True
        return False





def _file_read_cache_key(cwd: str, payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    try:
        resolved = str(candidate.resolve())
    except Exception:
        resolved = str(candidate)
    start_line = payload.get("requested_start_line", payload.get("start_line"))
    end_line = payload.get("requested_end_line", payload.get("end_line"))
    max_bytes = payload.get("max_bytes", 100_000)
    return f"{resolved}|{start_line}|{end_line}|{max_bytes}"


def _consolidate_shell_attempt_family(
    *,
    state: LoopState,
    artifact_id: str,
    result: ToolEnvelope,
) -> None:
    arguments = result.metadata.get("arguments") if isinstance(result.metadata, dict) else None
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or "").strip()
    if not command and isinstance(result.metadata, dict):
        command = str(result.metadata.get("command") or "").strip()
    if not command:
        return

    family_key = _shell_attempt_family_key(command)
    if not family_key:
        return

    family_state = state.scratchpad.setdefault("_shell_attempt_families", {})
    if not isinstance(family_state, dict):
        family_state = {}
        state.scratchpad["_shell_attempt_families"] = family_state

    record = family_state.get(family_key)
    if not isinstance(record, dict):
        record = {
            "tool_name": "shell_exec",
            "members": [],
            "canonical_artifact_id": None,
            "resolved": False,
        }
        family_state[family_key] = record

    members = record.get("members")
    if not isinstance(members, list):
        members = []
        record["members"] = members

    is_diagnostic = _shell_attempt_is_diagnostic(command)
    root = _shell_command_root(command)
    canonical_artifact_id = record.get("canonical_artifact_id")
    canonical_artifact_id = canonical_artifact_id if isinstance(canonical_artifact_id, str) and canonical_artifact_id else None

    artifact = state.artifacts.get(artifact_id)
    if artifact is None:
        return

    artifact.metadata["attempt_family"] = family_key
    if root:
        artifact.metadata["attempt_family_root"] = root

    if canonical_artifact_id:
        artifact.metadata["attempt_status"] = "redundant"
        artifact.metadata["superseded_by"] = canonical_artifact_id
        artifact.metadata["canonical_attempt_artifact_id"] = canonical_artifact_id
        members.append(artifact_id)
        return

    previous_members = [member_id for member_id in members if member_id != artifact_id]
    members.append(artifact_id)

    if result.success and not is_diagnostic:
        record["resolved"] = True
        record["canonical_artifact_id"] = artifact_id
        artifact.metadata["attempt_status"] = "canonical"
        artifact.metadata["canonical_attempt_artifact_id"] = artifact_id
        for member_id in previous_members:
            _mark_artifact_superseded(
                state=state,
                artifact_id=member_id,
                superseded_by=artifact_id,
                family_key=family_key,
                reason="resolved_by_success",
            )
        return

    artifact.metadata["attempt_status"] = "diagnostic" if is_diagnostic and result.success else "failed"
    for member_id in previous_members:
        _mark_artifact_superseded(
            state=state,
            artifact_id=member_id,
            superseded_by=artifact_id,
            family_key=family_key,
            reason="replaced_by_new_attempt",
        )


def _mark_artifact_superseded(
    *,
    state: LoopState,
    artifact_id: str,
    superseded_by: str,
    family_key: str,
    reason: str,
) -> None:
    artifact = state.artifacts.get(artifact_id)
    if artifact is None:
        return
    artifact.metadata["attempt_family"] = family_key
    artifact.metadata["superseded_by"] = superseded_by
    artifact.metadata["attempt_status"] = "superseded"
    artifact.metadata["superseded_reason"] = reason


def _shell_attempt_family_key(command: str) -> str | None:
    root = _shell_command_root(command)
    if not root:
        return None
    return f"shell_exec:{root}"


def _shell_attempt_is_diagnostic(command: str) -> bool:
    tokens = _shell_tokens(command)
    if not tokens:
        return False
    first = tokens[0].lower()
    wrapper_tokens = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell", "cmd", "cmd.exe", "sudo"}
    if first in wrapper_tokens:
        inner = _shell_unwrap_command(tokens)
        return _shell_attempt_is_diagnostic(inner)
    lowered = [token.lower() for token in tokens]
    return any(token in {"-h", "--help", "/?"} for token in lowered) or "help" in lowered[1:]


def _shell_command_root(command: str) -> str | None:
    tokens = _shell_tokens(command)
    if not tokens:
        return None
    first = tokens[0].lower()
    wrapper_tokens = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell", "cmd", "cmd.exe", "sudo"}
    if first in wrapper_tokens:
        inner = _shell_unwrap_command(tokens)
        if inner == command:
            return first
        return _shell_command_root(inner)
    for token in tokens:
        if token.startswith("-"):
            continue
        if "=" in token and token.split("=", 1)[0].isidentifier():
            continue
        return token.lower()
    return first


def _shell_unwrap_command(tokens: list[str]) -> str:
    if not tokens:
        return ""
    first = tokens[0].lower()
    if first == "sudo":
        inner_tokens = tokens[1:]
        while inner_tokens and inner_tokens[0].startswith("-"):
            inner_tokens = inner_tokens[1:]
        return " ".join(inner_tokens)
    if len(tokens) < 2:
        return tokens[0]
    if tokens[1] in {"-c", "-lc", "/c", "-Command", "-command"}:
        return " ".join(tokens[2:])
    return " ".join(tokens[1:])


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _coerce_usage_token_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _trim_recent_messages_window(
    messages: list[ConversationMessage],
    *,
    limit: int,
) -> list[ConversationMessage]:
    if len(messages) <= limit:
        return list(messages)
    last_user_index = max(
        (index for index, message in enumerate(messages) if message.role == "user"),
        default=-1,
    )
    if last_user_index == -1:
        trimmed = list(messages[-limit:])
    else:
        suffix = list(messages[last_user_index + 1 :])
        trimmed_suffix = suffix[-max(limit - 1, 0) :]
        while trimmed_suffix and trimmed_suffix[0].role == "tool":
            trimmed_suffix.pop(0)
        trimmed = [messages[last_user_index], *trimmed_suffix]
        if len(trimmed) > limit:
            trimmed = [trimmed[0], *trimmed[-(limit - 1) :]]
    return trimmed[-limit:]
