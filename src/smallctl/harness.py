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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from .client import OpenAICompatClient
from .context import (
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
from .guards import GuardConfig
from .logging_utils import RunLogger, log_kv
from .models.conversation import ConversationMessage
from .models.tool_result import ToolEnvelope
from .models.events import UIEvent, UIEventType
from .phases import normalize_phase
from .state import (
    ExperienceMemory,
    LoopState,
    _coerce_experience_memory,
    align_memory_entries,
    clip_string_list,
    clip_text_value,
    _dedupe_keep_tail,
    json_safe_value,
)
from .plans import write_plan_file
from .tools import ToolDispatcher, build_registry
from .tools.ansible import AnsibleRunnerAdapter, SessionInventory
from .tools.profiles import classify_tool_profiles
from .memory.taxonomy import (
    PHASE_MISMATCH,
    PREMATURE_TASK_COMPLETE,
    SCHEMA_VALIDATION_ERROR,
    TOOL_NOT_CALLED,
    ZERO_ARG_TOOL_ARG_LEAK,
    REPEATED_TOOL_LOOP,
    WRONG_TOOL_CALLED,
    UNKNOWN_FAILURE,
)


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
        summarizer_endpoint: str | None = None,
        summarizer_model: str | None = None,
        summarizer_api_key: str | None = None,
        run_logger: RunLogger | None = None,
        artifact_start_index: int | None = None,
        tool_result_inline_token_limit: int = 250,
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
        self.state = LoopState(
            current_phase=normalized_phase,
            strategy=strategy,
            planning_mode_enabled=bool(planning_mode),
        )
        self.client = OpenAICompatClient(
            base_url=endpoint,
            model=model,
            api_key=api_key,
            chat_endpoint=chat_endpoint,
            provider_profile=provider_profile,
            runtime_context_probe=runtime_context_probe,
            run_logger=run_logger,
        )
        self.summarizer_client = None
        if summarizer_endpoint:
            self.summarizer_client = OpenAICompatClient(
                base_url=summarizer_endpoint,
                model=summarizer_model or model,
                api_key=summarizer_api_key or api_key,
                chat_endpoint=chat_endpoint,
                runtime_context_probe=False,
                run_logger=run_logger,
            )
        self.reasoning_mode = reasoning_mode
        self.thinking_visibility = thinking_visibility
        self.thinking_start_tag = thinking_start_tag
        self.thinking_end_tag = thinking_end_tag
        self._active_processes: set[asyncio.subprocess.Process] = set()
        self.strategy_prompt = strategy_prompt
        self.event_handler = None
        self.allow_interactive_shell_approval = bool(allow_interactive_shell_approval)
        self.shell_approval_session_default = bool(shell_approval_session_default)
        self._shell_approval_waiters: dict[str, asyncio.Future[bool]] = {}
        self._configured_tool_profiles = list(tool_profiles) if tool_profiles else None
        self._strategy_prompt = strategy_prompt
        self._indexer = indexer
        self._use_ansible = use_ansible
        self._harness_kwargs = {
            "endpoint": endpoint,
            "model": model,
            "phase": normalized_phase,
            "provider_profile": provider_profile,
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
            "context_limit": context_limit,
            "max_prompt_tokens": max_prompt_tokens,
            "reserve_completion_tokens": reserve_completion_tokens,
            "reserve_tool_tokens": reserve_tool_tokens,
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
        self.provider_profile = provider_profile
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
            phase=normalized_phase,
            ansible_check_mode_in_plan=ansible_check_mode_in_plan,
            tier2_adapter=tier2_adapter,
            run_logger=run_logger,
        )
        if not max_prompt_tokens and context_limit:
            # Automatic budgeting: leave 25% headroom for completions (thinking + output), min 1024
            headroom = max(1024, context_limit // 4)
            max_prompt_tokens = context_limit - headroom

        self.context_policy = policy or ContextPolicy(
            max_prompt_tokens=max_prompt_tokens or context_limit,
            reserve_completion_tokens=reserve_completion_tokens,
            reserve_tool_tokens=reserve_tool_tokens,
            summarize_at_ratio=summarize_at_ratio,
            recent_message_limit=recent_message_limit,
            max_summary_items=max_summary_items,
            max_artifact_snippets=max_artifact_snippets,
            artifact_snippet_token_limit=artifact_snippet_token_limit,
            tool_result_inline_token_limit=tool_result_inline_token_limit,
        )
        self.state.recent_message_limit = self.context_policy.recent_message_limit
        self.prompt_assembler = PromptAssembler(self.context_policy)
        self.retriever = LexicalRetriever(self.context_policy)
        self.summarizer = ContextSummarizer(self.context_policy)
        self.subtask_runner = SubtaskRunner(max_child_depth=1)
        self.guards = GuardConfig()
        self.server_context_limit: int | None = max_prompt_tokens or context_limit
        self.conversation_id = uuid.uuid4().hex[:8]
        if not self.state.thread_id:
            self.state.thread_id = self.conversation_id
        artifact_base_dir = Path(self.state.cwd).resolve() / ".smallctl" / "artifacts"
        self.artifact_store = ArtifactStore(
            artifact_base_dir, 
            self.conversation_id,
            artifact_start_index=artifact_start_index
        )
        memory_base_dir = Path(self.state.cwd).resolve() / ".smallctl" / "memory"
        from .memory_store import ExperienceStore
        self.warm_memory_store = ExperienceStore(memory_base_dir / "warm-experiences.jsonl")
        self.cold_memory_store = ExperienceStore(memory_base_dir / "cold-experiences.jsonl")
        # Fresh harness starts should begin empty. Prior experience is restored
        # only through an explicit graph resume, which hydrates the loop state.

        self._cancel_requested = False
        self._active_dispatch_task: asyncio.Task[Any] | None = None
        self._runlog(
            "harness_started",
            "harness initialized",
            endpoint=endpoint,
            model=model,
            phase=normalized_phase,
        )
        self._use_ansible = use_ansible
        self._configured_tool_profiles = list(tool_profiles) if tool_profiles else None

    def set_interactive_shell_approval(self, enabled: bool) -> None:
        self.allow_interactive_shell_approval = bool(enabled)
        self._harness_kwargs["allow_interactive_shell_approval"] = self.allow_interactive_shell_approval

    def set_shell_approval_session_default(self, enabled: bool) -> None:
        self.shell_approval_session_default = bool(enabled)
        self._harness_kwargs["shell_approval_session_default"] = self.shell_approval_session_default

    async def run_task(self, task: str) -> dict[str, Any]:
        return await self.run_task_with_events(task)

    async def run_auto(
        self,
        task: str,
        *,
        event_handler: Callable[[UIEvent], Awaitable[None]] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        from .graph.runtime import (
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
        from .graph.runtime import LoopGraphRuntime

        runtime = LoopGraphRuntime.from_harness(self)
        return runtime.restore(thread_id=thread_id)

    async def run_subtask(
        self,
        brief: str,
        *,
        phase: str | None = None,
        depth: int = 1,
        max_prompt_tokens: int | None = None,
        recent_message_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        harness_factory: Callable[..., "Harness"] | None = None,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> ChildRunResult:
        request = ChildRunRequest(
            brief=brief,
            phase=normalize_phase(phase or self.state.current_phase),
            depth=depth,
            max_prompt_tokens=max_prompt_tokens,
            recent_message_limit=recent_message_limit or self.context_policy.recent_message_limit,
            metadata=metadata or {},
        )
        self._runlog(
            "subtask_start",
            "starting child run",
            brief=brief,
            phase=request.phase,
            depth=request.depth,
            max_prompt_tokens=request.max_prompt_tokens,
            recent_message_limit=request.recent_message_limit,
        )
        await self._emit(
            event_handler,
            UIEvent(
                event_type=UIEventType.SYSTEM,
                content=f"Starting subtask: {brief}",
                data={"phase": request.phase, "depth": request.depth},
            ),
        )
        from .graph.subgraphs import ChildSubgraphRunner

        result = await ChildSubgraphRunner().execute(
            parent=self,
            request=request,
            harness_factory=harness_factory,
        )
        self.subtask_runner.merge_result(parent_state=self.state, request=request, result=result)
        self._runlog(
            "subtask_complete",
            "child run finished",
            status=result.status,
            summary=result.summary,
            artifact_ids=result.artifact_ids,
            files_touched=result.files_touched,
        )
        await self._emit(
            event_handler,
            UIEvent(
                event_type=UIEventType.SYSTEM,
                content=f"Subtask {result.status}: {result.summary}",
                data={
                    "status": result.status,
                    "artifact_ids": result.artifact_ids,
                    "files_touched": result.files_touched,
                },
            ),
        )
        return result

    async def decide_run_mode(self, task: str) -> str:
        plan_request = self._extract_planning_request(task)
        if plan_request is not None:
            output_path, output_format = plan_request
            self._set_planning_request(output_path=output_path, output_format=output_format)
            self._runlog(
                "mode_decision",
                "selected run mode",
                mode="planning",
                raw="planning_intent",
                output_path=output_path,
                output_format=output_format,
            )
            return "planning"
        if self.state.planning_mode_enabled and not (self.state.active_plan and self.state.active_plan.approved):
            self._runlog(
                "mode_decision",
                "selected run mode",
                mode="planning",
                raw="planning_mode_enabled",
            )
            return "planning"
        if self._is_smalltalk(task):
            self._runlog("mode_decision", "selected run mode", mode="chat", raw="smalltalk_heuristic")
            return "chat"
        if self._needs_contextual_loop_escalation(task):
            self._runlog(
                "mode_decision",
                "selected run mode",
                mode="loop",
                raw="contextual_execution_followup",
            )
            return "loop"
        if (
            self._needs_loop_for_content_lookup(task)
            or self._looks_like_action_request(task)
            or self._looks_like_shell_request(task)
        ):
            self._runlog(
                "mode_decision",
                "selected run mode",
                mode="loop",
                raw="action_lookup_heuristic",
            )
            return "loop"
        prompt = (
            "Decide whether the user request requires tool usage in a coding harness. "
            "Reply with exactly one word: chat or loop."
        )
        messages = [
            ConversationMessage(role="system", content=prompt).to_dict(),
            ConversationMessage(role="user", content=task).to_dict(),
        ]
        chunks: list[dict[str, Any]] = []
        try:
            async for event in self.client.stream_chat(messages=messages, tools=[]):
                chunks.append(event)
        except Exception:
            self._runlog("mode_decision_fallback", "mode decision failed, using loop")
            return "loop"
        decision_result = OpenAICompatClient.collect_stream(
            chunks,
            reasoning_mode="off",
            thinking_start_tag=self.thinking_start_tag,
            thinking_end_tag=self.thinking_end_tag,
        )
        decision = decision_result.assistant_text.strip().lower()
        if not decision:
            decision = decision_result.thinking_text.strip().lower()
        mode = "loop" if decision.startswith("loop") else "chat"
        self._runlog("mode_decision", "selected run mode", mode=mode, raw=decision)
        return mode

    def _set_planning_request(self, *, output_path: str | None = None, output_format: str | None = None) -> None:
        self.state.planning_mode_enabled = True
        self.state.planner_requested_output_path = str(output_path or "").strip()
        self.state.planner_requested_output_format = str(output_format or "").strip().lower()
        self.state.planner_resume_target_mode = "loop"
        self.state.touch()
        self.state.sync_plan_mirror()

    def _extract_planning_request(self, task: str) -> tuple[str | None, str | None] | None:
        lowered = (task or "").lower()
        if "plan" not in lowered:
            return None
        output_path: str | None = None
        output_format: str | None = None
        import re

        path_match = re.search(r"([^\s]+?\.(?:md|txt|text))\b", task, flags=re.IGNORECASE)
        if path_match:
            output_path = path_match.group(1)
            suffix = Path(output_path).suffix.lower()
            if suffix == ".md":
                output_format = "markdown"
            elif suffix in {".txt", ".text"}:
                output_format = "text"
        if "plan.md" in lowered and output_path is None:
            output_format = "markdown"
        if any(
            phrase in lowered
            for phrase in (
                "make a plan",
                "make a short plan",
                "create a plan",
                "create a short plan",
                "create a brief plan",
                "plan this",
                "plan this out",
                "make a plan first",
                "plan out",
                "before doing anything, create a short plan",
                "before doing anything, create a plan",
                "before doing anything, plan",
            )
        ):
            return output_path, output_format
        if output_path:
            return output_path, output_format
        return None

    async def run_chat_with_events(
        self,
        task: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        from .graph.runtime import ChatGraphRuntime

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
        from .graph.runtime import AutoGraphRuntime

        self.event_handler = event_handler
        runtime = AutoGraphRuntime.from_harness(
            self,
            event_handler=event_handler,
        )

        return await runtime.run(task)

    def cancel(self) -> None:
        self._cancel_requested = True
        self._reject_pending_shell_approvals()
        if self._active_dispatch_task and not self._active_dispatch_task.done():
            self._active_dispatch_task.cancel()
        log_kv(self.log, logging.INFO, "harness_cancel_requested")
        # Direct cleanup on cancel helps avoid abandoned processes
        asyncio.create_task(self.teardown())

    async def teardown(self) -> None:
        """Kill and cleanup any remaining active processes efficiently."""
        self._reject_pending_shell_approvals()
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
    ) -> bool:
        if not self.allow_interactive_shell_approval or getattr(self, "event_handler", None) is None:
            return True
        if self.shell_approval_session_default:
            return True

        approval_id = f"shell-{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._shell_approval_waiters[approval_id] = future
        event = UIEvent(
            event_type=UIEventType.ALERT,
            content="Approve shell command?",
            data={
                "ui_kind": "approve_prompt",
                "approval_id": approval_id,
                "command": command,
                "cwd": cwd,
                "timeout_sec": timeout_sec,
                "status_activity": "awaiting approval...",
            },
        )
        try:
            await self._emit(self.event_handler, event)
            return await future
        except Exception:
            self._reject_shell_approval(approval_id)
            raise
        finally:
            self._shell_approval_waiters.pop(approval_id, None)

    def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
        future = self._shell_approval_waiters.get(approval_id)
        if future is None or future.done():
            return
        future.set_result(bool(approved))

    def _reject_pending_shell_approvals(self) -> None:
        for approval_id in list(self._shell_approval_waiters.keys()):
            self._reject_shell_approval(approval_id)

    def _reject_shell_approval(self, approval_id: str) -> None:
        future = self._shell_approval_waiters.get(approval_id)
        if future is None or future.done():
            return
        future.set_result(False)

    async def run_task_with_events(
        self,
        task: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        from .graph.runtime import LoopGraphRuntime

        self.event_handler = event_handler
        # Reset task-boundary state
        self.state.scratchpad.pop("_action_stalls", None)
        self.state.scratchpad.pop("_no_tool_nudges", None)
        self.state.scratchpad.pop("_chat_rounds", None)
        # Deep-purge state to isolate context for the NEW task
        self.state.recent_messages = []
        self.state.experiences = []
        self.state.working_memory.known_facts = []
        self.state.artifacts = {}
        self.state.episodic_summaries = []
        self.state.retrieval_cache = []
        self.state.scratchpad.pop("suppressed_truncated_artifact_ids", None)
        self.state.retrieved_experience_ids = []
        self.state.tool_execution_records = {}
        self.state.recent_errors = []
        
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
        from .graph.runtime import LoopGraphRuntime, PlanningGraphRuntime

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
        self._runlog("task_finalize", "task finished", result=result)
        self._record_terminal_experience(result)
        self._rewrite_active_plan_export()
        if self.checkpoint_on_exit:
            self._persist_checkpoint(result)
            
        # Add runtime metrics for benchmarking/AHO
        result["step_count"] = self.state.step_count
        result["inactive_steps"] = self.state.inactive_steps
        result["token_usage"] = self.state.token_usage
        
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
        child_kwargs = dict(self._harness_kwargs)
        child_kwargs["phase"] = request.phase
        child_kwargs["checkpoint_on_exit"] = False
        child_kwargs["checkpoint_path"] = None
        child_kwargs["artifact_start_index"] = artifact_start_index
        if request.max_prompt_tokens is not None:
            child_kwargs["max_prompt_tokens"] = request.max_prompt_tokens
            child_kwargs["context_limit"] = request.max_prompt_tokens
            child_kwargs["reserve_completion_tokens"] = min(
                self.context_policy.reserve_completion_tokens,
                max(64, request.max_prompt_tokens // 5),
            )
            child_kwargs["reserve_tool_tokens"] = min(
                self.context_policy.reserve_tool_tokens,
                max(64, request.max_prompt_tokens // 8),
            )
        child_recent_limit = request.recent_message_limit
        if request.max_prompt_tokens is not None and request.max_prompt_tokens <= 1024:
            child_recent_limit = min(child_recent_limit, 2)
        child_kwargs["recent_message_limit"] = child_recent_limit
        factory = harness_factory or self.__class__
        child = factory(**child_kwargs)
        child.state.cwd = self.state.cwd
        child.state.inventory_state = dict(self.state.inventory_state)
        return child

    @staticmethod
    def _build_subtask_result(
        *,
        child: "Harness",
        request: ChildRunRequest,
        result: dict[str, Any],
    ) -> ChildRunResult:
        del request
        raw_status = str(result.get("status", "unknown"))
        summary = _clean_subtask_summary(
            _extract_subtask_summary_value(result) or raw_status
        )
        status = _normalize_subtask_status(result=result, summary=summary)
        file_sources = [
            artifact.source
            for artifact in child.state.artifacts.values()
            if artifact.source and artifact.kind in {"file_read", "shell_exec", "grep", "yaml_read"}
        ]
        return ChildRunResult(
            status=status,
            summary=str(summary),
            artifact_ids=list(child.state.artifacts.keys())[-15:],
            files_touched=file_sources[-8:],
            decisions=child.state.working_memory.decisions[-6:],
            remaining_plan=child.state.working_memory.next_actions[-6:],
            artifacts={aid: child.state.artifacts[aid] for aid in list(child.state.artifacts.keys())[-15:]},
            metadata={
                "current_phase": child.state.current_phase,
                "step_count": child.state.step_count,
                "token_usage": child.state.token_usage,
                "result": result,
            },
        )

    def _persist_checkpoint(self, result: dict[str, Any]) -> None:
        path = (
            Path(self.checkpoint_path).resolve()
            if self.checkpoint_path
            else Path(self.state.cwd).resolve() / ".smallctl-checkpoint.json"
        )
        payload = {
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

    @staticmethod
    def _stream_print(text: str) -> None:
        try:
            print(text, end="", flush=True)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe, end="", flush=True)

    async def _ensure_context_limit(self) -> None:
        if self.server_context_limit is None and hasattr(self.client, "fetch_model_context_limit"):
            try:
                self.server_context_limit = await self.client.fetch_model_context_limit()
            except Exception:
                self.server_context_limit = None
            if self.server_context_limit is not None:
                limit: int = self.server_context_limit
                if not self.context_policy.max_prompt_tokens:
                    # Apply automatic 25% headroom to discovered server limit
                    headroom = max(1024, limit // 4)
                    self.context_policy.max_prompt_tokens = limit - headroom
                
                # Cascade update to all internal section quotas (Phase I)
                self.context_policy.recalculate_quotas(limit)
                self.state.recent_message_limit = self.context_policy.recent_message_limit

                self._runlog(
                    "context_limit",
                    "server context limit detected",
                    context_limit=limit,
                    max_prompt_tokens=self.context_policy.max_prompt_tokens,
                    hot_message_limit=self.context_policy.hot_message_limit,
                )

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
        self._update_working_memory()
        query = self._select_retrieval_query()
        self._runlog("retrieval_query", "selected retrieval query", query=query)
        await self._maybe_compact_context(
            query=query,
            system_prompt=system_prompt,
            event_handler=event_handler,
        )
        fresh_run_active = self.fresh_run and self._fresh_run_turns_remaining > 0
        cold_store = None if fresh_run_active else self.cold_memory_store
        retrieval_bundle = self.retriever.retrieve_bundle(
            state=self.state,
            query=query,
            cold_store=cold_store,
            include_experiences=not fresh_run_active,
        )
        summaries = retrieval_bundle.summaries
        artifacts = retrieval_bundle.artifacts
        experiences = [] if fresh_run_active else retrieval_bundle.experiences
        self._runlog(
            "retrieval_selected",
            "retrieval candidates selected",
            query=retrieval_bundle.query,
            initial_query=retrieval_bundle.initial_query,
            refined=retrieval_bundle.refined,
            refinement_reason=retrieval_bundle.refinement_reason,
            best_scores=retrieval_bundle.best_scores,
            candidate_counts=retrieval_bundle.candidate_counts,
            summaries=[
                {
                    "summary_id": summary.summary_id,
                    "artifact_ids": summary.artifact_ids,
                    "files_touched": summary.files_touched,
                }
                for summary in summaries
            ],
            artifacts=[
                {
                    "artifact_id": artifact.artifact_id,
                    "score": artifact.score,
                    "preview": artifact.text[:160],
                }
                for artifact in artifacts
            ],
            experiences=[
                {
                    "memory_id": exp.memory_id,
                    "intent": exp.intent,
                    "tool": exp.tool_name,
                    "outcome": exp.outcome,
                }
                for exp in experiences
            ],
        )
        recent_limit = self.context_policy.recent_message_limit
        include_structured_sections = bool(summaries or artifacts or experiences or self.state.episodic_summaries)
        soft_limit = self.context_policy.soft_prompt_token_limit
        assembly = self.prompt_assembler.build_messages(
            state=self.state,
            system_prompt=system_prompt,
            retrieved_summaries=summaries,
            retrieved_artifacts=artifacts,
            retrieved_experiences=experiences,
            recent_message_limit=recent_limit,
            include_structured_sections=include_structured_sections,
            token_budget=soft_limit,
        )
        if fresh_run_active and self._fresh_run_turns_remaining > 0:
            self._fresh_run_turns_remaining -= 1
        
        self.state.scratchpad["context_used_tokens"] = assembly.estimated_prompt_tokens

        self._runlog(
            "prompt_budget",
            "prompt assembly estimate (bidding complete)",
            estimated_prompt_tokens=assembly.estimated_prompt_tokens,
            sections=assembly.section_tokens,
            message_count=len(assembly.messages),
            retrieval_cache=self.state.retrieval_cache,
        )

        # 7. Hard Budget Guard (Phase III)
        limit = self.context_policy.max_prompt_tokens
        if limit and assembly.estimated_prompt_tokens > limit:
             raise RuntimeError(f"PROMPT BUDGET OVERFLOW: {assembly.estimated_prompt_tokens} tokens assembled, which exceeds the max prompt limit of {limit}.")

        return assembly.messages

    async def _record_tool_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        result: ToolEnvelope,
        arguments: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        from .context.policy import estimate_text_tokens
        
        # Ensure arguments are preserved in metadata for loop detection
        if arguments is not None and "arguments" not in result.metadata:
             result.metadata["arguments"] = arguments
        
        if result.metadata.get("cache_hit"):
            artifact_id = str(result.metadata.get("artifact_id", "")).strip()
            artifact = self.state.artifacts.get(artifact_id)
            if artifact is None:
                compact_content = json.dumps(json_safe_value(result.to_dict()), ensure_ascii=True)
            else:
                self.state.retrieval_cache = [artifact.artifact_id]
                compact_content = format_reused_artifact_message(artifact)
                self._runlog(
                    "artifact_reused",
                    "tool result satisfied from cached artifact",
                    artifact_id=artifact.artifact_id,
                    tool_name=tool_name,
                    source=artifact.source,
                )
            return ConversationMessage(
                role="tool",
                name=tool_name,
                tool_call_id=tool_call_id,
                content=compact_content,
                metadata={"artifact_id": artifact_id, "cache_hit": True},
            )
        artifact = None
        # Optimization: Do NOT create new artifacts for tool snapshots of existing artifacts (e.g. artifact_read).
        # This prevents 'A0001 -> artifact_read -> A0002 -> artifact_read -> A0003' chains that confuse the model.
        if tool_name == "artifact_read" and result.success and result.metadata.get("artifact_id"):
             # Use the existing backing artifact from tool metadata
             artifact_id = str(result.metadata["artifact_id"])
             artifact = self.state.artifacts.get(artifact_id)
             if artifact is not None and isinstance(result.output, str):
                 artifact.preview_text = result.output[:self.artifact_store.policy.preview_char_limit]
        
        if artifact is None:
             artifact = self.artifact_store.persist_tool_result(tool_name=tool_name, result=result)
        
        # Automatic summarization for large outputs (avoiding context flood)
        if result.success and result.output and artifact:
            out_str = str(result.output)
            tokens = estimate_text_tokens(out_str)
            if tokens > self.context_policy.artifact_summarization_threshold and self.summarizer_client:
                self._runlog(
                    "context_summarize_request",
                    "requesting summarization for large tool result",
                    tool_name=tool_name,
                    tokens=tokens,
                )
                try:
                    distilled = await self.summarizer.summarize_artifact_async(
                        client=self.summarizer_client,
                        artifact_id=artifact.artifact_id,
                        content=out_str,
                        label=artifact.source or tool_name
                    )
                    if distilled:
                        artifact.summary = f"Distilled: {distilled}"
                        # Also override preview text to be the summary for better retrieval/context
                        artifact.preview_text = distilled[:self.artifact_store.policy.preview_char_limit]
                        # Mark as summarized so messages.py suppresses the artifact_read followup hint.
                        # Without this, the compact tool message tells the model to call artifact_read
                        # while the system prompt says not to — a direct signal conflict.
                        artifact.metadata["summarized"] = True
                except Exception as exc:
                    self.log.warning("Automatic context summarization failed: %s", exc)

        if artifact:
            self.state.artifacts[artifact.artifact_id] = artifact
            self.state.retrieval_cache = [artifact.artifact_id]
            if tool_name == "shell_exec":
                _consolidate_shell_attempt_family(
                    state=self.state,
                    artifact_id=artifact.artifact_id,
                    result=result,
                )
            if tool_name == "file_read" and result.success:
                cache_key = _file_read_cache_key(self.state.cwd, result.metadata)
                if cache_key:
                    cache = self.state.scratchpad.setdefault("file_read_cache", {})
                    if isinstance(cache, dict):
                        cache[cache_key] = artifact.artifact_id

            if tool_name == "memory_update" and result.success:
                section = str(result.metadata.get("section", "")).strip().lower()
                if section == "plan":
                    self.state.plan_artifact_id = artifact.artifact_id
                    self.state.plan_resolved = True
            elif tool_name == "artifact_read" and result.success:
                artifact_id = str(result.metadata.get("artifact_id", "")).strip()
                if artifact_id:
                    if artifact_id == self.state.plan_artifact_id:
                        self.state.plan_resolved = True
                    elif (
                        not self.state.plan_artifact_id
                        and artifact.tool_name == "memory_update"
                        and str(artifact.metadata.get("section", "")).strip().lower() == "plan"
                    ):
                        self.state.plan_artifact_id = artifact_id
                        self.state.plan_resolved = True
                    if result.metadata.get("truncated"):
                        suppressed = self.state.scratchpad.get("suppressed_truncated_artifact_ids", [])
                        if isinstance(suppressed, list):
                            self.state.scratchpad["suppressed_truncated_artifact_ids"] = _dedupe_keep_tail(
                                suppressed + [artifact_id],
                                limit=12,
                            )
                        else:
                            self.state.scratchpad["suppressed_truncated_artifact_ids"] = [artifact_id]

            if tool_name != "shell_exec":
                fact_label = artifact.summary or tool_name
                self.state.working_memory.known_facts = _dedupe_keep_tail(
                    self.state.working_memory.known_facts + [f"{tool_name}: {fact_label}"],
                    limit=12,
                )
            # Success clears consecutive error count
            self.state.recent_errors = []
            
        if not result.success and result.error:
            # Skip recording hallucinations as permanent failures to keep the retrieval context clean
            if not result.metadata.get("hallucination") and not result.metadata.get("approval_denied"):
                self.state.working_memory.failures = _dedupe_keep_tail(
                    self.state.working_memory.failures + [f"{tool_name}: {result.error}"],
                    limit=8,
                )
                # Track consecutive errors for guard tripping
                self.state.recent_errors.append(f"{tool_name}: {result.error}")
        
        request_text = self.state.run_brief.original_task or self._current_user_task()
        compact_content = (
            self.artifact_store.compact_tool_message(
                artifact,
                result,
                request_text=request_text,
            )
            if artifact
            else str(result.output)
        )

        self._runlog(
            "artifact_created",
            "tool result processed",
            artifact_id=artifact.artifact_id if artifact else None,
            tool_name=tool_name,
            source=artifact.source if artifact else tool_name,
            size_bytes=artifact.size_bytes if artifact else 0,
            inline=bool(artifact and artifact.inline_content is not None),
        )

        msg = ConversationMessage(
            role="tool",
            name=tool_name,
            tool_call_id=tool_call_id,
            content=compact_content,
            metadata={"artifact_id": artifact.artifact_id} if artifact else {},
        )

        # Record tool history fingerprint for loop/deadlock detection
        args_str = json.dumps(result.metadata.get("arguments", {}), sort_keys=True)
        outcome = "success" if result.success else f"error:{result.error}"
        fingerprint = f"{tool_name}|{args_str}|{outcome}"
        self.state.append_tool_history(fingerprint)

        return msg

    def _record_assistant_message(
        self,
        *,
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        self.state.append_message(
            ConversationMessage(
                role="assistant",
                content=assistant_text or None,
                tool_calls=tool_calls,
            )
        )

    def _initialize_run_brief(self, task: str) -> None:
        self.state.run_brief.original_task = task
        self.state.run_brief.current_phase_objective = f"{self.state.current_phase}: {task}"
        self.state.working_memory.current_goal = task
        self.state.working_memory.next_actions = _dedupe_keep_tail(
            self.state.working_memory.next_actions + [f"{self.state.current_phase}: gather the next missing fact for {task}"],
                limit=6,
            )

    async def _maybe_compact_context(
        self,
        query: str,
        system_prompt: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> None:
        soft_limit = self.context_policy.soft_prompt_token_limit
        tier_manager = MessageTierManager(self.context_policy)
        
        # 1. Check Triggers (Phase II + IV)
        should_compact = (
            tier_manager.should_compact(self.state) or 
            tier_manager.should_compact_predictive(self.state, soft_limit)
        )
        
        if not should_compact:
            # Fallback for hard message limits (Phase II)
            if len(self.state.recent_messages) <= tier_manager.hot_window:
                return

        # 2. Structured Tier Compaction
        msg = "Context almost exhausted; initiating compression..."
        self._runlog("compaction_trigger", msg)
        if event_handler:
             asyncio.create_task(event_handler(UIEvent(event_type=UIEventType.ALERT, content=msg)))

        if self.summarizer_client:
            try:
                brief = await tier_manager.compact_to_warm(
                    state=self.state,
                    summarizer=self.summarizer,
                    client=self.summarizer_client,
                    artifact_store=self.artifact_store
                )
                if brief:
                    self._runlog("compaction_complete", "Compressed messages into warm brief", brief_id=brief.brief_id)
                    tier_manager.promote_to_cold(self.state, artifact_store=self.artifact_store)
                    return # Successfully compacted
            except Exception as e:
                self._runlog("compaction_error", "Structured tier compaction failed", error=str(e))

        # 3. Ratio-based Safety Fallback (Legacy/Emergency Truncation)
        # If tiering failed or isn't enough, we might still need to truncate.
        if len(self.state.recent_messages) > self.context_policy.recent_message_limit:
            self._runlog("compaction_emergency", "Emergency truncation triggered (message limit exceeded)")
            self.state.recent_messages = self.state.recent_messages[-self.context_policy.recent_message_limit:]
        
        # Re-evaluate after potential emergency truncation
        probe = self.prompt_assembler.build_messages(
            state=self.state,
            system_prompt=system_prompt,
            retrieved_summaries=self.retriever.retrieve_summaries(state=self.state, query=query),
            retrieved_artifacts=self.retriever.retrieve_artifacts(state=self.state, query=query),
            recent_message_limit=self.context_policy.recent_message_limit,
            include_structured_sections=True,
        )
        effective_soft_limit = soft_limit or self.context_policy.max_prompt_tokens or 4096
        threshold = int(effective_soft_limit * self.context_policy.summarize_at_ratio)
        
        if probe.estimated_prompt_tokens <= threshold:
            return # No further compaction needed
        
        self._runlog(
            "compaction_trigger",
            "context compaction triggered (budget pressure)",
            estimated_prompt_tokens=probe.estimated_prompt_tokens,
            summarize_threshold=threshold,
            recent_messages=len(self.state.recent_messages),
        )
        keep_recent = max(4, min(10, self.context_policy.recent_message_limit))
        if self.summarizer_client:
            self._runlog("compaction_start", "AI-based summarization pass started")
            await self._emit(
                event_handler,
                UIEvent(
                    UIEventType.SYSTEM,
                    "Long context detected; summarization pass activated",
                    data={"status_activity": "summarizing..."},
                ),
            )
            try:
                summary = await self.summarizer.compact_recent_messages_async(
                    state=self.state,
                    client=self.summarizer_client,
                    keep_recent=keep_recent,
                    artifact_store=self.artifact_store,
                )
            except Exception as e:
                self._runlog(
                    "compaction_error",
                    "AI summarization failed, falling back to heuristic compaction",
                    error=str(e),
                )
                summary = self.summarizer.compact_recent_messages(
                    state=self.state,
                    keep_recent=keep_recent,
                    artifact_store=self.artifact_store,
                )
        else:
            summary = self.summarizer.compact_recent_messages(
                state=self.state,
                keep_recent=keep_recent,
                artifact_store=self.artifact_store,
            )

        if summary:
            self._runlog(
                "summary_created",
                "compacted recent context",
                summary_id=summary.summary_id,
                artifact_ids=summary.artifact_ids,
                files_touched=summary.files_touched,
            )

    def _update_working_memory(self) -> None:
        self.state.run_brief.current_phase_objective = self.state.current_phase
        if not self.state.working_memory.current_goal:
            self.state.working_memory.current_goal = (
                self.state.run_brief.current_phase_objective or self.state.run_brief.original_task
            )
        if self.state.active_plan is not None or self.state.draft_plan is not None:
            self.state.sync_plan_mirror()
        self._refresh_active_intent()
        self.state.prune_stale_meta(limit=self.context_policy.memory_staleness_step_limit)
        self.state.align_meta_to_content()
        self.state.working_memory.open_questions = clip_string_list(
            self.state.working_memory.open_questions,
            limit=4,
            item_char_limit=240,
        )[0]
        self.state.working_memory.plan = clip_string_list(
            self.state.working_memory.plan,
            limit=10,
            item_char_limit=400,
        )[0]
        self.state.working_memory.decisions = clip_string_list(
            self.state.working_memory.decisions,
            limit=10,
            item_char_limit=400,
        )[0]
        self.state.working_memory.known_facts = clip_string_list(
            self.state.working_memory.known_facts,
            limit=12,
            item_char_limit=320,
        )[0]
        self.state.working_memory.failures = clip_string_list(
            self.state.working_memory.failures,
            limit=8,
            item_char_limit=320,
        )[0]
        
        has_recent_tool_evidence = any(message.role == "tool" for message in self.state.recent_messages[-4:])
        has_known_facts = bool(self.state.working_memory.known_facts)
        task_guidance = self._next_action_for_task(self.state.run_brief.original_task or "")

        # Once we have evidence, nudge toward closure instead of repeating discovery boilerplate.
        if self.state.current_phase == "verify" or has_recent_tool_evidence or has_known_facts:
            next_action = self._completion_next_action()
            if self.state.working_memory.next_actions and self.state.working_memory.next_actions[-1] == task_guidance:
                self.state.working_memory.next_actions.pop()
        else:
            next_action = task_guidance
        next_action, clipped = clip_text_value(next_action, limit=240)
        self.state.working_memory.next_actions = clip_string_list(
            _dedupe_keep_tail(self.state.working_memory.next_actions + [next_action], limit=6),
            limit=6,
            item_char_limit=240,
        )[0]
        self.state.working_memory.next_action_meta = align_memory_entries(
            self.state.working_memory.next_actions,
            self.state.working_memory.next_action_meta,
            current_step=self.state.step_count,
            current_phase=self.state.current_phase,
            confidence=0.7,
        )
        self.state.recent_messages = _trim_recent_messages_window(
            self.state.recent_messages,
            limit=self.context_policy.recent_message_limit,
        )

    def _refresh_active_intent(self) -> None:
        task = self.state.run_brief.original_task or self._current_user_task()
        primary, secondary, tags = self._extract_intent_state(task)
        self.state.active_intent = primary
        self.state.secondary_intents = secondary
        self.state.intent_tags = tags

    def _extract_intent_state(self, task: str) -> tuple[str, list[str], list[str]]:
        text = (task or "").lower()
        secondary: list[str] = []
        tags: list[str] = []
        requested_tool = self._infer_requested_tool_name(task)
        
        # Primary Intent logic from memory-upgrade.md
        primary = "general_task"
        if requested_tool:
            primary = f"use_{requested_tool}"
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
        return f"{self.state.current_phase}: gather the next missing fact for {clip_text_value(task, limit=40)[0]}"

    def _record_experience(
        self,
        *,
        tool_name: str,
        result: ToolEnvelope,
        evidence_refs: list[str] | None = None,
        notes: str = "",
        source: str = "observed",
    ) -> ExperienceMemory:
        failure_mode = self._normalize_failure_mode(result.error, tool_name=tool_name, success=result.success)
        outcome = "success" if result.success else "failure"
        confidence = 0.85 if result.success else 0.60
        if tool_name == "task_complete" and result.success:
            confidence = 0.95
        
        # Operational Notes Guidance based on memory-upgrade.md
        op_notes = notes
        if not op_notes:
            if result.success:
                if tool_name == "task_complete":
                    op_notes = "Task finished successfully. Call task_complete when objectives met."
                else:
                    args_meta = result.metadata.get("arguments") or {}
                    if not args_meta:
                        op_notes = f"Successfully called {tool_name} with no arguments."
                    else:
                        op_notes = f"Successfully called {tool_name}. Key pattern: {list(args_meta.keys())}."
            else:
                if failure_mode == ZERO_ARG_TOOL_ARG_LEAK:
                    op_notes = f"Do not send placeholder arguments to {tool_name}. Call it empty."
                elif failure_mode == SCHEMA_VALIDATION_ERROR:
                    op_notes = f"Argument mismatch for {tool_name}: {result.error}"
                else:
                    op_notes = str(result.error or result.output or "").strip()

        correction_ids = [
            memory.memory_id
            for memory in self.state.warm_experiences
            if memory.intent == self.state.active_intent and memory.tool_name == tool_name and memory.outcome == "failure"
        ]
        memory = ExperienceMemory(
            memory_id=f"mem-{uuid.uuid4().hex[:10]}",
            tier="warm",
            source=source,
            run_id=self.state.thread_id,
            phase=self.state.current_phase,
            intent=self.state.active_intent,
            intent_tags=list(self.state.intent_tags),
            environment_tags=self._infer_environment_tags(),
            entity_tags=self._infer_entity_tags(self.state.run_brief.original_task),
            action_type="tool_call",
            tool_name=tool_name,
            arguments_fingerprint=self._argument_fingerprint(result.metadata.get("arguments")),
            outcome=outcome,
            failure_mode=failure_mode,
            confidence=confidence,
            notes=op_notes,
            evidence_refs=evidence_refs or [],
            supersedes=correction_ids if result.success else [],
        )
        stored = self.state.upsert_experience(memory)
        self.warm_memory_store.upsert(stored)
        self._reinforce_retrieved_experiences(tool_name=tool_name, success=result.success)
        if stored.confidence >= 0.9 or stored.reuse_count >= 3:
            promoted = _coerce_experience_memory(json_safe_value(stored))
            promoted.tier = "cold"
            self.cold_memory_store.upsert(promoted)
        return stored

    def _normalize_failure_mode(self, error: Any, *, tool_name: str, success: bool) -> str:
        if success:
            return ""
        text = str(error or "").lower()
        if "missing required field" in text or "expected type" in text or "schema" in text:
            return SCHEMA_VALIDATION_ERROR
        if "unknown tool" in text:
            return WRONG_TOOL_CALLED
        if "zero-argument" in text or ("scratch_list" in tool_name and "missing" in text):
            return ZERO_ARG_TOOL_ARG_LEAK
        if "loop" in text:
            return REPEATED_TOOL_LOOP
        if "premature" in text:
            return PREMATURE_TASK_COMPLETE
        if "phase" in text:
            return PHASE_MISMATCH
        if "not called" in text:
            return TOOL_NOT_CALLED
        return UNKNOWN_FAILURE

    def _reinforce_retrieved_experiences(self, *, tool_name: str, success: bool) -> None:
        if not self.state.retrieved_experience_ids:
            return
        for memory in self.state.warm_experiences:
            if memory.memory_id in self.state.retrieved_experience_ids and memory.tool_name == tool_name:
                self.state.reinforce_experience(memory.memory_id, success=success)
                self.warm_memory_store.upsert(memory)
                if memory.confidence >= 0.9 or memory.reuse_count >= 3:
                     promoted = _coerce_experience_memory(json_safe_value(memory))
                     promoted.tier = "cold"
                     self.cold_memory_store.upsert(promoted)

    def _record_terminal_experience(self, result: dict[str, Any]) -> None:
        status = str(result.get("status", "") or "")
        if not status:
            return
        success = status in {"completed", "chat_completed"}
        reason = str(result.get("reason", "") or result.get("message", "") or status)
        from .models.tool_result import ToolEnvelope
        payload = ToolEnvelope(
            success=success,
            output={"status": status, "message": reason},
            error=None if success else reason,
            metadata={"status": status},
        )
        self._record_experience(
            tool_name="task_complete" if success else "task_fail",
            result=payload,
            notes=reason,
        )

    def _argument_fingerprint(self, arguments: Any) -> str:
        payload = json.dumps(json_safe_value(arguments or {}), sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _derive_task_contract(self, task: str) -> str:
        lowered = (task or "").lower()
        if "contract" in lowered or "plan" in lowered:
            return "high_fidelity"
        return "general"

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
            self._runlog(
                "chat_tool_selection",
                "chat tool exposure suppressed",
                task=task,
                reason="non_lookup_chat",
            )
            return []
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
        from .context.policy import estimate_text_tokens

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
            from .models.tool_result import ToolEnvelope
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
                message.content = compact
                compacted_any = True
        return compacted_any

    def _current_user_task(self) -> str:
        for message in reversed(self.state.recent_messages):
            if message.role == "user" and message.content:
                return message.content
        last_task = self.state.scratchpad.get("_last_task_text")
        if isinstance(last_task, str) and last_task:
            return last_task
        return self.state.run_brief.original_task

    def _chat_mode_requires_tools(self, task: str) -> bool:
        if self._is_smalltalk(task):
            return False
        if self._needs_loop_for_content_lookup(task):
            return True
        return self._looks_like_readonly_chat_request(task) or self._looks_like_action_request(task)

    def _looks_like_action_request(self, task: str) -> bool:
        text = task.strip().lower()
        action_markers = ("run", "exec", "shell", "terminal", "ping", "curl", "wget", "git")
        return any(m in text for m in action_markers)

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



def _clean_subtask_summary(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if isinstance(value, dict):
        nested = value.get("message") or value.get("output") or value.get("status")
        if nested is not None and nested is not value:
            return _clean_subtask_summary(nested)
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, dict):
            nested = parsed.get("message") or parsed.get("output") or parsed.get("status")
            if nested is not None:
                return _clean_subtask_summary(nested)
    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
        if tail:
            text = tail
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) >= 24 or len(lines) == 1:
            return _strip_summary_markup(last_line)
    return _strip_summary_markup(text)


def _normalize_subtask_status(*, result: dict[str, Any], summary: str) -> str:
    status = str(result.get("status", "unknown"))
    if status != "stopped":
        return status
    if str(result.get("reason", "")) != "no_tool_calls":
        return status
    if summary:
        return "completed"
    return status


def _extract_subtask_summary_value(result: dict[str, Any]) -> Any:
    return (
        result.get("message")
        or result.get("assistant")
        or result.get("reason")
        or result.get("status", "")
    )


def _strip_summary_markup(text: str) -> str:
    cleaned = text.strip()
    for marker in ("**", "__", "`"):
        if cleaned.startswith(marker) and cleaned.endswith(marker) and len(cleaned) > len(marker) * 2:
            cleaned = cleaned[len(marker) : -len(marker)].strip()
    if cleaned.startswith("#"):
        cleaned = cleaned.lstrip("#").strip()
    cleaned = cleaned.replace("`", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    return cleaned


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
