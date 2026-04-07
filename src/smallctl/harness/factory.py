from __future__ import annotations

import logging
from typing import Any, Callable, TYPE_CHECKING

from ..context import ChildRunRequest, ChildRunResult

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.factory")

class SubtaskService:
    def __init__(self, harness: Harness):
        self.harness = harness

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
        request = ChildRunRequest(
            brief=brief,
            phase=phase,
            depth=depth,
            max_prompt_tokens=max_prompt_tokens,
            recent_message_limit=recent_message_limit,
            metadata=metadata or {},
        )

        # Emit starting event
        from ..models.events import UIEventType, UIEvent
        await self.harness._emit(
            event_handler,
            UIEvent(UIEventType.SYSTEM, f"Starting subtask: {brief}")
        )

        child = self.create_child_harness(
            request=request, 
            harness_factory=harness_factory,
            artifact_start_index=artifact_start_index
        )
        child.event_handler = event_handler
        
        # Core execution logic that matched original run_subtask
        try:
            result_dict = await child.run_task(request.brief)
            result = self.build_subtask_result(child=child, request=request, result=result_dict)
            self.harness.subtask_runner.merge_result(
                parent_state=self.harness.state,
                request=request,
                result=result
            )
            await self.harness._emit(
                event_handler,
                UIEvent(UIEventType.SYSTEM, f"Subtask completed: {brief}")
            )
            return result
        finally:
            await child.teardown()

    def create_child_harness(
        self,
        *,
        request: ChildRunRequest,
        harness_factory: Callable[..., "Harness"] | None = None,
        artifact_start_index: int | None = None,
    ) -> "Harness":
        child_kwargs = dict(self.harness._harness_kwargs)
        child_kwargs["phase"] = request.phase
        child_kwargs["checkpoint_on_exit"] = False
        child_kwargs["checkpoint_path"] = None
        child_kwargs["artifact_start_index"] = artifact_start_index
        if getattr(self.harness, "server_context_limit", None) is not None:
            child_kwargs["context_limit"] = self.harness.server_context_limit
        
        if request.max_prompt_tokens is not None:
            child_kwargs["max_prompt_tokens"] = request.max_prompt_tokens
            child_kwargs["reserve_completion_tokens"] = min(
                self.harness.context_policy.reserve_completion_tokens,
                max(64, request.max_prompt_tokens // 5),
            )
            child_kwargs["reserve_tool_tokens"] = min(
                self.harness.context_policy.reserve_tool_tokens,
                max(64, request.max_prompt_tokens // 8),
            )
            
        child_recent_limit = request.recent_message_limit
        if request.max_prompt_tokens is not None and request.max_prompt_tokens <= 1024:
            child_recent_limit = min(child_recent_limit, 2)
        child_kwargs["recent_message_limit"] = child_recent_limit
        
        factory = harness_factory or self.harness.__class__
        child = factory(**child_kwargs)
        child.state.cwd = self.harness.state.cwd
        child.state.inventory_state = dict(self.harness.state.inventory_state)
        return child

    def build_subtask_result(
        self,
        *,
        child: "Harness",
        request: ChildRunRequest,
        result: dict[str, Any],
    ) -> ChildRunResult:
        # These helpers moved to normalization.py
        from ..normalization import (
            clean_subtask_summary,
            normalize_subtask_status,
            extract_subtask_summary_value,
        )

        del request
        raw_status = str(result.get("status", "unknown"))
        summary = clean_subtask_summary(
            extract_subtask_summary_value(result) or raw_status
        )
        status = normalize_subtask_status(result=result, summary=summary)
        
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
