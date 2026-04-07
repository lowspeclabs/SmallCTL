from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TYPE_CHECKING

from ..models.events import UIEvent, UIEventType
from ..context import MessageTierManager

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.compaction")

class CompactionService:
    def __init__(self, harness: Harness):
        self.harness = harness

    async def maybe_compact_context(
        self,
        query: str,
        system_prompt: str,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> None:
        soft_limit = self.harness.context_policy.soft_prompt_token_limit
        tier_manager = MessageTierManager(self.harness.context_policy)
        
        # 1. Check Triggers
        should_compact = (
            tier_manager.should_compact(self.harness.state) or 
            tier_manager.should_compact_predictive(self.harness.state, soft_limit)
        )
        
        if not should_compact:
            if len(self.harness.state.recent_messages) <= tier_manager.hot_window:
                return

        # 2. Structured Tier Compaction
        msg = "Context almost exhausted; initiating compression..."
        self.harness._runlog("compaction_trigger", msg)
        if event_handler:
             asyncio.create_task(event_handler(UIEvent(event_type=UIEventType.ALERT, content=msg)))

        if self.harness.summarizer_client:
            try:
                brief = await tier_manager.compact_to_warm(
                    state=self.harness.state,
                    summarizer=self.harness.summarizer,
                    client=self.harness.summarizer_client,
                    artifact_store=self.harness.artifact_store
                )
                if brief:
                    self.harness._runlog("compaction_complete", "Compressed messages into warm brief", brief_id=brief.brief_id)
                    tier_manager.promote_to_cold(self.harness.state, artifact_store=self.harness.artifact_store)
                    return # Successfully compacted
            except Exception as e:
                self.harness._runlog("compaction_error", "Structured tier compaction failed", error=str(e))

        # 3. Ratio-based Safety Fallback (Legacy/Emergency Truncation)
        if len(self.harness.state.recent_messages) > self.harness.context_policy.recent_message_limit:
            self.harness._runlog("compaction_emergency", "Emergency truncation triggered (message limit exceeded)")
            self.harness.state.recent_messages = self.harness.state.recent_messages[-self.harness.context_policy.recent_message_limit:]
        
        probe = self.harness.prompt_assembler.build_messages(
            state=self.harness.state,
            system_prompt=system_prompt,
            retrieved_summaries=self.harness.retriever.retrieve_summaries(state=self.harness.state, query=query),
            retrieved_artifacts=self.harness.retriever.retrieve_artifacts(state=self.harness.state, query=query),
            recent_message_limit=self.harness.context_policy.recent_message_limit,
            include_structured_sections=True,
        )
        effective_soft_limit = soft_limit or self.harness.context_policy.max_prompt_tokens or 4096
        threshold = int(effective_soft_limit * self.harness.context_policy.summarize_at_ratio)
        
        if probe.estimated_prompt_tokens <= threshold:
            return 
        
        self.harness._runlog(
            "compaction_trigger",
            "context compaction triggered (budget pressure)",
            estimated_prompt_tokens=probe.estimated_prompt_tokens,
            summarize_threshold=threshold,
            recent_messages=len(self.harness.state.recent_messages),
        )
        keep_recent = max(4, min(10, self.harness.context_policy.recent_message_limit))
        
        if self.harness.summarizer_client:
            self.harness._runlog("compaction_start", "AI-based summarization pass started")
            await self.harness._emit(
                event_handler,
                UIEvent(
                    UIEventType.SYSTEM,
                    "Long context detected; summarization pass activated",
                    data={"status_activity": "summarizing..."},
                ),
            )
            try:
                summary = await self.harness.summarizer.compact_recent_messages_async(
                    state=self.harness.state,
                    client=self.harness.summarizer_client,
                    keep_recent=keep_recent,
                    artifact_store=self.harness.artifact_store,
                )
            except Exception as e:
                self.harness._runlog(
                    "compaction_error",
                    "AI summarization failed, falling back to heuristic compaction",
                    error=str(e),
                )
                summary = self.harness.summarizer.compact_recent_messages(
                    state=self.harness.state,
                    keep_recent=keep_recent,
                    artifact_store=self.harness.artifact_store,
                )
        else:
            summary = self.harness.summarizer.compact_recent_messages(
                state=self.harness.state,
                keep_recent=keep_recent,
                artifact_store=self.harness.artifact_store,
            )

        if summary:
            self.harness._runlog(
                "summary_created",
                "compacted recent context",
                summary_id=summary.summary_id,
                artifact_ids=summary.artifact_ids,
                files_touched=summary.files_touched,
            )
