from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from ..models.events import UIEvent, UIEventType
from ..context import MessageTierManager
from ..context.summarizer import CompactionAttemptResult

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.compaction")

FALLBACK_INITIAL_KEEP_RECENT_MIN = 4
FALLBACK_INITIAL_KEEP_RECENT_MAX = 10
FALLBACK_MIN_KEEP_RECENT = 1
FALLBACK_STOP_REASONS = {
    "below_threshold",
    "minimum_recent_window_reached",
    "no_compactable_messages",
    "summarizer_noop",
    "summarizer_error",
}

class CompactionService:
    def __init__(self, harness: Harness):
        self.harness = harness

    def _record_compaction_snapshot(
        self,
        *,
        estimated_prompt_tokens_before: int,
        estimated_prompt_tokens_after: int,
        threshold: int,
        recent_messages_before: int,
        recent_messages_after: int,
        keep_recent_initial: int,
        keep_recent_final: int,
        messages_compacted: int,
        compaction_attempt_count: int,
        compaction_stopped_reason: str,
    ) -> None:
        snapshot = self.harness.state.prompt_budget
        snapshot.compaction_estimated_prompt_tokens_before = int(estimated_prompt_tokens_before)
        snapshot.compaction_estimated_prompt_tokens_after = int(estimated_prompt_tokens_after)
        snapshot.compaction_threshold = int(threshold)
        snapshot.compaction_recent_messages_before = int(recent_messages_before)
        snapshot.compaction_recent_messages_after = int(recent_messages_after)
        snapshot.compaction_keep_recent_initial = int(keep_recent_initial)
        snapshot.compaction_keep_recent_final = int(keep_recent_final)
        snapshot.compaction_messages_compacted = int(messages_compacted)
        snapshot.compaction_attempt_count = int(compaction_attempt_count)
        snapshot.compaction_stopped_reason = str(compaction_stopped_reason or "")
        snapshot.pressure_level = str(compaction_stopped_reason or "")
        snapshot.message_count = int(recent_messages_after)

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
        recent_messages_before = len(self.harness.state.recent_messages)
        msg = "Context almost exhausted; initiating compression..."
        self.harness._runlog("compaction_trigger", msg)
        if event_handler:
             asyncio.create_task(event_handler(UIEvent(event_type=UIEventType.ALERT, content=msg)))

        if should_compact:
            try:
                brief = await tier_manager.compact_to_warm(
                    state=self.harness.state,
                    summarizer=self.harness.summarizer,
                    client=self.harness.summarizer_client,
                    artifact_store=self.harness.artifact_store
                )
                demotion = self.harness.state.scratchpad.get("_last_compaction_demotion")
                if isinstance(demotion, dict):
                    self.harness._runlog(
                        "compaction_level_demoted",
                        "context demoted to lower compaction level",
                        from_level=demotion.get("from_level", ""),
                        to_level=demotion.get("to_level", ""),
                        bundle_id=demotion.get("bundle_id", ""),
                        bundle_ids=demotion.get("bundle_ids", []),
                        brief_id=demotion.get("brief_id", ""),
                        messages_compacted=demotion.get("messages_compacted", 0),
                    )
                if brief or (isinstance(demotion, dict) and int(demotion.get("messages_compacted", 0) or 0) > 0):
                    after_count = len(self.harness.state.recent_messages)
                    self._record_compaction_snapshot(
                        estimated_prompt_tokens_before=0,
                        estimated_prompt_tokens_after=0,
                        threshold=0,
                        recent_messages_before=recent_messages_before,
                        recent_messages_after=after_count,
                        keep_recent_initial=tier_manager.hot_window,
                        keep_recent_final=tier_manager.hot_window,
                        messages_compacted=max(0, recent_messages_before - after_count),
                        compaction_attempt_count=1,
                        compaction_stopped_reason=(
                            "structured_tier_compacted" if brief else "structured_l0_l1_demoted"
                        ),
                    )
                    if brief:
                        self.harness._runlog("compaction_complete", "Compressed messages into warm brief", brief_id=brief.brief_id)
                    else:
                        self.harness._runlog(
                            "compaction_complete",
                            "Demoted messages into turn bundle",
                            demotion=demotion,
                        )
                    tier_manager.promote_to_cold(self.harness.state, artifact_store=self.harness.artifact_store)
                    return # Successfully compacted
            except Exception as e:
                self.harness._runlog("compaction_error", "Structured tier compaction failed", error=str(e))

        # 3. Ratio-based Safety Fallback (Legacy/Emergency Truncation)
        if len(self.harness.state.recent_messages) > self.harness.context_policy.recent_message_limit:
            self.harness._runlog("compaction_emergency", "Emergency truncation triggered (message limit exceeded)")
            self.harness.state.recent_messages = self.harness.state.recent_messages[-self.harness.context_policy.recent_message_limit:]

        def _build_probe() -> Any:
            return self.harness.prompt_assembler.build_messages(
                state=self.harness.state,
                system_prompt=system_prompt,
                retrieved_summaries=self.harness.retriever.retrieve_summaries(state=self.harness.state, query=query),
                retrieved_artifacts=self.harness.retriever.retrieve_artifacts(state=self.harness.state, query=query),
                recent_message_limit=self.harness.context_policy.recent_message_limit,
                include_structured_sections=True,
            )

        probe = _build_probe()
        effective_soft_limit = soft_limit or self.harness.context_policy.max_prompt_tokens or 4096
        threshold = int(effective_soft_limit * self.harness.context_policy.summarize_at_ratio)
        
        if probe.estimated_prompt_tokens <= threshold:
            self._record_compaction_snapshot(
                estimated_prompt_tokens_before=probe.estimated_prompt_tokens,
                estimated_prompt_tokens_after=probe.estimated_prompt_tokens,
                threshold=threshold,
                recent_messages_before=len(self.harness.state.recent_messages),
                recent_messages_after=len(self.harness.state.recent_messages),
                keep_recent_initial=max(
                    FALLBACK_INITIAL_KEEP_RECENT_MIN,
                    min(FALLBACK_INITIAL_KEEP_RECENT_MAX, self.harness.context_policy.recent_message_limit),
                ),
                keep_recent_final=max(
                    FALLBACK_INITIAL_KEEP_RECENT_MIN,
                    min(FALLBACK_INITIAL_KEEP_RECENT_MAX, self.harness.context_policy.recent_message_limit),
                ),
                messages_compacted=0,
                compaction_attempt_count=0,
                compaction_stopped_reason="below_threshold",
            )
            return 
        
        self.harness._runlog(
            "compaction_trigger",
            "context compaction triggered (budget pressure)",
            estimated_prompt_tokens=probe.estimated_prompt_tokens,
            summarize_threshold=threshold,
            recent_messages=len(self.harness.state.recent_messages),
        )
        initial_keep_recent = max(
            FALLBACK_INITIAL_KEEP_RECENT_MIN,
            min(FALLBACK_INITIAL_KEEP_RECENT_MAX, self.harness.context_policy.recent_message_limit),
        )
        min_keep_recent = FALLBACK_MIN_KEEP_RECENT
        keep_recent = initial_keep_recent
        recent_messages_before = len(self.harness.state.recent_messages)
        estimated_prompt_tokens_before = probe.estimated_prompt_tokens
        messages_compacted = 0
        compaction_attempt_count = 0
        compaction_stopped_reason = "below_threshold"

        async def _attempt_compaction(current_keep_recent: int) -> CompactionAttemptResult:
            if self.harness.summarizer_client:
                try:
                    return await self.harness.summarizer.compact_recent_messages_async_with_status(
                        state=self.harness.state,
                        client=self.harness.summarizer_client,
                        keep_recent=current_keep_recent,
                        artifact_store=self.harness.artifact_store,
                    )
                except Exception as exc:
                    self.harness._runlog(
                        "compaction_error",
                        "AI summarization failed, falling back to heuristic compaction",
                        error=str(exc),
                        keep_recent=current_keep_recent,
                    )
            try:
                return self.harness.summarizer.compact_recent_messages_with_status(
                    state=self.harness.state,
                    keep_recent=current_keep_recent,
                    artifact_store=self.harness.artifact_store,
                )
            except Exception as exc:
                self.harness._runlog(
                    "compaction_error",
                    "Heuristic summarization failed",
                    error=str(exc),
                    keep_recent=current_keep_recent,
                )
                return CompactionAttemptResult(noop_reason="summarizer_error")

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

        while True:
            probe = _build_probe()
            estimated_before = probe.estimated_prompt_tokens
            if estimated_before <= threshold:
                compaction_stopped_reason = "below_threshold"
                break

            current_recent_messages = len(self.harness.state.recent_messages)
            if current_recent_messages <= min_keep_recent:
                compaction_stopped_reason = "minimum_recent_window_reached"
                break

            if keep_recent >= current_recent_messages:
                keep_recent = max(min_keep_recent, current_recent_messages - 1)

            if keep_recent < min_keep_recent:
                compaction_stopped_reason = "minimum_recent_window_reached"
                break

            compaction_attempt_count += 1
            self.harness._runlog(
                "compaction_attempt",
                "legacy fallback compaction attempt",
                attempt=compaction_attempt_count,
                keep_recent=keep_recent,
                estimated_prompt_tokens=estimated_before,
                recent_messages=current_recent_messages,
                threshold=threshold,
            )

            before_count = current_recent_messages
            attempt = await _attempt_compaction(keep_recent)
            summary = attempt.summary
            if summary:
                after_count = len(self.harness.state.recent_messages)
                messages_compacted += max(0, before_count - after_count)
                self.harness._runlog(
                    "summary_created",
                    "compacted recent context",
                    summary_id=summary.summary_id,
                    artifact_ids=summary.artifact_ids,
                    files_touched=summary.files_touched,
                    keep_recent=keep_recent,
                    messages_compacted=max(0, before_count - after_count),
                )
                post_probe = _build_probe()
                if post_probe.estimated_prompt_tokens <= threshold:
                    compaction_stopped_reason = "below_threshold"
                    break
                if len(self.harness.state.recent_messages) <= min_keep_recent:
                    compaction_stopped_reason = "minimum_recent_window_reached"
                    break
                next_keep_recent = max(
                    min_keep_recent,
                    min(keep_recent - 1, len(self.harness.state.recent_messages) - 1),
                )
                if next_keep_recent >= keep_recent:
                    compaction_stopped_reason = "minimum_recent_window_reached"
                    break
                keep_recent = next_keep_recent
                continue

            noop_reason = attempt.noop_reason or "summarizer_noop"
            if noop_reason == "summarizer_error":
                compaction_stopped_reason = "summarizer_error"
                break

            if keep_recent > min_keep_recent:
                next_keep_recent = max(
                    min_keep_recent,
                    min(keep_recent - 1, current_recent_messages - 1),
                )
                if next_keep_recent < keep_recent:
                    keep_recent = next_keep_recent
                    continue

            compaction_stopped_reason = noop_reason
            break

        final_probe = _build_probe()
        self.harness._runlog(
            "compaction_fallback_complete",
            "legacy fallback compaction finished",
            estimated_prompt_tokens_before=estimated_prompt_tokens_before,
            estimated_prompt_tokens_after=final_probe.estimated_prompt_tokens,
            threshold=threshold,
            recent_messages_before=recent_messages_before,
            recent_messages_after=len(self.harness.state.recent_messages),
            keep_recent_initial=initial_keep_recent,
            keep_recent_final=keep_recent,
            keep_recent_floor=min_keep_recent,
            initial_keep_recent_min=FALLBACK_INITIAL_KEEP_RECENT_MIN,
            initial_keep_recent_max=FALLBACK_INITIAL_KEEP_RECENT_MAX,
            messages_compacted=messages_compacted,
            compaction_attempt_count=compaction_attempt_count,
            compaction_stopped_reason=compaction_stopped_reason,
            compaction_contract={
                "min_keep_recent": min_keep_recent,
                "repeated_passes_allowed": True,
                "allowed_stop_reasons": sorted(FALLBACK_STOP_REASONS),
            },
        )
        self._record_compaction_snapshot(
            estimated_prompt_tokens_before=estimated_prompt_tokens_before,
            estimated_prompt_tokens_after=final_probe.estimated_prompt_tokens,
            threshold=threshold,
            recent_messages_before=recent_messages_before,
            recent_messages_after=len(self.harness.state.recent_messages),
            keep_recent_initial=initial_keep_recent,
            keep_recent_final=keep_recent,
            messages_compacted=messages_compacted,
            compaction_attempt_count=compaction_attempt_count,
            compaction_stopped_reason=compaction_stopped_reason,
        )
