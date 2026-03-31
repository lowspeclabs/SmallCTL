from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from ..state import LoopState, ContextBrief
    from .summarizer import ContextSummarizer
from .policy import estimate_text_tokens

class MessageTierManager:
    def __init__(self, policy: Any = None) -> None:
        self.policy = policy
        # Fallback defaults if no policy exists/passed
        self.warm_limit = getattr(policy, "warm_brief_limit", 3)
        self.compaction_interval = getattr(policy, "compaction_step_interval", 8)

    @property
    def hot_window(self) -> int:
        return getattr(self.policy, "hot_message_limit", 8)

    def should_compact(self, state: LoopState) -> bool:
        """Determines if the current message transcript should be folded into a warm brief."""
        if not state.recent_messages:
            return False
            
        # We always keep at least the hot window verbatim.
        if len(state.recent_messages) <= self.hot_window:
            return False
            
        # Condition 1 — temporal (existing)
        last_compaction_step = state.scratchpad.get("_last_tier_compaction_step", 0)
        steps_since = state.step_count - last_compaction_step
        if steps_since >= self.compaction_interval:
            return True

        # Condition 2 — density / volatility (Phase II)
        # If the current transcript is already burning through the token budget,
        # compact now instead of waiting for the interval.
        transcript_limit = getattr(self.policy, "transcript_token_limit", 1400)
        transcript_tokens = sum(
            estimate_text_tokens(m.content or "") for m in state.recent_messages
        )
        if transcript_tokens > transcript_limit * 0.85:
            return True
            
        return False

    async def compact_to_warm(
        self, 
        *,
        state: LoopState, 
        summarizer: ContextSummarizer,
        client: Any,
        artifact_store: Any = None
    ) -> ContextBrief | None:
        """Compresses messages beyond the hot window into a new ContextBrief."""
        # The messages to be compressed (everything older than the last N messages)
        to_compact = state.recent_messages[:-self.hot_window]
        if not to_compact:
            return None
            
        start_step = state.scratchpad.get("_last_tier_compaction_step", 0) + 1
        end_step = state.step_count
        
        # Produce a new structured brief. We assume summarizer has been updated to support this.
        # If it hasn't yet, we'll need to update it in the next phase.
        brief = await summarizer.compact_to_brief_async(
            state=state,
            client=client,
            messages=to_compact,
            step_range=(start_step, end_step),
            artifact_store=artifact_store
        )
        
        if brief:
            state.context_briefs.append(brief)
            # Remove the now-summarized messages from the active transcript.
            state.recent_messages = state.recent_messages[-self.hot_window:]
            # Mark the high-water mark for the next interval calculation.
            state.scratchpad["_last_tier_compaction_step"] = state.step_count
            
        return brief

    def promote_to_cold(self, state: LoopState, artifact_store: Any = None) -> None:
        """Rolls the oldest warm brief into the cold storage layer (working memory facts)
        and preserves overflow in the artifact store. (Phase II)"""
        limit = getattr(self.policy, "cold_fact_limit", 12)
        while len(state.context_brief_ids if hasattr(state, 'context_brief_ids') else state.context_briefs) > self.warm_limit:
            oldest = state.context_briefs.pop(0)
            source_id = oldest.brief_id or f"steps-{oldest.step_range[0]}-{oldest.step_range[1]}"
            new_facts = [f"Fact (from {source_id}): {i}" for i in oldest.key_discoveries]
            new_facts += [f"Blocker (from {source_id}): {i}" for i in oldest.blockers]
            state.working_memory.known_facts.extend(new_facts)
            
            # Persist overflow to artifact store so retriever can surface it later (Phase II)
            if len(state.working_memory.known_facts) > limit and artifact_store:
                overflow = state.working_memory.known_facts[:-limit]
                overflow_text = "\n".join(overflow)
                artifact_store.persist_thinking(
                    raw_thinking=overflow_text,
                    summary=f"Cold overflow from {source_id}",
                    source="tier_promote",
                )
            state.working_memory.known_facts = state.working_memory.known_facts[-limit:]

    def should_compact_predictive(self, state: LoopState, soft_limit: int) -> bool:
        """Compact early if the next turn is likely to overflow the window. (Phase IV)"""
        # Use actual token usage from the last streamed response if available
        last_completion = int(state.scratchpad.get("last_completion_tokens", 0))
        current_usage = int(state.scratchpad.get("context_used_tokens", 0))
        
        # Rolling average (store in scratchpad)
        history: list[int] = list(state.scratchpad.get("_completion_history", []))
        if last_completion:
            history.append(last_completion)
            history = history[-8:]
            state.scratchpad["_completion_history"] = history
            
        avg_completion = int(sum(history) / len(history)) if history else last_completion or 512
        
        # Heuristic: current context + 1.5x avg completion (to account for tool-call spikes)
        estimated_next = current_usage + int(avg_completion * 1.5)
        
        soft = soft_limit
        if soft is None:
            # If no limit is set, we can't reliably predict overflow
            return False
            
        threshold = getattr(self.policy, "summarize_at_ratio", 0.8) * soft
        
        return estimated_next > threshold
