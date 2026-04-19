from __future__ import annotations
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from ..state import EpisodicSummary
if TYPE_CHECKING:
    from ..state import LoopState, ContextBrief
    from .summarizer import ContextSummarizer
from .policy import estimate_text_tokens

class MessageTierManager:
    def __init__(self, policy: Any = None) -> None:
        self.policy = policy
        # Fallback defaults if no policy exists/passed
        self.warm_limit = getattr(policy, "warm_brief_limit", 3)
        self.turn_bundle_limit = getattr(policy, "turn_bundle_limit", 6)
        self.compaction_interval = getattr(policy, "compaction_step_interval", 8)

    @property
    def hot_window(self) -> int:
        return getattr(self.policy, "hot_message_limit", 8)

    @staticmethod
    def _record_demotion(state: "LoopState", payload: dict[str, Any]) -> None:
        state.scratchpad["_last_compaction_demotion"] = payload
        queue = state.scratchpad.get("_compaction_demotion_events")
        if not isinstance(queue, list):
            queue = []
        queue.append(dict(payload))
        state.scratchpad["_compaction_demotion_events"] = queue[-24:]

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
        """Demotes L0 transcript context to L1 turn bundles, then L1 to L2 briefs as needed."""
        # The messages to be compressed (everything older than the last N messages)
        to_compact = state.recent_messages[:-self.hot_window]
        if not to_compact:
            return None
            
        start_step = state.scratchpad.get("_last_tier_compaction_step", 0) + 1
        end_step = state.step_count
        turn_bundle = summarizer.compact_to_turn_bundle(
            state=state,
            messages=to_compact,
            step_range=(start_step, end_step),
        )
        if turn_bundle is None:
            return None
        state.turn_bundles.append(turn_bundle)
        state.recent_messages = state.recent_messages[-self.hot_window:]
        state.scratchpad["_last_tier_compaction_step"] = state.step_count
        self._record_demotion(state, {
            "from_level": "L0",
            "to_level": "L1",
            "bundle_id": turn_bundle.bundle_id,
            "messages_compacted": len(to_compact),
        })

        if len(state.turn_bundles) <= self.turn_bundle_limit:
            return None

        promote_count = len(state.turn_bundles) - self.turn_bundle_limit
        bundles_to_promote = list(state.turn_bundles[:promote_count])
        if not bundles_to_promote:
            return None
        promote_step_start = bundles_to_promote[0].step_range[0]
        promote_step_end = bundles_to_promote[-1].step_range[1]
        brief = summarizer.compact_turn_bundles_to_brief(
            state=state,
            bundles=bundles_to_promote,
            step_range=(promote_step_start, promote_step_end),
            artifact_store=artifact_store,
        )
        if brief is not None:
            state.context_briefs.append(brief)
            state.turn_bundles = state.turn_bundles[promote_count:]
            self._record_demotion(state, {
                "from_level": "L1",
                "to_level": "L2",
                "bundle_ids": [bundle.bundle_id for bundle in bundles_to_promote if bundle.bundle_id],
                "brief_id": brief.brief_id,
                "messages_compacted": len(to_compact),
            })
        return brief

    def promote_to_cold(self, state: LoopState, artifact_store: Any = None) -> None:
        """Rolls the oldest warm brief into the cold storage layer (working memory facts)
        and preserves overflow in the artifact store. (Phase II)"""
        limit = getattr(self.policy, "cold_fact_limit", 12)
        while len(state.context_brief_ids if hasattr(state, 'context_brief_ids') else state.context_briefs) > self.warm_limit:
            oldest = state.context_briefs.pop(0)
            summary_id = f"S{len(state.episodic_summaries) + 1:04d}"
            summary_notes = []
            if oldest.new_facts:
                summary_notes.extend(oldest.new_facts[:3])
            if oldest.state_changes:
                summary_notes.extend(oldest.state_changes[:3])
            if not summary_notes:
                summary_notes.extend(oldest.key_discoveries[:3])
            summary: EpisodicSummary = EpisodicSummary(
                summary_id=summary_id,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                decisions=(oldest.decision_deltas[:3] or oldest.key_discoveries[:3]),
                files_touched=list(oldest.files_touched[:8]),
                failed_approaches=list(oldest.blockers[:4]),
                remaining_plan=([oldest.next_action_hint] if oldest.next_action_hint else []),
                artifact_ids=list(oldest.artifact_ids[:10]),
                notes=summary_notes[:6],
                full_summary_artifact_id=oldest.full_artifact_id,
            )
            state.episodic_summaries.append(summary)
            self._record_demotion(state, {
                "from_level": "L2",
                "to_level": "L3",
                "brief_id": oldest.brief_id,
                "summary_id": summary.summary_id,
                "full_artifact_id": summary.full_summary_artifact_id or oldest.full_artifact_id or "",
                "messages_compacted": 0,
            })
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
