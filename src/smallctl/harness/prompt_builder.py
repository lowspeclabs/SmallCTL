from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from ..models.events import UIEvent

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.prompt_builder")

class PromptBuilderService:
    def __init__(self, harness: Harness):
        self.harness = harness

    def _prompt_pressure_threshold(self) -> int:
        soft_limit = self.harness.context_policy.soft_prompt_token_limit
        if soft_limit is None:
            return 12000
        return max(256, int(soft_limit * 0.8))

    def _dynamic_recent_message_limit(self) -> int:
        base_limit = max(1, int(getattr(self.harness.context_policy, "recent_message_limit", 1) or 1))
        state = self.harness.state
        if str(getattr(state, "task_mode", "") or "").strip().lower() != "remote_execute":
            return base_limit

        pressure_reasons: list[str] = []
        recent_count = len(getattr(state, "recent_messages", []) or [])
        if recent_count >= max(8, base_limit):
            pressure_reasons.append("recent_messages_growth")
        prompt_budget = getattr(state, "prompt_budget", None)
        estimated_prompt_tokens = int(getattr(prompt_budget, "estimated_prompt_tokens", 0) or 0)
        prompt_pressure_threshold = self._prompt_pressure_threshold()
        high_prompt_pressure = estimated_prompt_tokens >= prompt_pressure_threshold
        if high_prompt_pressure:
            pressure_reasons.append("high_prompt_budget")
        observation_staleness = state.scratchpad.get("_observation_staleness")
        stale_observation_count = 0
        if isinstance(observation_staleness, dict):
            stale_observation_count = sum(
                1 for marker in observation_staleness.values() if isinstance(marker, dict) and marker.get("stale")
            )
        if stale_observation_count >= 6:
            pressure_reasons.append("observation_invalidation_churn")
        if (
            pressure_reasons
            and str(getattr(state, "current_phase", "") or "").strip().lower() in {"repair", "verify"}
        ):
            pressure_reasons.append("remote_repair_cycle")
        if not pressure_reasons:
            return base_limit

        adjusted_limit = min(base_limit, 6)
        if high_prompt_pressure or "observation_invalidation_churn" in pressure_reasons:
            adjusted_limit = min(adjusted_limit, 4)
        adjusted_limit = max(2, adjusted_limit)
        self.harness._runlog(
            "recent_message_limit_tuned",
            "reduced recent message window for remote prompt pressure",
            task_mode=state.task_mode,
            active_phase=state.current_phase,
            base_limit=base_limit,
            adjusted_limit=adjusted_limit,
            reasons=pressure_reasons,
            recent_messages=recent_count,
            estimated_prompt_tokens=estimated_prompt_tokens,
            prompt_pressure_threshold=prompt_pressure_threshold,
            stale_observation_count=stale_observation_count,
        )
        return adjusted_limit

    async def build_messages(
        self,
        system_prompt: str,
        *,
        event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None,
    ) -> list[dict[str, Any]]:
        self.harness.state.scratchpad["_last_system_prompt"] = system_prompt
        recent_limit = self._dynamic_recent_message_limit()
        self.harness.memory.update_working_memory(recent_limit)
        query = self.harness._select_retrieval_query()
        self.harness._runlog("retrieval_query", "selected retrieval query", query=query)
        
        # Use compaction service directly
        await self.harness.compaction.maybe_compact_context(
            query=query,
            system_prompt=system_prompt,
            event_handler=event_handler,
        )
        
        fresh_run_active = self.harness.fresh_run and self.harness._fresh_run_turns_remaining > 0
        cold_store = None if fresh_run_active else self.harness.cold_memory_store
        
        retrieval_bundle = self.harness.retriever.retrieve_bundle(
            state=self.harness.state,
            query=query,
            cold_store=cold_store,
            include_experiences=not fresh_run_active,
        )
        def _stale_count(key: str) -> int:
            payload = self.harness.state.scratchpad.get(key)
            if not isinstance(payload, dict):
                return 0
            return sum(
                1
                for marker in payload.values()
                if isinstance(marker, dict) and bool(marker.get("stale", False))
            )
        stale_lane_counts = {
            "turn_bundles": _stale_count("_turn_bundle_staleness"),
            "context_briefs": _stale_count("_context_brief_staleness"),
            "episodic_summaries": _stale_count("_summary_staleness"),
            "artifact_snippets": _stale_count("_artifact_staleness"),
            "experience_memories": _stale_count("_experience_staleness"),
            "normalized_observations": _stale_count("_observation_staleness"),
        }
        
        summaries = retrieval_bundle.summaries
        artifacts = retrieval_bundle.artifacts
        experiences = [] if fresh_run_active else retrieval_bundle.experiences
        
        self.harness._runlog(
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
            lane_routes=retrieval_bundle.lane_routes,
            stale_lane_counts=stale_lane_counts,
        )
        self.harness._runlog(
            "retrieval_ranked_with_intent",
            "retrieval ranked using phase/intent/subsystem routing",
            active_phase=self.harness.state.current_phase,
            active_intent=self.harness.state.active_intent,
            secondary_intents=list(self.harness.state.secondary_intents),
            failure_mode=self.harness.state.last_failure_class,
            write_session_target=(
                self.harness.state.write_session.write_target_path
                if self.harness.state.write_session is not None
                else ""
            ),
            lane_routes=retrieval_bundle.lane_routes,
            score_gaps=retrieval_bundle.score_gaps,
            best_scores=retrieval_bundle.best_scores,
            stale_lane_counts=stale_lane_counts,
            selected_artifact_ids=[artifact.artifact_id for artifact in artifacts],
            selected_summary_ids=[summary.summary_id for summary in summaries],
            selected_experience_ids=[memory.memory_id for memory in experiences],
        )
        
        include_structured_sections = bool(
            summaries
            or artifacts
            or experiences
            or self.harness.state.episodic_summaries
            or self.harness.state.turn_bundles
            or self.harness.state.context_briefs
            or self.harness.state.reasoning_graph.evidence_records
        )
        soft_limit = self.harness.context_policy.soft_prompt_token_limit
        
        assembly = self.harness.prompt_assembler.build_messages(
            state=self.harness.state,
            system_prompt=system_prompt,
            retrieved_summaries=summaries,
            retrieved_artifacts=artifacts,
            retrieved_experiences=experiences,
            recent_message_limit=recent_limit,
            include_structured_sections=include_structured_sections,
            token_budget=soft_limit,
        )
        
        if fresh_run_active and self.harness._fresh_run_turns_remaining > 0:
            self.harness._fresh_run_turns_remaining -= 1
        
        self.harness.state.scratchpad["context_used_tokens"] = assembly.estimated_prompt_tokens

        self.harness._runlog(
            "prompt_budget",
            "prompt assembly estimate (bidding complete)",
            estimated_prompt_tokens=assembly.estimated_prompt_tokens,
            sections=assembly.section_tokens,
            message_count=len(assembly.messages),
            retrieval_cache=self.harness.state.retrieval_cache,
        )
        frame = getattr(assembly, "frame", None)
        if frame is not None:
            included_lane_counts = frame.included_lane_counts()
            dropped_lane_counts = frame.dropped_lane_counts()
            self.harness._runlog(
                "prompt_state_frame_compiled",
                "compiled deterministic prompt-state frame",
                active_phase=frame.spine.current_phase,
                active_intent=frame.spine.active_intent,
                coding_profile_enabled=bool(getattr(self.harness.context_policy, "coding_profile_enabled", True)),
                coding_anchor_count=len(frame.spine.coding_anchor_lines),
                included_lane_counts=included_lane_counts,
                dropped_lane_counts=dropped_lane_counts,
                selected_artifact_ids=[item.artifact_id for item in frame.artifact_packet.snippets],
                selected_experience_ids=[item.memory_id for item in frame.experience_packet.memories],
                selected_turn_bundle_ids=[item.bundle_id for item in frame.evidence_packet.turn_bundles],
                selected_brief_ids=[item.brief_id for item in frame.evidence_packet.context_briefs],
                selected_summary_ids=[item.summary_id for item in frame.evidence_packet.summaries],
            )
            if frame.spine.fama_capsule_lines:
                self.harness._runlog(
                    "fama_capsule_rendered",
                    "FAMA capsule rendered into working memory",
                    line_count=len(frame.spine.fama_capsule_lines),
                    token_estimate=assembly.section_tokens.get("fama_capsules", 0),
                )
            for lane, count in included_lane_counts.items():
                self.harness._runlog(
                    "context_lane_selected",
                    "context lane selected for prompt frame",
                    lane=lane,
                    count=count,
                    active_phase=frame.spine.current_phase,
                    active_intent=frame.spine.active_intent,
                )
            for drop in frame.drop_log:
                self.harness._runlog(
                    "context_lane_dropped",
                    "context lane dropped from prompt frame",
                    lane=drop.lane,
                    reason=drop.reason,
                    dropped_count=drop.dropped_count,
                    dropped_ids=list(drop.dropped_ids),
                    active_phase=frame.spine.current_phase,
                    active_intent=frame.spine.active_intent,
                )

        limit = self.harness.context_policy.max_prompt_tokens
        if limit and assembly.estimated_prompt_tokens > limit:
             raise RuntimeError(f"PROMPT BUDGET OVERFLOW: {assembly.estimated_prompt_tokens} tokens assembled, which exceeds the max prompt limit of {limit}.")

        return assembly.messages

    async def ensure_context_limit(self) -> None:
        if not hasattr(self.harness.client, "fetch_model_context_limit"):
            return
        if not bool(getattr(self.harness.client, "runtime_context_probe", True)):
            return

        probe_attempted = bool(getattr(self.harness, "_runtime_context_probe_attempted", False))
        if self.harness.server_context_limit is not None and probe_attempted:
            return

        try:
            context_limit = await self.harness.client.fetch_model_context_limit()
        except Exception:
            if self.harness.server_context_limit is not None:
                self.harness._runtime_context_probe_attempted = True
            return

        self.harness._runtime_context_probe_attempted = True
        if context_limit is not None:
            self.harness._apply_server_context_limit(context_limit, source="runtime_probe")
