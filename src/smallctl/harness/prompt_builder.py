from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from ..models.events import UIEvent

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.prompt_builder")

_PROMPT_BUDGET_REMEDIATION = (
    "reduce the tools exposed for this turn, clear stale artifacts/summaries, "
    "lower the recent-message limit, or raise the max prompt token limit / backend context size"
)


def prompt_budget_overflow_error(
    *,
    estimated_tokens: int,
    limit: int,
    section_tokens: dict[str, int],
) -> RuntimeError:
    """Build a RuntimeError carrying actionable prompt-budget diagnostics."""
    top_contributors = [
        {"section": name, "tokens": tokens}
        for name, tokens in sorted(section_tokens.items(), key=lambda item: item[1], reverse=True)
        if tokens > 0
    ][:5]
    contributor_text = ", ".join(
        f"{item['section']}={item['tokens']}" for item in top_contributors
    ) or "none"
    message = (
        f"PROMPT BUDGET OVERFLOW: {estimated_tokens} tokens assembled, which exceeds the max prompt limit of {limit}. "
        f"Top contributors: {contributor_text}. "
        f"Remediation: {_PROMPT_BUDGET_REMEDIATION}."
    )
    error = RuntimeError(message)
    error.prompt_budget_details = {  # type: ignore[attr-defined]
        "type": "prompt_budget_failure",
        "estimated_prompt_tokens": estimated_tokens,
        "max_prompt_tokens": limit,
        "top_contributors": top_contributors,
        "remediation": _PROMPT_BUDGET_REMEDIATION,
    }
    return error


def is_prompt_budget_overflow(exc: BaseException) -> bool:
    return getattr(exc, "prompt_budget_details", None) is not None or str(exc).startswith(
        "PROMPT BUDGET OVERFLOW"
    )

class PromptBuilderService:
    def __init__(self, harness: Harness):
        self.harness = harness

    def _prompt_pressure_threshold(self) -> int:
        soft_limit = self.harness.context_policy.soft_prompt_token_limit
        if soft_limit is None:
            return 10000
        return max(256, int(soft_limit * 0.7))

    def _dynamic_recent_message_limit(self) -> int:
        base_limit = max(1, int(getattr(self.harness.context_policy, "recent_message_limit", 1) or 1))
        state = self.harness.state
        if self._llamacpp_small_repair_pressure():
            adjusted_limit = min(base_limit, 3)
            self.harness._runlog(
                "recent_message_limit_tuned",
                "reduced recent message window for llamacpp small-model repair pressure",
                task_mode=state.task_mode,
                active_phase=state.current_phase,
                base_limit=base_limit,
                adjusted_limit=adjusted_limit,
                reasons=["llamacpp_small_repair_pressure"],
                recent_messages=len(getattr(state, "recent_messages", []) or []),
            )
            return max(2, adjusted_limit)
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
        # Phase 3B: enforce floor of 8 for remote repair so failure history is preserved
        if str(getattr(state, "current_phase", "") or "").strip().lower() in {"execute", "repair", "verify"}:
            adjusted_limit = max(8, adjusted_limit)
        # Remote execute mode needs even more context to maintain coherent task state
        if str(getattr(state, "task_mode", "") or "").strip().lower() == "remote_execute":
            adjusted_limit = max(10, adjusted_limit)
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

    def _llamacpp_small_repair_pressure(self) -> bool:
        state = self.harness.state
        provider_profile = str(getattr(self.harness, "provider_profile", "") or "").strip().lower()
        if provider_profile != "llamacpp":
            return False
        scratchpad = getattr(state, "scratchpad", {}) or {}
        if not bool(scratchpad.get("_model_is_small")):
            return False
        if str(getattr(state, "current_phase", "") or "").strip().lower() not in {"repair", "verify", "author"}:
            return False
        if scratchpad.get("_stream_chunk_error_auto_resume_signature"):
            return True
        if scratchpad.get("_terminal_write_session_repair_signatures"):
            return True
        if scratchpad.get("_last_write_session_schema_failure"):
            return True
        return False

    def _smalltalk_experience_suppressed(self) -> bool:
        state = self.harness.state
        scratchpad = getattr(state, "scratchpad", {}) or {}
        labels = {
            str(getattr(state, "active_intent", "") or "").strip().lower(),
            str(scratchpad.get("_chat_runtime_intent") or "").strip().lower(),
        }
        return "smalltalk" in labels

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
        
        suppress_llamacpp_repair_experiences = self._llamacpp_small_repair_pressure()
        suppress_smalltalk_experiences = self._smalltalk_experience_suppressed()
        include_experiences = (
            not fresh_run_active
            and not suppress_llamacpp_repair_experiences
            and not suppress_smalltalk_experiences
        )
        retrieval_bundle = self.harness.retriever.retrieve_bundle(
            state=self.harness.state,
            query=query,
            cold_store=cold_store,
            include_experiences=include_experiences,
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
        experiences = [] if not include_experiences else retrieval_bundle.experiences
        
        self.harness._runlog(
            "retrieval_selected",
            "retrieval candidates selected",
            query=retrieval_bundle.query,
            initial_query=retrieval_bundle.initial_query,
            refined=retrieval_bundle.refined,
            refinement_reason=retrieval_bundle.refinement_reason,
            best_scores=retrieval_bundle.best_scores,
            candidate_counts=retrieval_bundle.candidate_counts,
            ranked_candidates=retrieval_bundle.ranked_candidates,
            miss_reasons=retrieval_bundle.miss_reasons,
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
            repeat_counts=retrieval_bundle.repeat_counts,
            applied_decays=retrieval_bundle.applied_decays,
            experience_suppression_reason=(
                "fresh_run"
                if fresh_run_active
                else "llamacpp_small_repair_pressure"
                if suppress_llamacpp_repair_experiences
                else "smalltalk"
                if suppress_smalltalk_experiences
                else ""
            ),
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
            ranked_candidates=retrieval_bundle.ranked_candidates,
            miss_reasons=retrieval_bundle.miss_reasons,
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
        
        previous_sections = dict(getattr(getattr(self.harness.state, "prompt_budget", None), "sections", {}) or {})
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
        current_sections = dict(assembly.section_tokens)
        deltas = {
            key: {
                "before": previous_sections.get(key, 0),
                "after": value,
                "delta": value - previous_sections.get(key, 0),
            }
            for key, value in current_sections.items()
            if value != previous_sections.get(key, 0)
        }
        if previous_sections and deltas:
            self.harness._runlog(
                "prompt_section_delta",
                "prompt section token changes since previous prompt",
                deltas=deltas,
                total_before=sum(previous_sections.values()),
                total_after=sum(current_sections.values()),
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

        # The assembler recounts and enforces the tighter of the policy max and
        # the caller budget (soft_limit, passed as token_budget) on the final
        # emitted message list; an estimate above the effective budget here
        # means mandatory content alone could not fit, so surface the typed
        # budget failure instead of emitting an over-budget prompt.
        limit = self.harness.context_policy.max_prompt_tokens
        effective_limit = limit if limit and limit > 0 else None
        if soft_limit and soft_limit > 0 and (effective_limit is None or soft_limit < effective_limit):
            effective_limit = soft_limit
        if effective_limit and assembly.estimated_prompt_tokens > effective_limit:
            raise prompt_budget_overflow_error(
                estimated_tokens=assembly.estimated_prompt_tokens,
                limit=int(effective_limit),
                section_tokens=current_sections,
            )

        return assembly.messages

    async def ensure_context_limit(self) -> None:
        if not hasattr(self.harness.client, "fetch_model_context_limit"):
            return
        if not bool(getattr(self.harness.client, "runtime_context_probe", True)):
            return

        probe_attempted = bool(getattr(self.harness, "_runtime_context_probe_attempted", False))
        provider_profile = str(getattr(self.harness, "provider_profile", "") or "").strip().lower()
        if (
            provider_profile != "llamacpp"
            and self.harness.server_context_limit is not None
            and probe_attempted
        ):
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
