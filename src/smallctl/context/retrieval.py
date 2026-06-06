from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..memory_namespace import (
    NamespaceRouting,
    infer_memory_namespace,
    namespace_bucket,
    namespace_preferences_for_task_mode,
    normalize_memory_namespace,
)
from ..normalization import coerce_datetime as _coerce_datetime, tokenize as _tokens
from ..redaction import redact_sensitive_text
from ..guards import is_over_twenty_b_model_name, is_seven_b_or_under_model_name
from ..state import (
    ArtifactRecord,
    ArtifactSnippet,
    EpisodicSummary,
    ExperienceMemory,
    LoopState,
    normalize_intent_label,
)
from .artifact_visibility import is_prompt_visible_artifact, is_retrieval_visible_artifact, is_superseded_artifact
from .artifact_read_coverage import fully_read_artifact_ids
from .policy import ContextPolicy, estimate_text_tokens
from .retrieval_artifact_helpers import (
    artifact_body_excerpt,
    artifact_category,
    artifact_contains_interactive_prompt,
    artifact_dedupe_key,
    artifact_failure_text,
    artifact_has_resolved_successor,
    artifact_host,
    artifact_path,
    artifact_success,
    artifact_text,
    artifact_tool_name,
    file_like_paths,
    handoff_recent_research_artifact_ids,
    is_causal_remote_failure_artifact,
    is_remote_repair_state,
    latest_causal_remote_failure_artifact_id,
    query_requests_specific_detail,
    should_pin_recent_research_artifacts,
)
from .retrieval_query import (
    build_refined_retrieval_query as _build_refined_retrieval_query,
    build_retrieval_query as _build_retrieval_query,
)
from .retrieval_constants import (
    CHAT_SUPPRESSED_TOOL_NAMES,
    REMOTE_FILE_TOOLS,
)
from .retrieval_scoring import score_artifact
from .retrieval_state_helpers import (
    durably_stale_ids,
    effective_current_goal,
    is_durably_stale_experience,
    is_generic_retrieval_tag,
    is_generic_terminal_memory,
    is_model_terminal_claim,
    normalized_goal_text,
    path_match,
    prompt_visible_memory_tags,
    query_requests_live_remote_correction,
    state_entity_tags,
    state_environment_tags,
    state_target_paths,
    state_touched_symbols,
)

@dataclass
class RetrievalBundle:
    query: str
    summaries: list[EpisodicSummary]
    artifacts: list[ArtifactSnippet]
    experiences: list[ExperienceMemory]
    initial_query: str = ""
    refined_query: str = ""
    refined: bool = False
    refinement_reason: str = ""
    candidate_counts: dict[str, int] = field(default_factory=dict)
    best_scores: dict[str, float] = field(default_factory=dict)
    score_gaps: dict[str, float] = field(default_factory=dict)
    lane_routes: dict[str, list[str]] = field(default_factory=dict)
    ranked_candidates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    miss_reasons: dict[str, list[str]] = field(default_factory=dict)


class LexicalRetriever:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()

    _file_like_paths = staticmethod(file_like_paths)
    _artifact_text = staticmethod(artifact_text)
    _artifact_body_excerpt = staticmethod(artifact_body_excerpt)
    _artifact_category = staticmethod(artifact_category)
    _artifact_dedupe_key = staticmethod(artifact_dedupe_key)
    _artifact_success = staticmethod(artifact_success)
    _artifact_host = staticmethod(artifact_host)
    _artifact_path = staticmethod(artifact_path)
    _artifact_has_resolved_successor = staticmethod(artifact_has_resolved_successor)
    _query_requests_specific_detail = staticmethod(query_requests_specific_detail)
    _handoff_recent_research_artifact_ids = staticmethod(handoff_recent_research_artifact_ids)
    _should_pin_recent_research_artifacts = staticmethod(should_pin_recent_research_artifacts)
    _artifact_tool_name = staticmethod(artifact_tool_name)
    _artifact_failure_text = staticmethod(artifact_failure_text)
    _artifact_contains_interactive_prompt = staticmethod(artifact_contains_interactive_prompt)
    _is_remote_repair_state = staticmethod(is_remote_repair_state)
    _is_causal_remote_failure_artifact = staticmethod(is_causal_remote_failure_artifact)
    _latest_causal_remote_failure_artifact_id = staticmethod(latest_causal_remote_failure_artifact_id)
    _normalized_goal_text = staticmethod(normalized_goal_text)
    _effective_current_goal = staticmethod(effective_current_goal)
    _state_environment_tags = staticmethod(state_environment_tags)
    _is_generic_retrieval_tag = staticmethod(is_generic_retrieval_tag)
    _prompt_visible_memory_tags = staticmethod(prompt_visible_memory_tags)
    _is_generic_terminal_memory = staticmethod(is_generic_terminal_memory)
    _query_requests_live_remote_correction = staticmethod(query_requests_live_remote_correction)
    _is_model_terminal_claim = staticmethod(is_model_terminal_claim)
    _state_entity_tags = staticmethod(state_entity_tags)
    _state_target_paths = staticmethod(state_target_paths)
    _state_touched_symbols = staticmethod(state_touched_symbols)
    _path_match = staticmethod(path_match)

    @staticmethod
    def _state_model_name(state: LoopState) -> str:
        scratchpad = getattr(state, "scratchpad", {})
        if isinstance(scratchpad, dict):
            return str(scratchpad.get("_model_name") or "").strip()
        return ""

    @classmethod
    def _uses_light_artifact_policy(cls, state: LoopState) -> bool:
        return is_over_twenty_b_model_name(cls._state_model_name(state))

    @classmethod
    def _artifact_signal_threshold(cls, state: LoopState) -> float:
        return 5.0 if cls._uses_light_artifact_policy(state) else 2.5

    def retrieve_bundle(
        self,
        *,
        state: LoopState,
        query: str,
        cold_store: Any = None,
        artifact_token_budget: int | None = None,
        summary_token_budget: int | None = None,
        experience_limit: int = 5,
        include_experiences: bool = True,
    ) -> RetrievalBundle:
        base_query = (query or "").strip() or build_retrieval_query(state)
        first_pass = self._retrieve_pass(
            state=state,
            query=base_query,
            cold_store=cold_store,
            artifact_token_budget=artifact_token_budget,
            summary_token_budget=summary_token_budget,
            experience_limit=experience_limit,
            include_experiences=include_experiences,
            update_state=False,
        )
        needs_refinement, reason = self._needs_refinement(state=state, bundle=first_pass)
        if not needs_refinement:
            state.retrieval_cache = [snippet.artifact_id for snippet in first_pass.artifacts]
            state.retrieved_experience_ids = [memory.memory_id for memory in first_pass.experiences]
            return first_pass

        refined_query = self._build_refined_query(state=state, bundle=first_pass)
        if not refined_query or refined_query == base_query:
            state.retrieval_cache = [snippet.artifact_id for snippet in first_pass.artifacts]
            state.retrieved_experience_ids = [memory.memory_id for memory in first_pass.experiences]
            return RetrievalBundle(
                query=base_query,
                initial_query=base_query,
                summaries=first_pass.summaries,
                artifacts=first_pass.artifacts,
                experiences=first_pass.experiences,
                refined=False,
                refinement_reason=reason,
                candidate_counts=first_pass.candidate_counts,
                best_scores=first_pass.best_scores,
                score_gaps=first_pass.score_gaps,
                lane_routes=first_pass.lane_routes,
                ranked_candidates=first_pass.ranked_candidates,
                miss_reasons={**first_pass.miss_reasons, "refinement": [f"refinement_triggered:{reason}"]},
            )

        second_pass = self._retrieve_pass(
            state=state,
            query=refined_query,
            cold_store=cold_store,
            artifact_token_budget=artifact_token_budget,
            summary_token_budget=summary_token_budget,
            experience_limit=experience_limit,
            include_experiences=include_experiences,
            update_state=False,
        )
        if self._bundle_quality(second_pass) >= self._bundle_quality(first_pass):
            second_pass.initial_query = base_query
            second_pass.refined_query = refined_query
            second_pass.refined = True
            second_pass.refinement_reason = reason
            second_pass.miss_reasons.setdefault("refinement", []).append(f"refinement_triggered:{reason}")
            state.retrieval_cache = [snippet.artifact_id for snippet in second_pass.artifacts]
            state.retrieved_experience_ids = [memory.memory_id for memory in second_pass.experiences]
            return second_pass

        first_pass.refinement_reason = reason
        first_pass.miss_reasons.setdefault("refinement", []).append(f"refinement_triggered:{reason}")
        state.retrieval_cache = [snippet.artifact_id for snippet in first_pass.artifacts]
        state.retrieved_experience_ids = [memory.memory_id for memory in first_pass.experiences]
        return first_pass

    def retrieve_artifacts(
        self,
        *,
        state: LoopState,
        query: str,
        token_budget: int | None = None,
    ) -> list[ArtifactSnippet]:
        ranked = self._rank_artifacts(state=state, query=query)
        snippets = self._select_artifact_snippets(
            ranked,
            state=state,
            query=query,
            token_budget=token_budget,
        )
        state.retrieval_cache = [snippet.artifact_id for snippet in snippets]
        return snippets

    def retrieve_summaries(
        self,
        *,
        state: LoopState,
        query: str,
        token_budget: int | None = None,
    ) -> list[EpisodicSummary]:
        ranked = self._rank_summaries(state=state, query=query)
        return self._select_summaries(ranked, token_budget=token_budget)

    def retrieve_experiences(
        self,
        *,
        state: LoopState,
        cold_store: Any = None,
        limit: int = 5,
    ) -> list[ExperienceMemory]:
        ranked = self._rank_experiences(state=state, cold_store=cold_store)
        distinct = self._select_distinct_experiences([memory for _, memory in ranked], limit=limit)
        state.retrieved_experience_ids = [memory.memory_id for memory in distinct]
        return distinct

    def _retrieve_pass(
        self,
        *,
        state: LoopState,
        query: str,
        cold_store: Any = None,
        artifact_token_budget: int | None = None,
        summary_token_budget: int | None = None,
        experience_limit: int = 5,
        include_experiences: bool = True,
        update_state: bool = True,
    ) -> RetrievalBundle:
        ranked_artifacts = self._rank_artifacts(state=state, query=query)
        ranked_summaries = self._rank_summaries(state=state, query=query)
        ranked_experiences = (
            self._rank_experiences(state=state, cold_store=cold_store, query_override=query)
            if include_experiences
            else []
        )

        artifacts = self._select_artifact_snippets(
            ranked_artifacts,
            state=state,
            query=query,
            token_budget=artifact_token_budget,
        )
        summaries = self._select_summaries(ranked_summaries, token_budget=summary_token_budget)
        experiences = (
            self._select_distinct_experiences(
                [memory for _, memory in ranked_experiences],
                limit=experience_limit,
            )
            if include_experiences
            else []
        )

        if update_state:
            state.retrieval_cache = [snippet.artifact_id for snippet in artifacts]
            state.retrieved_experience_ids = [memory.memory_id for memory in experiences]
        miss_reasons = self._retrieval_miss_reasons(
            state=state,
            ranked_artifacts=ranked_artifacts,
            ranked_summaries=ranked_summaries,
            ranked_experiences=ranked_experiences,
            artifacts=artifacts,
            summaries=summaries,
            experiences=experiences,
            include_experiences=include_experiences,
        )
        return RetrievalBundle(
            query=query,
            initial_query=query,
            summaries=summaries,
            artifacts=artifacts,
            experiences=experiences,
            candidate_counts={
                "artifacts": len(ranked_artifacts),
                "summaries": len(ranked_summaries),
                "experiences": len(ranked_experiences),
            },
            best_scores={
                "artifacts": ranked_artifacts[0][0] if ranked_artifacts else 0.0,
                "summaries": ranked_summaries[0][0] if ranked_summaries else 0.0,
                "experiences": ranked_experiences[0][0] if ranked_experiences else 0.0,
            },
            score_gaps={
                "artifacts": self._score_gap(ranked_artifacts),
                "summaries": self._score_gap(ranked_summaries),
                "experiences": self._score_gap(ranked_experiences),
            },
            lane_routes={
                "evidence_packet": [summary.summary_id for summary in summaries if summary.summary_id],
                "artifact_packet": [snippet.artifact_id for snippet in artifacts if snippet.artifact_id],
                "experience_packet": [memory.memory_id for memory in experiences if memory.memory_id],
            },
            ranked_candidates={
                "artifacts": [self._artifact_candidate_preview(score, artifact) for score, artifact in ranked_artifacts[:5]],
                "summaries": [self._summary_candidate_preview(score, summary) for score, summary in ranked_summaries[:5]],
                "experiences": [
                    self._experience_candidate_preview(score, memory) for score, memory in ranked_experiences[:5]
                ],
            },
            miss_reasons=miss_reasons,
        )

    def _retrieval_miss_reasons(
        self,
        *,
        state: LoopState,
        ranked_artifacts: list[tuple[float, ArtifactRecord]],
        ranked_summaries: list[tuple[float, EpisodicSummary]],
        ranked_experiences: list[tuple[float, ExperienceMemory]],
        artifacts: list[ArtifactSnippet],
        summaries: list[EpisodicSummary],
        experiences: list[ExperienceMemory],
        include_experiences: bool,
    ) -> dict[str, list[str]]:
        reasons: dict[str, list[str]] = {}
        if not ranked_artifacts and getattr(state, "artifacts", None):
            reasons.setdefault("artifacts", []).append("no_candidates_after_filtering")
        if ranked_artifacts and not artifacts:
            top_score, top_artifact = ranked_artifacts[0]
            if self._artifact_category(top_artifact) != "verifier" and top_score < self._artifact_signal_threshold(state):
                reasons.setdefault("artifacts", []).append("below_artifact_signal_threshold")
        if not ranked_summaries and getattr(state, "episodic_summaries", None):
            reasons.setdefault("summaries", []).append("no_candidates_after_filtering")
        if not include_experiences:
            reasons.setdefault("experiences", []).append("experiences_disabled")
        elif not ranked_experiences and getattr(state, "warm_experiences", None):
            reasons.setdefault("experiences", []).append("no_candidates_after_filtering")
        return reasons

    def _artifact_candidate_preview(self, score: float, artifact: ArtifactRecord) -> dict[str, Any]:
        return {
            "artifact_id": artifact.artifact_id,
            "score": score,
            "category": self._artifact_category(artifact),
            "source": artifact.source,
            "tool_name": artifact.tool_name,
        }

    @staticmethod
    def _summary_candidate_preview(score: float, summary: EpisodicSummary) -> dict[str, Any]:
        return {
            "summary_id": summary.summary_id,
            "score": score,
            "files_touched": list(summary.files_touched),
        }

    @staticmethod
    def _experience_candidate_preview(score: float, memory: ExperienceMemory) -> dict[str, Any]:
        return {
            "memory_id": memory.memory_id,
            "score": score,
            "intent": memory.intent,
            "tool_name": memory.tool_name,
            "outcome": memory.outcome,
        }

    def _rank_artifacts(self, *, state: LoopState, query: str) -> list[tuple[float, ArtifactRecord]]:
        query_tokens = _tokens(query)
        recent_artifact_ids = {
            message.metadata.get("artifact_id")
            for message in state.recent_messages
            if isinstance(message.metadata, dict) and message.metadata.get("artifact_id")
        }
        recently_retrieved_ids = set(state.retrieval_cache)
        detail_requested = self._query_requests_specific_detail(query)
        fully_read_artifact_ids = self._fully_read_artifact_ids(state)
        stale_artifact_ids = self._durably_stale_ids(state, key="_artifact_staleness")
        suppressed_raw = state.scratchpad.get("suppressed_truncated_artifact_ids", [])
        if isinstance(suppressed_raw, (list, tuple, set)):
            suppressed_truncated_ids = {
                str(artifact_id).strip()
                for artifact_id in suppressed_raw
                if str(artifact_id).strip()
            }
        else:
            suppressed_truncated_ids = set()

        scored: list[tuple[float, ArtifactRecord]] = []
        latest_causal_remote_artifact_id = self._latest_causal_remote_failure_artifact_id(state)
        for index, artifact_id in enumerate(state.artifacts.keys()):
            artifact = state.artifacts[artifact_id]
            force_include_remote_repair_artifact = (
                self._is_remote_repair_state(state)
                and artifact_id == latest_causal_remote_artifact_id
                and self._is_causal_remote_failure_artifact(artifact)
            )
            if (
                (
                    artifact_id in recent_artifact_ids
                    or artifact_id in recently_retrieved_ids
                    or (artifact_id in fully_read_artifact_ids and not detail_requested)
                    or artifact_id in suppressed_truncated_ids
                    or artifact_id in stale_artifact_ids
                )
                and not force_include_remote_repair_artifact
            ):
                continue
            if is_superseded_artifact(artifact):
                continue
            if not is_retrieval_visible_artifact(artifact):
                continue
            score = self._score_artifact(
                artifact=artifact,
                query=query,
                query_tokens=query_tokens,
                recency=index,
                state=state,
            )
            if force_include_remote_repair_artifact:
                score = max(score, 32.0 + index * 0.05)
            if score > 0:
                scored.append((score, artifact))
        scored.sort(key=lambda item: item[0], reverse=True)
        deduped: list[tuple[float, ArtifactRecord]] = []
        seen_keys: set[str] = set()
        for score, artifact in scored:
            dedupe_key = self._artifact_dedupe_key(artifact)
            if dedupe_key and dedupe_key in seen_keys:
                continue
            if dedupe_key:
                seen_keys.add(dedupe_key)
            deduped.append((score, artifact))
        return deduped

    @staticmethod
    def _fully_read_artifact_ids(state: LoopState) -> set[str]:
        return fully_read_artifact_ids(state)

    def _rank_summaries(
        self,
        *,
        state: LoopState,
        query: str,
    ) -> list[tuple[float, EpisodicSummary]]:
        query_tokens = _tokens(query)
        stale_summary_ids = self._durably_stale_ids(state, key="_summary_staleness")
        scored: list[tuple[float, EpisodicSummary]] = []
        for index, summary in enumerate(state.episodic_summaries):
            if summary.summary_id and summary.summary_id in stale_summary_ids:
                continue
            score = self._score_summary(
                summary=summary,
                query_tokens=query_tokens,
                recency=index,
                state=state,
            )
            if score > 0:
                scored.append((score, summary))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _rank_experiences(
        self,
        *,
        state: LoopState,
        cold_store: Any = None,
        query_override: str | None = None,
    ) -> list[tuple[float, ExperienceMemory]]:
        all_pool = list(state.warm_experiences)
        if cold_store:
            all_pool.extend(cold_store.list())

        superseded_ids = set()
        for memory in all_pool:
            if memory.supersedes:
                superseded_ids.update(memory.supersedes)

        all_memories = [memory for memory in all_pool if memory.memory_id not in superseded_ids]
        available_namespaces = {
            namespace
            for memory in all_memories
            if (namespace := self._resolved_memory_namespace(memory, state=state))
        }
        routing = namespace_preferences_for_task_mode(
            str(getattr(state, "task_mode", "") or ""),
            available_namespaces=available_namespaces,
        )
        scored: list[tuple[float, ExperienceMemory]] = []
        for memory in all_memories:
            score = self._score_experience(
                memory,
                state,
                query_override=query_override,
                routing=routing,
            )
            if score > 0:
                scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _select_artifact_snippets(
        self,
        ranked: list[tuple[float, ArtifactRecord]],
        *,
        state: LoopState,
        query: str,
        token_budget: int | None = None,
    ) -> list[ArtifactSnippet]:
        snippets: list[ArtifactSnippet] = []
        remote_profile = self._uses_remote_artifact_profile(state=state, query=query)
        multi_file_profile = self._uses_multi_file_artifact_profile(state=state, query=query)
        snippet_limit = self.policy.max_artifact_snippets
        primary_file_limit = 1
        mutation_result_limit = 1
        verifier_limit = 1
        other_limit = 1
        if remote_profile:
            snippet_limit = max(snippet_limit, self.policy.remote_task_artifact_snippet_limit)
            primary_file_limit = max(primary_file_limit, self.policy.remote_task_primary_file_limit)
            mutation_result_limit = 2
        if multi_file_profile:
            snippet_limit = max(snippet_limit, self.policy.multi_file_artifact_snippet_limit)
            primary_file_limit = max(primary_file_limit, self.policy.multi_file_primary_file_limit)
        budget = token_budget or (self.policy.artifact_snippet_token_limit * snippet_limit)
        used_tokens = 0
        detail_requested = self._query_requests_specific_detail(query)
        light_artifact_policy = self._uses_light_artifact_policy(state)
        verifier_passed = False
        verdict = state.current_verifier_verdict()
        if isinstance(verdict, dict):
            verifier_passed = str(verdict.get("verdict") or "").strip().lower() == "pass"

        primary_file_count = 0
        verifier_count = 0
        mutation_result_count = 0
        other_count = 0
        semantic_limit = snippet_limit if detail_requested else min(2, snippet_limit)
        if remote_profile or multi_file_profile:
            semantic_limit = min(snippet_limit, max(semantic_limit, primary_file_limit + mutation_result_limit + verifier_limit))
        if light_artifact_policy and not detail_requested and not (remote_profile or multi_file_profile):
            semantic_limit = min(semantic_limit, 1)

        if ranked and light_artifact_policy and not detail_requested and not (remote_profile or multi_file_profile):
            top_score, top_artifact = ranked[0]
            top_is_verifier = self._artifact_category(top_artifact) == "verifier"
            if not top_is_verifier and top_score < self._artifact_signal_threshold(state):
                return []

        for score, artifact in ranked[: snippet_limit * 4]:
            category = self._artifact_category(artifact)
            if verifier_passed and not detail_requested and category == "primary_file":
                continue
            if not detail_requested:
                if category == "primary_file" and primary_file_count >= primary_file_limit:
                    continue
                if category == "verifier" and verifier_count >= verifier_limit:
                    continue
                if category == "mutation_result" and mutation_result_count >= mutation_result_limit:
                    continue
                if category == "other" and other_count >= other_limit:
                    continue
                if len(snippets) >= semantic_limit:
                    break
            text = self._artifact_text(artifact)
            text_tokens = estimate_text_tokens(text)
            if snippets and used_tokens + text_tokens > budget:
                continue
            used_tokens += text_tokens
            snippets.append(ArtifactSnippet(artifact_id=artifact.artifact_id, text=text, score=score))
            if category == "primary_file":
                primary_file_count += 1
            elif category == "verifier":
                verifier_count += 1
            elif category == "mutation_result":
                mutation_result_count += 1
            else:
                other_count += 1
            if len(snippets) >= snippet_limit and token_budget is None:
                break
        return snippets

    def _select_summaries(
        self,
        ranked: list[tuple[float, EpisodicSummary]],
        *,
        token_budget: int | None = None,
    ) -> list[EpisodicSummary]:
        if token_budget is None:
            return [summary for _, summary in ranked[: self.policy.max_summary_items]]

        winners: list[EpisodicSummary] = []
        used = 0
        for _, summary in ranked:
            estimated = 20 + len(summary.decisions) * 10 + len(summary.files_touched) * 5
            if used + estimated > token_budget and winners:
                break
            used += estimated
            winners.append(summary)
        return winners

    def _select_distinct_experiences(
        self,
        memories: list[ExperienceMemory],
        limit: int,
    ) -> list[ExperienceMemory]:
        seen_keys = set()
        distinct: list[ExperienceMemory] = []
        for memory in memories:
            key = (memory.intent, memory.tool_name, memory.outcome)
            if key in seen_keys:
                continue
            distinct.append(memory)
            seen_keys.add(key)
            if len(distinct) >= limit:
                break
        return distinct

    def _needs_refinement(self, *, state: LoopState, bundle: RetrievalBundle) -> tuple[bool, str]:
        reasons: list[str] = []
        art_score = bundle.best_scores.get("artifacts", 0.0)
        sum_score = bundle.best_scores.get("summaries", 0.0)
        exp_score = bundle.best_scores.get("experiences", 0.0)

        if bundle.candidate_counts.get("artifacts", 0) >= 2:
            art_gap = bundle.score_gaps.get("artifacts", 999.0)
            if art_gap <= 1.25:
                reasons.append("artifact routing is ambiguous")
        if bundle.candidate_counts.get("summaries", 0) >= 2:
            summary_gap = bundle.score_gaps.get("summaries", 999.0)
            if summary_gap <= 0.75:
                reasons.append("summary routing is ambiguous")
        if bundle.candidate_counts.get("experiences", 0) >= 2:
            experience_gap = bundle.score_gaps.get("experiences", 999.0)
            if experience_gap <= 1.0:
                # Short-circuit: in repair phase, ambiguous prior-outcome routing
                # is usually caused by a harness policy rejection, not a genuine
                # memory gap. Refinement just burns tokens without helping.
                if state.current_phase != "repair":
                    reasons.append("prior outcome routing is ambiguous")

        if not reasons and not (bundle.artifacts or bundle.summaries or bundle.experiences):
            if state.artifacts or state.episodic_summaries or state.warm_experiences:
                reasons.append("first pass found no strong match")

        if not reasons and art_score < self._artifact_signal_threshold(state) and (
            bundle.candidate_counts.get("artifacts", 0) > 1 or state.artifacts
        ):
            reasons.append("artifact signal is weak")

        if not reasons and sum_score < 1.5 and bundle.candidate_counts.get("summaries", 0) > 1:
            reasons.append("summary signal is weak")

        if not reasons and exp_score < 2.0 and bundle.candidate_counts.get("experiences", 0) > 1:
            reasons.append("experience signal is weak")

        return (bool(reasons), "; ".join(reasons[:2]))

    def _bundle_quality(self, bundle: RetrievalBundle) -> float:
        return (
            bundle.best_scores.get("artifacts", 0.0)
            + 0.5 * bundle.best_scores.get("summaries", 0.0)
            + 0.35 * bundle.best_scores.get("experiences", 0.0)
            + 0.05 * len(bundle.artifacts)
            + 0.03 * len(bundle.summaries)
            + 0.02 * len(bundle.experiences)
        )

    def _build_refined_query(self, *, state: LoopState, bundle: RetrievalBundle) -> str:
        return build_refined_retrieval_query(state, base_query=bundle.query, bundle=bundle)

    def _score_experience(
        self,
        m: ExperienceMemory,
        state: LoopState,
        *,
        query_override: str | None = None,
        routing: NamespaceRouting | None = None,
    ) -> float:
        task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
        if task_mode == "chat" and str(m.tool_name or "").strip().lower() in CHAT_SUPPRESSED_TOOL_NAMES:
            return 0.0
        if self._is_durably_stale_experience(state, m):
            return 0.0
        memory_namespace = self._resolved_memory_namespace(m, state=state)
        if routing is None:
            available_namespaces = {memory_namespace} if memory_namespace else set()
            routing = namespace_preferences_for_task_mode(task_mode, available_namespaces=available_namespaces)
        if namespace_bucket(memory_namespace, routing) == "blocked":
            return 0.0
        query_text = query_override or build_retrieval_query(state)
        query_tokens = _tokens(query_text)
        safe_notes = redact_sensitive_text(m.notes)
        item_tokens = _tokens(f"{normalize_intent_label(m.intent)} {m.tool_name} {safe_notes} {m.failure_mode}")
        active_intent = normalize_intent_label(state.active_intent)

        score = 0.0
        if namespace_bucket(memory_namespace, routing) == "preferred":
            score += routing.preferred_bonus
        elif namespace_bucket(memory_namespace, routing) == "allowed":
            score += routing.allowed_bonus
        else:
            score += routing.neutral_penalty
        if normalize_intent_label(m.intent) == normalize_intent_label(state.active_intent):
            score += 15.0
        secondary_intents = {
            normalize_intent_label(intent)
            for intent in (getattr(state, "secondary_intents", []) or [])
            if normalize_intent_label(intent)
        }
        if normalize_intent_label(m.intent) in secondary_intents:
            score += 6.0
        requested_tool = self._infer_requested_tool(state)
        if m.tool_name and requested_tool and m.tool_name == requested_tool:
            score += 10.0

        tag_overlap = len(set(m.intent_tags) & set(state.intent_tags))
        score += tag_overlap * 2.5

        environment_overlap = len(set(m.environment_tags) & self._state_environment_tags(state))
        score += environment_overlap * 1.5

        entity_overlap = len(set(m.entity_tags) & self._state_entity_tags(state))
        score += entity_overlap * 1.5

        overlap = len(query_tokens & item_tokens)
        score += overlap * 1.5

        if m.phase == state.current_phase:
            score += 3.0

        if m.failure_mode and m.failure_mode in query_text.lower():
            score += 2.0
        state_failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
        if state_failure_mode and str(m.failure_mode or "").strip().lower() == state_failure_mode:
            score += 4.0

        if m.pinned:
            score += 5.0

        if m.tier == "warm":
            score += 2.0

        if m.expires_at and _coerce_datetime(m.expires_at) is not None:
            expires_at = _coerce_datetime(m.expires_at)
            if expires_at is not None and expires_at <= datetime.now(timezone.utc):
                return 0.0

        now = datetime.now(timezone.utc)
        created_at = _coerce_datetime(m.created_at)
        if created_at is not None:
            age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
            score += max(0.0, 2.0 - min(2.0, age_days * 0.2))

        last_reinforced = _coerce_datetime(m.last_reinforced_at)
        if last_reinforced is not None:
            reinforcement_days = max(0.0, (now - last_reinforced).total_seconds() / 86400.0)
            score += max(0.0, 1.5 - min(1.5, reinforcement_days * 0.3))

        confidence = m.confidence if m.confidence is not None else 0.6
        score *= 0.6 + confidence

        if m.confidence < self.policy.memory_low_confidence_threshold and not m.pinned:
            score *= 0.75

        write_target = str(getattr(getattr(state, "write_session", None), "write_target_path", "") or "").strip()
        if write_target:
            write_target_tokens = _tokens(write_target)
            if write_target_tokens & item_tokens:
                score += 3.0
        touched_symbols = self._state_touched_symbols(state)
        if touched_symbols:
            symbol_overlap = len(touched_symbols & item_tokens)
            if symbol_overlap:
                score += min(2.5, symbol_overlap * 0.8)

        # Terminal summaries are useful for coarse history, but they should not
        # outrank actionable tool patterns during a live execution/resteer turn.
        if m.tool_name == "task_complete":
            score *= 0.7
            if self._is_model_terminal_claim(m):
                score *= 0.8
            if requested_tool in {"shell_exec", "ssh_exec"}:
                score *= 0.7
            if requested_tool is None and active_intent == "general_task":
                score *= 0.35
            if self._query_requests_live_remote_correction(query_text):
                score *= 0.45

        # De-prioritize identical-task memories for small models (≤7B) to avoid
        # confusing the model into thinking the current task is already done.
        if is_seven_b_or_under_model_name(self._state_model_name(state)):
            current_task = str(state.run_brief.original_task or "").strip().lower()
            memory_notes = str(m.notes or "").strip().lower()
            memory_intent = str(m.intent or "").strip().lower()
            if current_task and (current_task in memory_notes or current_task in memory_intent or memory_notes in current_task):
                score *= 0.3

        return score

    _durably_stale_ids = staticmethod(durably_stale_ids)
    _is_durably_stale_experience = staticmethod(is_durably_stale_experience)

    @staticmethod
    def _resolved_memory_namespace(memory: ExperienceMemory, *, state: LoopState) -> str:
        namespace = normalize_memory_namespace(getattr(memory, "namespace", ""))
        if namespace:
            return namespace
        namespace = infer_memory_namespace(
            task_mode=str(getattr(state, "task_mode", "") or ""),
            tool_name=memory.tool_name,
            intent=memory.intent,
            intent_tags=memory.intent_tags,
            environment_tags=memory.environment_tags,
            entity_tags=memory.entity_tags,
            notes=memory.notes,
            original_task=state.run_brief.original_task,
        )
        memory.namespace = namespace
        return namespace

    def _score_summary(
        self,
        summary: EpisodicSummary,
        *,
        query_tokens: set[str],
        recency: int,
        state: LoopState,
    ) -> float:
        haystack = " ".join(
            summary.decisions
            + summary.files_touched
            + summary.failed_approaches
            + summary.remaining_plan
            + summary.notes
            + summary.artifact_ids
        )
        haystack_tokens = _tokens(haystack)
        overlap = len(query_tokens & haystack_tokens)
        if overlap <= 0:
            return 0.0
        score = float(overlap)
        target_paths = self._state_target_paths(state)
        touched_paths = {
            Path(path).as_posix().lower()
            for path in summary.files_touched
            if str(path).strip()
        }
        if summary.files_touched:
            if touched_paths & target_paths:
                score += 3.0
            elif target_paths and any(
                self._path_match(touched, target)
                for touched in touched_paths
                for target in target_paths
            ):
                score += 2.2
        if summary.decisions:
            score += 0.5
        if summary.files_touched:
            score += 0.5
        if summary.failed_approaches:
            score += 0.25
            failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
            if failure_mode and any(failure_mode in str(item).lower() for item in summary.failed_approaches):
                score += 2.0
            elif failure_mode and any(failure_mode in str(item).lower() for item in summary.notes):
                score += 1.2
        if summary.artifact_ids:
            score += 0.5
        if summary.remaining_plan:
            score += 0.5
        active_intent = normalize_intent_label(getattr(state, "active_intent", "") or "")
        if active_intent and active_intent in _tokens(" ".join(summary.remaining_plan + summary.notes)):
            score += 1.0
        secondary_intents = {
            normalize_intent_label(intent)
            for intent in getattr(state, "secondary_intents", []) or []
            if normalize_intent_label(intent)
        }
        if secondary_intents and secondary_intents & _tokens(" ".join(summary.notes)):
            score += 0.8
        write_target = str(getattr(getattr(state, "write_session", None), "write_target_path", "") or "").strip()
        if write_target:
            write_target_path = Path(write_target).as_posix().lower()
            if touched_paths and any(self._path_match(path, write_target_path) for path in touched_paths):
                score += 2.6
            write_target_tokens = _tokens(write_target_path)
            if write_target_tokens and write_target_tokens & haystack_tokens:
                score += 0.9
        touched_symbols = self._state_touched_symbols(state)
        if touched_symbols:
            symbol_overlap = len(touched_symbols & haystack_tokens)
            if symbol_overlap:
                score += min(1.8, symbol_overlap * 0.6)
        entity_overlap = len(self._state_entity_tags(state) & haystack_tokens)
        if entity_overlap:
            score += min(1.5, entity_overlap * 0.4)
        score += max(0.0, 0.05 * recency)
        return score

    def _score_gap(self, ranked: list[tuple[float, Any]]) -> float:
        if len(ranked) < 2:
            return 999.0
        return ranked[0][0] - ranked[1][0]

    def _infer_requested_tool(self, state: LoopState) -> str | None:
        if state.working_memory.next_actions:
            # Try to find a tool name in the next actions list
            for action in state.working_memory.next_actions:
                if "(" in action:
                    return action.split("(")[0].strip()
        return None

    def _uses_remote_artifact_profile(self, *, state: LoopState, query: str) -> bool:
        requested_tool = self._infer_requested_tool(state)
        if requested_tool in REMOTE_FILE_TOOLS:
            return True
        task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
        if task_mode == "remote_execute":
            return True
        tags = {str(tag).strip().lower() for tag in getattr(state, "intent_tags", []) or []}
        if tags & REMOTE_FILE_TOOLS:
            return True
        text = " ".join(
            [
                str(query or ""),
                str(getattr(state.run_brief, "original_task", "") or ""),
                str(getattr(state.working_memory, "current_goal", "") or ""),
            ]
        ).lower()
        return any(token in text for token in ("ssh", "remote", "host", "server")) and bool(self._file_like_paths(text))

    def _uses_multi_file_artifact_profile(self, *, state: LoopState, query: str) -> bool:
        paths = set(self._file_like_paths(query))
        paths |= self._state_target_paths(state)
        return len(paths) > 1

    _score_artifact = staticmethod(score_artifact)

def build_retrieval_query(state: LoopState) -> str:
    return _build_retrieval_query(state, retriever_cls=LexicalRetriever)


def build_refined_retrieval_query(
    state: LoopState,
    *,
    base_query: str,
    bundle: RetrievalBundle,
) -> str:
    return _build_refined_retrieval_query(
        state,
        base_query=base_query,
        bundle=bundle,
        retriever_cls=LexicalRetriever,
    )
