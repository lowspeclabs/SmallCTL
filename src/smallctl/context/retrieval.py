from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..experience_tags import PHASE_TAG_PREFIX, is_generic_experience_tag
from ..memory_namespace import (
    NamespaceRouting,
    infer_memory_namespace,
    namespace_bucket,
    namespace_preferences_for_task_mode,
    normalize_memory_namespace,
)
from ..normalization import coerce_datetime as _coerce_datetime, tokenize as _tokens
from ..redaction import redact_sensitive_text
from ..retrieval_safety import build_retrieval_safe_text, format_failure_tag
from ..state import (
    ArtifactRecord,
    ArtifactSnippet,
    EpisodicSummary,
    ExperienceMemory,
    LoopState,
    MemoryEntry,
    memory_entry_is_stale,
    normalize_intent_label,
)
from .artifact_visibility import is_prompt_visible_artifact, is_superseded_artifact
from .policy import ContextPolicy, estimate_text_tokens

_CHAT_SUPPRESSED_TOOL_NAMES = {
    "shell_exec",
    "ssh_exec",
    "file_write",
    "file_append",
    "file_patch",
    "file_delete",
    "process_kill",
    "http_post",
    "file_download",
}
_CHAT_SUPPRESSED_MEMORY_TAGS = {
    "shell_exec",
    "ssh_exec",
    "scripts",
    "bash",
    "terminal",
    "command",
    "command_line",
}

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


class LexicalRetriever:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()

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
            state.retrieval_cache = [snippet.artifact_id for snippet in second_pass.artifacts]
            state.retrieved_experience_ids = [memory.memory_id for memory in second_pass.experiences]
            return second_pass

        first_pass.refinement_reason = reason
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
        )

    def _rank_artifacts(self, *, state: LoopState, query: str) -> list[tuple[float, ArtifactRecord]]:
        query_tokens = _tokens(query)
        recent_artifact_ids = {
            message.metadata.get("artifact_id")
            for message in state.recent_messages
            if isinstance(message.metadata, dict) and message.metadata.get("artifact_id")
        }
        recently_retrieved_ids = set(state.retrieval_cache)
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
        for index, artifact_id in enumerate(state.artifacts.keys()):
            if (
                artifact_id in recent_artifact_ids
                or artifact_id in recently_retrieved_ids
                or artifact_id in suppressed_truncated_ids
            ):
                continue
            artifact = state.artifacts[artifact_id]
            if is_superseded_artifact(artifact):
                continue
            if not _is_retrieval_visible_artifact(artifact):
                continue
            score = self._score_artifact(
                artifact=artifact,
                query_tokens=query_tokens,
                recency=index,
                state=state,
            )
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

    def _rank_summaries(
        self,
        *,
        state: LoopState,
        query: str,
    ) -> list[tuple[float, EpisodicSummary]]:
        query_tokens = _tokens(query)
        scored: list[tuple[float, EpisodicSummary]] = []
        for index, summary in enumerate(state.episodic_summaries):
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
        budget = token_budget or (self.policy.artifact_snippet_token_limit * self.policy.max_artifact_snippets)
        used_tokens = 0
        detail_requested = self._query_requests_specific_detail(query)
        verifier_passed = False
        verdict = state.current_verifier_verdict()
        if isinstance(verdict, dict):
            verifier_passed = str(verdict.get("verdict") or "").strip().lower() == "pass"

        primary_file_count = 0
        verifier_count = 0
        other_count = 0
        semantic_limit = self.policy.max_artifact_snippets if detail_requested else min(2, self.policy.max_artifact_snippets)

        for score, artifact in ranked[: self.policy.max_artifact_snippets * 4]:
            category = self._artifact_category(artifact)
            if verifier_passed and not detail_requested and category == "primary_file":
                continue
            if not detail_requested:
                if category == "primary_file" and primary_file_count >= 1:
                    continue
                if category == "verifier" and verifier_count >= 1:
                    continue
                if category == "other" and other_count >= 1:
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
            else:
                other_count += 1
            if len(snippets) >= self.policy.max_artifact_snippets and token_budget is None:
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
                reasons.append("prior outcome routing is ambiguous")

        if not reasons and not (bundle.artifacts or bundle.summaries or bundle.experiences):
            if state.artifacts or state.episodic_summaries or state.warm_experiences:
                reasons.append("first pass found no strong match")

        if not reasons and art_score < 2.5 and (bundle.candidate_counts.get("artifacts", 0) > 1 or state.artifacts):
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
        if task_mode == "chat" and str(m.tool_name or "").strip().lower() in _CHAT_SUPPRESSED_TOOL_NAMES:
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

        # Terminal summaries are useful for coarse history, but they should not
        # outrank actionable tool patterns during a live execution/resteer turn.
        if m.tool_name == "task_complete":
            score *= 0.7
            if requested_tool in {"shell_exec", "ssh_exec"}:
                score *= 0.7
            if requested_tool is None and active_intent == "general_task":
                score *= 0.35

        return score

    @staticmethod
    def _is_durably_stale_experience(state: LoopState, memory: ExperienceMemory) -> bool:
        memory_id = str(getattr(memory, "memory_id", "") or "").strip()
        if not memory_id:
            return False
        payload = state.scratchpad.get("_experience_staleness")
        if not isinstance(payload, dict):
            return False
        marker = payload.get(memory_id)
        if not isinstance(marker, dict):
            return False
        return bool(marker.get("stale", False))

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
        entity_overlap = len(self._state_entity_tags(state) & haystack_tokens)
        if entity_overlap:
            score += min(1.5, entity_overlap * 0.4)
        score += max(0.0, 0.05 * recency)
        return score

    def _score_gap(self, ranked: list[tuple[float, Any]]) -> float:
        if len(ranked) < 2:
            return 999.0
        return ranked[0][0] - ranked[1][0]

    @staticmethod
    def _normalized_goal_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).lower()

    @classmethod
    def _effective_current_goal(cls, state: LoopState) -> str:
        current_goal = str(getattr(state.working_memory, "current_goal", "") or "").strip()
        if not current_goal:
            return ""
        previous_task = str(state.scratchpad.get("_task_boundary_previous_task") or "").strip()
        if previous_task and cls._normalized_goal_text(previous_task) == cls._normalized_goal_text(current_goal):
            return ""
        return current_goal

    @staticmethod
    def _state_environment_tags(state: LoopState) -> set[str]:
        phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        if not phase:
            return set()
        return {f"{PHASE_TAG_PREFIX}{phase}"}

    @staticmethod
    def _is_generic_retrieval_tag(tag: str) -> bool:
        return is_generic_experience_tag(tag)

    @classmethod
    def _prompt_visible_memory_tags(cls, state: LoopState, memory: ExperienceMemory) -> list[str]:
        task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
        active_intent = normalize_intent_label(getattr(state, "active_intent", "") or "")
        state_tags = {
            str(tag).strip().lower()
            for tag in (getattr(state, "intent_tags", []) or [])
            if str(tag).strip()
        }
        visible: list[str] = []
        for tag in getattr(memory, "intent_tags", []) or []:
            normalized = str(tag or "").strip()
            lowered = normalized.lower()
            if not normalized or cls._is_generic_retrieval_tag(lowered):
                continue
            if task_mode == "chat" and lowered in _CHAT_SUPPRESSED_MEMORY_TAGS:
                continue
            if lowered.startswith(PHASE_TAG_PREFIX):
                continue
            if lowered == active_intent or lowered in state_tags or lowered.endswith("_exec"):
                visible.append(normalized)
                continue
            if lowered.startswith(("task_", "tool_")):
                visible.append(normalized)
        return visible

    @classmethod
    def _is_generic_terminal_memory(cls, state: LoopState, memory: ExperienceMemory) -> bool:
        if str(memory.tool_name or "").strip().lower() != "task_complete":
            return False
        if normalize_intent_label(memory.intent) != "general_task":
            return False
        if normalize_intent_label(getattr(state, "active_intent", "") or "") != "general_task":
            return False
        if cls._prompt_visible_memory_tags(state, memory):
            return False
        current_goal = cls._effective_current_goal(state)
        if current_goal and (_tokens(current_goal) & _tokens(memory.notes or "")):
            return False
        return True

    @classmethod
    def _state_entity_tags(cls, state: LoopState) -> set[str]:
        return _tokens(
            " ".join(
                filter(
                    None,
                    [
                        state.run_brief.original_task,
                        cls._effective_current_goal(state),
                        " ".join(state.working_memory.open_questions),
                    ],
                )
            )
        )

    @staticmethod
    def _state_target_paths(state: LoopState) -> set[str]:
        paths: set[str] = set()
        for value in list(getattr(state, "files_changed_this_cycle", []) or []):
            text = str(value or "").strip()
            if text:
                paths.add(Path(text).as_posix().lower())
        task_targets = state.scratchpad.get("_task_target_paths")
        if isinstance(task_targets, list):
            for value in task_targets:
                text = str(value or "").strip()
                if text:
                    paths.add(Path(text).as_posix().lower())
        write_session = getattr(state, "write_session", None)
        if write_session is not None:
            for key in ("write_target_path", "write_staging_path"):
                text = str(getattr(write_session, key, "") or "").strip()
                if text:
                    paths.add(Path(text).as_posix().lower())
        return paths

    @staticmethod
    def _path_match(left: str, right: str) -> bool:
        lhs = str(left or "").strip().lower()
        rhs = str(right or "").strip().lower()
        if not lhs or not rhs:
            return False
        if lhs == rhs:
            return True
        return lhs.endswith(rhs) or rhs.endswith(lhs)

    def _infer_requested_tool(self, state: LoopState) -> str | None:
        if state.working_memory.next_actions:
            # Try to find a tool name in the next actions list
            for action in state.working_memory.next_actions:
                if "(" in action:
                    return action.split("(")[0].strip()
        return None

    @staticmethod
    def _artifact_text(artifact: ArtifactRecord) -> str:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        verifier_verdict = str(metadata.get("verifier_verdict") or "").strip()
        if verifier_verdict:
            target = str(metadata.get("verifier_target") or artifact.source or artifact.tool_name or "").strip()
            exit_code = metadata.get("verifier_exit_code")
            stdout = str(metadata.get("verifier_stdout") or "").strip()
            stderr = str(metadata.get("verifier_stderr") or "").strip()
            transcript = stdout or stderr
            if len(transcript) > 320:
                transcript = f"{transcript[:320].rstrip()}..."
            details = [f"Verifier {verifier_verdict}: {artifact.summary or target or artifact.tool_name}"]
            if target:
                details.append(f"Target: {target}")
            if exit_code not in ("", None):
                details.append(f"Exit code: {exit_code}")
            if transcript:
                details.append(f"Key output: {transcript}")
            return "\n".join(details)[:900]

        base = f"{artifact.source or artifact.tool_name} | {artifact.summary}"
        preview = artifact.preview_text or artifact.inline_content or ""
        if metadata.get("complete_file") and preview:
            preview = f"Full file already captured; excerpt below is preview only.\n{preview[:500].rstrip()}"
        combined = f"{base}\n{preview}".strip()
        return combined[:900]

    @staticmethod
    def _artifact_category(artifact: ArtifactRecord) -> str:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        if str(metadata.get("verifier_verdict") or "").strip():
            return "verifier"
        if artifact.kind == "file_read" and bool(metadata.get("complete_file")):
            return "primary_file"
        return "other"

    @staticmethod
    def _artifact_dedupe_key(artifact: ArtifactRecord) -> str:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        category = LexicalRetriever._artifact_category(artifact)
        if category == "verifier":
            family = str(metadata.get("attempt_family") or "").strip()
            if family:
                return f"verifier:{family}"
            target = str(
                metadata.get("verifier_target")
                or metadata.get("command")
                or artifact.source
                or artifact.summary
                or ""
            ).strip()
            return f"verifier:{target.lower()}"
        path = str(metadata.get("path") or artifact.source or "").strip()
        if path:
            normalized = Path(path).as_posix().lower()
            return f"{category}:{normalized}"
        return f"{category}:{artifact.artifact_id}"

    @staticmethod
    def _query_requests_specific_detail(query: str) -> bool:
        lowered = str(query or "").lower()
        if not lowered:
            return False
        detail_markers = (
            "specific line",
            "specific lines",
            "line-level",
            "line level",
            "line numbers",
            "line number",
            "start_line",
            "end_line",
            "artifact_read",
            "quote the line",
            "show the line",
            "show lines",
            "inspect lines",
            "page forward",
            "narrow excerpt",
            "exact excerpt",
            "specific excerpt",
        )
        return any(marker in lowered for marker in detail_markers)

    @staticmethod
    def _score_artifact(
        *,
        artifact: ArtifactRecord,
        query_tokens: set[str],
        recency: int,
        state: LoopState,
    ) -> float:
        expanded_query_tokens = set(query_tokens)
        for token in list(query_tokens):
            stripped = token.lstrip("./")
            if len(stripped) > 1:
                expanded_query_tokens.add(stripped)
            for part in re.split(r"[\\/]+", stripped):
                if len(part) > 1:
                    expanded_query_tokens.add(part)
        source_name = Path(artifact.source).name.lower() if artifact.source else ""
        source_tokens = _tokens(artifact.source)
        summary_tokens = _tokens(artifact.summary)
        keyword_tokens = {token.lower() for token in artifact.keywords}
        path_tokens = {token.lower() for token in artifact.path_tags}
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        metadata_tokens = _tokens(" ".join(
            str(metadata.get(key, ""))
            for key in ("intent", "path", "url", "command", "content_type", "source_type", "confidence")
        ))
        overlap = len(expanded_query_tokens & (source_tokens | summary_tokens | keyword_tokens | path_tokens | metadata_tokens))
        filename_bonus = 5.0 if source_name and source_name in expanded_query_tokens else 0.0
        path_bonus = 3.0 if artifact.source and any(token in artifact.source.lower() for token in expanded_query_tokens) else 0.0
        tool_bonus = 2.0 if artifact.tool_name.lower() in expanded_query_tokens else 0.0
        verifier_bonus = 2.5 if str(metadata.get("verifier_verdict") or "").strip() else 0.0
        confidence_bonus = 0.0
        confidence = metadata.get("confidence")
        try:
            confidence_bonus = max(0.0, min(1.5, float(confidence) * 1.5)) if confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence_bonus = 0.0
        phase_bonus = 0.0
        metadata_phase = str(metadata.get("phase") or "").strip().lower()
        current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        if metadata_phase and current_phase and metadata_phase == current_phase:
            phase_bonus = 2.5

        intent_bonus = 0.0
        active_intent = normalize_intent_label(getattr(state, "active_intent", "") or "")
        metadata_intent = normalize_intent_label(metadata.get("intent"))
        if active_intent and metadata_intent and metadata_intent == active_intent:
            intent_bonus += 2.5
        secondary_intents = {
            normalize_intent_label(intent)
            for intent in getattr(state, "secondary_intents", []) or []
            if normalize_intent_label(intent)
        }
        if metadata_intent and metadata_intent in secondary_intents:
            intent_bonus += 1.2

        target_path_bonus = 0.0
        target_paths = LexicalRetriever._state_target_paths(state)
        source_path = Path(artifact.source).as_posix().lower() if artifact.source else ""
        if source_path and target_paths and any(LexicalRetriever._path_match(source_path, target) for target in target_paths):
            target_path_bonus += 4.0
        write_target = str(getattr(getattr(state, "write_session", None), "write_target_path", "") or "").strip()
        if write_target and source_path and LexicalRetriever._path_match(source_path, Path(write_target).as_posix().lower()):
            target_path_bonus += 2.0

        entity_bonus = 0.0
        entity_overlap = len(LexicalRetriever._state_entity_tags(state) & (keyword_tokens | path_tokens | metadata_tokens))
        if entity_overlap:
            entity_bonus = min(2.0, entity_overlap * 0.5)

        failure_bonus = 0.0
        failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
        if failure_mode:
            failure_haystack = " ".join(
                [
                    str(artifact.summary or ""),
                    str(metadata.get("failure_mode") or ""),
                    str(metadata.get("error") or ""),
                ]
            ).lower()
            if failure_mode in failure_haystack:
                failure_bonus += 2.0

        relevance = (
            overlap
            + filename_bonus
            + path_bonus
            + tool_bonus
            + verifier_bonus
            + confidence_bonus
            + phase_bonus
            + intent_bonus
            + target_path_bonus
            + entity_bonus
            + failure_bonus
        )
        if relevance <= 0:
            return 0.0
        recency_bonus = recency * 0.05
        return relevance + recency_bonus


def _dedupe_nonempty_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _retrieval_failure_texts(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for text in values:
        normalized.append(format_failure_tag(text))
    return _dedupe_nonempty_texts(normalized)


def _is_recovery_nudge_message(message: Any) -> bool:
    metadata = getattr(message, "metadata", {})
    return isinstance(metadata, dict) and bool(metadata.get("is_recovery_nudge"))


def _retrieval_message_text(message: Any) -> str:
    retrieval_safe_text = str(getattr(message, "retrieval_safe_text", "") or "").strip()
    if retrieval_safe_text:
        return retrieval_safe_text
    return build_retrieval_safe_text(
        role=str(getattr(message, "role", "") or ""),
        content=getattr(message, "content", ""),
        name=getattr(message, "name", ""),
        metadata=getattr(message, "metadata", {}),
    )


def _is_execution_oriented_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "shell_exec",
            "ssh_exec",
            "run ",
            "execute",
            "exec ",
            "command",
            "script",
            "terminal",
            "pytest",
            "apt-get",
            "git ",
        )
    )


def build_retrieval_query(state: LoopState) -> str:
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    parts = [
        state.run_brief.original_task,
        state.run_brief.task_contract,
        state.run_brief.current_phase_objective,
    ]
    plan = state.active_plan or state.draft_plan
    if plan is not None:
        parts.append(f"Plan goal: {plan.goal}")
        parts.append(f"Plan status: {plan.status}")
        if plan.requested_output_path:
            parts.append(f"Plan export: {plan.requested_output_path}")
    if task_mode:
        parts.append(f"Task mode: {task_mode}")
    if state.active_intent:
        parts.append(f"Intent: {normalize_intent_label(state.active_intent)}")
    if state.intent_tags:
        parts.append(f"Tags: {' '.join(state.intent_tags)}")
    current_goal = LexicalRetriever._effective_current_goal(state)
    if current_goal:
        parts.append(f"Current goal: {current_goal}")
    parts.extend(state.working_memory.plan[-3:])
    parts.extend(state.working_memory.decisions[-3:])
    parts.extend(
        _visible_memory_texts(
            state.working_memory.known_facts,
            state.working_memory.known_fact_meta,
            current_step=state.step_count,
            current_phase=state.current_phase,
        )[-4:]
    )
    parts.extend(state.working_memory.open_questions[-2:])
    parts.extend(
        _retrieval_failure_texts(
            _visible_memory_texts(
                state.working_memory.failures,
                state.working_memory.failure_meta,
                current_step=state.step_count,
                current_phase=state.current_phase,
            )[-4:]
        )
    )
    parts.extend(
        _visible_memory_texts(
            state.working_memory.next_actions,
            state.working_memory.next_action_meta,
            current_step=state.step_count,
            current_phase=state.current_phase,
        )[-3:]
    )
    if task_mode == "chat":
        parts = [part for part in parts if not _is_execution_oriented_text(part)]
    for content in _dedupe_nonempty_texts([
        _retrieval_message_text(message)
        for message in state.recent_messages[-3:]
    ]):
        parts.append(content)
    return "\n".join(part for part in parts if part)


def build_refined_retrieval_query(
    state: LoopState,
    *,
    base_query: str,
    bundle: RetrievalBundle,
) -> str:
    parts = [base_query]
    if state.run_brief.task_contract:
        parts.append(f"Contract: {state.run_brief.task_contract}")
    current_goal = LexicalRetriever._effective_current_goal(state)
    if current_goal:
        parts.append(f"Current goal: {current_goal}")
    if bundle.artifacts:
        top_snippet = bundle.artifacts[0]
        top_artifact = state.artifacts.get(top_snippet.artifact_id)
        if top_artifact:
            parts.append(f"Top artifact: {top_artifact.artifact_id} | {top_artifact.source} | {top_artifact.summary}")
            if top_artifact.path_tags:
                parts.append("Artifact path tags: " + " ".join(top_artifact.path_tags))
            if top_artifact.tool_name:
                parts.append(f"Artifact tool: {top_artifact.tool_name}")
    if bundle.summaries:
        summary = bundle.summaries[0]
        if summary.files_touched:
            parts.append("Summary files: " + " ".join(summary.files_touched[:4]))
        if summary.remaining_plan:
            parts.append("Summary next steps: " + " ".join(summary.remaining_plan[:3]))
        if summary.notes:
            parts.append("Summary notes: " + " ".join(summary.notes[:2]))
    if bundle.experiences:
        memory = bundle.experiences[0]
        if not LexicalRetriever._is_generic_terminal_memory(state, memory):
            parts.append(
                f"Prior outcome: {normalize_intent_label(memory.intent)} / {memory.tool_name} / {memory.outcome}"
            )
            memory_namespace = LexicalRetriever._resolved_memory_namespace(memory, state=state)
            if (
                memory_namespace in {"ssh_remote", "local_shell", "planning", "debugging", "incidents"}
                and (
                    memory.tool_name in {"task_complete", "task_fail", "memory_update", "artifact_read", "file_read", "dir_list"}
                    or bundle.score_gaps.get("experiences", 999.0) <= 1.0
                )
            ):
                parts.append(f"Memory namespace: {memory_namespace}")
            if memory.failure_mode:
                parts.append(f"Failure mode: {memory.failure_mode}")
            visible_memory_tags = LexicalRetriever._prompt_visible_memory_tags(state, memory)
            if visible_memory_tags:
                parts.append("Memory tags: " + " ".join(visible_memory_tags[:4]))
    if state.working_memory.open_questions:
        parts.append("Open questions: " + " ".join(state.working_memory.open_questions[-2:]))
    retrieval_failures = _retrieval_failure_texts(state.working_memory.failures[-2:])
    if retrieval_failures:
        parts.append("Recent failures: " + " ".join(retrieval_failures))
    if state.recent_messages:
        last_user = next(
            (
                message.content
                for message in reversed(state.recent_messages)
                if message.role == "user"
                and message.content
                and not _is_recovery_nudge_message(message)
            ),
            "",
        )
        if last_user:
            parts.append(f"Latest user context: {last_user[:240]}")
    return "\n".join(part for part in parts if part)


def _is_retrieval_visible_artifact(artifact: ArtifactRecord) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    if metadata.get("model_visible", True) is not False:
        return True
    return bool(str(metadata.get("verifier_verdict") or "").strip())




def _visible_memory_texts(
    values: list[str],
    entries: list[MemoryEntry],
    *,
    current_step: int,
    current_phase: str,
) -> list[str]:
    visible: list[str] = []
    for index, text in enumerate(values):
        entry = entries[index] if index < len(entries) else None
        if entry is not None:
            if memory_entry_is_stale(
                entry,
                current_step=current_step,
                current_phase=current_phase,
            ):
                continue
            if entry.confidence is not None and entry.confidence < 0.6:
                continue
        visible.append(text)
    return visible
