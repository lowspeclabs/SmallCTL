from __future__ import annotations

from typing import Iterable, Any

from ..state import (
    ArtifactSnippet,
    ContextBrief,
    EpisodicSummary,
    ExperienceMemory,
    LoopState,
    TurnBundle,
    normalize_intent_label,
)
from .frame_recovery_rendering import (
    fresh_read_loop_recovery_lines,
    fresh_schema_validation_hint_lines,
    guard_trip_recovery_lines,
    remote_mutation_directory_checks,
    render_recovery_guidance as _render_recovery_guidance,
    render_remote_mutation_next_action,
    render_ssh_tool_call,
    repair_continuity_lines,
)
from .frame_run_rendering import (
    active_ssh_session_labels,
    artifact_evidence_rows,
    coding_anchor_lines,
    continuation_anchor_lines,
    render_task_ground_truth,
    render_run_brief,
    run_boundary_lines,
    should_render_active_ssh_sessions,
)
from .frame_session_rendering import (
    render_session_notepad,
    render_write_session,
    render_write_session_next_action,
)
from .frame_working_memory_rendering import (
    render_sub4b_top_web_findings,
    render_working_memory,
)
from .frame_state_helpers import (
    collect_files_in_play,
    dedupe_nonempty,
    invalidated_fact_hints,
    state_model_name,
)
from .frame_invalidation_utils import (
    coerce_datetime,
    durably_stale_ids,
    guard_trip_preserved_ids,
    is_optimistic_statement,
    path_matches_any,
    path_tokens,
    recent_invalidation_events,
)
from .frame_invalidation_filtering import (
    artifact_invalidated,
    brief_invalidated,
    bundle_invalidated,
    durably_stale_artifact_ids,
    durably_stale_brief_ids,
    durably_stale_experience_ids,
    durably_stale_observation_ids,
    durably_stale_summary_ids,
    durably_stale_turn_bundle_ids,
    experience_invalidated,
    filter_invalidated_artifact_snippets,
    filter_invalidated_context_briefs,
    filter_invalidated_experiences,
    filter_invalidated_observations,
    filter_invalidated_summaries,
    filter_invalidated_turn_bundles,
    optimistic_text_invalidated_by_verifier,
    summary_invalidated,
    verifier_failure_paths,
    verifier_failure_related_to_text,
)
from .artifact_visibility import (
    artifact_path_candidates,
    is_read_only_artifact,
)
from .frame_phase_rendering import render_phase_context
from .frame import (
    PromptArtifactPacket,
    PromptEvidencePacket,
    PromptExperiencePacket,
    PromptPhasePacket,
    PromptStateFrame,
    PromptStateSpine,
)
from .observations import build_observation_packets
from .policy import ContextPolicy

class PromptStateFrameCompiler:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()

    def compile(
        self,
        *,
        state: LoopState,
        retrieved_summaries: Iterable[EpisodicSummary] = (),
        retrieved_artifacts: Iterable[ArtifactSnippet] = (),
        retrieved_experiences: Iterable[ExperienceMemory] = (),
    ) -> PromptStateFrame:
        phase_lines = self._render_phase_context(state)
        coding_anchor_lines = self._coding_anchor_lines(state) if self.policy.coding_profile_enabled else []
        run_brief_text = self._render_run_brief(state)
        from ..fama.capsules import render_fama_capsules

        fama_capsule_lines = render_fama_capsules(
            state,
            token_budget=int(state.scratchpad.get("_fama_config", {}).get("capsule_token_budget", 180))
            if isinstance(state.scratchpad.get("_fama_config"), dict)
            else 180,
        )
        recovery_guidance_lines = self.render_recovery_guidance(state)
        working_memory_text = self._render_working_memory(
            state=state,
            phase_lines=phase_lines,
            coding_anchor_lines=coding_anchor_lines,
            fama_capsule_lines=fama_capsule_lines,
            recovery_guidance_lines=recovery_guidance_lines,
        )

        plan = state.active_plan or state.draft_plan
        active_step = ""
        if plan is not None:
            step = plan.active_step()
            if step is not None:
                active_step = f"{step.step_id} [{step.status}] {step.title}".strip()

        checklist = state.acceptance_checklist()
        unmet_acceptance = [
            str(item.get("criterion") or "").strip()
            for item in checklist
            if item.get("satisfied") is False and str(item.get("criterion") or "").strip()
        ]

        context_briefs, dropped_brief_ids = self._filter_invalidated_context_briefs(
            state=state,
            briefs=list(state.context_briefs),
        )
        latest_brief = context_briefs[-1] if context_briefs else None
        observations, dropped_observation_ids, dropped_observation_count = self._filter_invalidated_observations(
            state=state,
            observations=build_observation_packets(state, limit=8),
        )
        turn_bundles, dropped_turn_bundle_ids = self._filter_invalidated_turn_bundles(
            state=state,
            bundles=list(state.turn_bundles[-4:]),
        )
        summaries, dropped_summary_ids = self._filter_invalidated_summaries(
            state=state,
            summaries=list(retrieved_summaries),
        )
        experiences, dropped_experience_ids = self._filter_invalidated_experiences(
            state=state,
            experiences=list(retrieved_experiences),
        )
        from ..harness.task_classifier import task_is_local_coding_target
        from ..memory_namespace import is_remote_only_memory
        task_text = str(getattr(state.run_brief, "original_task", "") or "")
        if task_is_local_coding_target(task_text):
            filtered_experiences = []
            dropped_remote_ids = []
            for e in experiences:
                if is_remote_only_memory(e):
                    dropped_remote_ids.append(getattr(e, "id", str(e)))
                else:
                    filtered_experiences.append(e)
            if dropped_remote_ids:
                state.scratchpad["_remote_memory_suppressed_for_local_task"] = {
                    "count": len(dropped_remote_ids),
                    "reason": "local_coding_task",
                }
            experiences = filtered_experiences
        artifacts, dropped_artifact_ids = self._filter_invalidated_artifact_snippets(
            state=state,
            snippets=list(retrieved_artifacts),
        )
        invalidated_hints = self._invalidated_fact_hints(state)
        known_good_facts = self._dedupe(
            list(state.working_memory.known_facts)
            + list(getattr(latest_brief, "facts_confirmed", []) or [])
            + [packet.summary for packet in observations if not packet.stale and packet.kind in {"file_fact", "verifier_verdict"}][:3]
        )
        if invalidated_hints:
            known_good_facts = [fact for fact in known_good_facts if fact not in set(invalidated_hints)]
        current_blockers = self._dedupe(
            list(state.working_memory.failures)
            + list(getattr(latest_brief, "blockers", []) or [])
            + ([state.last_failure_class] if state.last_failure_class else [])
            + [f"Invalidated: {item}" for item in invalidated_hints[:3]]
        )
        remote_mutation_action = self._render_remote_mutation_next_action(state)
        if remote_mutation_action:
            current_blockers = self._dedupe(
                current_blockers
                + ["Remote mutation verification is pending before completion."]
            )
        next_allowed_action = ""
        if remote_mutation_action:
            next_allowed_action = remote_mutation_action
        if not next_allowed_action and state.write_session is not None:
            next_allowed_action = self._render_write_session_next_action(state.write_session)
        if not next_allowed_action and state.working_memory.next_actions:
            next_allowed_action = state.working_memory.next_actions[-1]
        if next_allowed_action and task_is_local_coding_target(task_text):
            lowered_next = next_allowed_action.lower()
            if any(t in lowered_next for t in ("ssh_exec", "ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between")):
                next_allowed_action = state.run_brief.current_phase_objective or state.run_brief.original_task
                state.scratchpad["_remote_next_action_suppressed_for_local_task"] = {
                    "original_next_action": state.working_memory.next_actions[-1] if state.working_memory.next_actions else None,
                    "reason": "local_coding_task",
                }
        if not next_allowed_action:
            next_allowed_action = state.run_brief.current_phase_objective or state.run_brief.original_task

        files_in_play = self._collect_files_in_play(state=state, latest_brief=latest_brief)
        write_session_summary = self._render_write_session(state) if state.write_session else ""

        _maybe_inject_argparse_subcommand_note(state)

        spine = PromptStateSpine(
            cwd=state.cwd,
            task_goal=state.run_brief.original_task,
            task_contract=state.run_brief.task_contract,
            current_phase=state.contract_phase(),
            phase_focus=state.run_brief.current_phase_objective,
            active_step=active_step,
            active_intent=normalize_intent_label(state.active_intent),
            unmet_acceptance_criteria=unmet_acceptance,
            known_good_facts=known_good_facts,
            current_blockers=current_blockers,
            next_allowed_action=next_allowed_action,
            files_in_play=files_in_play,
            write_session_summary=write_session_summary,
            coding_anchor_lines=coding_anchor_lines,
            constraints=list(state.run_brief.constraints),
            acceptance_criteria=state.active_acceptance_criteria(),
            run_brief_text=run_brief_text,
            working_memory_text=working_memory_text,
            fama_capsule_lines=fama_capsule_lines,
            recovery_guidance_lines=recovery_guidance_lines,
        )
        frame = PromptStateFrame(
            spine=spine,
            phase_packet=PromptPhasePacket(lines=phase_lines),
            evidence_packet=PromptEvidencePacket(
                observations=observations,
                turn_bundles=turn_bundles,
                context_briefs=context_briefs,
                summaries=summaries,
            ),
            experience_packet=PromptExperiencePacket(memories=experiences),
            artifact_packet=PromptArtifactPacket(snippets=artifacts),
        )
        if dropped_turn_bundle_ids:
            frame.add_drop(
                lane="turn_bundles",
                reason="context_invalidated",
                dropped_count=len(dropped_turn_bundle_ids),
                dropped_ids=dropped_turn_bundle_ids,
            )
        if dropped_observation_count > 0:
            frame.add_drop(
                lane="normalized_observations",
                reason="context_invalidated",
                dropped_count=dropped_observation_count,
                dropped_ids=dropped_observation_ids,
            )
        if dropped_brief_ids:
            frame.add_drop(
                lane="context_briefs",
                reason="context_invalidated",
                dropped_count=len(dropped_brief_ids),
                dropped_ids=dropped_brief_ids,
            )
        if dropped_summary_ids:
            frame.add_drop(
                lane="episodic_summaries",
                reason="context_invalidated",
                dropped_count=len(dropped_summary_ids),
                dropped_ids=dropped_summary_ids,
            )
        if dropped_experience_ids:
            frame.add_drop(
                lane="experience_memories",
                reason="context_invalidated",
                dropped_count=len(dropped_experience_ids),
                dropped_ids=dropped_experience_ids,
            )
        if dropped_artifact_ids:
            frame.add_drop(
                lane="artifact_snippets",
                reason="context_invalidated",
                dropped_count=len(dropped_artifact_ids),
                dropped_ids=dropped_artifact_ids,
            )
        return frame

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return dedupe_nonempty(values)

    @staticmethod
    def _collect_files_in_play(
        *,
        state: LoopState,
        latest_brief: ContextBrief | None,
    ) -> list[str]:
        return collect_files_in_play(state=state, latest_brief=latest_brief)

    @staticmethod
    def _invalidated_fact_hints(state: LoopState) -> list[str]:
        return invalidated_fact_hints(state)

    @staticmethod
    def _state_model_name(state: LoopState) -> str:
        return state_model_name(state)

    @classmethod
    def _filter_invalidated_turn_bundles(
        cls,
        *,
        state: LoopState,
        bundles: list[TurnBundle],
    ) -> tuple[list[TurnBundle], list[str]]:
        return filter_invalidated_turn_bundles(state=state, bundles=bundles)

    @classmethod
    def _filter_invalidated_observations(
        cls,
        *,
        state: LoopState,
        observations: list[Any],
    ) -> tuple[list[Any], list[str], int]:
        return filter_invalidated_observations(state=state, observations=observations)

    @classmethod
    def _filter_invalidated_context_briefs(
        cls,
        *,
        state: LoopState,
        briefs: list[ContextBrief],
    ) -> tuple[list[ContextBrief], list[str]]:
        return filter_invalidated_context_briefs(state=state, briefs=briefs)

    @classmethod
    def _filter_invalidated_summaries(
        cls,
        *,
        state: LoopState,
        summaries: list[EpisodicSummary],
    ) -> tuple[list[EpisodicSummary], list[str]]:
        return filter_invalidated_summaries(state=state, summaries=summaries)

    @classmethod
    def _filter_invalidated_experiences(
        cls,
        *,
        state: LoopState,
        experiences: list[ExperienceMemory],
    ) -> tuple[list[ExperienceMemory], list[str]]:
        return filter_invalidated_experiences(state=state, experiences=experiences)

    @classmethod
    def _filter_invalidated_artifact_snippets(
        cls,
        *,
        state: LoopState,
        snippets: list[ArtifactSnippet],
    ) -> tuple[list[ArtifactSnippet], list[str]]:
        return filter_invalidated_artifact_snippets(state=state, snippets=snippets)

    @staticmethod
    def _recent_invalidation_events(state: LoopState) -> list[dict[str, Any]]:
        return recent_invalidation_events(state)

    @staticmethod
    def _durably_stale_ids(state: LoopState, key: str) -> set[str]:
        return durably_stale_ids(state, key)

    @staticmethod
    def _guard_trip_preserved_ids(state: LoopState, key: str) -> set[str]:
        return guard_trip_preserved_ids(state, key)

    @classmethod
    def _durably_stale_experience_ids(cls, state: LoopState) -> set[str]:
        return durably_stale_experience_ids(state)

    @classmethod
    def _durably_stale_turn_bundle_ids(cls, state: LoopState) -> set[str]:
        return durably_stale_turn_bundle_ids(state)

    @classmethod
    def _durably_stale_brief_ids(cls, state: LoopState) -> set[str]:
        return durably_stale_brief_ids(state)

    @classmethod
    def _durably_stale_summary_ids(cls, state: LoopState) -> set[str]:
        return durably_stale_summary_ids(state)

    @classmethod
    def _durably_stale_artifact_ids(cls, state: LoopState) -> set[str]:
        return durably_stale_artifact_ids(state)

    @classmethod
    def _durably_stale_observation_ids(cls, state: LoopState) -> set[str]:
        return durably_stale_observation_ids(state)

    @staticmethod
    def _path_matches_any(target: str, changed_paths: list[str]) -> bool:
        return path_matches_any(target, changed_paths)

    @staticmethod
    def _path_tokens(text: str) -> list[str]:
        return path_tokens(text)

    @classmethod
    def _verifier_failure_paths(cls, event: dict[str, Any]) -> list[str]:
        return verifier_failure_paths(event)

    @classmethod
    def _verifier_failure_related_to_text(cls, text: str, event: dict[str, Any]) -> bool:
        return verifier_failure_related_to_text(text, event)

    @classmethod
    def _optimistic_text_invalidated_by_verifier(cls, text: str, event: dict[str, Any]) -> bool:
        return optimistic_text_invalidated_by_verifier(text, event)

    @classmethod
    def _bundle_invalidated(
        cls,
        *,
        state: LoopState,
        bundle: TurnBundle,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        return bundle_invalidated(state=state, bundle=bundle, invalidations=invalidations)

    @classmethod
    def _brief_invalidated(
        cls,
        *,
        state: LoopState,
        brief: ContextBrief,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        return brief_invalidated(state=state, brief=brief, invalidations=invalidations)

    @classmethod
    def _summary_invalidated(
        cls,
        *,
        state: LoopState,
        summary: EpisodicSummary,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        return summary_invalidated(state=state, summary=summary, invalidations=invalidations)

    @classmethod
    def _experience_invalidated(
        cls,
        *,
        state: LoopState,
        memory: ExperienceMemory,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        return experience_invalidated(state=state, memory=memory, invalidations=invalidations)

    @classmethod
    def _artifact_invalidated(
        cls,
        *,
        state: LoopState,
        artifact: Any,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        return artifact_invalidated(state=state, artifact=artifact, invalidations=invalidations)

    @staticmethod
    def _is_read_only_artifact(artifact: Any) -> bool:
        return is_read_only_artifact(artifact)

    @staticmethod
    def _artifact_path_candidates(artifact: Any, metadata: dict[str, Any]) -> list[str]:
        return artifact_path_candidates(artifact, metadata)

    @staticmethod
    def _is_optimistic_statement(value: str) -> bool:
        return is_optimistic_statement(value)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        return coerce_datetime(value)

    @staticmethod
    def _coding_anchor_lines(state: LoopState) -> list[str]:
        return coding_anchor_lines(state)

    @staticmethod
    def _artifact_evidence_rows(state: LoopState, *, limit: int = 8) -> list[str]:
        return artifact_evidence_rows(state, limit=limit)

    @staticmethod
    def _render_run_brief(state: LoopState) -> str:
        return render_run_brief(state)

    @staticmethod
    def _run_boundary_lines(state: LoopState) -> list[str]:
        return run_boundary_lines(state)

    @staticmethod
    def _active_ssh_session_labels(state: LoopState) -> list[str]:
        return active_ssh_session_labels(state)

    @staticmethod
    def _continuation_anchor_lines(state: LoopState) -> list[str]:
        return continuation_anchor_lines(state)

    @staticmethod
    def _should_render_active_ssh_sessions(state: LoopState) -> bool:
        return should_render_active_ssh_sessions(state)

    @staticmethod
    def _render_task_ground_truth(state: LoopState) -> str:
        return render_task_ground_truth(state)

    @staticmethod
    def _render_remote_mutation_next_action(state: LoopState) -> str:
        return render_remote_mutation_next_action(state)

    @staticmethod
    def _render_ssh_tool_call(tool_name: str, *, host: str, user: str = "", path: str = "", command: str = "") -> str:
        return render_ssh_tool_call(tool_name, host=host, user=user, path=path, command=command)

    @staticmethod
    def _remote_mutation_directory_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
        return remote_mutation_directory_checks(requirement)

    @staticmethod
    def render_recovery_guidance(state: LoopState, token_budget: int = 500) -> list[str]:
        return _render_recovery_guidance(state, token_budget=token_budget)

    @staticmethod
    def _fresh_schema_validation_hint_lines(state: LoopState) -> list[str]:
        return fresh_schema_validation_hint_lines(state)

    @staticmethod
    def _fresh_read_loop_recovery_lines(state: LoopState) -> list[str]:
        return fresh_read_loop_recovery_lines(state)

    @staticmethod
    def _repair_continuity_lines(state: LoopState) -> list[str]:
        return repair_continuity_lines(state)

    @staticmethod
    def _guard_trip_recovery_lines(state: LoopState) -> list[str]:
        return guard_trip_recovery_lines(state)

    @staticmethod
    def _render_working_memory(
        state: LoopState,
        *,
        phase_lines: list[str],
        coding_anchor_lines: list[str],
        fama_capsule_lines: list[str] | None = None,
        recovery_guidance_lines: list[str] | None = None,
    ) -> str:
        return render_working_memory(
            state,
            phase_lines=phase_lines,
            coding_anchor_lines=coding_anchor_lines,
            fama_capsule_lines=fama_capsule_lines,
            recovery_guidance_lines=recovery_guidance_lines,
        )

    @staticmethod
    def _render_sub4b_top_web_findings(state: LoopState) -> str:
        return render_sub4b_top_web_findings(state)

    @staticmethod
    def _render_session_notepad(state: LoopState) -> str:
        return render_session_notepad(state)

    @staticmethod
    def _render_write_session(state: LoopState) -> str:
        return render_write_session(state)

    @staticmethod
    def _render_write_session_next_action(session: Any) -> str:
        return render_write_session_next_action(session)

    @staticmethod
    def _render_phase_context(state: LoopState) -> list[str]:
        return render_phase_context(state)


def _maybe_inject_argparse_subcommand_note(state: LoopState) -> None:
    """Detect argparse subparser structure in recently read .py artifacts and inject a hint."""
    note = (
        "CLI subcommand ordering: this script uses argparse subcommands. "
        "Place global flags (e.g., --url, --token) BEFORE the subcommand, not after."
    )
    known_facts = list(state.working_memory.known_facts) if state.working_memory.known_facts else []
    if any(note in fact for fact in known_facts):
        return
    for artifact in (state.artifacts or {}).values():
        if isinstance(artifact, dict):
            kind = str(artifact.get("kind") or artifact.get("tool_name") or "").strip()
            source = str(artifact.get("source") or "").strip()
            content = str(artifact.get("inline_content") or artifact.get("preview_text") or "").strip()
        else:
            kind = str(getattr(artifact, "kind", "") or getattr(artifact, "tool_name", "") or "").strip()
            source = str(getattr(artifact, "source", "") or "").strip()
            content = str(getattr(artifact, "inline_content", "") or getattr(artifact, "preview_text", "") or "").strip()
        if kind not in {"file_read", "ssh_file_read"}:
            continue
        if not source.endswith(".py"):
            continue
        if not content:
            continue
        if "add_argument" in content and "subparsers.add_parser" in content:
            state.working_memory.known_facts.append(note)
            return
