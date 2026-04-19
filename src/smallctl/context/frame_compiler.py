from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

from ..state import (
    ArtifactSnippet,
    ContextBrief,
    EpisodicSummary,
    ExperienceMemory,
    LoopState,
    TurnBundle,
    clip_string_list,
    clip_text_value,
    normalize_intent_label,
)
from .artifact_visibility import is_prompt_visible_artifact, is_superseded_artifact
from .frame import (
    PromptArtifactPacket,
    PromptEvidencePacket,
    PromptExperiencePacket,
    PromptPhasePacket,
    PromptStateFrame,
    PromptStateSpine,
)
from .observations import build_observation_packets
from .retrieval import LexicalRetriever


class PromptStateFrameCompiler:
    def compile(
        self,
        *,
        state: LoopState,
        retrieved_summaries: Iterable[EpisodicSummary] = (),
        retrieved_artifacts: Iterable[ArtifactSnippet] = (),
        retrieved_experiences: Iterable[ExperienceMemory] = (),
    ) -> PromptStateFrame:
        phase_lines = self._render_phase_context(state)
        coding_anchor_lines = self._coding_anchor_lines(state)
        run_brief_text = self._render_run_brief(state)
        working_memory_text = self._render_working_memory(
            state=state,
            phase_lines=phase_lines,
            coding_anchor_lines=coding_anchor_lines,
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
            build_observation_packets(state, limit=8)
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
        next_allowed_action = ""
        if state.write_session is not None:
            next_allowed_action = self._render_write_session_next_action(state.write_session)
        if not next_allowed_action and state.working_memory.next_actions:
            next_allowed_action = state.working_memory.next_actions[-1]
        if not next_allowed_action:
            next_allowed_action = state.run_brief.current_phase_objective or state.run_brief.original_task

        files_in_play = self._collect_files_in_play(state=state, latest_brief=latest_brief)
        write_session_summary = self._render_write_session(state) if state.write_session else ""

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
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _collect_files_in_play(
        *,
        state: LoopState,
        latest_brief: ContextBrief | None,
    ) -> list[str]:
        candidates: list[str] = []
        task_targets = state.scratchpad.get("_task_target_paths")
        if isinstance(task_targets, list):
            candidates.extend(str(path or "").strip() for path in task_targets)
        candidates.extend(state.files_changed_this_cycle)
        if latest_brief is not None:
            candidates.extend(latest_brief.files_touched)
        plan = state.active_plan or state.draft_plan
        if plan is not None and plan.requested_output_path:
            candidates.append(str(plan.requested_output_path))
        if state.write_session is not None:
            if state.write_session.write_target_path:
                candidates.append(state.write_session.write_target_path)
            if state.write_session.write_staging_path:
                candidates.append(state.write_session.write_staging_path)
        return PromptStateFrameCompiler._dedupe(candidates)

    @staticmethod
    def _invalidated_fact_hints(state: LoopState) -> list[str]:
        queued = state.scratchpad.get("_invalidated_facts_queue")
        if not isinstance(queued, list):
            return []
        return PromptStateFrameCompiler._dedupe([str(item).strip() for item in queued if str(item).strip()])

    @classmethod
    def _filter_invalidated_turn_bundles(
        cls,
        *,
        state: LoopState,
        bundles: list[TurnBundle],
    ) -> tuple[list[TurnBundle], list[str]]:
        invalidations = cls._recent_invalidation_events(state)
        if not invalidations or not bundles:
            return bundles, []
        kept: list[TurnBundle] = []
        dropped_ids: list[str] = []
        for bundle in bundles:
            if cls._bundle_invalidated(state=state, bundle=bundle, invalidations=invalidations):
                if bundle.bundle_id:
                    dropped_ids.append(bundle.bundle_id)
                continue
            kept.append(bundle)
        return kept, dropped_ids

    @staticmethod
    def _filter_invalidated_observations(
        observations: list[Any],
    ) -> tuple[list[Any], list[str], int]:
        kept: list[Any] = []
        dropped_ids: list[str] = []
        dropped_count = 0
        for packet in observations:
            if bool(getattr(packet, "stale", False)):
                dropped_count += 1
                observation_id = str(getattr(packet, "observation_id", "") or "").strip()
                if observation_id:
                    dropped_ids.append(observation_id)
                continue
            kept.append(packet)
        return kept, dropped_ids, dropped_count

    @classmethod
    def _filter_invalidated_context_briefs(
        cls,
        *,
        state: LoopState,
        briefs: list[ContextBrief],
    ) -> tuple[list[ContextBrief], list[str]]:
        invalidations = cls._recent_invalidation_events(state)
        if not invalidations or not briefs:
            return briefs, []
        kept: list[ContextBrief] = []
        dropped_ids: list[str] = []
        for brief in briefs:
            if cls._brief_invalidated(state=state, brief=brief, invalidations=invalidations):
                if brief.brief_id:
                    dropped_ids.append(brief.brief_id)
                continue
            kept.append(brief)
        return kept, dropped_ids

    @classmethod
    def _filter_invalidated_summaries(
        cls,
        *,
        state: LoopState,
        summaries: list[EpisodicSummary],
    ) -> tuple[list[EpisodicSummary], list[str]]:
        invalidations = cls._recent_invalidation_events(state)
        if not invalidations or not summaries:
            return summaries, []
        kept: list[EpisodicSummary] = []
        dropped_ids: list[str] = []
        for summary in summaries:
            if cls._summary_invalidated(state=state, summary=summary, invalidations=invalidations):
                if summary.summary_id:
                    dropped_ids.append(summary.summary_id)
                continue
            kept.append(summary)
        return kept, dropped_ids

    @classmethod
    def _filter_invalidated_experiences(
        cls,
        *,
        state: LoopState,
        experiences: list[ExperienceMemory],
    ) -> tuple[list[ExperienceMemory], list[str]]:
        invalidations = cls._recent_invalidation_events(state)
        if not invalidations or not experiences:
            return experiences, []
        kept: list[ExperienceMemory] = []
        dropped_ids: list[str] = []
        for memory in experiences:
            if cls._experience_invalidated(state=state, memory=memory, invalidations=invalidations):
                if memory.memory_id:
                    dropped_ids.append(memory.memory_id)
                continue
            kept.append(memory)
        return kept, dropped_ids

    @classmethod
    def _filter_invalidated_artifact_snippets(
        cls,
        *,
        state: LoopState,
        snippets: list[ArtifactSnippet],
    ) -> tuple[list[ArtifactSnippet], list[str]]:
        invalidations = cls._recent_invalidation_events(state)
        if not invalidations or not snippets:
            return snippets, []
        artifacts = getattr(state, "artifacts", {}) if isinstance(getattr(state, "artifacts", {}), dict) else {}
        kept: list[ArtifactSnippet] = []
        dropped_ids: list[str] = []
        for snippet in snippets:
            artifact = artifacts.get(snippet.artifact_id)
            if cls._artifact_invalidated(state=state, artifact=artifact, invalidations=invalidations):
                if snippet.artifact_id:
                    dropped_ids.append(snippet.artifact_id)
                continue
            kept.append(snippet)
        return kept, dropped_ids

    @staticmethod
    def _recent_invalidation_events(state: LoopState) -> list[dict[str, Any]]:
        payload = state.scratchpad.get("_context_invalidations")
        if not isinstance(payload, list):
            return []
        return [item for item in payload[-24:] if isinstance(item, dict)]

    @staticmethod
    def _path_matches_any(target: str, changed_paths: list[str]) -> bool:
        normalized_target = Path(str(target or "").strip()).as_posix().lower()
        if not normalized_target:
            return False
        for changed in changed_paths:
            normalized_changed = Path(str(changed or "").strip()).as_posix().lower()
            if not normalized_changed:
                continue
            if (
                normalized_target == normalized_changed
                or normalized_target.endswith(normalized_changed)
                or normalized_changed.endswith(normalized_target)
            ):
                return True
            changed_name = Path(normalized_changed).name
            if changed_name and changed_name in normalized_target:
                return True
        return False

    @classmethod
    def _bundle_invalidated(
        cls,
        *,
        state: LoopState,
        bundle: TurnBundle,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        for event in invalidations:
            reason = str(event.get("reason") or "").strip().lower()
            paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
            if reason in {"file_changed", "write_session_target_changed"} and paths:
                if any(cls._path_matches_any(path, paths) for path in bundle.files_touched):
                    return True
            if reason in {"phase_advanced", "environment_changed"}:
                if bundle.phase and str(bundle.phase).strip().lower() != current_phase:
                    return True
            if reason == "verifier_failed" and any(cls._is_optimistic_statement(line) for line in bundle.summary_lines):
                return True
        return False

    @classmethod
    def _brief_invalidated(
        cls,
        *,
        state: LoopState,
        brief: ContextBrief,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        for event in invalidations:
            reason = str(event.get("reason") or "").strip().lower()
            paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
            if reason in {"file_changed", "write_session_target_changed"} and paths:
                if any(cls._path_matches_any(path, paths) for path in brief.files_touched):
                    return True
            if reason in {"phase_advanced", "environment_changed"}:
                if brief.current_phase and str(brief.current_phase).strip().lower() != current_phase:
                    return True
            if reason == "verifier_failed":
                if any(cls._is_optimistic_statement(line) for line in brief.key_discoveries):
                    return True
                if any(cls._is_optimistic_statement(line) for line in brief.new_facts):
                    return True
        return False

    @classmethod
    def _summary_invalidated(
        cls,
        *,
        state: LoopState,
        summary: EpisodicSummary,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
        for event in invalidations:
            reason = str(event.get("reason") or "").strip().lower()
            paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
            if reason in {"file_changed", "write_session_target_changed"} and paths:
                if any(cls._path_matches_any(path, paths) for path in summary.files_touched):
                    return True
            if reason == "verifier_failed":
                if any(cls._is_optimistic_statement(line) for line in summary.notes):
                    return True
                if failure_mode and any(failure_mode in str(line).strip().lower() for line in summary.failed_approaches):
                    return True
        return False

    @classmethod
    def _experience_invalidated(
        cls,
        *,
        state: LoopState,
        memory: ExperienceMemory,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
        notes = str(memory.notes or "").strip()
        for event in invalidations:
            reason = str(event.get("reason") or "").strip().lower()
            paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
            if reason in {"file_changed", "write_session_target_changed"} and paths:
                if any(cls._path_matches_any(notes, [path]) for path in paths):
                    return True
            if reason in {"phase_advanced", "environment_changed"}:
                if memory.phase and str(memory.phase).strip().lower() != current_phase:
                    return True
            if reason == "verifier_failed":
                if str(memory.outcome or "").strip().lower() == "success" and cls._is_optimistic_statement(notes):
                    return True
                if failure_mode and str(memory.failure_mode or "").strip().lower() == failure_mode:
                    return True
        return False

    @classmethod
    def _artifact_invalidated(
        cls,
        *,
        state: LoopState,
        artifact: Any,
        invalidations: list[dict[str, Any]],
    ) -> bool:
        if artifact is None:
            return False
        metadata = getattr(artifact, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        artifact_paths = cls._artifact_path_candidates(artifact, metadata)
        current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        metadata_phase = str(metadata.get("phase") or metadata.get("created_phase") or "").strip().lower()
        verifier_verdict = str(metadata.get("verifier_verdict") or "").strip().lower()
        for event in invalidations:
            reason = str(event.get("reason") or "").strip().lower()
            paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
            if reason in {"file_changed", "write_session_target_changed"} and paths:
                if any(cls._path_matches_any(candidate, paths) for candidate in artifact_paths):
                    return True
            if reason in {"phase_advanced", "environment_changed"} and metadata_phase:
                if metadata_phase != current_phase:
                    return True
            if reason == "verifier_failed" and verifier_verdict == "pass":
                return True
        return False

    @staticmethod
    def _artifact_path_candidates(artifact: Any, metadata: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        source = str(getattr(artifact, "source", "") or "").strip()
        if source:
            candidates.append(source)
        for key in ("path", "target_path", "write_target_path"):
            value = str(metadata.get(key) or "").strip()
            if value:
                candidates.append(value)
        path_tags = getattr(artifact, "path_tags", [])
        if isinstance(path_tags, list):
            candidates.extend(str(item).strip() for item in path_tags if str(item).strip())
        return candidates

    @staticmethod
    def _is_optimistic_statement(value: str) -> bool:
        lowered = str(value or "").strip().lower()
        if not lowered:
            return False
        return any(token in lowered for token in ("pass", "verified", "success", "fixed", "resolved"))

    @staticmethod
    def _coding_anchor_lines(state: LoopState) -> list[str]:
        anchors: list[str] = []
        task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
        is_coding_mode = bool(state.write_session) or bool(state.files_changed_this_cycle) or task_mode in {
            "analysis",
            "local_execute",
            "debug_inspect",
        }
        if not is_coding_mode:
            return anchors

        if state.write_session is not None:
            session = state.write_session
            if session.write_target_path:
                anchors.append(f"target_file={session.write_target_path}")
            anchors.append(f"write_mode={session.write_session_mode}")
            if session.write_sections_completed:
                anchors.append("completed_sections=" + ", ".join(session.write_sections_completed[:5]))
            if session.write_failed_local_patches:
                anchors.append(f"failed_sections={session.write_failed_local_patches} local patch failures")
        if state.files_changed_this_cycle:
            anchors.append("files_changed=" + ", ".join(state.files_changed_this_cycle[-5:]))
        verdict = state.current_verifier_verdict()
        if isinstance(verdict, dict):
            anchors.append("verifier_status=" + str(verdict.get("verdict") or "unknown"))
            command = str(verdict.get("command") or "").strip()
            if command:
                anchors.append(f"verifier_command={command}")
        checklist = state.acceptance_checklist()
        unmet = [
            str(item.get("criterion") or "").strip()
            for item in checklist
            if item.get("satisfied") is False and str(item.get("criterion") or "").strip()
        ]
        if unmet:
            anchors.append("acceptance_deltas=" + "; ".join(unmet[:4]))
        symbols = state.scratchpad.get("_touched_symbols")
        if isinstance(symbols, list):
            touched = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
            if touched:
                anchors.append("touched_symbols=" + ", ".join(touched[:8]))
        return PromptStateFrameCompiler._dedupe(anchors)

    @staticmethod
    def _render_run_brief(state: LoopState) -> str:
        brief = state.run_brief
        has_content = any(
            [
                brief.original_task,
                brief.task_contract,
                brief.current_phase_objective,
                state.active_intent,
                bool(brief.constraints),
                bool(brief.acceptance_criteria),
            ]
        )
        if not has_content:
            return ""
        parts = ["Run brief:"]
        parts.append(f"  CWD: {state.cwd}")

        if brief.original_task:
            parts.append(f"  Goal: {brief.original_task}")

        if brief.task_contract:
            parts.append(f"  Contract: {brief.task_contract}")

        if brief.current_phase_objective:
            parts.append(f"  Phase focus: {brief.current_phase_objective}")

        if state.active_intent:
            parts.append(f"  Active intent: {normalize_intent_label(state.active_intent)}")

        if brief.constraints:
            parts.append("  Constraints: " + "; ".join(brief.constraints))

        if brief.acceptance_criteria:
            parts.append("  Acceptance criteria: " + "; ".join(brief.acceptance_criteria))

        if len(parts) == 1:
            return ""
        return "\n".join(parts)

    @staticmethod
    def _render_working_memory(
        state: LoopState,
        *,
        phase_lines: list[str],
        coding_anchor_lines: list[str],
    ) -> str:
        memory = state.working_memory
        plan = state.active_plan or state.draft_plan
        has_content = any(
            [
                memory.current_goal,
                memory.plan,
                memory.decisions,
                memory.open_questions,
                memory.known_facts,
                memory.failures,
                memory.next_actions,
                bool((state.scratchpad.get("_session_notepad") or {}).get("entries"))
                if isinstance(state.scratchpad.get("_session_notepad"), dict)
                else False,
                plan is not None,
                bool(state.artifacts),
                state.write_session is not None,
                bool(phase_lines),
                bool(coding_anchor_lines),
            ]
        )
        if not has_content:
            return ""
        sections = [f"Current CWD: {state.cwd}"]
        task_targets = state.scratchpad.get("_task_target_paths")
        if isinstance(task_targets, list):
            cleaned_targets = [str(path).strip() for path in task_targets if str(path).strip()]
            if cleaned_targets:
                sections.append("Task targets: " + " | ".join(cleaned_targets[:3]))
        current_goal = LexicalRetriever._effective_current_goal(state)
        if current_goal:
            sections.append("Current goal: " + current_goal)
        if memory.plan:
            sections.append("Plan: " + " | ".join(memory.plan))
        if memory.decisions:
            sections.append("Decisions: " + " | ".join(memory.decisions))
        if memory.open_questions:
            sections.append("Open questions: " + " | ".join(memory.open_questions))
        if memory.known_facts:
            sections.append("Known facts: " + " | ".join(memory.known_facts))
        if memory.failures:
            sections.append("Known failures: " + " | ".join(memory.failures))
        if memory.next_actions:
            sections.append("Next actions: " + " | ".join(memory.next_actions))
        notepad_section = PromptStateFrameCompiler._render_session_notepad(state)
        if notepad_section:
            sections.append(notepad_section)
        if phase_lines:
            sections.extend(phase_lines)
        if coding_anchor_lines:
            sections.append("Coding anchors:")
            sections.extend(f"  {line}" for line in coding_anchor_lines[:8])
        if plan is not None:
            sections.append("Plan summary: " + plan.goal)
            sections.append(f"Plan status: {plan.status}")
            sections.append(
                "Plan resolved: "
                + ("yes" if state.plan_resolved else "no")
                + (f" | Plan artifact: {state.plan_artifact_id}" if state.plan_artifact_id else "")
            )
            if state.plan_artifact_id:
                sections.append(
                    f"Plan playbook artifact: {state.plan_artifact_id} (use this as the staged implementation checklist)"
                )
            active_step = plan.active_step()
            if active_step is not None:
                sections.append(f"Active step: {active_step.step_id} [{active_step.status}] {active_step.title}")
            for step in plan.steps[:6]:
                sections.append(f"Plan step: [{step.status}] {step.step_id} {step.title}")
            if plan.requested_output_path:
                sections.append(f"Plan export: {plan.requested_output_path}")
        elif state.plan_resolved and memory.plan:
            sections.append(
                "Plan resolved: yes"
                + (f" | Plan artifact: {state.plan_artifact_id}" if state.plan_artifact_id else "")
            )
            if state.plan_artifact_id:
                sections.append(
                    f"Plan playbook artifact: {state.plan_artifact_id} (use this as the staged implementation checklist)"
                )
        if state.contract_phase() == "repair":
            repair_bits = [f"Repair phase: {state.contract_phase()}"]
            if state.last_failure_class:
                repair_bits.append(f"Failure class: {state.last_failure_class}")
            if state.repair_cycle_id:
                repair_bits.append(
                    f"System repair cycle: {state.repair_cycle_id} (diagnostic only; not a write_session_id)"
                )
            if state.files_changed_this_cycle:
                repair_bits.append(
                    "Files changed this cycle: " + ", ".join(state.files_changed_this_cycle[-5:])
                )
            if state.stagnation_counters:
                counters = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(state.stagnation_counters.items())
                    if count
                )
                if counters:
                    repair_bits.append(f"Stagnation: {counters}")
            sections.append("Repair focus: " + " | ".join(repair_bits))
        if state.write_session:
            sections.append(PromptStateFrameCompiler._render_write_session(state))

        if state.artifacts:
            art_lines = []
            for aid, art in state.artifacts.items():
                if is_superseded_artifact(art):
                    continue
                if not is_prompt_visible_artifact(art):
                    continue
                summary_snippet = (art.summary or art.tool_name or "").strip()[:90]
                art_lines.append(f"  - {aid}: {summary_snippet}")
            if art_lines:
                sections.append(
                    "Available Artifacts (compressed summaries already in context; page forward with artifact_read(start_line=...) only if you need more unseen lines):\n"
                    + "\n".join(art_lines)
                )
        if not sections:
            return ""
        return "Working memory:\n" + "\n".join(sections)

    @staticmethod
    def _render_session_notepad(state: LoopState) -> str:
        payload = state.scratchpad.get("_session_notepad")
        if not isinstance(payload, dict):
            return ""
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            return ""

        entries, clipped = clip_string_list(
            raw_entries,
            limit=5,
            item_char_limit=160,
        )
        if not entries:
            return ""

        bits = ["Session notepad: " + " | ".join(entries)]
        bits.append("Keep this brief; the notepad is durable session memory, not a transcript dump.")
        if clipped or len(entries) < len([item for item in raw_entries if str(item).strip()]):
            bits.append("Some entries were clipped to keep the session note concise.")
        return " ".join(bits)

    @staticmethod
    def _render_write_session(state: LoopState) -> str:
        session = state.write_session
        ws_bits = [f"Session: {session.write_session_id}"]
        if session.write_target_path:
            ws_bits.append(f"Target: {session.write_target_path}")
        if session.write_staging_path:
            ws_bits.append(f"Staging: {session.write_staging_path}")
        if session.write_staging_path:
            ws_bits.append("Reminder: the staging path is for read/verify only; write to the target path.")
        else:
            ws_bits.append("Reminder: write to the target path; staged copies are for read/verify.")
        ws_bits.append(f"Mode: {session.write_session_mode}")
        ws_bits.append(f"Intent: {session.write_session_intent}")
        ws_bits.append(f"Status: {session.status}")
        if session.write_current_section:
            ws_bits.append(f"Current section: {session.write_current_section}")
        if session.write_next_section:
            ws_bits.append(f"Next section: {session.write_next_section}")
        if session.write_sections_completed:
            ws_bits.append(f"Completed sections: {', '.join(session.write_sections_completed)}")
        if session.suggested_sections:
            ws_bits.append(f"Suggested sections: {', '.join(session.suggested_sections)}")
        if session.write_failed_local_patches:
            ws_bits.append(f"Local patch failures: {session.write_failed_local_patches}")
        if session.write_pending_finalize:
            ws_bits.append("Pending finalize: yes")
        else:
            ws_bits.append("Pending finalize: no")
        next_action = PromptStateFrameCompiler._render_write_session_next_action(session)
        if next_action:
            ws_bits.append(f"Next action: {next_action}")
        verifier = session.write_last_verifier or {}
        if verifier:
            ws_bits.append(f"Last verifier verdict: {verifier.get('verdict', 'unknown')}")
            command = str(verifier.get("command", "") or "").strip()
            if command:
                ws_bits.append(f"Verifier command: {command}")
            verifier_output, clipped = clip_text_value(str(verifier.get("output", "") or "").strip(), limit=180)
            if verifier_output:
                suffix = " [truncated]" if clipped else ""
                ws_bits.append(f"Verifier output: {verifier_output}{suffix}")
        return "Active Write Session: " + " | ".join(ws_bits)

    @staticmethod
    def _render_write_session_next_action(session: Any) -> str:
        if bool(getattr(session, "write_pending_finalize", False)):
            return "Finalize the staged copy after verification."
        next_section = str(getattr(session, "write_next_section", "") or "").strip()
        if next_section:
            return f"Continue with section {next_section}."
        current_section = str(getattr(session, "write_current_section", "") or "").strip()
        if current_section:
            return f"Complete section {current_section} or move to verification once it is ready."
        completed = list(getattr(session, "write_sections_completed", []) or [])
        if completed:
            return "Choose the next section or verify the completed staged file."
        return "Continue authoring the active target file."

    @staticmethod
    def _render_phase_context(state: LoopState) -> list[str]:
        phase = state.contract_phase()
        sections: list[str] = []
        reasoning = state.reasoning_graph
        if phase == "explore":
            sections.append("Phase handoff: explore -> plan")
            if state.context_briefs:
                brief = state.context_briefs[-1]
                brief_bits = [f"brief={brief.brief_id}", f"phase={brief.current_phase}"]
                if brief.key_discoveries:
                    brief_bits.append("facts=" + "; ".join(brief.key_discoveries[:4]))
                if brief.open_questions:
                    brief_bits.append("questions=" + "; ".join(brief.open_questions[:3]))
                if brief.evidence_refs:
                    brief_bits.append("evidence=" + ", ".join(brief.evidence_refs[:5]))
                sections.append("Explore brief: " + " | ".join(brief_bits))
        elif phase == "plan":
            sections.append("Phase handoff: plan from compressed evidence")
            evidence_bits = []
            for record in reasoning.evidence_records[-5:]:
                statement = record.statement.strip()
                if statement:
                    evidence_bits.append(f"{record.evidence_id}: {statement}")
            if evidence_bits:
                sections.append("Evidence packet: " + " | ".join(evidence_bits))
            if reasoning.claim_records:
                claim_bits = []
                for claim in reasoning.claim_records[-4:]:
                    claim_bits.append(f"{claim.claim_id} [{claim.status}] {claim.statement}".strip())
                if claim_bits:
                    sections.append("Claims: " + " | ".join(claim_bits))
            plan = state.active_plan or state.draft_plan
            if plan is not None and getattr(plan, "claim_refs", None):
                sections.append("Plan claim refs: " + ", ".join(plan.claim_refs[:5]))
        elif phase == "author":
            sections.append("Phase handoff: plan -> author")
            plan = state.active_plan or state.draft_plan
            if plan is not None:
                sections.append(f"Authoring plan: {plan.plan_id} | {plan.goal}")
                active_step = plan.active_step()
                if active_step is not None:
                    sections.append(
                        f"Current step: {active_step.step_id} [{active_step.status}] {active_step.title}"
                    )
                    if active_step.claim_refs:
                        sections.append("Current step claims: " + ", ".join(active_step.claim_refs[:5]))
                if plan.acceptance_criteria:
                    sections.append("Acceptance: " + "; ".join(plan.acceptance_criteria[:4]))
                if getattr(plan, "claim_refs", None):
                    sections.append("Plan claims: " + ", ".join(plan.claim_refs[:5]))
                if plan.requested_output_path:
                    sections.append(f"Target output: {plan.requested_output_path}")
        elif phase == "execute":
            sections.append("Phase handoff: author -> execute")
            plan = state.active_plan or state.draft_plan
            if plan is not None:
                sections.append(f"Execution plan: {plan.plan_id} [{plan.status}]")
                if getattr(plan, "claim_refs", None):
                    sections.append("Plan claims: " + ", ".join(plan.claim_refs[:5]))
            if state.reasoning_graph.evidence_records:
                evidence_ids = [record.evidence_id for record in state.reasoning_graph.evidence_records[-5:]]
                sections.append("Evidence refs: " + ", ".join(evidence_ids))
        elif phase == "verify":
            sections.append("Phase handoff: execute -> verify")
            if state.current_verifier_verdict():
                sections.append("Verifier verdict present")
            if state.run_brief.acceptance_criteria:
                sections.append("Acceptance criteria: " + "; ".join(state.run_brief.acceptance_criteria[:4]))
        elif phase == "repair":
            sections.append("Phase handoff: verify -> repair")
            if state.last_failure_class:
                sections.append(f"Failure class: {state.last_failure_class}")
            if state.files_changed_this_cycle:
                sections.append("Files changed: " + ", ".join(state.files_changed_this_cycle[-5:]))
            if state.write_session:
                sections.append(f"Write session: {state.write_session.write_session_id}")
        return sections
