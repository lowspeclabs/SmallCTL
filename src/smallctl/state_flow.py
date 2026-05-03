from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experience_tags import normalize_experience_tags
from .memory_namespace import infer_memory_namespace, normalize_memory_namespace
from .models.conversation import ConversationMessage
from .state_memory import (
    align_memory_entries,
    memory_entry_is_stale,
)
from .state_schema import ExperienceMemory, MemoryEntry
from .state_support import clip_string_list, normalize_intent_label


def _coerce_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class LoopStateFlowMixin:
    @staticmethod
    def _normalize_paths(paths: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for path in paths or []:
            text = str(path or "").strip()
            if not text:
                continue
            normalized.append(Path(text).as_posix().lower())
        deduped: list[str] = []
        for path in normalized:
            if path not in deduped:
                deduped.append(path)
        return deduped

    @staticmethod
    def _text_matches_any_path(text: str, paths: list[str]) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        for path in paths:
            if not path:
                continue
            if path in lowered or lowered.endswith(path):
                return True
            basename = Path(path).name
            if basename and basename in lowered:
                return True
        return False

    @staticmethod
    def _invalidate_memory_entry(entry: MemoryEntry) -> None:
        if entry.freshness == "pinned":
            return
        entry.freshness = "stale"
        if entry.confidence is None:
            entry.confidence = 0.6
        entry.confidence = max(0.0, min(1.0, float(entry.confidence) - 0.3))

    @staticmethod
    def _is_optimistic_statement(value: str) -> bool:
        lowered = str(value or "").strip().lower()
        if not lowered:
            return False
        return any(token in lowered for token in ("pass", "verified", "success", "fixed", "resolved"))

    def _lane_staleness_index(self, key: str) -> dict[str, dict[str, Any]]:
        payload = self.scratchpad.get(key)
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for raw_id, marker in payload.items():
            item_id = str(raw_id or "").strip()
            if not item_id or not isinstance(marker, dict):
                continue
            normalized[item_id] = dict(marker)
        return normalized

    def _clear_lane_staleness(self, key: str, item_id: str) -> None:
        normalized_id = str(item_id or "").strip()
        if not normalized_id:
            return
        index = self._lane_staleness_index(key)
        if normalized_id not in index:
            return
        index.pop(normalized_id, None)
        if index:
            self.scratchpad[key] = index
        else:
            self.scratchpad.pop(key, None)

    def _mark_lane_stale(
        self,
        *,
        key: str,
        item_id: str,
        reason: str,
        paths: list[str],
        now: str,
    ) -> None:
        normalized_id = str(item_id or "").strip()
        if not normalized_id:
            return
        index = self._lane_staleness_index(key)
        existing_meta = index.get(normalized_id)
        if not isinstance(existing_meta, dict):
            existing_meta = {}
        reasons = [
            str(item).strip().lower()
            for item in (existing_meta.get("reasons") or [])
            if str(item).strip()
        ]
        if reason not in reasons:
            reasons.append(reason)
        merged_paths = [
            str(item).strip().lower()
            for item in (existing_meta.get("paths") or [])
            if str(item).strip()
        ]
        for path in paths:
            if path not in merged_paths:
                merged_paths.append(path)
        index[normalized_id] = {
            "stale": True,
            "reason": reason,
            "reasons": reasons[-6:],
            "paths": merged_paths[-12:],
            "updated_at": now,
            "phase": str(self.current_phase or ""),
        }
        self.scratchpad[key] = index

    def _experience_staleness_index(self) -> dict[str, dict[str, Any]]:
        return self._lane_staleness_index("_experience_staleness")

    def _clear_experience_staleness(self, memory_id: str) -> None:
        self._clear_lane_staleness("_experience_staleness", memory_id)

    def sync_plan_mirror(self) -> None:
        plan = self.active_plan or self.draft_plan
        if plan is None:
            self.plan_resolved = bool(self.working_memory.plan)
            if not self.plan_resolved:
                self.working_memory.plan = []
            return
        self.working_memory.plan = clip_string_list(
            plan.compact_lines(),
            limit=12,
            item_char_limit=220,
        )[0]
        self.working_memory.current_goal = plan.goal
        self.run_brief.current_phase_objective = plan.goal
        self.plan_resolved = True
        self.run_brief.task_contract = plan.spec_summary()
        self.run_brief.inputs = list(plan.inputs)
        self.run_brief.outputs = list(plan.outputs)
        self.run_brief.constraints = list(plan.constraints)
        self.run_brief.acceptance_criteria = list(plan.acceptance_criteria)
        self.run_brief.implementation_plan = list(plan.implementation_plan)

    def active_acceptance_criteria(self) -> list[str]:
        plan = self.active_plan or self.draft_plan
        if plan is not None and plan.acceptance_criteria:
            return list(plan.acceptance_criteria)
        if self.run_brief.acceptance_criteria:
            return list(self.run_brief.acceptance_criteria)
        return []

    def acceptance_checklist(self) -> list[dict[str, Any]]:
        criteria = self.active_acceptance_criteria()
        if not criteria:
            return []
        ledger = self.acceptance_ledger.copy()
        scratch_ledger = self.scratchpad.get("_acceptance_ledger")
        if isinstance(scratch_ledger, dict):
            for key, value in scratch_ledger.items():
                ledger[str(key)] = str(value)
        waived_items = {item for item in self.acceptance_waivers if item}
        if self.acceptance_waived:
            waived_items.update(criteria)
        checklist: list[dict[str, Any]] = []
        for criterion in criteria:
            status = str(ledger.get(criterion, "pending") or "pending")
            if criterion in waived_items:
                status = "waived"
            checklist.append(
                {
                    "criterion": criterion,
                    "status": status,
                    "satisfied": status in {"done", "passed", "complete", "completed", "waived"},
                }
            )
        return checklist

    def acceptance_ready(self) -> bool:
        checklist = self.acceptance_checklist()
        return not checklist or all(item["satisfied"] for item in checklist)

    def current_verifier_verdict(self) -> dict[str, Any] | None:
        verdict = self.last_verifier_verdict
        if isinstance(verdict, dict):
            return verdict
        scratch_verdict = self.scratchpad.get("_last_verifier_verdict")
        return scratch_verdict if isinstance(scratch_verdict, dict) else None

    def contract_flow_active(self) -> bool:
        if self.repair_cycle_id or self.current_verifier_verdict() is not None:
            return True
        if self.active_plan is not None or self.draft_plan is not None:
            return True
        if self.run_brief.acceptance_criteria or self.run_brief.implementation_plan:
            return True
        normalized_intent = normalize_intent_label(self.active_intent)
        if normalized_intent in {
            "author_write",
            "write_file",
            "requested_write_file",
            "requested_file_write",
            "requested_file_append",
            "requested_file_patch",
            "requested_ast_patch",
            "requested_file_delete",
        }:
            return True

        intent_tags = {
            str(tag).strip().lower()
            for tag in (self.intent_tags or [])
            if str(tag).strip()
        }
        if intent_tags & {"write_file", "file_write", "file_patch", "ast_patch", "mutate_repo"}:
            return True

        target_paths = self.scratchpad.get("_task_target_paths")
        has_targets = isinstance(target_paths, list) and any(str(path).strip() for path in target_paths)
        task_bits = [
            str(self.run_brief.original_task or "").strip(),
            str(self.working_memory.current_goal or "").strip(),
        ]
        task_text = " ".join(bit for bit in task_bits if bit).lower()
        if not task_text:
            return False

        write_markers = (
            "write ",
            "edit ",
            "patch ",
            "create ",
            "build ",
            "implement ",
            "update ",
            "append ",
            "replace ",
            "refactor ",
            "fix ",
            "setup ",
            "deploy ",
            "configure ",
        )
        artifact_markers = (
            "script",
            "file",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".json",
            ".md",
            "unittest",
            "test suite",
            "tests",
        )
        is_setup_or_deploy = any(
            marker in task_text for marker in ("setup ", "deploy ", "configure ")
        )
        return is_setup_or_deploy or (
            any(marker in task_text for marker in write_markers)
            and (has_targets or any(marker in task_text for marker in artifact_markers))
        )

    def contract_phase(self) -> str:
        scratch_phase = str(self.scratchpad.get("_contract_phase") or "").strip()
        if scratch_phase:
            return scratch_phase
        plan = self.active_plan or self.draft_plan
        if self.planning_mode_enabled and not (plan and plan.approved):
            return self.current_phase
        if self.repair_cycle_id:
            return "repair"
        if not self.contract_flow_active():
            return self.current_phase
        if self.current_verifier_verdict() is None:
            return "author"
        return "verify" if not self.acceptance_ready() else "execute"

    def upsert_experience(self, memory: ExperienceMemory) -> ExperienceMemory:
        memory.namespace = normalize_memory_namespace(memory.namespace) or infer_memory_namespace(
            tool_name=memory.tool_name,
            intent=memory.intent,
            intent_tags=memory.intent_tags,
            environment_tags=memory.environment_tags,
            entity_tags=memory.entity_tags,
            notes=memory.notes,
        )
        memory.intent_tags = normalize_experience_tags(memory.intent_tags)
        memory.environment_tags = normalize_experience_tags(memory.environment_tags)
        memory.entity_tags = normalize_experience_tags(memory.entity_tags)
        for i, existing in enumerate(self.warm_experiences):
            if existing.memory_id == memory.memory_id:
                self.warm_experiences[i] = memory
                if memory.outcome == "success":
                    self._clear_experience_staleness(memory.memory_id)
                self.touch()
                return memory
        self.warm_experiences.append(memory)
        if memory.outcome == "success":
            self._clear_experience_staleness(memory.memory_id)
        self.touch()
        return memory

    def reinforce_experience(self, memory_id: str, *, success: bool) -> None:
        for memory in self.warm_experiences:
            if memory.memory_id == memory_id:
                memory.reuse_count += 1
                modifier = 0.05 if success else -0.15
                memory.confidence = max(0.0, min(1.0, memory.confidence + modifier))
                memory.last_reinforced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                if success:
                    self._clear_experience_staleness(memory_id)
                self.touch()
                break

    def decay_experiences(self, *, rate: float = 0.005) -> None:
        remaining = []
        now = datetime.now(timezone.utc)
        for memory in self.warm_experiences:
            if memory.pinned:
                remaining.append(memory)
                continue

            expires_at = _coerce_datetime(memory.expires_at)
            if expires_at is not None and expires_at <= now:
                continue

            last_reinforced_at = _coerce_datetime(memory.last_reinforced_at)
            age_penalty = rate
            if last_reinforced_at is not None:
                age_days = max(0.0, (now - last_reinforced_at).total_seconds() / 86400.0)
                age_penalty += min(0.03, age_days * rate)
            else:
                created_at = _coerce_datetime(memory.created_at)
                if created_at is not None:
                    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
                    age_penalty += min(0.02, age_days * rate * 0.5)

            memory.confidence = max(0.0, memory.confidence - age_penalty)
            if memory.confidence > 0.1:
                remaining.append(memory)
        self.warm_experiences = remaining
        self.touch()

    def prune_stale_meta(self, *, limit: int = 15) -> None:
        self.working_memory.known_fact_meta = [
            m for m in self.working_memory.known_fact_meta
            if not memory_entry_is_stale(
                m,
                current_step=self.step_count,
                current_phase=self.current_phase,
                staleness_step_limit=limit,
            )
        ]
        self.working_memory.failure_meta = [
            m for m in self.working_memory.failure_meta
            if not memory_entry_is_stale(
                m,
                current_step=self.step_count,
                current_phase=self.current_phase,
                staleness_step_limit=limit,
            )
        ]
        self.working_memory.next_action_meta = [
            m for m in self.working_memory.next_action_meta
            if not memory_entry_is_stale(
                m,
                current_step=self.step_count,
                current_phase=self.current_phase,
                staleness_step_limit=limit,
            )
        ]
        self.align_meta_to_content()

    def prune_context_staleness_indexes(self) -> None:
        def _prune(key: str, active_ids: set[str]) -> None:
            index = self._lane_staleness_index(key)
            if not index:
                self.scratchpad.pop(key, None)
                return
            filtered = {
                item_id: payload
                for item_id, payload in index.items()
                if item_id in active_ids
            }
            if filtered:
                self.scratchpad[key] = filtered
            else:
                self.scratchpad.pop(key, None)

        _prune(
            "_experience_staleness",
            {str(memory.memory_id).strip() for memory in self.warm_experiences if str(memory.memory_id).strip()},
        )
        _prune(
            "_turn_bundle_staleness",
            {str(bundle.bundle_id).strip() for bundle in self.turn_bundles if str(bundle.bundle_id).strip()},
        )
        _prune(
            "_context_brief_staleness",
            {str(brief.brief_id).strip() for brief in self.context_briefs if str(brief.brief_id).strip()},
        )
        _prune(
            "_summary_staleness",
            {str(summary.summary_id).strip() for summary in self.episodic_summaries if str(summary.summary_id).strip()},
        )
        _prune(
            "_artifact_staleness",
            {str(artifact_id).strip() for artifact_id in self.artifacts if str(artifact_id).strip()},
        )
        _prune(
            "_observation_staleness",
            {
                str(record.evidence_id).strip()
                for record in self.reasoning_graph.evidence_records
                if str(record.evidence_id).strip()
            },
        )

    def align_meta_to_content(self) -> None:
        self.working_memory.known_fact_meta = align_memory_entries(
            self.working_memory.known_facts,
            self.working_memory.known_fact_meta,
            current_step=self.step_count,
            current_phase=self.current_phase,
        )
        self.working_memory.failure_meta = align_memory_entries(
            self.working_memory.failures,
            self.working_memory.failure_meta,
            current_step=self.step_count,
            current_phase=self.current_phase,
        )
        self.working_memory.next_action_meta = align_memory_entries(
            self.working_memory.next_actions,
            self.working_memory.next_action_meta,
            current_step=self.step_count,
            current_phase=self.current_phase,
        )

    def memory_entries(self, section: str) -> list[MemoryEntry]:
        if section == "known_facts":
            return self.working_memory.known_fact_meta
        if section == "failures":
            return self.working_memory.failure_meta
        if section == "next_actions":
            return self.working_memory.next_action_meta
        return []

    def set_memory_entries(self, section: str, entries: list[MemoryEntry]) -> None:
        if section == "known_facts":
            self.working_memory.known_facts = [e.content for e in entries]
            self.working_memory.known_fact_meta = entries
        elif section == "failures":
            self.working_memory.failures = [e.content for e in entries]
            self.working_memory.failure_meta = entries
        elif section == "next_actions":
            self.working_memory.next_actions = [e.content for e in entries]
            self.working_memory.next_action_meta = entries
        self.touch()

    def append_memory_entry(self, section: str, entry: MemoryEntry) -> None:
        entries = self.memory_entries(section)
        entries.append(entry)
        self.set_memory_entries(section, entries)

    def remove_memory_entry(self, section: str, marker: str) -> bool:
        entries = self.memory_entries(section)
        initial_len = len(entries)
        new_entries = [e for e in entries if marker not in e.content]
        if len(new_entries) < initial_len:
            self.set_memory_entries(section, new_entries)
            return True
        return False

    def invalidate_context(
        self,
        *,
        reason: str,
        paths: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reason_label = str(reason or "").strip().lower() or "unspecified"
        path_hints = self._normalize_paths(paths)
        detail_payload = dict(details or {})
        invalidated_facts: list[str] = []
        invalidated_memory_ids: list[str] = []

        def _maybe_invalidate_entry(entry: MemoryEntry, content: str) -> None:
            if entry.freshness == "pinned":
                return
            if content and content not in invalidated_facts:
                invalidated_facts.append(content)
            self._invalidate_memory_entry(entry)

        known_entries = self.memory_entries("known_facts")
        for idx, entry in enumerate(known_entries):
            content = entry.content
            should_invalidate = False
            if reason_label in {"file_changed", "write_session_target_changed"} and path_hints:
                should_invalidate = self._text_matches_any_path(content, path_hints)
            elif reason_label == "phase_advanced":
                should_invalidate = str(entry.created_phase or "").strip() != str(self.current_phase or "").strip()
            elif reason_label == "verifier_failed":
                lowered = str(content or "").lower()
                should_invalidate = any(token in lowered for token in ("pass", "verified", "success", "fixed"))
            elif reason_label == "environment_changed":
                should_invalidate = bool(entry.created_phase and entry.created_phase != self.current_phase)
            if should_invalidate:
                _maybe_invalidate_entry(entry, content)
                known_entries[idx] = entry
        self.set_memory_entries("known_facts", known_entries)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        invalidated_turn_bundle_ids: list[str] = []
        invalidated_brief_ids: list[str] = []
        invalidated_summary_ids: list[str] = []
        invalidated_artifact_ids: list[str] = []
        invalidated_observation_ids: list[str] = []

        for memory in self.warm_experiences:
            should_downgrade = False
            if reason_label in {"file_changed", "write_session_target_changed"} and path_hints:
                should_downgrade = self._text_matches_any_path(memory.notes, path_hints)
            elif reason_label == "verifier_failed":
                should_downgrade = memory.outcome == "success"
            elif reason_label == "phase_advanced":
                should_downgrade = bool(memory.phase and memory.phase != self.current_phase)
            elif reason_label == "environment_changed":
                phase_tag = f"phase_{self.current_phase}".lower()
                env_tags = {str(tag).strip().lower() for tag in (memory.environment_tags or []) if str(tag).strip()}
                should_downgrade = bool(env_tags) and phase_tag not in env_tags
            if not should_downgrade:
                continue
            memory.confidence = max(0.0, min(1.0, float(memory.confidence or 0.0) - 0.2))
            memory.last_reinforced_at = now
            if memory.memory_id:
                invalidated_memory_ids.append(memory.memory_id)
                self._mark_lane_stale(
                    key="_experience_staleness",
                    item_id=memory.memory_id,
                    reason=reason_label,
                    paths=path_hints,
                    now=now,
                )

        for bundle in self.turn_bundles:
            bundle_id = str(getattr(bundle, "bundle_id", "") or "").strip()
            if not bundle_id:
                continue
            should_mark_stale = False
            if reason_label in {"file_changed", "write_session_target_changed"} and path_hints:
                should_mark_stale = any(
                    self._text_matches_any_path(str(path), path_hints)
                    for path in (getattr(bundle, "files_touched", []) or [])
                )
            elif reason_label in {"phase_advanced", "environment_changed"}:
                should_mark_stale = bool(
                    str(getattr(bundle, "phase", "") or "").strip()
                    and str(getattr(bundle, "phase", "") or "").strip() != str(self.current_phase or "").strip()
                )
            elif reason_label == "verifier_failed":
                should_mark_stale = any(
                    self._is_optimistic_statement(str(line))
                    for line in (getattr(bundle, "summary_lines", []) or [])
                )
            if should_mark_stale:
                invalidated_turn_bundle_ids.append(bundle_id)
                self._mark_lane_stale(
                    key="_turn_bundle_staleness",
                    item_id=bundle_id,
                    reason=reason_label,
                    paths=path_hints,
                    now=now,
                )

        for brief in self.context_briefs:
            brief_id = str(getattr(brief, "brief_id", "") or "").strip()
            if not brief_id:
                continue
            should_mark_stale = False
            if reason_label in {"file_changed", "write_session_target_changed"} and path_hints:
                should_mark_stale = any(
                    self._text_matches_any_path(str(path), path_hints)
                    for path in (getattr(brief, "files_touched", []) or [])
                )
            elif reason_label in {"phase_advanced", "environment_changed"}:
                should_mark_stale = bool(
                    str(getattr(brief, "current_phase", "") or "").strip()
                    and str(getattr(brief, "current_phase", "") or "").strip() != str(self.current_phase or "").strip()
                )
            elif reason_label == "verifier_failed":
                should_mark_stale = any(
                    self._is_optimistic_statement(str(line))
                    for line in ((getattr(brief, "key_discoveries", []) or []) + (getattr(brief, "new_facts", []) or []))
                )
            if should_mark_stale:
                invalidated_brief_ids.append(brief_id)
                self._mark_lane_stale(
                    key="_context_brief_staleness",
                    item_id=brief_id,
                    reason=reason_label,
                    paths=path_hints,
                    now=now,
                )

        failure_mode = str(getattr(self, "last_failure_class", "") or "").strip().lower()
        for summary in self.episodic_summaries:
            summary_id = str(getattr(summary, "summary_id", "") or "").strip()
            if not summary_id:
                continue
            should_mark_stale = False
            if reason_label in {"file_changed", "write_session_target_changed"} and path_hints:
                should_mark_stale = any(
                    self._text_matches_any_path(str(path), path_hints)
                    for path in (getattr(summary, "files_touched", []) or [])
                )
            elif reason_label == "verifier_failed":
                notes = [str(line) for line in (getattr(summary, "notes", []) or [])]
                failed_lines = [str(line).strip().lower() for line in (getattr(summary, "failed_approaches", []) or [])]
                should_mark_stale = any(self._is_optimistic_statement(line) for line in notes)
                if failure_mode and any(failure_mode in line for line in failed_lines):
                    should_mark_stale = True
            if should_mark_stale:
                invalidated_summary_ids.append(summary_id)
                self._mark_lane_stale(
                    key="_summary_staleness",
                    item_id=summary_id,
                    reason=reason_label,
                    paths=path_hints,
                    now=now,
                )

        for artifact_id, artifact in self.artifacts.items():
            normalized_artifact_id = str(artifact_id or "").strip()
            if not normalized_artifact_id:
                continue
            metadata = getattr(artifact, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            should_mark_stale = False
            if reason_label in {"file_changed", "write_session_target_changed"} and path_hints:
                artifact_paths: list[str] = []
                source = str(getattr(artifact, "source", "") or "").strip()
                if source:
                    artifact_paths.append(source)
                for key in ("path", "target_path", "write_target_path"):
                    value = str(metadata.get(key) or "").strip()
                    if value:
                        artifact_paths.append(value)
                for path_tag in (getattr(artifact, "path_tags", []) or []):
                    value = str(path_tag or "").strip()
                    if value:
                        artifact_paths.append(value)
                should_mark_stale = any(self._text_matches_any_path(candidate, path_hints) for candidate in artifact_paths)
            elif reason_label in {"phase_advanced", "environment_changed"}:
                metadata_phase = str(metadata.get("phase") or metadata.get("created_phase") or "").strip()
                should_mark_stale = bool(metadata_phase and metadata_phase != str(self.current_phase or "").strip())
            elif reason_label == "verifier_failed":
                should_mark_stale = str(metadata.get("verifier_verdict") or "").strip().lower() == "pass"
            if should_mark_stale:
                invalidated_artifact_ids.append(normalized_artifact_id)
                self._mark_lane_stale(
                    key="_artifact_staleness",
                    item_id=normalized_artifact_id,
                    reason=reason_label,
                    paths=path_hints,
                    now=now,
                )

        for evidence in self.reasoning_graph.evidence_records:
            evidence_id = str(getattr(evidence, "evidence_id", "") or "").strip()
            if not evidence_id:
                continue
            metadata = getattr(evidence, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            should_mark_stale = False
            if reason_label in {"file_changed", "write_session_target_changed"} and path_hints:
                path_candidates = [
                    str(getattr(evidence, "statement", "") or ""),
                    str(getattr(evidence, "source", "") or ""),
                    str(metadata.get("path") or ""),
                    str(metadata.get("target_path") or ""),
                ]
                should_mark_stale = any(self._text_matches_any_path(candidate, path_hints) for candidate in path_candidates)
            elif reason_label in {"phase_advanced", "environment_changed"}:
                evidence_phase = str(getattr(evidence, "phase", "") or "").strip()
                should_mark_stale = bool(evidence_phase and evidence_phase != str(self.current_phase or "").strip())
                if should_mark_stale:
                    created_at_step = int(getattr(evidence, "created_at_step", 0) or 0)
                    if created_at_step >= max(0, self.step_count - 1):
                        should_mark_stale = False
            elif reason_label == "verifier_failed":
                should_mark_stale = (
                    str(metadata.get("verifier_verdict") or "").strip().lower() == "pass"
                    or self._is_optimistic_statement(str(getattr(evidence, "statement", "") or ""))
                )
            if should_mark_stale:
                invalidated_observation_ids.append(evidence_id)
                self._mark_lane_stale(
                    key="_observation_staleness",
                    item_id=evidence_id,
                    reason=reason_label,
                    paths=path_hints,
                    now=now,
                )

        self.prune_context_staleness_indexes()

        invalidation_event: dict[str, Any] = {
            "reason": reason_label,
            "step": int(self.step_count),
            "phase": str(self.current_phase or ""),
            "paths": path_hints,
            "invalidated_facts": invalidated_facts[:8],
            "invalidated_fact_count": len(invalidated_facts),
            "invalidated_memory_ids": invalidated_memory_ids[:8],
            "invalidated_memory_count": len(invalidated_memory_ids),
            "invalidated_turn_bundle_ids": invalidated_turn_bundle_ids[:8],
            "invalidated_turn_bundle_count": len(invalidated_turn_bundle_ids),
            "invalidated_brief_ids": invalidated_brief_ids[:8],
            "invalidated_brief_count": len(invalidated_brief_ids),
            "invalidated_summary_ids": invalidated_summary_ids[:8],
            "invalidated_summary_count": len(invalidated_summary_ids),
            "invalidated_artifact_ids": invalidated_artifact_ids[:8],
            "invalidated_artifact_count": len(invalidated_artifact_ids),
            "invalidated_observation_ids": invalidated_observation_ids[:8],
            "invalidated_observation_count": len(invalidated_observation_ids),
            "details": detail_payload,
            "created_at": now,
        }
        history = self.scratchpad.get("_context_invalidations")
        if not isinstance(history, list):
            history = []
        history.append(invalidation_event)
        self.scratchpad["_context_invalidations"] = history[-40:]

        if invalidated_facts:
            queued = self.scratchpad.get("_invalidated_facts_queue")
            if not isinstance(queued, list):
                queued = []
            for fact in invalidated_facts:
                if fact not in queued:
                    queued.append(fact)
            self.scratchpad["_invalidated_facts_queue"] = queued[-25:]

        state_change = detail_payload.get("state_change")
        if state_change:
            state_changes = self.scratchpad.get("_state_change_queue")
            if not isinstance(state_changes, list):
                state_changes = []
            state_changes.append(str(state_change))
            self.scratchpad["_state_change_queue"] = state_changes[-25:]

        self.touch()
        return invalidation_event

    @property
    def conversation_history(self) -> list[ConversationMessage]:
        # Compatibility alias for older code and checkpoint consumers.
        return list(self.recent_messages)
