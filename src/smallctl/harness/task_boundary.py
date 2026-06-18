from __future__ import annotations

import json
import logging
import re
from time import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..interrupt_replies import is_interrupt_response, is_plan_approval_reply
from ..remote_scope import handoff_supports_remote_continuation, task_matches_remote_continuation
from ..models.conversation import ConversationMessage
from ..recovery_metrics import record_failure_event_metric
from ..state import (
    EpisodicSummary,
    PromptBudgetSnapshot,
    RunBrief,
    WorkingMemory,
    json_safe_value,
)
from ..recovery_schema import FailureEvent
from ..task_targets import extract_task_target_paths
from ..normalization import dedupe_keep_tail
from ..redaction import redact_sensitive_data
from ..state_memory import trim_recent_messages
from ..state_support import clip_text_value
from .followup_signals import (
    assistant_message_proposes_concrete_implementation,
    is_affirmative_followup,
    recent_assistant_requested_action_confirmation,
)
from .task_boundary_followups import (
    has_plan_execution_approval_context as _has_plan_execution_approval_context,
    is_continue_like_followup as _is_continue_like_followup,
)
from .task_classifier import (
    classify_task_mode,
    looks_like_write_file_request,
    looks_like_write_patch_request,
)
from .task_intent import derive_task_contract, next_action_for_task
from .task_transactions import (
    FollowupClassification,
    FollowupSignals,
    classify_followup_transaction,
)
from .task_boundary_summary import (
    clip_task_summary_text,
    extract_task_terminal_message,
    task_duration_seconds,
)
from .task_boundary_support import (
    base_task_from_task_chain,
    blocks_inherited_target,
    canonicalize_inline_task_wrapper,
    clean_option_title,
    collapse_task_chain,
    coerce_remote_target,
    extract_action_options_from_text,
    format_remote_target,
    is_remote_followup_wrapper,
    merge_action_options,
    merge_remote_targets,
    normalize_remote_host,
    normalize_task_text,
    parse_inline_task_wrapper,
    suspicious_remote_target_reason,
)
from .task_boundary_helpers import (
    extract_remote_absolute_paths,
    guard_trip_preserves_artifact,
    guard_trip_repeated_tool,
    normalize_target_path,
    ordinal_followup_index,
    remote_target_matches_known_target,
    strip_ordinal_prefix,
    target_paths_overlap,
)
from .task_boundary_semantic_tail import (
    message_is_semantic_tail_candidate,
    semantic_recent_tail_messages,
)
from .task_boundary_constants import (
    _ACTION_CONFIRMATION_PROMPTS,
    _AFFIRMATIVE_REMOTE_CONTINUATION_TEXT,
    _CONTEXTUAL_REFERENCE_RE,
    _CONTINUATION_ACTION_LEAD_RE,
    _CONTINUE_DIRECTIVE_RE,
    _CORRECTIVE_RESTEER_RE,
    _CORRECTIVE_TOOL_NAMES,
    _FOLLOWUP_ACTION_RE,
    _FOLLOWUP_FILLERS,
    _GENERIC_EDIT_LEAD_RE,
    _GENERIC_TARGET_RE,
    _GUARD_FAILURE_RE,
    _GUARD_RECOVERY_NUDGE_RE,
    _INLINE_NUMBERED_OPTION_RE,
    _INLINE_USER_WRAP_MARKER_RE,
    _IPV4_HOST_RE,
    _MARKDOWN_OPTION_RE,
    _NUMBERED_OPTION_RE,
    _ORDINAL_FOLLOWUP_RE,
    _ORDINAL_PREFIX_RE,
    _ORDINAL_WORDS,
    _ORDINAL_WORD_FOLLOWUP_RE,
    _OPTION_ACTION_WORDS,
    _QUALITY_FOLLOWUP_RE,
    _QUALITY_TARGET_RE,
    _REMOTE_ABSOLUTE_PATH_RE,
    _REMOTE_CLARIFICATION_PHRASES,
    _REMOTE_CORRECTIVE_CLEANUP_PHRASES,
    _REMOTE_DEPLOYMENT_CONTEXT_TARGETS,
    _REMOTE_DIAGNOSTIC_HINTS,
    _REMOTE_DIAGNOSTIC_QUESTION_RE,
    _REMOTE_DIAGNOSTIC_TARGETS,
    _REMOTE_LIVE_CORRECTION_HINTS,
    _REMOTE_LIVE_CORRECTION_PHRASES,
    _REMOTE_OPERATIONAL_TARGETS,
    _REMOTE_OPERATIONAL_VERBS,
    _REMOTE_PERMISSION_FOLLOWUP_RE,
    _REMOTE_RESIDUE_MARKERS,
    _REMOTE_SCRIPT_HINT_RE,
    _REMOTE_SITE_MUTATION_ACTION_RE,
    _REMOTE_SITE_MUTATION_TARGET_RE,
    _RESEARCH_CONTEXT_RE,
    _RETRY_FOLLOWUP_RE,
    _SEMANTIC_RECENT_TAIL_TOKEN_CAP,
    _SEQUENTIAL_REMOTE_FOLLOWUP_RE,
    _TARGET_LANGUAGE_RE,
    _TARGET_NEGATION_RE,
    _TARGET_REPLACEMENT_RE,
    _TASK_BOUNDARY_GUARD_SCRATCHPAD_KEYS,
    _USER_AT_HOST_RE,
    _WEB_RESEARCH_DIRECTIVE_RE,
)
from .task_boundary_remote_mixin import TaskBoundaryRemoteMixin
from .task_boundary_lifecycle_mixin import TaskBoundaryLifecycleMixin
from .task_boundary_classification_mixin import TaskBoundaryClassificationMixin
from .task_boundary_handoff_mixin import TaskBoundaryHandoffMixin


class TaskBoundaryService(
    TaskBoundaryRemoteMixin,
    TaskBoundaryLifecycleMixin,
    TaskBoundaryClassificationMixin,
    TaskBoundaryHandoffMixin,
):
    def __init__(self, harness: Any):
        self.harness = harness

    def _consume_session_restored_flag(self) -> bool:
        scratchpad = getattr(getattr(self.harness, "state", None), "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return False
        return bool(scratchpad.pop("_session_restored", False))

    def _active_task_scope_payload(self) -> dict[str, Any] | None:
        payload = getattr(self.harness, "_active_task_scope", None)
        if isinstance(payload, dict) and payload:
            return dict(payload)
        state = getattr(self.harness, "state", None)
        scratchpad = getattr(state, "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return None
        stored = scratchpad.get("_active_task_scope")
        if not isinstance(stored, dict) or not stored:
            return None
        restored = dict(stored)
        self.harness._active_task_scope = restored
        sequence = stored.get("sequence")
        try:
            restored_sequence = int(sequence)
        except (TypeError, ValueError):
            restored_sequence = 0
        current_sequence = int(getattr(self.harness, "_task_sequence", 0) or 0)
        if restored_sequence > current_sequence:
            self.harness._task_sequence = restored_sequence
        return restored

    def _clip_task_summary_text(self, value: Any, *, limit: int = 240) -> str:
        return clip_task_summary_text(value, limit=limit)

    def _extract_task_terminal_message(self, result: dict[str, Any] | None) -> str:
        return extract_task_terminal_message(result)

    def _task_duration_seconds(self, started_at: str, finished_at: str) -> float:
        return task_duration_seconds(started_at, finished_at)

    def _write_task_summary(self, payload: dict[str, Any]) -> str:
        summary_path_text = str(payload.get("summary_path") or "").strip()
        if not summary_path_text:
            return ""
        path = Path(summary_path_text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(redact_sensitive_data(json_safe_value(payload)), indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            logger = getattr(self.harness, "log", logging.getLogger("smallctl.harness"))
            if logger is not None:
                logger.exception("failed to write task summary")
            return ""

    def _append_task_episodic_summary(self, payload: dict[str, Any]) -> None:
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            return
        existing_ids = {
            str(getattr(summary, "summary_id", "") or "").strip()
            for summary in getattr(self.harness.state, "episodic_summaries", []) or []
        }
        summary_id = f"{task_id}-summary"
        if summary_id in existing_ids:
            return

        task_text = self._clip_task_summary_text(
            payload.get("effective_task") or payload.get("raw_task"),
            limit=180,
        )
        message = self._clip_task_summary_text(payload.get("message"), limit=180)
        reason = self._clip_task_summary_text(payload.get("reason"), limit=140)
        full_reason = str(payload.get("reason") or payload.get("message") or "").strip()
        status = str(payload.get("status") or "").strip()
        result_status = str(payload.get("result_status") or "").strip()
        is_guard_trip = status == "failed" and "guard tripped:" in full_reason.lower()

        # Classify outcome more honestly for future context retrieval.
        if status == "completed" and result_status == "completed":
            outcome_status = "completed"
        elif status == "completed" and result_status in {"failed", "stopped", "error"}:
            outcome_status = "failed"
        elif status == "aborted":
            outcome_status = "blocked"
        elif status == "failed" or is_guard_trip:
            outcome_status = "failed"
        else:
            outcome_status = status or "stopped"

        notes = [f"Task {task_id} {outcome_status}: {task_text}".strip()]
        if message:
            notes.append(message)
        elif reason:
            notes.append(f"Reason: {reason}")

        artifacts = [str(item).strip() for item in (payload.get("artifact_ids") or []) if str(item).strip()]
        if not artifacts:
            count = int(payload.get("artifact_count") or 0)
            if count:
                artifacts = list((getattr(self.harness.state, "artifacts", {}) or {}).keys())[-min(count, 5):]

        decisions = [f"status={outcome_status}"] if outcome_status else []
        remaining_plan: list[str] = []
        if payload.get("replacement_task"):
            remaining_plan.append(str(payload.get("replacement_task") or "").strip())
        if is_guard_trip:
            decisions.append("guard_trip_recovery")
            remaining_plan.append("Continue from preserved progress; do not retry the repeated tool call.")

        self.harness.state.episodic_summaries.append(
            EpisodicSummary(
                summary_id=summary_id,
                created_at=str(payload.get("finished_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
                decisions=decisions,
                files_touched=[
                    str(path).strip()
                    for path in (payload.get("target_paths") or [])
                    if str(path).strip()
                ],
                failed_approaches=[reason] if status in {"aborted", "failed"} and reason else [],
                remaining_plan=[item for item in remaining_plan if item],
                artifact_ids=artifacts,
                notes=notes,
                full_summary_artifact_id=None,
            )
        )
        self.harness.state.episodic_summaries = self.harness.state.episodic_summaries[-12:]
        if outcome_status == "failed" and task_text:
            normalized_task_text = normalize_task_text(task_text)
            if normalized_task_text not in {"status", "status?", "continue"}:
                self.harness.state.scratchpad["_last_failed_continuation_task"] = {
                    "task": task_text,
                    "reason": self._clip_task_summary_text(full_reason, limit=260),
                    "task_id": task_id,
                    "updated_at": str(payload.get("finished_at") or ""),
                }
        if is_guard_trip:
            self._record_guard_trip_recovery_context(
                summary_id=summary_id,
                artifact_ids=artifacts,
                reason=full_reason,
                task_text=task_text,
            )

    def _record_guard_trip_recovery_context(
        self,
        *,
        summary_id: str,
        artifact_ids: list[str],
        reason: str,
        task_text: str,
    ) -> None:
        state = self.harness.state
        scratchpad = state.scratchpad
        protected_summaries = [
            str(item).strip()
            for item in (scratchpad.get("_guard_trip_preserved_summary_ids") or [])
            if str(item).strip()
        ]
        if summary_id and summary_id not in protected_summaries:
            protected_summaries.append(summary_id)
        scratchpad["_guard_trip_preserved_summary_ids"] = protected_summaries[-8:]
        state._clear_lane_staleness("_summary_staleness", summary_id)

        protected_artifacts = [
            str(item).strip()
            for item in (scratchpad.get("_guard_trip_preserved_artifact_ids") or [])
            if str(item).strip()
        ]
        protected_observations = [
            str(item).strip()
            for item in (scratchpad.get("_guard_trip_preserved_observation_ids") or [])
            if str(item).strip()
        ]
        artifacts_by_id = getattr(state, "artifacts", {}) if isinstance(getattr(state, "artifacts", {}), dict) else {}
        for artifact_id in artifact_ids:
            artifact = artifacts_by_id.get(artifact_id)
            if not self._guard_trip_preserves_artifact(artifact):
                continue
            if artifact_id not in protected_artifacts:
                protected_artifacts.append(artifact_id)
            state._clear_lane_staleness("_artifact_staleness", artifact_id)
            metadata = getattr(artifact, "metadata", {}) if artifact is not None else {}
            if isinstance(metadata, dict):
                evidence_id = str(metadata.get("evidence_id") or "").strip()
                if evidence_id:
                    if evidence_id not in protected_observations:
                        protected_observations.append(evidence_id)
                    state._clear_lane_staleness("_observation_staleness", evidence_id)
        if protected_artifacts:
            scratchpad["_guard_trip_preserved_artifact_ids"] = protected_artifacts[-12:]
        if protected_observations:
            scratchpad["_guard_trip_preserved_observation_ids"] = protected_observations[-12:]

        repeated_tool = self._guard_trip_repeated_tool(reason)
        scratchpad["_guard_trip_recovery_capsule"] = {
            "created_at_step": int(getattr(state, "step_count", 0) or 0),
            "summary_id": summary_id,
            "failed_tool": repeated_tool,
            "reason": self._clip_task_summary_text(reason, limit=220),
            "goal": task_text,
            "preserved_artifact_ids": protected_artifacts[-6:],
        }

    @staticmethod
    def _guard_trip_preserves_artifact(artifact: Any) -> bool:
        return guard_trip_preserves_artifact(artifact)

    @staticmethod
    def _guard_trip_repeated_tool(reason: str) -> str:
        return guard_trip_repeated_tool(reason)

    @staticmethod
    def _normalize_target_path(value: Any) -> str:
        return normalize_target_path(value)

    def _target_paths_overlap(self, left: list[str], right: list[str]) -> bool:
        return target_paths_overlap(left, right)

    def _known_target_paths(self) -> list[str]:
        paths: list[str] = []
        handoff = self.last_task_handoff()
        for source in (
            handoff.get("target_paths"),
            handoff.get("remote_target_paths"),
            self.harness.state.scratchpad.get("_task_target_paths"),
            getattr(self.harness, "_active_task_scope", {}).get("target_paths")
            if isinstance(getattr(self.harness, "_active_task_scope", None), dict)
            else [],
        ):
            if not isinstance(source, list):
                continue
            paths.extend(str(path).strip() for path in source if str(path).strip())
        return dedupe_keep_tail(paths, limit=12)

    def _extract_remote_absolute_paths(self, *texts: Any) -> list[str]:
        return extract_remote_absolute_paths(*texts)

    def _recent_remote_target_paths(self, *, handoff: dict[str, Any] | None = None) -> list[str]:
        candidates: list[str] = []
        payload = handoff if isinstance(handoff, dict) else self.last_task_handoff()
        stored_paths = payload.get("remote_target_paths") if isinstance(payload, dict) else None
        if isinstance(stored_paths, list):
            candidates.extend(str(path).strip() for path in stored_paths if str(path).strip())

        for record in reversed(self._tool_execution_record_items()):
            if str(record.get("tool_name") or "").strip() != "ssh_exec":
                continue
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            if not (
                bool(result.get("success"))
                or bool(metadata.get("ssh_transport_succeeded"))
                or str(metadata.get("failure_kind") or "").strip() == "remote_command"
            ):
                continue
            args = record.get("args")
            if isinstance(args, dict):
                candidates.extend(self._extract_remote_absolute_paths(args.get("command")))
            candidates.extend(self._extract_remote_absolute_paths(metadata.get("command")))

        return dedupe_keep_tail(candidates, limit=12)

    def _session_ssh_targets(self) -> list[dict[str, str]]:
        scratchpad = getattr(self.harness.state, "scratchpad", {})
        targets = scratchpad.get("_session_ssh_targets")
        if not isinstance(targets, dict):
            return []
        collected: list[dict[str, str]] = []
        for key, value in targets.items():
            if not isinstance(value, dict):
                continue
            host = normalize_remote_host(value.get("host") or key)
            if not host:
                continue
            collected.append({"host": host, "user": str(value.get("user") or "").strip()})
        return merge_remote_targets(collected)

    def _confirmed_session_ssh_targets(self) -> list[dict[str, str]]:
        scratchpad = getattr(self.harness.state, "scratchpad", {})
        targets = scratchpad.get("_session_ssh_targets")
        if not isinstance(targets, dict):
            return []
        collected: list[dict[str, str]] = []
        for key, value in targets.items():
            if not isinstance(value, dict) or not bool(value.get("confirmed")):
                continue
            host = normalize_remote_host(value.get("host") or key)
            if not host:
                continue
            collected.append({"host": host, "user": str(value.get("user") or "").strip()})
        return merge_remote_targets(collected)

    def _remote_targets_from_texts(self, *texts: Any) -> list[dict[str, str]]:
        collected: list[dict[str, str]] = []
        for text_value in texts:
            text = str(text_value or "").strip()
            if not text:
                continue
            seen_hosts: set[str] = set()
            for match in _USER_AT_HOST_RE.finditer(text):
                host = normalize_remote_host(match.group("host"))
                user = str(match.group("user") or "").strip()
                if not host:
                    continue
                reason = suspicious_remote_target_reason(host=host, user=user)
                prefix = text[max(0, match.start() - 24):match.start()].lower()
                suffix = text[match.end(): min(len(text), match.end() + 8)]
                if not reason and "password" in prefix and (suffix.startswith('"') or suffix.startswith("'")):
                    reason = "password_context"
                if reason:
                    self.harness._runlog(
                        "ssh_target_candidate_rejected",
                        "rejected suspicious SSH target candidate from task text",
                        candidate=f"{user}@{host}" if user else host,
                        reason=reason,
                    )
                    continue
                seen_hosts.add(host)
                collected.append({"host": host, "user": user})
            for match in _IPV4_HOST_RE.finditer(text):
                host = normalize_remote_host(match.group(0))
                if not host or host in seen_hosts:
                    continue
                seen_hosts.add(host)
                collected.append({"host": host, "user": ""})
        return merge_remote_targets(collected)

    @staticmethod
    def _remote_target_matches_known_target(
        candidate: dict[str, Any],
        known_targets: list[dict[str, str]],
    ) -> bool:
        return remote_target_matches_known_target(candidate, known_targets)

    def _handoff_remote_targets(self, handoff: dict[str, Any]) -> list[dict[str, str]]:
        stored_targets = handoff.get("ssh_targets")
        if isinstance(stored_targets, list):
            coerced = merge_remote_targets(stored_targets)
            if coerced:
                return coerced

        session_by_host = {
            target["host"]: target
            for target in self._session_ssh_targets()
            if target.get("host")
        }
        inferred = self._remote_targets_from_texts(
            handoff.get("effective_task"),
            handoff.get("current_goal"),
            handoff.get("raw_task"),
        )
        if inferred:
            enriched: list[dict[str, str]] = []
            for target in inferred:
                session_target = session_by_host.get(target["host"])
                if session_target is not None:
                    enriched.append(session_target)
                else:
                    enriched.append(target)
            return merge_remote_targets(enriched)

        if str(handoff.get("task_mode") or "").strip() == "remote_execute":
            session_targets = self._session_ssh_targets()
            if len(session_targets) == 1:
                return session_targets
        return []

    def _handoff_mentions_remote_context(self, handoff: dict[str, Any]) -> bool:
        texts = (
            handoff.get("effective_task"),
            handoff.get("current_goal"),
            handoff.get("raw_task"),
        )
        for text_value in texts:
            text = str(text_value or "").strip().lower()
            if not text:
                continue
            if "ssh" in text or "remote host" in text or "remote" in text:
                return True
            if _USER_AT_HOST_RE.search(text) or _IPV4_HOST_RE.search(text):
                return True
        return False

    def _handoff_has_remote_context(self, handoff: dict[str, Any]) -> bool:
        task_mode = str(handoff.get("task_mode") or "").strip()
        if task_mode == "remote_execute":
            return True
        ssh_target = handoff.get("ssh_target")
        if isinstance(ssh_target, dict) and str(ssh_target.get("host") or "").strip():
            return True
        ssh_targets = handoff.get("ssh_targets")
        if isinstance(ssh_targets, list) and any(
            isinstance(target, dict) and str(target.get("host") or "").strip()
            for target in ssh_targets
        ):
            return True
        remote_target_paths = handoff.get("remote_target_paths")
        if isinstance(remote_target_paths, list) and any(str(path).strip() for path in remote_target_paths):
            return True
        next_required_tool = handoff.get("next_required_tool")
        if isinstance(next_required_tool, dict) and str(next_required_tool.get("tool_name") or "").strip() == "ssh_exec":
            return True
        if task_mode == "debug_inspect" and (
            self._handoff_mentions_remote_context(handoff) or bool(self._handoff_remote_targets(handoff))
        ):
            return True

        inferred_mode = classify_task_mode(
            str(handoff.get("effective_task") or handoff.get("current_goal") or handoff.get("raw_task") or "")
        )
        if inferred_mode == "remote_execute":
            return True
        return inferred_mode == "debug_inspect" and (
            self._handoff_mentions_remote_context(handoff) or bool(self._handoff_remote_targets(handoff))
        )



















    def has_task_local_context(self) -> bool:
        return self.has_resettable_context() or self.has_durable_context()

    def has_resettable_context(self) -> bool:
        return bool(
            self.harness.state.recent_messages
            or self.harness.state.recent_errors
            or self.harness.state.run_brief.task_contract
            or self.harness.state.run_brief.current_phase_objective
            or self.harness.state.working_memory.current_goal
            or self.harness.state.working_memory.plan
            or self.harness.state.working_memory.open_questions
            or self.harness.state.working_memory.next_actions
            or self.harness.state.acceptance_ledger
            or self.harness.state.acceptance_waivers
            or self.harness.state.scratchpad.get("_task_complete")
            or self.harness.state.scratchpad.get("_task_failed")
            or self.harness.state.scratchpad.get("_tool_attempt_history")
        )

    def has_durable_context(self) -> bool:
        return bool(
            self.harness.state.artifacts
            or self.harness.state.episodic_summaries
            or self.harness.state.context_briefs
            or self.harness.state.working_memory.decisions
            or self.harness.state.working_memory.known_facts
            or self.harness.state.working_memory.failures
        )

    def last_task_handoff(self) -> dict[str, Any]:
        payload = self.harness.state.scratchpad.get("_last_task_handoff")
        if not isinstance(payload, dict):
            return {}
        return dict(payload)


    def _is_continue_like_followup(self, task: str) -> bool:
        return _is_continue_like_followup(task)


    def _is_affirmative_followup(self, task: str) -> bool:
        return is_affirmative_followup(task, fillers=_FOLLOWUP_FILLERS)

    def _recent_assistant_requested_action_confirmation(self) -> bool:
        return recent_assistant_requested_action_confirmation(
            list(self.harness.state.recent_messages),
            prompts=_ACTION_CONFIRMATION_PROMPTS,
        )


    def _has_plan_execution_approval_context(self) -> bool:
        return _has_plan_execution_approval_context(self.harness.state)

























    def _message_is_semantic_tail_candidate(self, message: Any) -> bool:
        return message_is_semantic_tail_candidate(message)

    def _semantic_recent_tail_messages(self, *, token_cap: int) -> list[Any]:
        return semantic_recent_tail_messages(self.harness.state.recent_messages, token_cap=token_cap)

    def _strip_ordinal_prefix(self, task: str) -> str:
        return strip_ordinal_prefix(task)












