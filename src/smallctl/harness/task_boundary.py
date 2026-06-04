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
    clip_text_value,
    json_safe_value,
)
from ..recovery_schema import FailureEvent
from ..task_targets import extract_task_target_paths
from ..normalization import dedupe_keep_tail
from ..state_memory import trim_recent_messages
from .followup_signals import (
    assistant_message_proposes_concrete_implementation,
    is_affirmative_followup,
    recent_assistant_requested_action_confirmation,
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


class TaskBoundaryService:
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
        text = str(value or "").strip()
        if not text:
            return ""
        clipped, truncated = clip_text_value(text, limit=limit)
        return f"{clipped} [truncated]" if truncated else clipped

    def _extract_task_terminal_message(self, result: dict[str, Any] | None) -> str:
        if not isinstance(result, dict) or not result:
            return ""
        message = result.get("message")
        if isinstance(message, dict):
            candidate = (
                message.get("message")
                or message.get("question")
                or message.get("status")
            )
            if candidate:
                return self._clip_task_summary_text(candidate)
        if isinstance(message, str) and message.strip():
            return self._clip_task_summary_text(message)
        reason = str(result.get("reason") or "").strip()
        if reason:
            return self._clip_task_summary_text(reason)
        error = result.get("error")
        if isinstance(error, dict):
            candidate = error.get("message")
            if candidate:
                return self._clip_task_summary_text(candidate)
        return ""

    def _task_duration_seconds(self, started_at: str, finished_at: str) -> float:
        try:
            started = datetime.fromisoformat(str(started_at))
            finished = datetime.fromisoformat(str(finished_at))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, round((finished - started).total_seconds(), 3))

    def _write_task_summary(self, payload: dict[str, Any]) -> str:
        summary_path_text = str(payload.get("summary_path") or "").strip()
        if not summary_path_text:
            return ""
        path = Path(summary_path_text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(json_safe_value(payload), indent=2, ensure_ascii=True) + "\n",
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
        is_guard_trip = status == "failed" and "guard tripped:" in full_reason.lower()
        notes = [f"Task {task_id} {status}: {task_text}".strip()]
        if message:
            notes.append(message)
        elif reason:
            notes.append(f"Reason: {reason}")

        artifacts = [str(item).strip() for item in (payload.get("artifact_ids") or []) if str(item).strip()]
        if not artifacts:
            count = int(payload.get("artifact_count") or 0)
            if count:
                artifacts = list((getattr(self.harness.state, "artifacts", {}) or {}).keys())[-min(count, 5):]

        decisions = [f"status={status}"] if status else []
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
                if not host:
                    continue
                seen_hosts.add(host)
                collected.append({"host": host, "user": str(match.group("user") or "").strip()})
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

    def _handoff_mentions_remote_deployment_context(self, handoff: dict[str, Any]) -> bool:
        texts = (
            handoff.get("effective_task"),
            handoff.get("current_goal"),
            handoff.get("raw_task"),
        )
        for text_value in texts:
            text = str(text_value or "").strip().lower()
            if not text:
                continue
            if any(target in text for target in _REMOTE_DEPLOYMENT_CONTEXT_TARGETS):
                return True
        return False

    def _looks_like_remote_operational_followup(self, task: str) -> bool:
        text = str(task or "").strip().lower()
        if not text:
            return False
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        if extract_task_target_paths(text) or self._extract_remote_absolute_paths(text):
            return False
        has_remote_verb = any(verb in text for verb in _REMOTE_OPERATIONAL_VERBS)
        has_remote_target = any(target in text for target in _REMOTE_OPERATIONAL_TARGETS)
        return has_remote_verb and has_remote_target

    def _looks_like_remote_clarification_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").strip().lower()
        if not text:
            return False
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        if extract_task_target_paths(text) or self._extract_remote_absolute_paths(text):
            return False
        if self._remote_targets_from_texts(text):
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        if not self._handoff_mentions_remote_deployment_context(handoff):
            return False
        has_clarification_phrase = any(phrase in text for phrase in _REMOTE_CLARIFICATION_PHRASES)
        has_deployment_target = any(target in text for target in _REMOTE_DEPLOYMENT_CONTEXT_TARGETS)
        return has_clarification_phrase and has_deployment_target

    def _looks_like_remote_live_correction_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").replace("\u2019", "'").strip().lower()
        if not text:
            return False
        if self._looks_like_remote_artifact_cleanup_followup(text, handoff):
            return True
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        if extract_task_target_paths(text) or self._extract_remote_absolute_paths(text):
            return False
        if not (self.has_task_local_context() or handoff):
            return False
        if not (
            self._handoff_has_remote_context(handoff)
            or self._handoff_mentions_remote_context(handoff)
            or bool(self._session_ssh_targets())
        ):
            return False
        if any(phrase in text for phrase in _REMOTE_LIVE_CORRECTION_PHRASES):
            return True
        has_live_correction_language = any(marker in text for marker in _REMOTE_LIVE_CORRECTION_HINTS)
        has_remote_anchor = any(token in text for token in ("ssh", "remote", "host", "server"))
        has_reliance_negation = any(
            phrase in text
            for phrase in (
                "don't rely",
                "do not rely",
                "dont rely",
                "do not trust",
                "don't trust",
            )
        )
        return has_live_correction_language and (has_remote_anchor or has_reliance_negation)

    def _looks_like_remote_diagnostic_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").replace("\u2019", "'").strip().lower()
        if not text:
            return False
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        explicit_paths = extract_task_target_paths(text)
        if explicit_paths:
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        if not (
            self._handoff_mentions_remote_deployment_context(handoff)
            or bool(self._recent_remote_target_paths(handoff=handoff))
            or bool(self._confirmed_session_ssh_targets())
        ):
            return False

        referenced_remote_paths = self._extract_remote_absolute_paths(text)
        if referenced_remote_paths:
            recent_remote_paths = self._recent_remote_target_paths(handoff=handoff)
            if recent_remote_paths and self._target_paths_overlap(referenced_remote_paths, recent_remote_paths):
                return True

        has_target = any(target in text for target in _REMOTE_DIAGNOSTIC_TARGETS)
        if not has_target:
            return False
        has_hint = any(hint in text for hint in _REMOTE_DIAGNOSTIC_HINTS)
        has_question = "?" in text or _REMOTE_DIAGNOSTIC_QUESTION_RE.search(text) is not None
        return has_hint or has_question

    def _looks_like_remote_artifact_cleanup_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text:
            return False
        if not (
            self._handoff_has_remote_context(handoff)
            or self._handoff_mentions_remote_context(handoff)
            or bool(self._session_ssh_targets())
        ):
            return False

        lowered = text.lower()
        recent_remote_paths = self._recent_remote_target_paths(handoff=handoff)
        referenced_remote_paths = self._extract_remote_absolute_paths(text)
        if referenced_remote_paths:
            if not recent_remote_paths:
                return False
            if not self._target_paths_overlap(referenced_remote_paths, recent_remote_paths):
                return False

        has_cleanup_phrase = any(phrase in lowered for phrase in _REMOTE_CORRECTIVE_CLEANUP_PHRASES)
        has_cleanup_verb = bool(re.search(r"\b(?:fix|remove|clean(?:\s+up)?|trim)\b", lowered))
        has_location_hint = any(
            phrase in lowered
            for phrase in ("bottom of the page", "end of the page", "very bottom", "very end", "trailing")
        )
        has_residue_marker = any(marker in lowered for marker in _REMOTE_RESIDUE_MARKERS)
        return (has_cleanup_phrase or (has_cleanup_verb and (has_location_hint or has_residue_marker))) and (
            has_residue_marker or bool(referenced_remote_paths)
        )

    def _looks_like_remote_permission_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").strip().lower()
        if not text:
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        if not bool(self._confirmed_session_ssh_targets()):
            return False
        has_permission_hint = _REMOTE_PERMISSION_FOLLOWUP_RE.search(text) is not None
        has_script_hint = _REMOTE_SCRIPT_HINT_RE.search(text) is not None
        return has_permission_hint and has_script_hint

    def _looks_like_remote_contextual_site_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").strip()
        if not text:
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        return task_matches_remote_continuation(self.harness.state, text)

    def _looks_like_remote_site_mutation_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").strip()
        if not text:
            return False
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        if extract_task_target_paths(text) or self._extract_remote_absolute_paths(text):
            return False
        if self._remote_targets_from_texts(text):
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        if not handoff_supports_remote_continuation(self.harness.state):
            return False
        recent_remote_paths = self._recent_remote_target_paths(handoff=handoff)
        has_remote_site_context = any(
            path.startswith("/var/www/")
            or path.endswith(".html")
            or path.endswith(".htm")
            for path in recent_remote_paths
        )
        if not has_remote_site_context:
            candidate_text = " ".join(
                str(handoff.get(key) or "").lower()
                for key in ("effective_task", "current_goal", "raw_task")
            )
            has_remote_site_context = any(
                marker in candidate_text
                for marker in (
                    "index.html",
                    "html",
                    "page",
                    "site",
                    "theme",
                    "website",
                    "/var/www/",
                )
            )
        if not has_remote_site_context:
            return False
        return bool(
            _REMOTE_SITE_MUTATION_ACTION_RE.search(text)
            and _REMOTE_SITE_MUTATION_TARGET_RE.search(text)
        )

    def _is_remote_correction_followup(self, task: str) -> bool:
        return self._looks_like_remote_live_correction_followup(task, self.last_task_handoff())

    def _remote_followup_resolution(self, task: str) -> dict[str, Any] | None:
        text = str(task or "").strip()
        if not text:
            return None

        handoff = self.last_task_handoff()
        if not handoff:
            return None

        is_operational_followup = self._looks_like_remote_operational_followup(text)
        is_clarification_followup = self._looks_like_remote_clarification_followup(text, handoff)
        is_live_correction_followup = self._looks_like_remote_live_correction_followup(text, handoff)
        is_diagnostic_followup = self._looks_like_remote_diagnostic_followup(text, handoff)
        is_artifact_cleanup_followup = self._looks_like_remote_artifact_cleanup_followup(text, handoff)
        is_contextual_site_followup = self._looks_like_remote_contextual_site_followup(text, handoff)
        is_site_mutation_followup = self._looks_like_remote_site_mutation_followup(text, handoff)
        is_permission_followup = self._looks_like_remote_permission_followup(text, handoff)
        if not (
            is_operational_followup
            or is_clarification_followup
            or is_live_correction_followup
            or is_diagnostic_followup
            or is_artifact_cleanup_followup
            or is_contextual_site_followup
            or is_site_mutation_followup
            or is_permission_followup
        ):
            return None
        if not handoff_supports_remote_continuation(self.harness.state):
            return None

        mission_task = self._current_or_handoff_continuity_task()
        chosen_targets = self._handoff_remote_targets(handoff)
        session_targets = self._session_ssh_targets()
        explicit_targets = self._remote_targets_from_texts(text)
        if is_operational_followup and not explicit_targets and not self._confirmed_session_ssh_targets():
            return None
        if (
            (
                is_live_correction_followup
                or is_diagnostic_followup
                or is_artifact_cleanup_followup
                or is_contextual_site_followup
                or is_site_mutation_followup
                or is_permission_followup
            )
            and not explicit_targets
            and not (chosen_targets or self._confirmed_session_ssh_targets())
        ):
            return None
        if explicit_targets:
            known_targets = merge_remote_targets(chosen_targets + session_targets)
            if not known_targets:
                return None
            if not all(
                self._remote_target_matches_known_target(target, known_targets)
                for target in explicit_targets
            ):
                return None
        session_labels = [ format_remote_target(target) for target in session_targets]
        session_labels = [label for label in session_labels if label]

        recent_remote_paths = self._recent_remote_target_paths(handoff=handoff)
        active_path_hint = ""
        if is_permission_followup and recent_remote_paths:
            active_path_hint = f" for {recent_remote_paths[-1]}"

        if len(chosen_targets) == 1:
            target = chosen_targets[0]
            label =  format_remote_target(target)
            effective_task = f"Continue remote task over SSH on {label}{active_path_hint}. User follow-up: {text}"
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if len(session_targets) == 1:
            target = session_targets[0]
            label =  format_remote_target(target)
            effective_task = f"Continue remote task over SSH on {label}{active_path_hint}. User follow-up: {text}"
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if session_labels:
            effective_task = (
                "Continue remote task over SSH. "
                f"Active SSH sessions: {', '.join(session_labels[:3])}. "
                f"User follow-up: {text}. Resolve which host before executing remote commands."
            )
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "ambiguous",
                "host": "",
                "user": "",
                "active_sessions": session_labels,
            }

        if (is_live_correction_followup or is_diagnostic_followup) and self._handoff_has_remote_context(handoff):
            effective_task = (
                "Continue remote task over SSH. "
                f"User follow-up: {text}. Resolve which host before executing remote commands."
            )
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "ambiguous",
                "host": "",
                "user": "",
                "active_sessions": session_labels,
            }

        return None

    def _apply_remote_followup_metadata(self, raw_task: str, resolution: dict[str, Any]) -> None:
        self.harness.state.scratchpad["_resolved_remote_followup"] = {
            "raw_task": str(raw_task or "").strip(),
            "effective_task": str(resolution.get("effective_task") or "").strip(),
            "mission_task": collapse_task_chain(resolution.get("mission_task") or ""),
            "target_status": str(resolution.get("target_status") or "").strip(),
            "host": normalize_remote_host(resolution.get("host")),
            "user": str(resolution.get("user") or "").strip(),
            "active_sessions": list(resolution.get("active_sessions") or []),
        }

    def _affirmative_remote_execution_followup_resolution(self, task: str) -> dict[str, Any] | None:
        text = str(task or "").strip()
        if not text or not self._is_affirmative_followup(text):
            return None

        handoff = self.last_task_handoff()
        if not handoff or not self._handoff_has_remote_context(handoff):
            return None
        if (
            not self._recent_assistant_requested_action_confirmation()
            and not self._can_assume_remote_affirmative_continuation()
            and not self._has_plan_execution_approval_context()
        ):
            return None

        mission_task = self._current_or_handoff_continuity_task()
        chosen_targets = self._handoff_remote_targets(handoff)
        session_targets = self._session_ssh_targets()
        session_labels = [ format_remote_target(target) for target in session_targets]
        session_labels = [label for label in session_labels if label]

        followup_text = (
            self._approved_plan_followup_text()
            if self._has_plan_execution_approval_context()
            else _AFFIRMATIVE_REMOTE_CONTINUATION_TEXT
        )
        plan = self.harness.state.active_plan or self.harness.state.draft_plan
        plan_goal = collapse_task_chain(getattr(plan, "goal", "") or "")
        if plan_goal:
            mission_task = plan_goal
        if len(chosen_targets) == 1:
            target = chosen_targets[0]
            label =  format_remote_target(target)
            return {
                "effective_task": f"Continue remote task over SSH on {label}. User follow-up: {followup_text}",
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if len(session_targets) == 1:
            target = session_targets[0]
            label =  format_remote_target(target)
            return {
                "effective_task": f"Continue remote task over SSH on {label}. User follow-up: {followup_text}",
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if session_labels:
            return {
                "effective_task": (
                    "Continue remote task over SSH. "
                    f"Active SSH sessions: {', '.join(session_labels[:3])}. "
                    f"User follow-up: {followup_text}. Resolve which host before executing remote commands."
                ),
                "mission_task": mission_task,
                "target_status": "ambiguous",
                "host": "",
                "user": "",
                "active_sessions": session_labels,
            }

        return {
            "effective_task": (
                "Continue remote task over SSH. "
                f"User follow-up: {followup_text}. Resolve which host before executing remote commands."
            ),
            "mission_task": mission_task,
            "target_status": "ambiguous",
            "host": "",
            "user": "",
            "active_sessions": session_labels,
        }

    def begin_task_scope(self, *, raw_task: str, effective_task: str) -> dict[str, Any]:
        self._consume_session_restored_flag()
        normalized_raw = str(raw_task or "").strip()
        normalized_effective = str(effective_task or normalized_raw).strip()

        # Don't abort the current scope if the user is responding to a pending interrupt
        pending_interrupt = getattr(getattr(self.harness, "state", None), "pending_interrupt", None)
        current = self._active_task_scope_payload()
        is_interrupt = is_interrupt_response(pending_interrupt, normalized_raw)
        # Defensive fallback for plan approval replies when pending_interrupt is missing
        is_plan_approval_fallback = (
            not is_interrupt
            and is_plan_approval_reply(normalized_raw)
            and self._has_plan_execution_approval_context()
        )
        if is_interrupt or is_plan_approval_fallback:
            if current:
                return current
        else:
            if current:
                current_effective = str(
                    current.get("effective_task") or current.get("raw_task") or ""
                ).strip()
                current_raw = str(current.get("raw_task") or "").strip()
                if (
                    (current_effective and current_effective == normalized_effective)
                    or (current_raw and current_raw == normalized_raw)
                ):
                    return current
                self.finalize_task_scope(
                    terminal_event="task_aborted",
                    status="aborted",
                    reason="replaced_by_new_task",
                    replacement_task=normalized_effective,
                )

        prior_sequence = getattr(self.harness, "_task_sequence", 0)
        if not prior_sequence:
            state = getattr(self.harness, "state", None)
            scratchpad = getattr(state, "scratchpad", None)
            if isinstance(scratchpad, dict):
                prior_sequence = scratchpad.get("_task_sequence", 0)
        try:
            sequence = int(prior_sequence) + 1
        except (TypeError, ValueError):
            sequence = 1
        self.harness._task_sequence = sequence

        task_id = f"task-{sequence:04d}"
        summary_path = ""
        if getattr(self.harness, "run_logger", None) is not None:
            summary_path = str(
                (self.harness.run_logger.run_dir / "tasks" / task_id / "task_summary.json").resolve()
            )
        scope = {
            "task_id": task_id,
            "sequence": sequence,
            "raw_task": normalized_raw,
            "effective_task": normalized_effective,
            "target_paths": extract_task_target_paths(normalized_effective),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "start_step_count": int(getattr(self.harness.state, "step_count", 0) or 0),
            "start_token_usage": int(getattr(self.harness.state, "token_usage", 0) or 0),
            "summary_path": summary_path,
        }
        self.harness._active_task_scope = dict(scope)
        self.harness.state.scratchpad["_task_sequence"] = sequence
        self.harness.state.scratchpad["_active_task_scope"] = json_safe_value(scope)
        self.harness.state.scratchpad["_active_task_id"] = task_id
        return dict(scope)

    def finalize_task_scope(
        self,
        *,
        terminal_event: str,
        status: str,
        reason: str = "",
        result: dict[str, Any] | None = None,
        replacement_task: str = "",
    ) -> dict[str, Any] | None:
        scope = self._active_task_scope_payload()
        if not scope:
            return None

        finished_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        result_status = str((result or {}).get("status") or "").strip().lower()
        summary_terminal_event = terminal_event or "task_finalize"
        summary_text = self._extract_task_terminal_message(result)
        if not summary_text and reason:
            summary_text = self._clip_task_summary_text(reason)

        start_step_count = int(scope.get("start_step_count") or 0)
        start_token_usage = int(scope.get("start_token_usage") or 0)
        current_step_count = int(getattr(self.harness.state, "step_count", 0) or 0)
        current_token_usage = int(getattr(self.harness.state, "token_usage", 0) or 0)

        payload = {
            "task_id": str(scope.get("task_id") or "").strip(),
            "sequence": int(scope.get("sequence") or 0),
            "raw_task": str(scope.get("raw_task") or "").strip(),
            "effective_task": str(scope.get("effective_task") or "").strip(),
            "terminal_event": summary_terminal_event,
            "status": str(status or result_status or "stopped").strip(),
            "result_status": result_status,
            "reason": self._clip_task_summary_text(reason),
            "message": summary_text,
            "started_at": str(scope.get("started_at") or "").strip(),
            "finished_at": finished_at,
            "duration_seconds": self._task_duration_seconds(
                str(scope.get("started_at") or "").strip(),
                finished_at,
            ),
            "step_count": max(0, current_step_count - start_step_count),
            "token_usage": max(0, current_token_usage - start_token_usage),
            "current_phase": str(getattr(self.harness.state, "current_phase", "") or "").strip(),
            "active_tool_profiles": list(getattr(self.harness.state, "active_tool_profiles", []) or []),
            "target_paths": list(scope.get("target_paths") or []),
            "artifact_count": len(getattr(self.harness.state, "artifacts", {}) or {}),
            "recent_error_count": len(getattr(self.harness.state, "recent_errors", []) or []),
            "last_recent_error": self._clip_task_summary_text(
                (getattr(self.harness.state, "recent_errors", []) or [""])[-1]
                if getattr(self.harness.state, "recent_errors", [])
                else "",
                limit=180,
            ),
            "summary_path": str(scope.get("summary_path") or "").strip(),
        }
        if replacement_task:
            payload["replacement_task"] = replacement_task
        error = (result or {}).get("error")
        if isinstance(error, dict):
            payload["error_type"] = str(error.get("type") or "").strip()

        summary_path = self._write_task_summary(payload)
        payload["summary_path"] = summary_path
        self._append_task_episodic_summary(payload)

        if terminal_event:
            self.harness._runlog(
                terminal_event,
                "task ended without normal completion",
                task_id=payload["task_id"],
                status=payload["status"],
                result_status=result_status,
                reason=payload["reason"],
                replacement_task=replacement_task,
                summary_path=summary_path,
                raw_task=payload["raw_task"],
                effective_task=payload["effective_task"],
            )

        self.harness._active_task_scope = None
        self.harness.state.scratchpad.pop("_active_task_scope", None)
        self.harness.state.scratchpad.pop("_active_task_id", None)
        return payload

    def reset_task_boundary_state(
        self,
        *,
        reason: str,
        new_task: str = "",
        previous_task: str = "",
        preserve_memory: bool = False,
        preserve_summaries: bool = False,
        preserve_recent_tail: bool = False,
        semantic_recent_tail: bool = False,
        preserve_guard_context: bool = False,
    ) -> None:
        preserved_previous_task = str(
            previous_task
            or self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        ).strip()
        preserved_scratchpad: dict[str, Any] = {}
        for key in (
            "_model_name",
            "_model_is_small",
            "_max_steps",
            "strategy",
            "_fama_config",
            "_recovery_config",
            "_recovery_metrics",
            "_session_notepad",
            "_session_ssh_targets",
            "_last_task_text",
            "_last_task_handoff",
            "_task_transaction",
            "_resolved_remote_followup",
            "_task_boundary_previous_task",
            "_task_sequence",
            "_web_result_index",
            "_web_search_artifact_results",
            "_web_last_search_result_ids",
            "_web_last_search_fetch_ids",
            "_web_last_search_artifact_id",
            "_web_fetch_id_counter",
            "_web_budget",
            "_artifact_staleness",
            "_observation_staleness",
            "_summary_staleness",
            "_context_brief_staleness",
            "_experience_staleness",
            "_turn_bundle_staleness",
        ):
            if key in self.harness.state.scratchpad:
                preserved_scratchpad[key] = self.harness.state.scratchpad[key]
        if preserve_guard_context:
            for key in _TASK_BOUNDARY_GUARD_SCRATCHPAD_KEYS:
                if key in self.harness.state.scratchpad:
                    preserved_scratchpad[key] = self.harness.state.scratchpad[key]
        if preserved_previous_task:
            preserved_scratchpad["_task_boundary_previous_task"] = preserved_previous_task

        background_processes = json_safe_value(self.harness.state.background_processes)
        preserved_artifacts = dict(self.harness.state.artifacts)
        if preserve_recent_tail and semantic_recent_tail:
            recent_tail = self._semantic_recent_tail_messages(token_cap=_SEMANTIC_RECENT_TAIL_TOKEN_CAP)
        else:
            recent_tail = trim_recent_messages(
                list(self.harness.state.recent_messages),
                limit=self.harness.state.recent_message_limit,
            )
        preserved_recent_errors = list(self.harness.state.recent_errors[-6:])
        preserved_summaries = list(self.harness.state.episodic_summaries)
        preserved_context_briefs = list(self.harness.state.context_briefs)
        preserved_tool_history = list(self.harness.state.tool_history)
        preserved_failure_events = list(getattr(self.harness.state, "failure_events", []) or [])
        preserved_reflexion_memory = list(getattr(self.harness.state, "reflexion_memory", []) or [])
        preserved_subtask_ledger = getattr(self.harness.state, "subtask_ledger", None)
        recovery_config = preserved_scratchpad.get("_recovery_config")
        recovery_config = recovery_config if isinstance(recovery_config, dict) else {}
        preserve_reflexion_memory = preserve_memory or bool(recovery_config.get("reflexion_persist_cross_task", False))
        current_memory = self.harness.state.working_memory
        preserved_memory = WorkingMemory(
            current_goal=str(current_memory.current_goal or ""),
            plan=list(current_memory.plan),
            decisions=list(current_memory.decisions),
            open_questions=list(current_memory.open_questions),
            known_facts=list(current_memory.known_facts),
            known_fact_meta=list(current_memory.known_fact_meta),
            failures=list(current_memory.failures),
            failure_meta=list(current_memory.failure_meta),
            next_actions=list(current_memory.next_actions),
            next_action_meta=list(current_memory.next_action_meta),
        )

        self.harness.state.current_phase = self.harness._initial_phase
        self.harness.state.step_count = 0
        self.harness.state.inactive_steps = 0
        self.harness.state.latest_verdict = None

        self.harness.state.scratchpad = preserved_scratchpad
        self.harness.state.recent_messages = recent_tail if preserve_recent_tail else []
        self.harness.state.recent_errors = preserved_recent_errors if preserve_guard_context else []
        self.harness.state.run_brief = RunBrief()
        if preserve_memory and preserved_previous_task:
            self.harness.state.run_brief.original_task = preserved_previous_task
        self.harness.state.working_memory = preserved_memory if preserve_memory else WorkingMemory()
        self.harness.state.acceptance_ledger = {}
        self.harness.state.acceptance_waivers = []
        self.harness.state.acceptance_waived = False
        self.harness.state.last_verifier_verdict = None
        self.harness.state.last_failure_class = ""
        self.harness.state.failure_events = preserved_failure_events if preserve_memory else []
        self.harness.state.reflexion_memory = preserved_reflexion_memory if preserve_reflexion_memory else []
        # Preserve ledger across task switches so the checklist history is continuous.
        # Mark old active/pending subtasks as done on hard resets; keep done/failed/
        # abandoned history so the UI shows a continuous trail.
        ledger = preserved_subtask_ledger
        if ledger is not None:
            for task in getattr(ledger, "subtasks", []) or []:
                if str(getattr(task, "status", "") or "").strip().lower() in {"active", "pending"}:
                    task.status = "done"
                    task.updated_at = time()
            new_task_text = str(new_task or "").strip()
            if new_task_text:
                from .subtask_ledger_service import _task_id

                ledger.task_id = _task_id(new_task_text)
        self.harness.state.subtask_ledger = ledger

        self.harness.state.files_changed_this_cycle = []
        self.harness.state.repair_cycle_id = ""
        self.harness.state.stagnation_counters = {}
        self.harness.state.scratchpad.pop("_confabulation_nudged", None)
        self.harness.state.draft_plan = None
        self.harness.state.active_plan = None
        self.harness.state.plan_resolved = False
        self.harness.state.plan_artifact_id = ""
        self.harness.state.planning_mode_enabled = self.harness._configured_planning_mode
        self.harness.state.planner_requested_output_path = ""
        self.harness.state.planner_requested_output_format = ""
        self.harness.state.planner_resume_target_mode = "loop"
        self.harness.state.planner_interrupt = None
        # Artifacts are durable session handles. Keep them across task switches so
        # follow-up and resteered tasks can still inspect outputs from earlier work.
        self.harness.state.artifacts = preserved_artifacts
        if self.harness.state.write_session and self.harness.state.write_session.status != "complete":
            self.harness._runlog(
                "write_session_abandoned",
                "incomplete write session abandoned on task switch",
                session_id=self.harness.state.write_session.write_session_id,
                stage_target=self.harness.state.write_session.write_target_path,
                status=self.harness.state.write_session.status,
            )
            from ..graph.tool_outcomes import _register_write_session_stage_artifact
            from ..write_session_fsm import archive_interrupted_write_session

            _register_write_session_stage_artifact(self.harness, self.harness.state.write_session)
            archive_interrupted_write_session(
                self.harness.state,
                reason="task_switch_abandoned",
            )
            self.harness.state.write_session = None
        self.harness.state.episodic_summaries = preserved_summaries if preserve_summaries else []
        self.harness.state.context_briefs = preserved_context_briefs if preserve_summaries else []
        self.harness.state.prompt_budget = PromptBudgetSnapshot()
        self.harness.state.retrieval_cache = []
        self.harness.state.task_mode = ""
        self.harness.state.active_intent = ""
        self.harness.state.secondary_intents = []
        self.harness.state.intent_tags = []
        self.harness.state.retrieved_experience_ids = []
        self.harness.state.tool_execution_records = {}
        self.harness.state.pending_interrupt = None
        self.harness.state.tool_history = preserved_tool_history if preserve_guard_context else []
        self.harness.state.background_processes = background_processes if isinstance(background_processes, dict) else {}
        self.harness.state.warm_experiences = []
        self.harness.state.touch()
        self.harness._runlog(
            "task_boundary_reset",
            "reset task-local state for new task",
            reason=reason,
            previous_task=previous_task,
            new_task=new_task,
            preserve_memory=preserve_memory,
            preserve_summaries=preserve_summaries,
            preserve_recent_tail=preserve_recent_tail,
            semantic_recent_tail=semantic_recent_tail,
            preserve_guard_context=preserve_guard_context,
        )

    def maybe_reset_for_new_task(self, task: str, *, raw_task: str | None = None) -> None:
        pending_interrupt = getattr(getattr(self.harness, "state", None), "pending_interrupt", None)
        if is_interrupt_response(pending_interrupt, raw_task or task):
            return
        # Defensive fallback for plan approval replies when pending_interrupt is missing
        if is_plan_approval_reply(raw_task or task) and self._has_plan_execution_approval_context():
            return

        previous_task = collapse_task_chain(
            self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )
        if not previous_task:
            return
        new_task = collapse_task_chain(task)
        if not new_task or normalize_task_text(new_task) == normalize_task_text(previous_task):
            return
        has_task_local_context = self.has_task_local_context()
        if has_task_local_context:
            session_restored = self._consume_session_restored_flag()
            remote_correction = self._is_remote_correction_followup(raw_task or task)
            same_scope_followup = self._is_same_scope_transition(
                raw_task=raw_task or task,
                effective_task=new_task,
                previous_task=previous_task,
            ) or remote_correction
            classification = self._build_followup_classification(
                raw_task=raw_task or task,
                effective_task=new_task,
                previous_task=previous_task,
                same_scope_followup=same_scope_followup,
                remote_correction=remote_correction,
            )
            transaction = self._store_task_transaction(classification)
            turn_type = str(transaction.get("turn_type") or classification.turn_type)
            policy = classification.reset_policy
            preserve_prior_result = bool(policy.keep_prior_result)
            preserve_recent_tail = preserve_prior_result or session_restored or remote_correction
            if previous_task:
                self.store_task_handoff(raw_task=previous_task, effective_task=previous_task)
            if turn_type == "CLARIFICATION":
                reset_reason = "task_clarification"
            elif preserve_prior_result:
                reset_reason = "task_soft_switch"
            else:
                reset_reason = "task_switch"
            if session_restored and not preserve_prior_result:
                reset_reason = "task_resume_switch"
            self.finalize_task_scope(
                terminal_event="task_aborted",
                status="aborted",
                reason="replaced_by_new_task",
                replacement_task=new_task,
            )
            self.reset_task_boundary_state(
                reason=reset_reason,
                new_task=new_task,
                previous_task=previous_task,
                preserve_memory=preserve_prior_result,
                preserve_summaries=preserve_prior_result,
                preserve_recent_tail=preserve_recent_tail,
                semantic_recent_tail=turn_type in {"ITERATION", "CORRECTION", "RETRY"} or remote_correction,
                preserve_guard_context=bool(policy.preserve_guard_context),
            )
            if preserve_prior_result:
                self.harness.state.scratchpad["_task_transaction"] = transaction
                self._apply_classified_reset_followup(classification)
                self._record_same_scope_resteer(
                    raw_task=raw_task or task,
                    effective_task=new_task,
                    turn_type=turn_type,
                )
            else:
                self.harness.state.scratchpad.pop("_task_transaction", None)

    def _record_same_scope_resteer(
        self,
        *,
        raw_task: str,
        effective_task: str,
        turn_type: str,
    ) -> None:
        state = self.harness.state
        text = str(raw_task or effective_task or "").strip()
        if not text:
            return
        subtask_id = ""
        ledger = getattr(state, "subtask_ledger", None)
        active = ledger.active() if ledger is not None and callable(getattr(ledger, "active", None)) else None
        if active is not None:
            subtask_id = str(getattr(active, "subtask_id", "") or "").strip()
        event = FailureEvent(
            event_id=f"resteer-{int(time() * 1000)}",
            timestamp=time(),
            failure_class="human_resteer",
            severity="warning",
            source="task_boundary",
            message=f"human_resteer: user redirected same-scope work ({turn_type.lower() or 'followup'})",
            evidence=[text[:240]],
            subtask_id=subtask_id or None,
            suggested_next_action=text[:240],
            metadata={"effective_task": str(effective_task or "").strip()[:240]},
        )
        state.failure_events.append(event)
        state.failure_events = state.failure_events[-40:]
        record_failure_event_metric(state, event)
        state.last_failure_class = "human_resteer"
        state.scratchpad["_last_failure_class"] = "human_resteer"
        if active is not None:
            active.next_action = text[:240]
            active.updated_at = event.timestamp
            if "human_resteer" not in active.failure_classes:
                active.failure_classes.append("human_resteer")
        reflexion = getattr(self.harness, "reflexion", None)
        maybe_create = getattr(reflexion, "maybe_create_reflection", None)
        if callable(maybe_create):
            maybe_create(event, ledger)
        self.harness._runlog(
            "recovery_human_resteer_recorded",
            "same-scope human resteer recorded",
            turn_type=turn_type,
            subtask_id=subtask_id,
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

    def maybe_nudge_internal_divergence(self) -> bool:
        """Detect internal task divergence and emit a recovery nudge."""
        if self.harness.state.scratchpad.get("_task_divergence_nudged"):
            return False
        handoff = self.last_task_handoff()
        task_mode = handoff.get("task_mode", "")
        # Only detect remote -> local divergence for now
        if task_mode != "remote_execute":
            return False
        history = self.harness.state.scratchpad.get("_tool_attempt_history", [])
        if not isinstance(history, list) or len(history) < 3:
            return False
        recent = history[-5:]
        local_only = 0
        for item in recent:
            tool_name = str(item.get("tool_name", ""))
            if tool_name in {"file_read", "dir_list", "find_files"}:
                local_only += 1
            elif tool_name.startswith("ssh_"):
                return False
        if local_only < 3:
            return False
        original_task = self._clip_task_summary_text(
            self.harness.state.run_brief.original_task or handoff.get("raw_task", "")
        )
        self.harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"TASK DIVERGENCE WARNING: Your original task is: {original_task}. "
                    "You appear to have switched to working on local files instead of the remote target. "
                    "Do NOT abandon the original task. Return to the remote work immediately."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "task_divergence",
                },
            )
        )
        self.harness._runlog(
            "task_divergence_nudge",
            "nudged model back to original remote task after internal task switch detected",
            original_task=original_task,
        )
        self.harness.state.scratchpad["_task_divergence_nudged"] = True
        return True

    def _is_continue_like_followup(self, task: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(task or "").strip().lower()).strip()
        if not normalized:
            return False
        tokens = [token for token in normalized.split() if token not in _FOLLOWUP_FILLERS]
        collapsed = " ".join(tokens)
        return collapsed in {
            "continue",
            "cntinue",
            "conitnue",
            "continune",
            "cotinue",
            "keep going",
            "resume",
            "proceed",
            "go on",
            "carry on",
        }

    def _is_continue_directive_followup(self, task: str) -> bool:
        text = str(task or "").strip()
        if not text:
            return False
        if _CONTINUE_DIRECTIVE_RE.match(text):
            return True
        if not _WEB_RESEARCH_DIRECTIVE_RE.search(text):
            return False
        if _CONTINUATION_ACTION_LEAD_RE.search(text):
            return True
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        context = " ".join(
            part
            for part in (
                self._current_or_handoff_continuity_task(),
                self._current_or_handoff_task(),
            )
            if part
        )
        return bool(context and _RESEARCH_CONTEXT_RE.search(context))

    def _is_affirmative_followup(self, task: str) -> bool:
        return is_affirmative_followup(task, fillers=_FOLLOWUP_FILLERS)

    def _recent_assistant_requested_action_confirmation(self) -> bool:
        return recent_assistant_requested_action_confirmation(
            list(self.harness.state.recent_messages),
            prompts=_ACTION_CONFIRMATION_PROMPTS,
        )

    def _can_assume_remote_affirmative_continuation(self) -> bool:
        handoff = self.last_task_handoff()
        if not handoff or not self._handoff_has_remote_context(handoff):
            return False
        if len(self._confirmed_session_ssh_targets()) != 1:
            return False
        for message in reversed(self.harness.state.recent_messages[-4:]):
            if getattr(message, "role", "") != "assistant":
                continue
            text = str(getattr(message, "content", "") or "").strip()
            if assistant_message_proposes_concrete_implementation(text):
                return True
        return False

    def _has_plan_execution_approval_context(self) -> bool:
        state = self.harness.state
        pending_interrupt = getattr(state, "pending_interrupt", None)
        if isinstance(pending_interrupt, dict) and pending_interrupt.get("kind") == "plan_execute_approval":
            return True
        planner_interrupt = getattr(state, "planner_interrupt", None)
        if str(getattr(planner_interrupt, "kind", "") or "").strip() == "plan_execute_approval":
            return True
        plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
        status = str(getattr(plan, "status", "") or "").strip().lower()
        return status == "awaiting_approval"

    def _approved_plan_followup_text(self) -> str:
        plan = self.harness.state.active_plan or self.harness.state.draft_plan
        goal = collapse_task_chain(getattr(plan, "goal", "") or "")
        if not goal:
            return _AFFIRMATIVE_REMOTE_CONTINUATION_TEXT
        plan_id = str(getattr(plan, "plan_id", "") or "").strip()
        lead = "execute approved plan"
        if plan_id:
            lead = f"{lead} {plan_id}"
        parts = [f"{lead}: {goal}"]
        summary = collapse_task_chain(getattr(plan, "summary", "") or "")
        if summary:
            parts.append(f"summary: {summary}")
        target_paths = self._extract_remote_absolute_paths(
            goal,
            *(getattr(plan, "inputs", []) or []),
            *(getattr(plan, "outputs", []) or []),
        )
        if target_paths:
            parts.append("targets: " + ", ".join(f"`{path}`" for path in target_paths[:3]))
        return ". ".join(parts)

    def _has_contextual_reference_to_current_task(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text:
            return False
        known_paths = self._known_target_paths()
        explicit_paths = extract_task_target_paths(text)
        if explicit_paths and known_paths and self._target_paths_overlap(explicit_paths, known_paths):
            return True
        recent_remote_paths = self._recent_remote_target_paths()
        explicit_remote_paths = self._extract_remote_absolute_paths(text)
        if (
            explicit_remote_paths
            and recent_remote_paths
            and self._target_paths_overlap(explicit_remote_paths, recent_remote_paths)
        ):
            return True
        if explicit_paths or explicit_remote_paths:
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        if self._is_continue_directive_followup(text):
            return True
        if _CONTEXTUAL_REFERENCE_RE.search(text) and (
            _FOLLOWUP_ACTION_RE.search(text) or _GENERIC_TARGET_RE.search(text)
        ):
            return True
        if known_paths and _GENERIC_TARGET_RE.search(text) and _FOLLOWUP_ACTION_RE.search(text):
            return True
        # If previous task had known paths and new task mentions same artifact type
        # without specifying any new paths, treat as contextual follow-up
        if not (explicit_paths or explicit_remote_paths) and known_paths and _GENERIC_TARGET_RE.search(text):
            return True
        return False

    def _has_recent_guard_failure_context(self) -> bool:
        candidate_texts: list[str] = []
        recent_errors = getattr(self.harness.state, "recent_errors", None)
        if isinstance(recent_errors, list):
            candidate_texts.extend(str(error or "") for error in recent_errors[-6:])
        scratchpad = getattr(self.harness.state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            for key in ("_task_failed_message", "_last_guard_error"):
                value = scratchpad.get(key)
                if value:
                    candidate_texts.append(str(value))
            for key in _TASK_BOUNDARY_GUARD_SCRATCHPAD_KEYS:
                if key in scratchpad:
                    candidate_texts.append(key)
        return any(_GUARD_FAILURE_RE.search(text) for text in candidate_texts if text)

    def _is_quality_followup(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text or not _QUALITY_FOLLOWUP_RE.search(text):
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        known_paths = self._known_target_paths()
        explicit_paths = extract_task_target_paths(text)
        explicit_remote_paths = self._extract_remote_absolute_paths(text)
        if explicit_paths and known_paths:
            return self._target_paths_overlap(explicit_paths, known_paths)
        if explicit_remote_paths and self._recent_remote_target_paths():
            return self._target_paths_overlap(explicit_remote_paths, self._recent_remote_target_paths())
        if explicit_paths or explicit_remote_paths:
            return False
        return bool(
            known_paths
            or _QUALITY_TARGET_RE.search(text)
            or _CONTEXTUAL_REFERENCE_RE.search(text)
        )

    def _is_guard_recovery_followup(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text or not self._has_recent_guard_failure_context():
            return False
        explicit_paths = extract_task_target_paths(text)
        known_paths = self._known_target_paths()
        if explicit_paths and known_paths and not self._target_paths_overlap(explicit_paths, known_paths):
            return False
        explicit_remote_paths = self._extract_remote_absolute_paths(text)
        recent_remote_paths = self._recent_remote_target_paths()
        if (
            explicit_remote_paths
            and recent_remote_paths
            and not self._target_paths_overlap(explicit_remote_paths, recent_remote_paths)
        ):
            return False
        return bool(_GUARD_RECOVERY_NUDGE_RE.search(text))

    def _is_corrective_resteer_followup(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text:
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        if _CORRECTIVE_RESTEER_RE.search(text):
            return True
        lowered = text.lower()
        if any(f"`{tool}` instead" in lowered or f"{tool} instead" in lowered for tool in _CORRECTIVE_TOOL_NAMES):
            return True
        return self._looks_like_remote_live_correction_followup(text, self.last_task_handoff())

    def _current_or_handoff_task(self) -> str:
        handoff = self.last_task_handoff()
        return base_task_from_task_chain(
            self.harness.state.working_memory.current_goal
            or handoff.get("current_goal")
            or handoff.get("effective_task")
            or self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )

    def _current_or_handoff_continuity_task(self) -> str:
        handoff = self.last_task_handoff()
        return collapse_task_chain(
            self.harness.state.working_memory.current_goal
            or handoff.get("current_goal")
            or self.harness.state.run_brief.original_task
            or handoff.get("effective_task")
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )

    def _remote_followup_mission_task(self, raw_task: str) -> str:
        resolved_remote = self.harness.state.scratchpad.get("_resolved_remote_followup")
        if not isinstance(resolved_remote, dict):
            return ""
        if normalize_task_text(resolved_remote.get("raw_task")) != normalize_task_text(raw_task):
            return ""
        mission_task = collapse_task_chain(resolved_remote.get("mission_task") or "")
        if mission_task:
            return mission_task
        return self._current_or_handoff_continuity_task() or collapse_task_chain(
            self.harness.state.scratchpad.get("_task_boundary_previous_task") or ""
        )

    def _resolved_corrective_resteer_task(self, raw_task: str) -> str:
        candidate = self._current_or_handoff_task()
        if not candidate:
            return str(raw_task or "").strip()
        correction = str(raw_task or "").strip()
        return f"Continue current task: {candidate}. User correction: {correction}"

    def _is_same_scope_transition(
        self,
        *,
        raw_task: str,
        effective_task: str,
        previous_task: str,
    ) -> bool:
        if self._looks_like_remote_artifact_cleanup_followup(raw_task, self.last_task_handoff()):
            return True
        if self._is_contextual_followup(raw_task):
            return True
        resolved = self.harness.state.scratchpad.get("_resolved_followup")
        if isinstance(resolved, dict) and resolved.get("target_inheritance") == "inherited":
            return True
        known_paths = self._known_target_paths()
        candidate_paths = (
            extract_task_target_paths(effective_task)
            or extract_task_target_paths(raw_task)
            or self._extract_remote_absolute_paths(effective_task)
            or self._extract_remote_absolute_paths(raw_task)
        )
        if candidate_paths and self._target_paths_overlap(candidate_paths, known_paths):
            return True
        previous_paths = extract_task_target_paths(previous_task) or self._extract_remote_absolute_paths(previous_task)
        if candidate_paths and self._target_paths_overlap(candidate_paths, previous_paths):
            return True
        # Remote same-directory / same-host transitions for multi-file operational sequences
        if candidate_paths and self._confirmed_session_ssh_targets():
            for cand in candidate_paths:
                if not cand.startswith("/"):
                    continue
                cand_parent = str(Path(cand).parent).lower()
                for known in known_paths:
                    if str(Path(known).parent).lower() == cand_parent:
                        return True
                for prev in previous_paths:
                    if str(Path(prev).parent).lower() == cand_parent:
                        return True
        # Sequential language ("now do ...", "next edit ...") on a confirmed remote session
        if candidate_paths and self._is_sequential_remote_followup(raw_task, candidate_paths):
            return True
        return False

    def _followup_allowed_paths(
        self,
        *,
        raw_task: str,
        effective_task: str,
        previous_task: str,
    ) -> list[str]:
        handoff = self.last_task_handoff()
        paths: list[str] = []
        for source in (
            extract_task_target_paths(effective_task),
            extract_task_target_paths(raw_task),
            self._extract_remote_absolute_paths(effective_task),
            self._extract_remote_absolute_paths(raw_task),
            handoff.get("target_paths") if isinstance(handoff, dict) else [],
            handoff.get("remote_target_paths") if isinstance(handoff, dict) else [],
            extract_task_target_paths(previous_task),
            self._extract_remote_absolute_paths(previous_task),
        ):
            if not isinstance(source, list):
                continue
            paths.extend(str(path).strip() for path in source if str(path).strip())
        return dedupe_keep_tail(paths, limit=12)

    def _followup_allowed_artifacts(self) -> list[str]:
        handoff = self.last_task_handoff()
        candidates: list[str] = []
        for key in ("last_good_artifact_ids", "recent_research_artifact_ids"):
            values = handoff.get(key)
            if isinstance(values, list):
                candidates.extend(str(item).strip() for item in values if str(item).strip())
        retrieval_cache = getattr(self.harness.state, "retrieval_cache", None)
        if isinstance(retrieval_cache, list):
            candidates.extend(str(item).strip() for item in retrieval_cache if str(item).strip())
        return dedupe_keep_tail(candidates, limit=8)

    def _followup_failure_summary(self) -> str:
        fragments: list[str] = []
        last_failed = self.last_task_handoff().get("last_failed_tool")
        if isinstance(last_failed, dict):
            tool_name = str(last_failed.get("tool_name") or "").strip()
            error = str(last_failed.get("error") or "").strip()
            if tool_name:
                fragments.append(f"{tool_name}: {error}" if error else tool_name)
        recent_errors = getattr(self.harness.state, "recent_errors", None)
        if isinstance(recent_errors, list):
            fragments.extend(str(error or "").strip() for error in recent_errors[-2:] if str(error or "").strip())
        scratchpad = getattr(self.harness.state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            for key in ("_task_failed_message", "_last_guard_error"):
                value = str(scratchpad.get(key) or "").strip()
                if value:
                    fragments.append(value)
        return self._clip_task_summary_text("; ".join(fragments), limit=220)

    def _followup_verification_hint(self, *, task_mode: str) -> str:
        handoff = self.last_task_handoff()
        next_required_tool = handoff.get("next_required_tool")
        if isinstance(next_required_tool, dict):
            tool_name = str(next_required_tool.get("tool_name") or "").strip()
            if tool_name:
                return f"Use `{tool_name}` if it is still the required next verifier."
        if task_mode == "remote_execute":
            return "Use a focused SSH read or command to verify the remote result."
        if task_mode == "local_execute":
            return "Run the focused local verifier for the changed target."
        return ""

    def _build_followup_classification(
        self,
        *,
        raw_task: str,
        effective_task: str,
        previous_task: str,
        same_scope_followup: bool | None = None,
        remote_correction: bool | None = None,
    ) -> FollowupClassification:
        raw = str(raw_task or "").strip()
        effective = str(effective_task or raw).strip()
        handoff = self.last_task_handoff()
        known_paths = self._known_target_paths()
        raw_explicit_paths = extract_task_target_paths(raw) or self._extract_remote_absolute_paths(raw)
        explicit_paths = raw_explicit_paths or (
            extract_task_target_paths(effective) or self._extract_remote_absolute_paths(effective)
        )
        previous_paths = extract_task_target_paths(previous_task) or self._extract_remote_absolute_paths(previous_task)
        has_overlap = bool(
            (explicit_paths and self._target_paths_overlap(explicit_paths, known_paths))
            or (explicit_paths and self._target_paths_overlap(explicit_paths, previous_paths))
        )
        explicit_conflicting_target = bool(
            explicit_paths
            and (known_paths or previous_paths)
            and not has_overlap
            and not same_scope_followup
        )
        selected_action = isinstance(self._selected_action_option(raw, handoff), dict)
        remote_clarification = self._looks_like_remote_clarification_followup(raw, handoff)
        corrective_resteer = self._is_corrective_resteer_followup(raw)
        guard_recovery_followup = self._is_guard_recovery_followup(raw)
        quality_followup = self._is_quality_followup(raw)
        remote_live_correction = (
            bool(remote_correction)
            or self._looks_like_remote_live_correction_followup(raw, handoff)
        )
        guard_failure_context = self._has_recent_guard_failure_context()
        retry_language = bool(_RETRY_FOLLOWUP_RE.search(raw))
        task_mode = classify_task_mode(effective)
        return classify_followup_transaction(
            raw_task=raw,
            effective_task=effective,
            previous_task=previous_task,
            task_mode=task_mode,
            signals=FollowupSignals(
                has_prior_task=bool(previous_task),
                has_overlap=has_overlap,
                explicit_conflicting_target=explicit_conflicting_target,
                selected_action_option=selected_action,
                contextual_reference=self._has_contextual_reference_to_current_task(raw)
                or self._is_continue_like_followup(raw),
                same_target_delta=bool(same_scope_followup) or has_overlap,
                corrective_resteer=corrective_resteer or guard_recovery_followup,
                quality_followup=quality_followup,
                remote_live_correction=remote_live_correction,
                remote_clarification=remote_clarification,
                guard_failure_context=guard_failure_context,
                retry_language=retry_language,
            ),
            allowed_paths=self._followup_allowed_paths(
                raw_task=raw,
                effective_task=effective,
                previous_task=previous_task,
            ),
            allowed_artifacts=self._followup_allowed_artifacts(),
            failure_summary=self._followup_failure_summary(),
            verification_hint=self._followup_verification_hint(task_mode=task_mode),
        )

    def _store_task_transaction(self, classification: FollowupClassification) -> dict[str, Any]:
        payload = classification.to_dict()
        self.harness.state.scratchpad["_task_transaction"] = payload
        return payload

    def _store_followup_transaction_for_resolution(
        self,
        *,
        raw_task: str,
        effective_task: str,
    ) -> None:
        previous_task = self._current_or_handoff_continuity_task() or collapse_task_chain(
            self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )
        if not previous_task:
            return
        same_scope_followup = self._is_same_scope_transition(
            raw_task=raw_task,
            effective_task=effective_task,
            previous_task=previous_task,
        )
        remote_correction = self._is_remote_correction_followup(raw_task)
        classification = self._build_followup_classification(
            raw_task=raw_task,
            effective_task=effective_task,
            previous_task=previous_task,
            same_scope_followup=same_scope_followup,
            remote_correction=remote_correction,
        )
        self._store_task_transaction(classification)

    def _clear_fresh_plan_state_for_transaction(self) -> None:
        memory = self.harness.state.working_memory
        memory.plan = []
        memory.next_actions = []
        memory.next_action_meta = []
        self.harness.state.draft_plan = None
        self.harness.state.active_plan = None
        self.harness.state.plan_resolved = False
        self.harness.state.plan_artifact_id = ""
        for key in (
            "_tool_attempt_history",
            "_progress_read_history",
            "_progress_prior_plan_step",
            "_progress_prior_verdict",
            "_retrieval_query",
            "_last_verifier_command",
        ):
            self.harness.state.scratchpad.pop(key, None)

    def _apply_classified_reset_followup(self, classification: FollowupClassification) -> None:
        policy = classification.reset_policy
        if policy.force_fresh_plan:
            self._clear_fresh_plan_state_for_transaction()

    def _is_sequential_remote_followup(self, task: str, candidate_paths: list[str]) -> bool:
        if not candidate_paths or not self._confirmed_session_ssh_targets():
            return False
        if not any(path.startswith("/") for path in candidate_paths):
            return False
        if not self.has_task_local_context():
            return False
        return bool(_SEQUENTIAL_REMOTE_FOLLOWUP_RE.search(task))

    def _is_contextual_followup(self, task: str) -> bool:
        if self._is_continue_like_followup(task):
            return True
        if self._ordinal_followup_index(task) is not None and self.last_task_handoff().get("action_options"):
            return True
        if self._remote_followup_resolution(task) is not None:
            return True
        if self._is_corrective_resteer_followup(task):
            return True
        if self._has_contextual_reference_to_current_task(task):
            return True
        if self._is_quality_followup(task):
            return True
        if self._is_guard_recovery_followup(task):
            return True
        if not self._is_affirmative_followup(task):
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        return self._recent_assistant_requested_action_confirmation()

    def _ordinal_followup_index(self, task: str) -> int | None:
        return ordinal_followup_index(task)

    def _selected_action_option(self, task: str, handoff: dict[str, Any]) -> dict[str, Any] | None:
        index = ordinal_followup_index(task)
        if index is None:
            return None
        options = handoff.get("action_options")
        if not isinstance(options, list):
            return None
        for option in options:
            if not isinstance(option, dict):
                continue
            try:
                option_index = int(option.get("index") or 0)
            except (TypeError, ValueError):
                option_index = 0
            if option_index == index:
                return dict(option)
        return None

    def _message_is_semantic_tail_candidate(self, message: Any) -> bool:
        return message_is_semantic_tail_candidate(message)

    def _semantic_recent_tail_messages(self, *, token_cap: int) -> list[Any]:
        return semantic_recent_tail_messages(self.harness.state.recent_messages, token_cap=token_cap)

    def _strip_ordinal_prefix(self, task: str) -> str:
        return strip_ordinal_prefix(task)

    def _resolve_option_target_paths(
        self,
        raw_task: str,
        option: dict[str, Any],
        handoff: dict[str, Any],
    ) -> dict[str, Any]:
        suffix = self._strip_ordinal_prefix(raw_task)
        explicit_paths = extract_task_target_paths(suffix)
        inherited_paths = option.get("target_paths")
        if not isinstance(inherited_paths, list) or not inherited_paths:
            inherited_paths = handoff.get("target_paths")
        if not isinstance(inherited_paths, list):
            inherited_paths = []

        if explicit_paths:
            return {
                "target_paths": list(explicit_paths),
                "target_inheritance": "explicit_override",
                "blocked_target_paths": [],
                "suffix": suffix,
            }

        if blocks_inherited_target(suffix, list(inherited_paths)):
            return {
                "target_paths": [],
                "target_inheritance": "blocked_by_user_constraint",
                "blocked_target_paths": list(inherited_paths),
                "suffix": suffix,
            }

        return {
            "target_paths": list(inherited_paths),
            "target_inheritance": "inherited",
            "blocked_target_paths": [],
            "suffix": suffix,
        }

    def _resolved_option_task(
        self,
        raw_task: str,
        option: dict[str, Any],
        handoff: dict[str, Any],
    ) -> str:
        title = str(option.get("title") or "").strip()
        target_info = self._resolve_option_target_paths(raw_task, option, handoff)
        path_text = ", ".join(
            str(path).strip() for path in target_info.get("target_paths", []) if str(path).strip()
        )
        suffix = str(target_info.get("suffix") or "").strip()
        pieces: list[str] = []
        if path_text:
            pieces.append(
                f"Patch {path_text} to implement proposal #{option.get('index')}: {title}."
            )
        else:
            pieces.append(f"Implement proposal #{option.get('index')}: {title}.")
        if target_info.get("target_inheritance") == "blocked_by_user_constraint":
            blocked = ", ".join(
                str(path).strip()
                for path in target_info.get("blocked_target_paths", [])
                if str(path).strip()
            )
            if suffix:
                pieces.append(f"User constraint: {suffix[0].upper() + suffix[1:]}.")
            if blocked:
                pieces.append(
                    f"Do not assume {blocked} is the edit target; identify the appropriate target or ask before editing."
                )
        elif suffix:
            pieces.append(suffix[0].upper() + suffix[1:])
        return " ".join(piece.strip() for piece in pieces if piece.strip()).strip()

    def _apply_resolved_followup_metadata(
        self,
        raw_task: str,
        option: dict[str, Any],
        target_info: dict[str, Any],
        effective_task: str,
    ) -> None:
        self.harness.state.scratchpad["_resolved_followup"] = {
            "raw_task": str(raw_task or "").strip(),
            "option_index": option.get("index"),
            "option_title": str(option.get("title") or "").strip(),
            "target_paths": list(target_info.get("target_paths") or []),
            "target_inheritance": str(target_info.get("target_inheritance") or "").strip(),
            "blocked_target_paths": list(target_info.get("blocked_target_paths") or []),
            "effective_task": str(effective_task or "").strip(),
        }

    def resolve_followup_task(self, task: str) -> str:
        raw_task = str(task or "").strip()
        if not raw_task:
            return raw_task
        self.harness.state.scratchpad.pop("_resolved_remote_followup", None)

        handoff = self.last_task_handoff()
        option = self._selected_action_option(raw_task, handoff)
        if option is not None:
            target_info = self._resolve_option_target_paths(raw_task, option, handoff)
            resolved = self._resolved_option_task(raw_task, option, handoff)
            self._apply_resolved_followup_metadata(raw_task, option, target_info, resolved)
            self._store_followup_transaction_for_resolution(raw_task=raw_task, effective_task=resolved)
            return resolved

        remote_resolution = self._remote_followup_resolution(raw_task)
        if remote_resolution is not None:
            self._apply_remote_followup_metadata(raw_task, remote_resolution)
            resolved = str(remote_resolution.get("effective_task") or raw_task).strip()
            self._store_followup_transaction_for_resolution(raw_task=raw_task, effective_task=resolved)
            return resolved

        affirmative_remote_resolution = self._affirmative_remote_execution_followup_resolution(raw_task)
        if affirmative_remote_resolution is not None:
            self._apply_remote_followup_metadata(raw_task, affirmative_remote_resolution)
            resolved = str(affirmative_remote_resolution.get("effective_task") or raw_task).strip()
            self._store_followup_transaction_for_resolution(raw_task=raw_task, effective_task=resolved)
            return resolved

        if self._is_corrective_resteer_followup(raw_task):
            resolved = self._resolved_corrective_resteer_task(raw_task)
            self.harness.state.scratchpad["_resolved_resteer"] = {
                "raw_task": raw_task,
                "effective_task": resolved,
                "kind": "corrective_tool_resteer",
            }
            self._store_followup_transaction_for_resolution(raw_task=raw_task, effective_task=resolved)
            return resolved

        if not self._is_contextual_followup(raw_task):
            return raw_task

        continuity_candidate = self._current_or_handoff_continuity_task()
        candidate = base_task_from_task_chain(
            handoff.get("current_goal")
            or handoff.get("effective_task")
            or self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )
        if self._is_continue_like_followup(raw_task):
            return continuity_candidate or candidate or raw_task
        if self._is_continue_directive_followup(raw_task) and (continuity_candidate or candidate):
            base = continuity_candidate or candidate
            resolved = f"Continue current task: {base}. User follow-up: {raw_task}"
            self._store_followup_transaction_for_resolution(raw_task=raw_task, effective_task=resolved)
            return resolved
        if not candidate:
            return raw_task

        if not (self.has_task_local_context() or handoff):
            return raw_task

        if (
            self._has_contextual_reference_to_current_task(raw_task)
            or self._is_quality_followup(raw_task)
            or self._is_guard_recovery_followup(raw_task)
        ):
            resolved = f"Continue current task: {candidate}. User follow-up: {raw_task}"
            self._store_followup_transaction_for_resolution(raw_task=raw_task, effective_task=resolved)
            return resolved

        self._store_followup_transaction_for_resolution(raw_task=raw_task, effective_task=candidate)
        return candidate

    def store_task_handoff(self, *, raw_task: str, effective_task: str) -> None:
        effective = collapse_task_chain(effective_task)
        if not effective:
            return
        target_paths = extract_task_target_paths(effective)
        current_goal = collapse_task_chain(self.harness.state.working_memory.current_goal or effective)
        task_mode = classify_task_mode(effective)
        active_scope = self._active_task_scope_payload()
        active_task_id = str(active_scope.get("task_id") or "").strip() if active_scope else ""
        ssh_targets = self._handoff_remote_targets(
            {
                "raw_task": str(raw_task or "").strip(),
                "effective_task": effective,
                "current_goal": current_goal,
                "task_mode": task_mode,
            }
        )
        remote_target_paths = self._recent_remote_target_paths()
        if task_mode == "remote_execute" or ssh_targets:
            remote_target_paths = dedupe_keep_tail(
                remote_target_paths
                + self._extract_remote_absolute_paths(effective, current_goal, raw_task),
                limit=12,
            )
        previous = self.last_task_handoff()
        previous_paths = previous.get("target_paths") if isinstance(previous.get("target_paths"), list) else []
        same_task = normalize_task_text(previous.get("effective_task")) == normalize_task_text(effective)
        same_target = bool(set(previous_paths) & set(target_paths))
        existing_options = previous.get("action_options") if (same_task or same_target) else []
        if not isinstance(existing_options, list):
            existing_options = []
        last_good_artifact_ids = self._continuation_artifact_ids(previous)
        recent_research_artifact_ids = self._continuation_research_artifact_ids(previous)
        next_required_tool = self._continuation_next_required_tool(previous)
        last_failed_tool = self._continuation_last_failed_tool(previous)
        ssh_target = self._continuation_primary_ssh_target(ssh_targets, previous)
        transaction = self.harness.state.scratchpad.get("_task_transaction")
        if not isinstance(transaction, dict):
            transaction = {}
        failure_summary = ""
        if isinstance(last_failed_tool, dict) and str(last_failed_tool.get("tool_name") or "").strip():
            failure_summary = self._clip_task_summary_text(
                f"{last_failed_tool.get('tool_name')}: {last_failed_tool.get('error') or ''}",
                limit=220,
            )
        allowed_paths = dedupe_keep_tail(list(target_paths) + list(remote_target_paths), limit=12)
        allowed_artifacts = dedupe_keep_tail(
            list(last_good_artifact_ids) + list(recent_research_artifact_ids),
            limit=8,
        )
        self.harness.state.scratchpad["_last_task_handoff"] = {
            "task_id": active_task_id or str(previous.get("task_id") or "").strip(),
            "status": str(previous.get("status") or "closed").strip(),
            "turn_type": str(transaction.get("turn_type") or previous.get("turn_type") or "NEW_TASK").strip(),
            "raw_task": str(raw_task or "").strip(),
            "effective_task": effective,
            "current_goal": current_goal,
            "user_goal": self._clip_task_summary_text(
                transaction.get("user_goal") if transaction.get("user_goal") else current_goal,
                limit=320,
            ),
            "success_condition": self._clip_task_summary_text(
                transaction.get("success_condition") or "Task is completed and relevant verification is captured.",
                limit=220,
            ),
            "previous_task_relevance": str(
                transaction.get("previous_task_relevance") or previous.get("previous_task_relevance") or "high"
            ).strip(),
            "task_mode": task_mode,
            "active_tool_profiles": list(getattr(self.harness.state, "active_tool_profiles", []) or []),
            "ssh_target": ssh_target,
            "ssh_targets": list(ssh_targets),
            "target_paths": list(target_paths),
            "remote_target_paths": list(remote_target_paths),
            "allowed_paths": list(allowed_paths),
            "allowed_artifacts": list(allowed_artifacts),
            "ignored_context": list(transaction.get("ignored_context") or []),
            "failure_summary": str(transaction.get("failure_summary") or failure_summary).strip(),
            "verification_hint": str(transaction.get("verification_hint") or "").strip(),
            "action_options": list(existing_options),
            "last_good_artifact_ids": list(last_good_artifact_ids),
            "recent_research_artifact_ids": list(recent_research_artifact_ids),
            "next_required_tool": next_required_tool,
            "last_failed_tool": last_failed_tool,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _tool_execution_record_items(self) -> list[dict[str, Any]]:
        records = getattr(self.harness.state, "tool_execution_records", None)
        if not isinstance(records, dict):
            return []
        items = [record for record in records.values() if isinstance(record, dict)]
        return sorted(
            items,
            key=lambda record: (
                int(record.get("step_count") or 0),
                str(record.get("operation_id") or ""),
            ),
        )

    def _continuation_artifact_ids(self, previous: dict[str, Any], *, limit: int = 4) -> list[str]:
        candidates: list[str] = []
        retrieval_cache = getattr(self.harness.state, "retrieval_cache", None)
        if isinstance(retrieval_cache, list):
            candidates.extend(str(item).strip() for item in retrieval_cache if str(item).strip())

        for record in reversed(self._tool_execution_record_items()):
            result = record.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            artifact_id = str(metadata.get("artifact_id") or "").strip()
            if artifact_id:
                candidates.append(artifact_id)

        previous_ids = previous.get("last_good_artifact_ids")
        if isinstance(previous_ids, list):
            candidates.extend(str(item).strip() for item in previous_ids if str(item).strip())
        return dedupe_keep_tail(candidates, limit=limit)

    def _continuation_research_artifact_ids(self, previous: dict[str, Any], *, limit: int = 2) -> list[str]:
        candidates: list[str] = []
        for record in reversed(self._tool_execution_record_items()):
            tool_name = str(record.get("tool_name") or "").strip()
            if tool_name not in {"web_search", "web_fetch"}:
                continue
            result = record.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            artifact_id = str(metadata.get("artifact_id") or "").strip()
            if artifact_id:
                candidates.append(artifact_id)

        previous_ids = previous.get("recent_research_artifact_ids")
        if isinstance(previous_ids, list):
            candidates.extend(str(item).strip() for item in previous_ids if str(item).strip())

        fallback_ids = previous.get("last_good_artifact_ids")
        if isinstance(fallback_ids, list):
            for artifact_id in fallback_ids:
                normalized_id = str(artifact_id or "").strip()
                artifact = self.harness.state.artifacts.get(normalized_id)
                if normalized_id and artifact is not None and str(getattr(artifact, "kind", "")).strip() in {
                    "web_search",
                    "web_fetch",
                }:
                    candidates.append(normalized_id)
        return dedupe_keep_tail(candidates, limit=limit)

    def _continuation_next_required_tool(self, previous: dict[str, Any]) -> dict[str, Any]:
        for record in reversed(self._tool_execution_record_items()):
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            next_required_tool = metadata.get("next_required_tool")
            if isinstance(next_required_tool, dict) and str(next_required_tool.get("tool_name") or "").strip():
                normalized = json_safe_value(next_required_tool)
                return normalized if isinstance(normalized, dict) else {}
        previous_tool = previous.get("next_required_tool")
        if isinstance(previous_tool, dict):
            normalized = json_safe_value(previous_tool)
            return normalized if isinstance(normalized, dict) else {}
        return {}

    def _continuation_last_failed_tool(self, previous: dict[str, Any]) -> dict[str, Any]:
        for record in reversed(self._tool_execution_record_items()):
            result = record.get("result")
            if not isinstance(result, dict) or bool(result.get("success")):
                continue
            tool_name = str(record.get("tool_name") or "").strip()
            if not tool_name:
                continue
            return {
                "tool_name": tool_name,
                "error": clip_text_value(str(result.get("error") or "").strip(), limit=220)[0],
            }
        previous_failed = previous.get("last_failed_tool")
        if isinstance(previous_failed, dict) and str(previous_failed.get("tool_name") or "").strip():
            normalized = json_safe_value(previous_failed)
            return normalized if isinstance(normalized, dict) else {}
        return {}

    def _continuation_primary_ssh_target(
        self,
        ssh_targets: list[dict[str, str]],
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_remote = self.harness.state.scratchpad.get("_resolved_remote_followup")
        if isinstance(resolved_remote, dict):
            host = str(resolved_remote.get("host") or "").strip().lower()
            user = str(resolved_remote.get("user") or "").strip()
            if host:
                payload: dict[str, Any] = {"host": host}
                if user:
                    payload["user"] = user
                return payload
        if len(ssh_targets) == 1:
            target = ssh_targets[0]
            host = str(target.get("host") or "").strip().lower()
            user = str(target.get("user") or "").strip()
            if host:
                payload = {"host": host}
                if user:
                    payload["user"] = user
                return payload
        previous_target = previous.get("ssh_target")
        if isinstance(previous_target, dict) and str(previous_target.get("host") or "").strip():
            normalized = json_safe_value(previous_target)
            return normalized if isinstance(normalized, dict) else {}
        return {}

    def refresh_task_handoff_action_options(self, assistant_text: str) -> None:
        handoff = self.last_task_handoff()
        if not handoff:
            return
        inherited_paths = handoff.get("target_paths")
        if not isinstance(inherited_paths, list):
            inherited_paths = []
        extracted_options = extract_action_options_from_text(assistant_text, list(inherited_paths))
        if not extracted_options:
            return
        existing_options = handoff.get("action_options")
        if not isinstance(existing_options, list):
            existing_options = []
        handoff["action_options"] = merge_action_options(existing_options, extracted_options)
        handoff["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.harness.state.scratchpad["_last_task_handoff"] = handoff

    def initialize_run_brief(self, task: str, *, raw_task: str | None = None) -> None:
        effective_task = collapse_task_chain(task)
        source_task = str(raw_task or effective_task).strip()
        continue_like = self._is_contextual_followup(source_task)
        remote_mission_task = self._remote_followup_mission_task(source_task)
        previous_task = collapse_task_chain(
            self.harness.state.scratchpad.pop("_task_boundary_previous_task", "") or ""
        )
        existing_task = collapse_task_chain(self.harness.state.run_brief.original_task or "")
        if remote_mission_task and "execute approved plan" in effective_task.lower():
            canonical_task = remote_mission_task
        else:
            canonical_task = effective_task or remote_mission_task or existing_task or previous_task

        self.harness.state.run_brief.original_task = canonical_task
        self.harness.state.run_brief.task_contract = derive_task_contract(canonical_task)
        resolved_remote_followup = isinstance(
            self.harness.state.scratchpad.get("_resolved_remote_followup"), dict
        )
        task_mode_source = canonical_task
        if resolved_remote_followup and is_remote_followup_wrapper(effective_task):
            self.harness.state.task_mode = "remote_execute"
        else:
            self.harness.state.task_mode = classify_task_mode(task_mode_source)

        existing_phase_objective = str(self.harness.state.run_brief.current_phase_objective or "").strip()
        if remote_mission_task and effective_task:
            self.harness.state.run_brief.current_phase_objective = (
                f"{self.harness.state.current_phase}: {effective_task}"
            )
        elif continue_like and existing_phase_objective:
            self.harness.state.run_brief.current_phase_objective = existing_phase_objective
        elif effective_task:
            self.harness.state.run_brief.current_phase_objective = (
                f"{self.harness.state.current_phase}: {effective_task}"
            )
        elif not existing_phase_objective:
            self.harness.state.run_brief.current_phase_objective = self.harness.state.current_phase

        existing_goal = collapse_task_chain(self.harness.state.working_memory.current_goal or "")
        plan = self.harness.state.active_plan or self.harness.state.draft_plan
        plan_goal = collapse_task_chain(getattr(plan, "goal", "") or "")
        if self._is_corrective_resteer_followup(source_task) and existing_goal:
            next_goal = existing_goal
        elif remote_mission_task:
            if is_remote_followup_wrapper(effective_task):
                next_goal = canonical_task
            elif plan_goal and normalize_task_text(plan_goal) == normalize_task_text(existing_goal):
                next_goal = plan_goal
            elif existing_goal and not is_remote_followup_wrapper(existing_goal):
                next_goal = existing_goal
            else:
                next_goal = remote_mission_task
        elif continue_like and plan_goal and normalize_task_text(plan_goal) == normalize_task_text(existing_goal):
            next_goal = plan_goal
        else:
            next_goal = canonical_task or existing_goal
        self.harness.state.working_memory.current_goal = next_goal

        self.harness.state.scratchpad["_task_target_paths"] = extract_task_target_paths(effective_task)
        self.store_task_handoff(raw_task=source_task, effective_task=effective_task)
        if self._ordinal_followup_index(source_task) is not None:
            handoff = self.last_task_handoff()
            option = self._selected_action_option(source_task, handoff)
            if option is not None:
                target_info = self._resolve_option_target_paths(source_task, option, handoff)
                self._apply_resolved_followup_metadata(
                    source_task,
                    option,
                    target_info,
                    self._resolved_option_task(source_task, option, handoff),
                )
        if hasattr(self.harness.memory, "prime_write_policy"):
            self.harness.memory.prime_write_policy(effective_task)
        self.harness.state.working_memory.next_actions = dedupe_keep_tail(
            self.harness.state.working_memory.next_actions + [next_action_for_task(self.harness, effective_task)],
            limit=6,
        )

    def current_user_task(self) -> str:
        for message in reversed(self.harness.state.recent_messages):
            if message.role == "user" and message.content:
                content = str(message.content or "").strip()
                resolved = self.resolve_followup_task(content)
                return collapse_task_chain(resolved or content)
        last_task = self.harness.state.scratchpad.get("_last_task_text")
        if isinstance(last_task, str) and last_task:
            return collapse_task_chain(last_task)
        return collapse_task_chain(self.harness.state.run_brief.original_task)
