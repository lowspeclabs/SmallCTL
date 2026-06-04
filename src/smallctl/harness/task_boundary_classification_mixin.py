from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from time import time
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


class TaskBoundaryClassificationMixin:

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
