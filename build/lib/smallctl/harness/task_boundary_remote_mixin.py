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


class TaskBoundaryRemoteMixin:

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
