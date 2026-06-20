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


class TaskBoundaryHandoffMixin:

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
            payload: dict[str, Any] = {
                "tool_name": tool_name,
                "error": clip_text_value(str(result.get("error") or "").strip(), limit=220)[0],
            }
            metadata = record.get("metadata")
            if isinstance(metadata, dict) and bool(metadata.get("approval_denied")):
                payload["approval_denied"] = True
            return payload
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
        resolved_followup = self.harness.state.scratchpad.get("_resolved_followup")
        resolved_followup_task = ""
        if isinstance(resolved_followup, dict):
            resolved_followup_task = collapse_task_chain(resolved_followup.get("effective_task") or "")
        if self._is_corrective_resteer_followup(source_task) and existing_goal:
            next_goal = existing_goal
        elif (
            continue_like
            and resolved_followup_task
            and normalize_task_text(resolved_followup_task) == normalize_task_text(effective_task)
        ):
            next_goal = resolved_followup_task
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
        goal_changed = normalize_task_text(existing_goal) != normalize_task_text(next_goal)
        self.harness.state.working_memory.current_goal = next_goal

        self.harness.state.scratchpad["_task_target_paths"] = extract_task_target_paths(effective_task)
        self.store_task_handoff(raw_task=source_task, effective_task=effective_task)
        transaction = self.harness.state.scratchpad.get("_task_transaction")
        if isinstance(transaction, dict) and goal_changed:
            transaction["previous_goal"] = str(existing_goal)[:500]
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
