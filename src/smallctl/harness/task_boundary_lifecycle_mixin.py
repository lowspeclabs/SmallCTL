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


class TaskBoundaryLifecycleMixin:

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
