from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..state import (
    PromptBudgetSnapshot,
    RunBrief,
    WorkingMemory,
    clip_text_value,
    json_safe_value,
)
from ..task_targets import extract_task_target_paths
from ..normalization import dedupe_keep_tail
from .task_classifier import classify_task_mode
from .task_intent import derive_task_contract, next_action_for_task

_SYSTEM_FOLLOW_UP_SPLIT_RE = re.compile(r"\nFollow-up:\s*", re.IGNORECASE)
_FOLLOWUP_FILLERS = {"please", "pls", "now", "again", "just", "then", "more", "further"}
_AFFIRMATIVE_FOLLOWUPS = {
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "do it",
    "please do",
    "go ahead",
    "run it",
    "execute",
    "approve",
    "approved",
}
_ACTION_CONFIRMATION_PROMPTS = (
    "would you like me to",
    "do you want me to",
    "should i",
    "shall i",
    "want me to",
    "ready for me to",
)


def _normalize_task_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _collapse_task_chain(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in _SYSTEM_FOLLOW_UP_SPLIT_RE.split(text) if part.strip()]
    return parts[-1] if parts else text


class TaskBoundaryService:
    def __init__(self, harness: Any):
        self.harness = harness

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

    def begin_task_scope(self, *, raw_task: str, effective_task: str) -> dict[str, Any]:
        normalized_raw = str(raw_task or "").strip()
        normalized_effective = str(effective_task or normalized_raw).strip()
        current = self._active_task_scope_payload()
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
            "_session_notepad",
            "_session_ssh_targets",
            "_last_task_text",
            "_last_task_handoff",
            "_task_boundary_previous_task",
            "_task_sequence",
        ):
            if key in self.harness.state.scratchpad:
                preserved_scratchpad[key] = self.harness.state.scratchpad[key]
        if preserved_previous_task:
            preserved_scratchpad["_task_boundary_previous_task"] = preserved_previous_task

        background_processes = json_safe_value(self.harness.state.background_processes)

        self.harness.state.current_phase = self.harness._initial_phase
        self.harness.state.step_count = 0
        self.harness.state.inactive_steps = 0
        self.harness.state.latest_verdict = None

        self.harness.state.scratchpad = preserved_scratchpad
        self.harness.state.recent_messages = []
        self.harness.state.recent_errors = []
        self.harness.state.run_brief = RunBrief()
        self.harness.state.working_memory = WorkingMemory()
        self.harness.state.acceptance_ledger = {}
        self.harness.state.acceptance_waivers = []
        self.harness.state.acceptance_waived = False
        self.harness.state.last_verifier_verdict = None
        self.harness.state.last_failure_class = ""

        self.harness.state.files_changed_this_cycle = []
        self.harness.state.repair_cycle_id = ""
        self.harness.state.stagnation_counters = {}
        self.harness.state.draft_plan = None
        self.harness.state.active_plan = None
        self.harness.state.plan_resolved = False
        self.harness.state.plan_artifact_id = ""
        self.harness.state.planning_mode_enabled = self.harness._configured_planning_mode
        self.harness.state.planner_requested_output_path = ""
        self.harness.state.planner_requested_output_format = ""
        self.harness.state.planner_resume_target_mode = "loop"
        self.harness.state.planner_interrupt = None
        self.harness.state.artifacts = {}
        if self.harness.state.write_session and self.harness.state.write_session.status != "complete":
            self.harness._runlog(
                "write_session_abandoned",
                "incomplete write session abandoned on task switch",
                session_id=self.harness.state.write_session.write_session_id,
                stage_target=self.harness.state.write_session.write_target_path,
                status=self.harness.state.write_session.status,
            )
            from ..graph.tool_outcomes import _register_write_session_stage_artifact
            _register_write_session_stage_artifact(self.harness, self.harness.state.write_session)
            self.harness.state.write_session = None
        self.harness.state.episodic_summaries = []
        self.harness.state.context_briefs = []
        self.harness.state.prompt_budget = PromptBudgetSnapshot()
        self.harness.state.retrieval_cache = []
        self.harness.state.task_mode = ""
        self.harness.state.active_intent = ""
        self.harness.state.secondary_intents = []
        self.harness.state.intent_tags = []
        self.harness.state.retrieved_experience_ids = []
        self.harness.state.tool_execution_records = {}
        self.harness.state.pending_interrupt = None
        self.harness.state.tool_history = []
        self.harness.state.background_processes = background_processes if isinstance(background_processes, dict) else {}
        self.harness.state.warm_experiences = []
        self.harness.state.touch()
        self.harness._runlog(
            "task_boundary_reset",
            "reset task-local state for new task",
            reason=reason,
            previous_task=previous_task,
            new_task=new_task,
        )

    def maybe_reset_for_new_task(self, task: str) -> None:
        previous_task = _collapse_task_chain(
            self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )
        if not previous_task:
            return
        new_task = _collapse_task_chain(task)
        if not new_task or _normalize_task_text(new_task) == _normalize_task_text(previous_task):
            return
        has_task_local_context = self.has_task_local_context()
        if has_task_local_context:
            self.finalize_task_scope(
                terminal_event="task_aborted",
                status="aborted",
                reason="replaced_by_new_task",
                replacement_task=new_task,
            )
            self.reset_task_boundary_state(
                reason="task_switch",
                new_task=new_task,
                previous_task=previous_task,
            )

    def has_task_local_context(self) -> bool:
        return bool(
            self.harness.state.recent_messages
            or self.harness.state.recent_errors
            or self.harness.state.artifacts
            or self.harness.state.episodic_summaries
            or self.harness.state.context_briefs
            or self.harness.state.run_brief.task_contract
            or self.harness.state.run_brief.current_phase_objective
            or self.harness.state.working_memory.current_goal
            or self.harness.state.working_memory.plan
            or self.harness.state.working_memory.decisions
            or self.harness.state.working_memory.open_questions
            or self.harness.state.working_memory.known_facts
            or self.harness.state.working_memory.failures
            or self.harness.state.working_memory.next_actions
            or self.harness.state.acceptance_ledger
            or self.harness.state.acceptance_waivers
            or self.harness.state.scratchpad.get("_task_complete")
            or self.harness.state.scratchpad.get("_task_failed")
        )

    def last_task_handoff(self) -> dict[str, Any]:
        payload = self.harness.state.scratchpad.get("_last_task_handoff")
        if not isinstance(payload, dict):
            return {}
        return dict(payload)

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

    def _is_affirmative_followup(self, task: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(task or "").strip().lower()).strip()
        if not normalized:
            return False
        tokens = [token for token in normalized.split() if token not in _FOLLOWUP_FILLERS]
        collapsed = " ".join(tokens)
        return collapsed in _AFFIRMATIVE_FOLLOWUPS

    def _recent_assistant_requested_action_confirmation(self) -> bool:
        for message in reversed(self.harness.state.recent_messages[-8:]):
            if getattr(message, "role", "") != "assistant":
                continue
            text = str(getattr(message, "content", "") or "").strip().lower()
            if not text:
                continue
            if any(prompt in text for prompt in _ACTION_CONFIRMATION_PROMPTS):
                return True
        return False

    def _is_contextual_followup(self, task: str) -> bool:
        if self._is_continue_like_followup(task):
            return True
        if not self._is_affirmative_followup(task):
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        return self._recent_assistant_requested_action_confirmation()

    def resolve_followup_task(self, task: str) -> str:
        raw_task = str(task or "").strip()
        if not raw_task or not self._is_contextual_followup(raw_task):
            return raw_task

        handoff = self.last_task_handoff()
        candidate = _collapse_task_chain(
            handoff.get("effective_task")
            or handoff.get("current_goal")
            or self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )
        if not candidate:
            return raw_task

        if not (self.has_task_local_context() or handoff):
            return raw_task

        return candidate

    def store_task_handoff(self, *, raw_task: str, effective_task: str) -> None:
        effective = _collapse_task_chain(effective_task)
        if not effective:
            return
        target_paths = extract_task_target_paths(effective)
        self.harness.state.scratchpad["_last_task_handoff"] = {
            "raw_task": str(raw_task or "").strip(),
            "effective_task": effective,
            "current_goal": _collapse_task_chain(self.harness.state.working_memory.current_goal or effective),
            "target_paths": list(target_paths),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def initialize_run_brief(self, task: str, *, raw_task: str | None = None) -> None:
        effective_task = _collapse_task_chain(task)
        source_task = str(raw_task or effective_task).strip()
        continue_like = self._is_contextual_followup(source_task)
        previous_task = _collapse_task_chain(
            self.harness.state.scratchpad.pop("_task_boundary_previous_task", "") or ""
        )
        existing_task = _collapse_task_chain(self.harness.state.run_brief.original_task or "")
        canonical_task = effective_task or existing_task or previous_task

        self.harness.state.run_brief.original_task = canonical_task
        self.harness.state.run_brief.task_contract = derive_task_contract(canonical_task)
        self.harness.state.task_mode = classify_task_mode(canonical_task)

        existing_phase_objective = str(self.harness.state.run_brief.current_phase_objective or "").strip()
        if continue_like and existing_phase_objective:
            self.harness.state.run_brief.current_phase_objective = existing_phase_objective
        elif effective_task:
            self.harness.state.run_brief.current_phase_objective = (
                f"{self.harness.state.current_phase}: {effective_task}"
            )
        elif not existing_phase_objective:
            self.harness.state.run_brief.current_phase_objective = self.harness.state.current_phase

        existing_goal = _collapse_task_chain(self.harness.state.working_memory.current_goal or "")
        plan = self.harness.state.active_plan or self.harness.state.draft_plan
        plan_goal = _collapse_task_chain(getattr(plan, "goal", "") or "")
        if continue_like and plan_goal and _normalize_task_text(plan_goal) == _normalize_task_text(existing_goal):
            next_goal = plan_goal
        else:
            next_goal = canonical_task or existing_goal
        self.harness.state.working_memory.current_goal = next_goal

        self.harness.state.scratchpad["_task_target_paths"] = extract_task_target_paths(effective_task)
        self.store_task_handoff(raw_task=source_task, effective_task=effective_task)
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
                return _collapse_task_chain(resolved or content)
        last_task = self.harness.state.scratchpad.get("_last_task_text")
        if isinstance(last_task, str) and last_task:
            return _collapse_task_chain(last_task)
        return _collapse_task_chain(self.harness.state.run_brief.original_task)
