from __future__ import annotations

import hashlib
from time import time
from typing import Any

from ..recovery_metrics import increment_metric
from ..recovery_schema import FailureEvent, Subtask, SubtaskLedger


class SubtaskLedgerService:
    def __init__(self, harness: Any) -> None:
        self.harness = harness

    def ensure_ledger(self, task_text: str) -> SubtaskLedger:
        state = self.harness.state
        if state.subtask_ledger is None:
            state.subtask_ledger = SubtaskLedger(
                task_id=_task_id(task_text),
                subtasks=[],
                active_subtask_id=None,
            )
        if not state.subtask_ledger.subtasks:
            root = Subtask(
                subtask_id="S1",
                title="Complete user task",
                goal=task_text,
                acceptance=["User request satisfied with tool-backed evidence when needed."],
                status="active",
            )
            state.subtask_ledger.subtasks.append(root)
            state.subtask_ledger.active_subtask_id = root.subtask_id
            increment_metric(state, "subtasks_created")
            _runlog(self.harness, "subtask_created", subtask=root)
        self._enforce_limits()
        return state.subtask_ledger

    def import_plan_if_needed(self, *, replace_synthetic_root: bool = False) -> None:
        state = self.harness.state
        plan = state.active_plan or state.draft_plan
        if plan is None:
            self._enforce_limits()
            return
        task_text = str(getattr(plan, "goal", "") or state.run_brief.original_task or "").strip()
        plan_steps = list(plan.iter_steps())
        if state.subtask_ledger is None:
            state.subtask_ledger = SubtaskLedger(
                task_id=_task_id(task_text),
                subtasks=[],
                active_subtask_id=None,
            )
        ledger = state.subtask_ledger
        transferred_evidence: list[str] = []
        transferred_blockers: list[str] = []
        transferred_failures: list[str] = []
        transferred_attempts = 0
        if replace_synthetic_root or plan_steps:
            retained: list[Subtask] = []
            for task in ledger.subtasks:
                if _is_synthetic_root(task):
                    transferred_evidence.extend(task.evidence)
                    transferred_blockers.extend(task.blockers)
                    transferred_failures.extend(task.failure_classes)
                    transferred_attempts += int(task.attempts or 0)
                    if ledger.active_subtask_id == task.subtask_id:
                        ledger.active_subtask_id = None
                    _log_transition(self.harness, task, task.status, "abandoned", reason="plan_import_replaced_root")
                    continue
                retained.append(task)
            ledger.subtasks = retained
        existing = {task.subtask_id for task in ledger.subtasks}
        existing_titles = {_subtask_display_key(task) for task in ledger.subtasks if _subtask_display_key(task)}
        plan_goal_key = _normalize_task_text(task_text)
        imported: list[Subtask] = []
        for index, step in enumerate(plan_steps, start=1):
            subtask_id = str(getattr(step, "step_id", "") or f"S{index}").strip()
            if subtask_id in existing:
                continue
            title = str(getattr(step, "title", "") or subtask_id)
            goal = str(getattr(step, "description", "") or getattr(step, "task", "") or getattr(step, "title", "") or "")
            display_key = _normalize_task_text(title or goal)
            if display_key and (display_key == plan_goal_key or display_key in existing_titles):
                continue
            status = _status_from_plan_status(str(getattr(step, "status", "") or "pending"))
            imported.append(
                Subtask(
                    subtask_id=subtask_id,
                    title=title,
                    goal=goal,
                    status=status,
                    acceptance=list(getattr(step, "acceptance", []) or []),
                )
            )
            if display_key:
                existing_titles.add(display_key)
        if imported:
            ledger.subtasks.extend(imported)
            increment_metric(state, "subtasks_created", len(imported))
            _runlog(self.harness, "subtasks_imported", count=len(imported))
        if (transferred_evidence or transferred_blockers or transferred_failures or transferred_attempts) and ledger.subtasks:
            target = next((task for task in ledger.subtasks if task.status in {"active", "pending"}), ledger.subtasks[0])
            target.evidence = _dedupe_list(target.evidence + transferred_evidence)[-6:]
            target.blockers = _dedupe_list(target.blockers + transferred_blockers)[-5:]
            target.failure_classes = _dedupe_list(target.failure_classes + transferred_failures)[-5:]
            target.attempts += transferred_attempts
        if not ledger.active_subtask_id or ledger.active() is None:
            active = next((task for task in ledger.subtasks if task.status == "active"), None)
            active = active or next((task for task in ledger.subtasks if task.status == "pending"), None)
            active = active or next((task for task in ledger.subtasks if task.status == "blocked"), None)
            if active is not None:
                old_status = active.status
                if active.status == "pending":
                    active.status = "active"
                ledger.active_subtask_id = active.subtask_id
                _log_transition(self.harness, active, old_status, active.status, reason="import_plan_activate")
        self._enforce_limits()

    def infer_or_create_active_subtask(self, graph_state: Any = None) -> Subtask:
        del graph_state
        state = self.harness.state
        task_text = str(state.run_brief.original_task or state.run_brief.current_phase_objective or "").strip()
        ledger = self.ensure_ledger(task_text or "Complete user task")
        active = ledger.active()
        if active is not None:
            return active
        for task in ledger.subtasks:
            if task.status == "pending":
                old_status = task.status
                task.status = "active"
                task.updated_at = time()
                ledger.active_subtask_id = task.subtask_id
                _log_transition(self.harness, task, old_status, task.status, reason="infer_active_subtask")
                return task
        for task in ledger.subtasks:
            if task.status == "blocked":
                ledger.active_subtask_id = task.subtask_id
                return task
        subtask = Subtask(
            subtask_id=f"S{len(ledger.subtasks) + 1}",
            title="Continue user task",
            goal=task_text or "Complete user task",
            status="active",
        )
        ledger.subtasks.append(subtask)
        ledger.active_subtask_id = subtask.subtask_id
        increment_metric(state, "subtasks_created")
        self._enforce_limits()
        _runlog(self.harness, "subtask_created", subtask=subtask)
        return subtask

    def mark_attempt(self, subtask_id: str) -> None:
        task = self._find(subtask_id)
        if task is None:
            return
        task.attempts += 1
        task.updated_at = time()

    def attach_evidence(self, subtask_id: str, evidence: str) -> None:
        task = self._find(subtask_id)
        text = str(evidence or "").strip()
        if task is None or not text:
            return
        if text not in task.evidence:
            task.evidence.append(text[:240])
            task.evidence = task.evidence[-6:]
        task.updated_at = time()

    def attach_failure(self, subtask_id: str, failure: FailureEvent) -> None:
        task = self._find(subtask_id)
        if task is None:
            return
        task.attempts += 1
        if failure.failure_class not in task.failure_classes:
            task.failure_classes.append(failure.failure_class)
        if failure.message and failure.message not in task.blockers:
            task.blockers.append(failure.message[:240])
            task.blockers = task.blockers[-5:]
        if failure.suggested_next_action:
            task.next_action = failure.suggested_next_action
        if task.status == "active" and task.attempts >= _blocked_attempt_threshold(self.harness):
            old_status = task.status
            task.status = "blocked"
            if not task.next_action:
                task.next_action = (
                    "Roadblock reached. Call escalate_to_bigger_model for stronger reasoning, "
                    "or ask_human if progress requires missing information or approval."
                )
            increment_metric(self.harness.state, "subtasks_blocked")
            _log_transition(self.harness, task, old_status, task.status, reason="failure_threshold")
        task.updated_at = time()

    def mark_done_if_verified(self, subtask_id: str, verifier: dict[str, Any] | None) -> bool:
        if not isinstance(verifier, dict):
            return False
        if str(verifier.get("verdict") or "").strip().lower() != "pass":
            return False
        task = self._find(subtask_id)
        if task is None:
            return False
        old_status = task.status
        task.status = "done"
        task.updated_at = time()
        increment_metric(self.harness.state, "subtasks_completed")
        _log_transition(self.harness, task, old_status, task.status, reason="verifier_passed")
        ledger = self.harness.state.subtask_ledger
        if ledger is not None and ledger.active_subtask_id == subtask_id:
            next_task = next((item for item in ledger.subtasks if item.status == "pending"), None)
            if next_task is not None:
                next_old_status = next_task.status
                next_task.status = "active"
                next_task.updated_at = time()
                ledger.active_subtask_id = next_task.subtask_id
                _log_transition(self.harness, next_task, next_old_status, next_task.status, reason="previous_done")
            else:
                ledger.active_subtask_id = None
        return True

    def handle_human_resteer(self, user_text: str) -> None:
        ledger = self.ensure_ledger(str(user_text or "").strip() or "Follow latest user direction")
        active = ledger.active()
        if active is not None:
            old_status = active.status
            active.status = "abandoned"
            active.updated_at = time()
            _log_transition(self.harness, active, old_status, active.status, reason="human_resteer")
        subtask = Subtask(
            subtask_id=f"S{len(ledger.subtasks) + 1}",
            title="Follow latest user direction",
            goal=str(user_text or "").strip(),
            status="active",
        )
        ledger.subtasks.append(subtask)
        ledger.active_subtask_id = subtask.subtask_id
        increment_metric(self.harness.state, "subtasks_created")
        self._enforce_limits()
        _runlog(self.harness, "subtask_created", subtask=subtask)

    def _find(self, subtask_id: str) -> Subtask | None:
        ledger = getattr(self.harness.state, "subtask_ledger", None)
        if ledger is None:
            return None
        wanted = str(subtask_id or "").strip()
        return next((task for task in ledger.subtasks if task.subtask_id == wanted), None)

    def _enforce_limits(self) -> None:
        ledger = getattr(self.harness.state, "subtask_ledger", None)
        if ledger is None:
            return
        config = getattr(self.harness, "config", None)
        try:
            max_active = max(1, int(getattr(config, "subtask_max_active", 1) or 1))
        except (TypeError, ValueError):
            max_active = 1
        active_seen = 0
        for task in ledger.subtasks:
            if task.status != "active":
                continue
            if active_seen < max_active:
                if active_seen == 0:
                    ledger.active_subtask_id = task.subtask_id
                active_seen += 1
                continue
            old_status = task.status
            task.status = "pending"
            task.updated_at = time()
            _log_transition(self.harness, task, old_status, task.status, reason="max_active_limit")
        if ledger.active_subtask_id and ledger.active() is None:
            ledger.active_subtask_id = None

        try:
            max_history = max(0, int(getattr(config, "subtask_max_history", 12) or 12))
        except (TypeError, ValueError):
            max_history = 12
        if max_history <= 0:
            return
        terminal_statuses = {"done", "failed", "abandoned"}
        terminal = [task for task in ledger.subtasks if task.status in terminal_statuses]
        if len(terminal) <= max_history:
            return
        remove_ids = {task.subtask_id for task in terminal[: len(terminal) - max_history]}
        ledger.subtasks = [task for task in ledger.subtasks if task.subtask_id not in remove_ids]


def _task_id(task_text: str) -> str | None:
    text = str(task_text or "").strip()
    if not text:
        return None
    return "task-" + hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _is_synthetic_root(task: Subtask) -> bool:
    return (
        str(task.subtask_id or "") == "S1"
        and str(task.title or "") == "Complete user task"
        and str(task.goal or "").strip()
        and list(task.acceptance or []) == ["User request satisfied with tool-backed evidence when needed."]
    )


def _dedupe_list(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _subtask_display_key(task: Subtask) -> str:
    title = str(task.title or "").strip()
    if title and title not in {"Complete user task", "Continue user task", "Follow latest user direction"}:
        return _normalize_task_text(title)
    return _normalize_task_text(str(task.goal or "").strip() or title)


def _normalize_task_text(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    for prefix in ("execute:", "planning:", "repair:", "verify:", "verification:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    return normalized


def _blocked_attempt_threshold(harness: Any) -> int:
    config = getattr(harness, "config", None)
    try:
        return max(1, int(getattr(config, "subtask_block_after_failures", 3) or 3))
    except (TypeError, ValueError):
        return 3


def _status_from_plan_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"done", "completed", "pass", "passed"}:
        return "done"
    if normalized in {"failed", "fail"}:
        return "failed"
    if normalized in {"blocked"}:
        return "blocked"
    if normalized in {"active", "in_progress", "running"}:
        return "active"
    return "pending"


def _runlog(harness: Any, event: str, *, subtask: Subtask | None = None, count: int = 0) -> None:
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            event,
            "Subtask ledger updated",
            subtask_id=getattr(subtask, "subtask_id", ""),
            title=getattr(subtask, "title", ""),
            count=count,
        )


def _log_transition(harness: Any, subtask: Subtask, old_status: str, new_status: str, *, reason: str = "") -> None:
    if old_status == new_status:
        return
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "subtask_transition",
            f"Subtask {subtask.subtask_id} {old_status} -> {new_status}",
            subtask_id=subtask.subtask_id,
            title=subtask.title,
            old_status=old_status,
            new_status=new_status,
            reason=reason,
            attempts=subtask.attempts,
            blockers=subtask.blockers[-3:],
        )
