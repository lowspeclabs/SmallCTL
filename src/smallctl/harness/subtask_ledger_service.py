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

    def import_plan_if_needed(self) -> None:
        state = self.harness.state
        plan = state.active_plan or state.draft_plan
        if plan is None:
            self._enforce_limits()
            return
        task_text = str(getattr(plan, "goal", "") or state.run_brief.original_task or "").strip()
        ledger = self.ensure_ledger(task_text)
        existing = {task.subtask_id for task in ledger.subtasks}
        imported: list[Subtask] = []
        for index, step in enumerate(plan.iter_steps(), start=1):
            subtask_id = str(getattr(step, "step_id", "") or f"S{index}").strip()
            if subtask_id in existing:
                continue
            status = _status_from_plan_status(str(getattr(step, "status", "") or "pending"))
            imported.append(
                Subtask(
                    subtask_id=subtask_id,
                    title=str(getattr(step, "title", "") or subtask_id),
                    goal=str(getattr(step, "description", "") or getattr(step, "task", "") or getattr(step, "title", "") or ""),
                    status=status,
                    acceptance=list(getattr(step, "acceptance", []) or []),
                )
            )
        if imported:
            ledger.subtasks.extend(imported)
            increment_metric(state, "subtasks_created", len(imported))
            _runlog(self.harness, "subtasks_imported", count=len(imported))
        if not ledger.active_subtask_id or ledger.active() is None:
            active = next((task for task in ledger.subtasks if task.status == "active"), None)
            active = active or next((task for task in ledger.subtasks if task.status == "pending"), None)
            if active is not None:
                active.status = "active"
                ledger.active_subtask_id = active.subtask_id
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
            if task.status in {"pending", "blocked"}:
                task.status = "active"
                task.updated_at = time()
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
        if task.status == "active":
            increment_metric(self.harness.state, "subtasks_blocked")
        task.updated_at = time()

    def mark_done_if_verified(self, subtask_id: str, verifier: dict[str, Any] | None) -> bool:
        if not isinstance(verifier, dict):
            return False
        if str(verifier.get("verdict") or "").strip().lower() != "pass":
            return False
        task = self._find(subtask_id)
        if task is None:
            return False
        task.status = "done"
        task.updated_at = time()
        increment_metric(self.harness.state, "subtasks_completed")
        ledger = self.harness.state.subtask_ledger
        if ledger is not None and ledger.active_subtask_id == subtask_id:
            next_task = next((item for item in ledger.subtasks if item.status == "pending"), None)
            if next_task is not None:
                next_task.status = "active"
                next_task.updated_at = time()
                ledger.active_subtask_id = next_task.subtask_id
            else:
                ledger.active_subtask_id = None
        return True

    def handle_human_resteer(self, user_text: str) -> None:
        ledger = self.ensure_ledger(str(user_text or "").strip() or "Follow latest user direction")
        active = ledger.active()
        if active is not None:
            active.status = "abandoned"
            active.updated_at = time()
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
            task.status = "pending"
            task.updated_at = time()
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
