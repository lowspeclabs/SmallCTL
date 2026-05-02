from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..state import ExecutionPlan, LoopState, PlanStep, StepEvidenceArtifact


@dataclass
class PlanValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class PlanExecutionEngine:
    def __init__(self, state: LoopState):
        self.state = state

    def validate_plan(self, plan: ExecutionPlan) -> PlanValidationResult:
        steps = plan.iter_steps()
        errors: list[str] = []
        if any(step.substeps for step in steps):
            errors.append("Staged execution does not support substeps yet; flatten the plan first.")
        by_id = {step.step_id: step for step in steps}
        if len(by_id) != len(steps):
            errors.append("Plan contains duplicate step IDs.")
        for step in steps:
            for dependency_id in step.depends_on:
                if dependency_id not in by_id:
                    errors.append(f"Step {step.step_id} depends on unknown step {dependency_id}.")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            if step_id in visiting:
                errors.append(f"Plan dependency cycle includes step {step_id}.")
                return
            visiting.add(step_id)
            step = by_id.get(step_id)
            if step is not None:
                for dependency_id in step.depends_on:
                    visit(dependency_id)
            visiting.discard(step_id)
            visited.add(step_id)

        for step_id in by_id:
            visit(step_id)
        return PlanValidationResult(valid=not errors, errors=errors)

    def ready_step_ids(self, plan: ExecutionPlan) -> list[str]:
        by_id = {step.step_id: step for step in plan.iter_steps()}
        ready: list[str] = []
        for step in plan.iter_steps():
            if step.status not in {"pending", ""}:
                continue
            dependencies_ready = True
            for dependency_id in step.depends_on:
                dependency = by_id.get(dependency_id)
                if dependency is None or dependency.status not in {"completed", "skipped"}:
                    dependencies_ready = False
                    break
            if dependencies_ready:
                ready.append(step.step_id)
        return ready

    def get_next_step(self, plan: ExecutionPlan) -> PlanStep | None:
        active_id = str(getattr(self.state, "active_step_id", "") or "").strip()
        if active_id:
            active = plan.find_step(active_id)
            if active is not None and active.status == "in_progress":
                return active
        ready = self.ready_step_ids(plan)
        return plan.find_step(ready[0]) if ready else None

    def activate_step(self, plan: ExecutionPlan, step_id: str) -> PlanStep:
        step = plan.find_step(step_id)
        if step is None:
            raise ValueError(f"Unknown plan step: {step_id}")
        self.state.plan_execution_mode = True
        self.state.active_plan = plan
        self.state.active_step_id = step.step_id
        self.state.active_step_run_id = f"{step.step_id}-{uuid.uuid4().hex[:10]}"
        self.state.step_sandbox_history = []
        for key in (
            "_step_complete_requested",
            "_step_complete_message",
            "_step_failed_requested",
            "_step_failed_message",
        ):
            self.state.scratchpad.pop(key, None)
        step.status = "in_progress"
        self._capture_file_baseline(step)
        plan.touch()
        self.state.sync_plan_mirror()
        self.state.touch()
        return step

    def complete_step(self, plan: ExecutionPlan, step_id: str, evidence: StepEvidenceArtifact) -> None:
        step = plan.find_step(step_id)
        if step is None:
            raise ValueError(f"Unknown plan step: {step_id}")
        step.status = "completed"
        step.evidence_refs = list(dict.fromkeys([*step.evidence_refs, *evidence.artifact_ids]))
        self.state.step_evidence[step_id] = evidence
        self._clear_active_step_flags()
        plan.touch()
        self.state.sync_plan_mirror()
        self.state.touch()

    def fail_step(self, plan: ExecutionPlan, step_id: str, reason: str) -> None:
        step = plan.find_step(step_id)
        if step is None:
            raise ValueError(f"Unknown plan step: {step_id}")
        step.retry_count = int(step.retry_count or 0) + 1
        if reason:
            step.failure_reasons = [*step.failure_reasons, reason][-10:]
        if step.retry_count < int(step.max_retries or 0):
            step.status = "pending"
            self._clear_active_step_flags()
        else:
            self.block_step(plan, step_id, reason or "Step failed.")
            return
        plan.touch()
        self.state.sync_plan_mirror()
        self.state.touch()

    def block_step(self, plan: ExecutionPlan, step_id: str, reason: str) -> None:
        step = plan.find_step(step_id)
        if step is None:
            raise ValueError(f"Unknown plan step: {step_id}")
        step.status = "blocked"
        if reason:
            step.failure_reasons = [*step.failure_reasons, reason][-10:]
        for dependent in plan.iter_steps():
            if step_id in dependent.depends_on and dependent.status == "pending":
                dependent.status = "blocked"
                dependent.failure_reasons = [*dependent.failure_reasons, f"Dependency {step_id} blocked."][-10:]
        self.state.pending_interrupt = {
            "kind": "staged_step_blocked",
            "question": f"Step {step_id} is blocked: {reason}",
            "step_id": step_id,
            "plan_id": plan.plan_id,
            "response_mode": "revise/skip/retry",
        }
        self._clear_active_step_flags()
        plan.touch()
        self.state.sync_plan_mirror()
        self.state.touch()

    def is_plan_complete(self, plan: ExecutionPlan) -> bool:
        return bool(plan.steps) and all(step.status in {"completed", "skipped"} for step in plan.iter_steps())

    def _clear_active_step_flags(self) -> None:
        self.state.active_step_id = ""
        self.state.active_step_run_id = ""
        self.state.step_sandbox_history = []
        for key in (
            "_step_complete_requested",
            "_step_complete_message",
            "_step_failed_requested",
            "_step_failed_message",
        ):
            self.state.scratchpad.pop(key, None)

    def _capture_file_baseline(self, step: PlanStep) -> None:
        baseline: dict[str, float] = {}
        for spec in step.verifiers:
            if spec.kind not in {"file_changed", "file_exists", "syntax_ok"}:
                continue
            path = str(spec.args.get("path") or spec.args.get("ref") or "").strip()
            if not path:
                continue
            resolved = Path(self.state.cwd) / path
            try:
                baseline[str(resolved)] = resolved.stat().st_mtime
            except OSError:
                baseline[str(resolved)] = 0.0
        self.state.scratchpad["_staged_step_file_baseline"] = baseline
