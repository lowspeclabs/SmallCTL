from __future__ import annotations

from typing import Any

from .tool_plan_schema import ToolPlan, ToolPlanStep


def build_execution_dag(plan: ToolPlan) -> list[list[ToolPlanStep]]:
    """Return topologically sorted batches; each batch can run in parallel."""
    steps_by_id: dict[str, ToolPlanStep] = {step.id: step for step in plan.steps}
    pending: set[str] = set(steps_by_id.keys())
    completed: set[str] = set()
    batches: list[list[ToolPlanStep]] = []

    while pending:
        batch = [
            steps_by_id[sid]
            for sid in pending
            if all(dep in completed for dep in steps_by_id[sid].depends_on)
        ]
        if not batch:
            # Cyclic dependency — fall back to serial order of remaining steps
            batch = [steps_by_id[sid] for sid in pending]
            pending.clear()
        else:
            for step in batch:
                pending.remove(step.id)
                completed.add(step.id)
        batches.append(batch)

    return batches
