from __future__ import annotations

from .tool_plan_schema import PARALLELIZABLE_TOOL_PLAN_TOOLS, ToolPlanStep


class NonParallelizableStepInDAGError(Exception):
    """Raised when a DAG batch contains a tool outside the parallel allowlist."""

    pass


class MutatingStepInDAGError(NonParallelizableStepInDAGError):
    """Compatibility alias for callers that still catch the old safety error."""

    pass


def assert_parallelizable_steps(batches: list[list[ToolPlanStep]]) -> None:
    """Hard-abort if any step is outside the parallel ToolPlan allowlist."""
    for batch in batches:
        for step in batch:
            if step.tool not in PARALLELIZABLE_TOOL_PLAN_TOOLS:
                raise NonParallelizableStepInDAGError(
                    f"DAG batch contains non-parallelizable tool '{step.tool}' in step '{step.id}'"
                )


def assert_no_mutating_steps(batches: list[list[ToolPlanStep]]) -> None:
    """Compatibility wrapper for the old blocklist-only DAG safety API."""
    try:
        assert_parallelizable_steps(batches)
    except NonParallelizableStepInDAGError as exc:
        raise MutatingStepInDAGError(str(exc)) from exc
