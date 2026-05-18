from __future__ import annotations

from typing import Any

from .tool_plan_schema import MUTATING_TOOL_PLAN_BLOCKLIST, ToolPlanStep


class MutatingStepInDAGError(Exception):
    """Raised when a DAG batch contains a mutating tool."""

    pass


def assert_no_mutating_steps(batches: list[list[ToolPlanStep]]) -> None:
    """Hard-abort if any step in any batch touches a mutating tool."""
    for batch in batches:
        for step in batch:
            if step.tool in MUTATING_TOOL_PLAN_BLOCKLIST:
                raise MutatingStepInDAGError(
                    f"DAG batch contains mutating tool '{step.tool}' in step '{step.id}'"
                )
