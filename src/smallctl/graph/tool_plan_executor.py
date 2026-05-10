from __future__ import annotations

from .state import GraphRunState, PendingToolCall
from .tool_plan_schema import ToolPlan


def prepare_tool_plan_dispatch(graph_state: GraphRunState, plan: ToolPlan) -> None:
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name=step.tool,
            args=dict(step.args),
            tool_call_id=f"toolplan:{step.id}",
            source="tool_plan",
            parser_metadata={"tool_plan_step_id": step.id, "reason": step.reason},
        )
        for step in plan.steps
    ]

