from __future__ import annotations

from smallctl.graph.state import GraphRunState, inflate_graph_state, serialize_graph_state
from smallctl.graph.tool_plan_executor import prepare_tool_plan_dispatch
from smallctl.graph.tool_plan_schema import ToolPlan, ToolPlanStep
from smallctl.state import LoopState


def test_prepare_tool_plan_dispatch_uses_runtime_owned_pending_calls(tmp_path) -> None:
    graph_state = GraphRunState(
        loop_state=LoopState(cwd=str(tmp_path)),
        thread_id="thread-test",
        run_mode="tool_plan",
    )
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect",
        steps=[ToolPlanStep("E1", "file_read", {"path": "src/app.py"}, reason="read seam")],
    )

    prepare_tool_plan_dispatch(graph_state, plan)

    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.tool_call_id == "toolplan:E1"
    assert pending.source == "tool_plan"
    assert pending.parser_metadata["reason"] == "read seam"


def test_tool_plan_pending_call_source_survives_state_round_trip(tmp_path) -> None:
    graph_state = GraphRunState(
        loop_state=LoopState(cwd=str(tmp_path)),
        thread_id="thread-test",
        run_mode="tool_plan",
    )
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect",
        steps=[ToolPlanStep("E1", "file_read", {"path": "src/app.py"}, reason="read seam")],
    )

    prepare_tool_plan_dispatch(graph_state, plan)
    inflated = inflate_graph_state(serialize_graph_state(graph_state))

    assert inflated.pending_tool_calls[0].source == "tool_plan"
    assert inflated.pending_tool_calls[0].parser_metadata["tool_plan_step_id"] == "E1"
