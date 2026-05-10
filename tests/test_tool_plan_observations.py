from __future__ import annotations

from smallctl.graph.state import ToolExecutionRecord
from smallctl.graph.tool_plan_observations import build_tool_plan_observations, render_tool_plan_observations
from smallctl.graph.tool_plan_schema import ToolPlan, ToolPlanStep
from smallctl.models.tool_result import ToolEnvelope


def test_tool_plan_observations_render_compact_success_and_failure() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect runtime",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E2", "grep", {"path": "src", "pattern": "missing"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(success=True, output="large body", metadata={"artifact_id": "A0001", "path": "src/app.py"}),
        ),
        ToolExecutionRecord(
            operation_id="op-2",
            tool_name="grep",
            args={"path": "src", "pattern": "missing"},
            tool_call_id="toolplan:E2",
            result=ToolEnvelope(success=False, error="no matches", metadata={"pattern": "missing"}),
        ),
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=200, max_chars_per_step=80)
    rendered = render_tool_plan_observations(plan.objective, observations)

    assert "TOOL PLAN OBSERVATIONS" in rendered
    assert "E1 file_read src/app.py" in rendered
    assert "- artifact: A0001" in rendered
    assert "E2 grep src" in rendered
    assert "- error: no matches" in rendered


def test_tool_plan_observations_dedupe_repeated_reads_and_respect_budget() -> None:
    long_output = "alpha " * 400
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect runtime",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E2", "file_read", {"path": "src/app.py"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(success=True, output=long_output, metadata={}),
        ),
        ToolExecutionRecord(
            operation_id="op-2",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E2",
            result=ToolEnvelope(success=True, output=long_output, metadata={}),
        ),
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=80, max_chars_per_step=600)
    rendered = render_tool_plan_observations(plan.objective, observations)

    assert observations[1].duplicate_of == "E1"
    assert "duplicate_of: E1" in rendered
    assert "Duplicate of E1" in rendered
    assert len(observations[0].summary) < len(long_output)
