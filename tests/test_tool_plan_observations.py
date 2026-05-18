from __future__ import annotations

from smallctl.graph.state import ToolExecutionRecord
from smallctl.graph.tool_plan_observations import (
    attach_tool_plan_observation_evidence,
    build_tool_plan_observations,
    observation_to_evidence_record,
    render_tool_plan_observations,
)
from smallctl.graph.tool_plan_schema import ToolPlan, ToolPlanStep
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


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


def test_tool_plan_observation_to_evidence_preserves_fields_and_dedupes() -> None:
    observation = build_tool_plan_observations(
        ToolPlan(
            mode="tool_plan",
            objective="inspect runtime",
            steps=[
                ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
                ToolPlanStep("E2", "file_read", {"path": "src/app.py"}),
            ],
        ),
        [
            ToolExecutionRecord(
                operation_id="op-1",
                tool_name="file_read",
                args={"path": "src/app.py"},
                tool_call_id="toolplan:E1",
                result=ToolEnvelope(success=True, output="body", metadata={"artifact_id": "A0001"}),
            )
        ],
        token_limit=200,
        max_chars_per_step=80,
    )[1]

    record = observation_to_evidence_record(
        observation,
        objective="inspect runtime",
        step_index=2,
        created_at_step=7,
    )
    assert record.evidence_id == "TP-E7-E2"
    assert record.operation_id == "op-1"
    assert record.artifact_id == "A0001"
    assert record.source == "src/app.py"
    assert record.metadata["duplicate_of"] == "E1"
    assert record.metadata["objective"] == "inspect runtime"

    state = LoopState(step_count=7)
    ids = attach_tool_plan_observation_evidence(state, objective="inspect runtime", observations=[observation])
    ids_again = attach_tool_plan_observation_evidence(state, objective="inspect runtime", observations=[observation])
    assert ids == ids_again == ["TP-E7-E2"]
    assert len(state.reasoning_graph.evidence_records) == 1
