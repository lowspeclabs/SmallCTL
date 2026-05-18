from __future__ import annotations

from smallctl.graph.tool_dag import build_execution_dag
from smallctl.graph.tool_plan_schema import ToolPlan, ToolPlanStep


def test_build_execution_dag_independent_steps_one_batch() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="read files",
        steps=[
            ToolPlanStep(id="E1", tool="file_read", args={"path": "a.py"}),
            ToolPlanStep(id="E2", tool="file_read", args={"path": "b.py"}),
            ToolPlanStep(id="E3", tool="file_read", args={"path": "c.py"}),
        ],
    )
    batches = build_execution_dag(plan)
    assert len(batches) == 1
    assert {s.id for s in batches[0]} == {"E1", "E2", "E3"}


def test_build_execution_dag_with_dependency_chain() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="chain",
        steps=[
            ToolPlanStep(id="E1", tool="file_read", args={"path": "a.py"}),
            ToolPlanStep(id="E2", tool="file_read", args={"path": "b.py"}, depends_on=["E1"]),
            ToolPlanStep(id="E3", tool="file_read", args={"path": "c.py"}, depends_on=["E2"]),
        ],
    )
    batches = build_execution_dag(plan)
    assert len(batches) == 3
    assert batches[0][0].id == "E1"
    assert batches[1][0].id == "E2"
    assert batches[2][0].id == "E3"


def test_build_execution_dag_mixed_dependencies() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="mixed",
        steps=[
            ToolPlanStep(id="E1", tool="file_read", args={"path": "a.py"}),
            ToolPlanStep(id="E2", tool="file_read", args={"path": "b.py"}),
            ToolPlanStep(id="E3", tool="file_read", args={"path": "c.py"}, depends_on=["E1", "E2"]),
        ],
    )
    batches = build_execution_dag(plan)
    assert len(batches) == 2
    assert {s.id for s in batches[0]} == {"E1", "E2"}
    assert batches[1][0].id == "E3"


def test_build_execution_dag_empty_plan() -> None:
    plan = ToolPlan(mode="tool_plan", objective="empty", steps=[])
    batches = build_execution_dag(plan)
    assert batches == []
