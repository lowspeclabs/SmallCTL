from __future__ import annotations

from smallctl.graph.plan_execution import PlanExecutionEngine
from smallctl.state import ExecutionPlan, LoopState, PlanStep, StepEvidenceArtifact


def test_deterministic_dag_order() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S3", title="third", depends_on=["S2"]),
            PlanStep(step_id="S1", title="first"),
            PlanStep(step_id="S2", title="second", depends_on=["S1"]),
        ],
    )
    engine = PlanExecutionEngine(state)

    assert engine.validate_plan(plan).valid
    # Steps are returned in plan definition order; only S1 has no pending deps
    assert engine.ready_step_ids(plan) == ["S1"]
    engine.activate_step(plan, "S1")
    engine.complete_step(plan, "S1", StepEvidenceArtifact(step_id="S1", summary="done"))
    assert engine.ready_step_ids(plan) == ["S2"]
    engine.activate_step(plan, "S2")
    engine.complete_step(plan, "S2", StepEvidenceArtifact(step_id="S2", summary="done"))
    assert engine.ready_step_ids(plan) == ["S3"]


def test_missing_dependency_detection() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[PlanStep(step_id="S1", title="step", depends_on=["missing"])],
    )
    engine = PlanExecutionEngine(state)

    result = engine.validate_plan(plan)
    assert result.valid is False
    assert any("missing" in err for err in result.errors)


def test_cycle_detection() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="a", depends_on=["S2"]),
            PlanStep(step_id="S2", title="b", depends_on=["S1"]),
        ],
    )
    engine = PlanExecutionEngine(state)

    result = engine.validate_plan(plan)
    assert result.valid is False
    assert any("cycle" in err.lower() for err in result.errors)


def test_retry_and_blocked_transitions() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[PlanStep(step_id="S1", title="step", max_retries=2)],
    )
    engine = PlanExecutionEngine(state)

    engine.activate_step(plan, "S1")
    engine.fail_step(plan, "S1", "fail 1")
    assert plan.find_step("S1").status == "pending"
    assert plan.find_step("S1").retry_count == 1

    engine.activate_step(plan, "S1")
    engine.fail_step(plan, "S1", "fail 2")
    # max_retries=2 means 2 attempts total; second failure exhausts retries
    assert plan.find_step("S1").status == "blocked"
    assert plan.find_step("S1").retry_count == 2
    assert state.pending_interrupt["kind"] == "staged_step_blocked"


def test_blocked_dependency_blocks_dependents() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="root", max_retries=0),
            PlanStep(step_id="S2", title="dependent", depends_on=["S1"]),
        ],
    )
    engine = PlanExecutionEngine(state)

    engine.activate_step(plan, "S1")
    engine.fail_step(plan, "S1", "root failure")
    assert plan.find_step("S1").status == "blocked"
    assert plan.find_step("S2").status == "blocked"
    assert "Dependency S1 blocked." in str(plan.find_step("S2").failure_reasons)


def test_substep_policy_rejected() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(
                step_id="S1",
                title="parent",
                substeps=[PlanStep(step_id="S1-1", title="child")],
            ),
        ],
    )
    engine = PlanExecutionEngine(state)

    result = engine.validate_plan(plan)
    assert result.valid is False
    assert any("substeps" in err.lower() for err in result.errors)


def test_get_next_step_returns_active_in_progress_step() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="first"),
            PlanStep(step_id="S2", title="second"),
        ],
    )
    engine = PlanExecutionEngine(state)

    engine.activate_step(plan, "S1")
    assert engine.get_next_step(plan).step_id == "S1"


def test_is_plan_complete_requires_all_steps_done() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="first"),
            PlanStep(step_id="S2", title="second"),
        ],
    )
    engine = PlanExecutionEngine(state)

    assert not engine.is_plan_complete(plan)
    engine.activate_step(plan, "S1")
    engine.complete_step(plan, "S1", StepEvidenceArtifact(step_id="S1", summary="done"))
    assert not engine.is_plan_complete(plan)
    engine.activate_step(plan, "S2")
    engine.complete_step(plan, "S2", StepEvidenceArtifact(step_id="S2", summary="done"))
    assert engine.is_plan_complete(plan)


def test_skip_step_counts_as_complete() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="first", status="skipped"),
            PlanStep(step_id="S2", title="second"),
        ],
    )
    engine = PlanExecutionEngine(state)

    assert not engine.is_plan_complete(plan)
    engine.activate_step(plan, "S2")
    engine.complete_step(plan, "S2", StepEvidenceArtifact(step_id="S2", summary="done"))
    assert engine.is_plan_complete(plan)
