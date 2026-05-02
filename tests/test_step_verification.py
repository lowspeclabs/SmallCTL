from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.plan_verification import StepCompletionGate, compact_step_evidence
from smallctl.state import (
    ExecutionPlan,
    LoopState,
    PlanStep,
    StepEvidenceArtifact,
    StepOutputSpec,
    StepVerificationResult,
    StepVerifierSpec,
)


def _harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(state=state, log=SimpleNamespace(info=lambda *a, **k: None))


def test_required_verifier_failure_fails_step() -> None:
    state = LoopState()
    step = PlanStep(
        step_id="S1",
        title="Step",
        verifiers=[StepVerifierSpec(kind="file_exists", args={"path": "nonexistent.txt"}, required=True)],
    )

    result = asyncio.run(StepCompletionGate().verify_step(_harness(state), step))

    assert result.passed is False
    assert "file_exists" in result.failed_criteria


def test_optional_verifier_failure_does_not_fail_step() -> None:
    state = LoopState()
    step = PlanStep(
        step_id="S1",
        title="Step",
        verifiers=[
            StepVerifierSpec(kind="file_exists", args={"path": "nonexistent.txt"}, required=False),
        ],
    )

    result = asyncio.run(StepCompletionGate().verify_step(_harness(state), step))

    assert result.passed is True
    assert result.failed_criteria == []


def test_required_output_missing_fails_step() -> None:
    state = LoopState()
    step = PlanStep(
        step_id="S1",
        title="Step",
        outputs_expected=[StepOutputSpec(kind="file", ref="missing.py", required=True)],
    )

    result = asyncio.run(StepCompletionGate().verify_step(_harness(state), step))

    assert result.passed is False
    assert "missing.py" in result.failed_criteria


def test_all_verifiers_passing_passes_step() -> None:
    state = LoopState()
    step = PlanStep(
        step_id="S1",
        title="Step",
        verifiers=[StepVerifierSpec(kind="file_exists", args={"path": "."}, required=True)],
    )

    result = asyncio.run(StepCompletionGate().verify_step(_harness(state), step))

    assert result.passed is True
    assert result.failed_criteria == []


def test_verification_result_includes_step_run_id() -> None:
    state = LoopState(active_step_run_id="run-abc-123")
    step = PlanStep(step_id="S1", title="Step")

    result = asyncio.run(StepCompletionGate().verify_step(_harness(state), step))

    assert result.step_run_id == "run-abc-123"


def test_compact_step_evidence_includes_provenance(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path), active_step_run_id="run-xyz-789")
    plan = ExecutionPlan(plan_id="plan-1", goal="goal", steps=[PlanStep(step_id="S1", title="Step")])
    state.active_plan = plan
    step = plan.find_step("S1")
    state.step_sandbox_history = []
    state.scratchpad["_step_complete_message"] = "All good"
    (tmp_path / "file.txt").write_text("hello")
    state.files_changed_this_cycle = ["file.txt"]

    vresult = StepVerificationResult(
        step_id="S1",
        step_run_id="run-xyz-789",
        passed=True,
        verifier_results=[{"kind": "file_exists", "passed": True}],
    )

    evidence = compact_step_evidence(_harness(state), step, vresult)

    assert evidence.step_id == "S1"
    assert evidence.step_run_id == "run-xyz-789"
    assert evidence.attempt == 1
    assert "All good" in evidence.summary
    assert evidence.files_touched == ["file.txt"]
    assert evidence.verifier_results == [{"kind": "file_exists", "passed": True}]


def test_compact_step_evidence_collects_step_run_tool_records() -> None:
    state = LoopState(active_step_run_id="run-abc")
    plan = ExecutionPlan(plan_id="plan-1", goal="goal", steps=[PlanStep(step_id="S1", title="Step")])
    state.active_plan = plan
    step = plan.find_step("S1")
    state.tool_execution_records = {
        "op-1": {"tool_name": "file_read", "step_run_id": "run-abc", "artifact_id": "art-1"},
        "op-2": {"tool_name": "shell_exec", "step_run_id": "run-other"},
        "op-3": {"tool_name": "dir_list", "step_run_id": "run-abc"},
    }

    vresult = StepVerificationResult(step_id="S1", step_run_id="run-abc", passed=True)
    evidence = compact_step_evidence(_harness(state), step, vresult)

    assert "op-1" in evidence.tool_operation_ids
    assert "op-3" in evidence.tool_operation_ids
    assert "op-2" not in evidence.tool_operation_ids
    assert "art-1" in evidence.artifact_ids
