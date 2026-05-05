from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness.subtask_ledger_service import SubtaskLedgerService
from smallctl.harness.tool_result_artifact_updates import _update_subtask_ledger_from_verifier
from smallctl.graph.state import ToolExecutionRecord
from smallctl.graph.tool_outcomes import _update_subtask_ledger_from_record
from smallctl.models.tool_result import ToolEnvelope
from smallctl.recovery_schema import FailureEvent
from smallctl.state import LoopState
from smallctl.state_schema import ExecutionPlan, PlanStep


def _service(state: LoopState) -> SubtaskLedgerService:
    return SubtaskLedgerService(SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None))


def _harness(state: LoopState) -> SimpleNamespace:
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(subtask_ledger_enabled=True),
        _runlog=lambda *args, **kwargs: None,
    )
    harness.subtask_ledger = SubtaskLedgerService(harness)
    return harness


def test_subtask_ledger_creates_root_subtask() -> None:
    state = LoopState()
    ledger = _service(state).ensure_ledger("Fix the thing")

    assert ledger.active() is not None
    assert ledger.active().title == "Complete user task"
    assert ledger.active().status == "active"


def test_subtask_ledger_imports_plan_steps() -> None:
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="P1",
        goal="Implement feature",
        steps=[
            PlanStep(step_id="S2", title="Patch code", description="Make the edit", acceptance=["tests pass"]),
            PlanStep(step_id="S3", title="Verify", status="pending"),
        ],
    )

    service = _service(state)
    service.import_plan_if_needed()

    assert {task.subtask_id for task in state.subtask_ledger.subtasks} >= {"S1", "S2", "S3"}


def test_subtask_ledger_attaches_failure_and_evidence() -> None:
    state = LoopState()
    service = _service(state)
    active = service.infer_or_create_active_subtask()
    service.attach_evidence(active.subtask_id, "file_read showed target")
    failure = FailureEvent(
        event_id="F1",
        timestamp=1.0,
        failure_class="wrong_path",
        severity="warning",
        source="fama",
        message="wrong_path: missing.py",
    )
    service.attach_failure(active.subtask_id, failure)

    assert active.evidence == ["file_read showed target"]
    assert active.attempts == 1
    assert active.failure_classes == ["wrong_path"]


def test_subtask_ledger_marks_done_and_advances() -> None:
    state = LoopState()
    service = _service(state)
    service.ensure_ledger("Task")
    state.subtask_ledger.subtasks.append(
        state.subtask_ledger.subtasks[0].__class__(subtask_id="S2", title="Next", goal="next")
    )

    assert service.mark_done_if_verified("S1", {"verdict": "pass"}) is True

    assert state.subtask_ledger.subtasks[0].status == "done"
    assert state.subtask_ledger.active_subtask_id == "S2"
    assert state.subtask_ledger.active().status == "active"
    assert state.scratchpad["_recovery_metrics"]["subtasks_created"] == 1
    assert state.scratchpad["_recovery_metrics"]["subtasks_completed"] == 1


def test_tool_outcome_success_attaches_active_subtask_evidence() -> None:
    state = LoopState()
    harness = _harness(state)
    record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="file_read",
        args={"path": "src/app.py"},
        tool_call_id="call-1",
        result=ToolEnvelope(success=True, output="content"),
    )

    _update_subtask_ledger_from_record(harness, record)

    active = state.subtask_ledger.active()
    assert active is not None
    assert active.evidence == ["file_read succeeded for src/app.py"]


def test_tool_outcome_failure_attaches_generic_failure_when_fama_did_not() -> None:
    state = LoopState()
    harness = _harness(state)
    record = ToolExecutionRecord(
        operation_id="op-2",
        tool_name="shell_exec",
        args={"command": "false"},
        tool_call_id="call-2",
        result=ToolEnvelope(success=False, error="exit 1"),
    )

    _update_subtask_ledger_from_record(harness, record)

    active = state.subtask_ledger.active()
    assert active is not None
    assert active.attempts == 1
    assert active.failure_classes == ["tool_execution_failed"]
    assert state.scratchpad["_recovery_metrics"]["failure_events_total"] == 1
    assert state.scratchpad["_recovery_metrics"]["failure_events_by_class"]["tool_execution_failed"] == 1


def test_subtask_ledger_prunes_terminal_history_to_configured_limit() -> None:
    state = LoopState()
    harness = _harness(state)
    harness.config.subtask_max_history = 2
    service = harness.subtask_ledger
    service.ensure_ledger("Task")
    for index in range(2, 6):
        state.subtask_ledger.subtasks.append(
            state.subtask_ledger.subtasks[0].__class__(
                subtask_id=f"S{index}",
                title=f"Done {index}",
                goal="done",
                status="done",
            )
        )

    service.import_plan_if_needed()

    assert [task.subtask_id for task in state.subtask_ledger.subtasks] == ["S1", "S4", "S5"]


def test_subtask_ledger_respects_configured_active_limit() -> None:
    state = LoopState()
    harness = _harness(state)
    harness.config.subtask_max_active = 2
    service = harness.subtask_ledger
    service.ensure_ledger("Task")
    state.subtask_ledger.subtasks.extend(
        [
            state.subtask_ledger.subtasks[0].__class__(
                subtask_id="S2",
                title="Second",
                goal="second",
                status="active",
            ),
            state.subtask_ledger.subtasks[0].__class__(
                subtask_id="S3",
                title="Third",
                goal="third",
                status="active",
            ),
        ]
    )

    service.import_plan_if_needed()

    assert [task.subtask_id for task in state.subtask_ledger.subtasks if task.status == "active"] == ["S1", "S2"]
    assert state.subtask_ledger.active_subtask_id == "S1"
    assert state.subtask_ledger.subtasks[2].status == "pending"


def test_verifier_pass_marks_active_subtask_done() -> None:
    state = LoopState()
    harness = _harness(state)
    service = SimpleNamespace(harness=harness)
    active = harness.subtask_ledger.infer_or_create_active_subtask()
    active.failure_classes.append("verifier_failed")

    _update_subtask_ledger_from_verifier(
        service,
        {"verdict": "pass", "command": "pytest tests/test_app.py"},
    )

    assert state.subtask_ledger.subtasks[0].subtask_id == active.subtask_id
    assert state.subtask_ledger.subtasks[0].status == "done"
    assert "verifier pass: pytest tests/test_app.py" in state.subtask_ledger.subtasks[0].evidence
    assert state.scratchpad["_recovery_metrics"]["verifier_fail_then_success_count"] == 1
