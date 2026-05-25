from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context.artifacts import ArtifactStore
from smallctl.harness.subtask_checklist import build_subtask_checklist_update, emit_subtask_checklist_if_changed
from smallctl.harness.subtask_ledger_service import SubtaskLedgerService
from smallctl.harness.tool_result_artifact_updates import _update_subtask_ledger_from_verifier
from smallctl.graph.state import ToolExecutionRecord
from smallctl.graph.tool_outcomes import _update_subtask_ledger_from_record
from smallctl.models.events import UIEventType
from smallctl.models.tool_result import ToolEnvelope
from smallctl.recovery_schema import FailureEvent, Subtask, SubtaskLedger
from smallctl.state import LoopState
from smallctl.state_schema import ExecutionPlan, PlanStep
from smallctl.tools import control, planning


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

    assert [task.subtask_id for task in state.subtask_ledger.subtasks] == ["S2", "S3"]
    assert state.subtask_ledger.active_subtask_id == "S2"
    assert state.subtask_ledger.active().status == "active"


def test_plan_import_replaces_synthetic_root_with_real_plan_steps() -> None:
    state = LoopState()
    service = _service(state)
    service.ensure_ledger("Implement feature")
    state.subtask_ledger.active().evidence.append("file_read succeeded for src/app.py")
    state.active_plan = ExecutionPlan(
        plan_id="P1",
        goal="Implement feature",
        steps=[
            PlanStep(step_id="S1", title="Patch code", description="Make the edit"),
            PlanStep(step_id="S2", title="Verify", description="Run tests"),
        ],
    )

    service.import_plan_if_needed(replace_synthetic_root=True)

    assert [task.title for task in state.subtask_ledger.subtasks] == ["Patch code", "Verify"]
    assert state.subtask_ledger.active_subtask_id == "S1"
    assert state.subtask_ledger.active().evidence == ["file_read succeeded for src/app.py"]


def test_plan_set_immediately_seeds_subtask_ledger(tmp_path) -> None:
    state = LoopState()
    harness = _harness(state)
    harness.artifact_store = ArtifactStore(tmp_path, "run-1")
    harness.log = SimpleNamespace(warning=lambda *args, **kwargs: None)

    result = asyncio.run(
        planning.plan_set(
            goal="Create a script and verify it",
            inputs=["User request"],
            outputs=["temp/example.py"],
            acceptance_criteria=["Verifier passes"],
            implementation_plan=["Write", "Verify"],
            steps=[
                {"task": "Write script", "tool_allowlist": ["file_write"]},
                {"task": "Run script", "tool_allowlist": ["shell_exec"]},
            ],
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is True
    assert state.subtask_ledger is not None
    assert [task.subtask_id for task in state.subtask_ledger.subtasks] == ["P1", "P2"]
    assert state.subtask_ledger.active_subtask_id == "P1"


def test_subtask_checklist_update_renders_goal_and_task_statuses() -> None:
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="P1",
        goal="Deploy app",
        steps=[
            PlanStep(step_id="S1", title="SSH to remote server"),
            PlanStep(step_id="S2", title="Run verifier", status="pending"),
        ],
    )
    harness = _harness(state)
    harness.subtask_ledger.import_plan_if_needed(replace_synthetic_root=True)

    first = build_subtask_checklist_update(state)

    assert "Goal Objective: Deploy app" in first
    assert "  ○ SSH to remote server" in first
    assert "  ○ Run verifier" in first
    assert build_subtask_checklist_update(state) == ""

    assert harness.subtask_ledger.mark_done_if_verified("S1", {"verdict": "pass"}) is True
    second = build_subtask_checklist_update(state)

    assert "  ✓ SSH to remote server" in second
    assert "  ○ Run verifier" in second


def test_subtask_checklist_does_not_render_parent_objective_as_child() -> None:
    objective = "SSH to the host, list enabled services, identify anything unusual"
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="P1",
        goal=objective,
        steps=[
            PlanStep(step_id="S1", title=objective, status="done"),
            PlanStep(step_id="S2", title=objective, status="pending"),
            PlanStep(step_id="S3", title="Save evidence and summarize"),
        ],
    )
    harness = _harness(state)
    harness.subtask_ledger.import_plan_if_needed(replace_synthetic_root=True)

    rendered = build_subtask_checklist_update(state)

    assert f"Goal Objective: {objective}" in rendered
    assert rendered.count(objective) == 1
    assert "  ○ Save evidence and summarize" in rendered


def test_subtask_checklist_treats_phase_prefixed_objective_as_duplicate_child() -> None:
    objective = "Build a self-contained Python script at ./temp/artifact_retention.py"
    state = LoopState()
    state.run_brief.current_phase_objective = f"execute: {objective}"
    state.subtask_ledger = SubtaskLedger(task_id="task-1", subtasks=[], active_subtask_id=None)

    state.subtask_ledger.subtasks.append(
        Subtask(subtask_id="S1", title=objective, goal=objective, status="active")
    )

    rendered = build_subtask_checklist_update(state)

    assert f"Goal Objective: execute: {objective}" in rendered
    assert rendered.count(objective) == 1
    assert "\n  ○ " not in rendered


def test_subtask_checklist_dedups_long_synthetic_root_against_original_task() -> None:
    objective = (
        "Build a self-contained Python script at ./temp/restart_backoff.py "
        "that simulates a tiny message processor with retry, backoff, and dead-letter queue behavior."
    )
    state = LoopState()
    state.run_brief.original_task = objective
    state.run_brief.current_phase_objective = f"repair: {objective}"
    state.subtask_ledger = SubtaskLedger(task_id="task-1", subtasks=[], active_subtask_id=None)
    state.subtask_ledger.subtasks.append(
        Subtask(
            subtask_id="S1",
            title="Complete user task",
            goal=objective,
            status="blocked",
            acceptance=["User request satisfied with tool-backed evidence when needed."],
        )
    )

    rendered = build_subtask_checklist_update(state)

    assert rendered.startswith("Goal Objective: repair: Build a self-contained Python script")
    lines = rendered.splitlines()
    assert len(lines) == 2  # goal + synthetic root child with short summary
    assert "⚠" in lines[1]
    assert "Build a self-contained Python script" in lines[1]  # extracted short summary, not full text


def test_plan_import_dedupes_duplicate_step_titles() -> None:
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="P1",
        goal="Review services",
        steps=[
            PlanStep(step_id="S1", title="List enabled services", status="done"),
            PlanStep(step_id="S2", title="List enabled services", status="pending"),
            PlanStep(step_id="S3", title="Save evidence"),
        ],
    )
    harness = _harness(state)

    harness.subtask_ledger.import_plan_if_needed(replace_synthetic_root=True)

    assert [task.title for task in state.subtask_ledger.subtasks] == [
        "List enabled services",
        "Save evidence",
    ]


def test_subtask_checklist_emits_visible_chat_progress_event() -> None:
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="P1",
        goal="Deploy app",
        steps=[
            PlanStep(step_id="S1", title="SSH to remote server"),
            PlanStep(step_id="S2", title="Run verifier", status="pending"),
        ],
    )
    harness = _harness(state)
    harness.subtask_ledger.import_plan_if_needed(replace_synthetic_root=True)
    events = []

    async def emit(handler, event, *, emit_status=True):
        del emit_status
        maybe = handler(event)
        if hasattr(maybe, "__await__"):
            await maybe

    harness._emit = emit

    asyncio.run(emit_subtask_checklist_if_changed(harness, events.append))
    asyncio.run(emit_subtask_checklist_if_changed(harness, events.append))

    assert len(events) == 1
    assert events[0].event_type == UIEventType.ALERT
    assert events[0].data["ui_kind"] == "subtask_checklist"
    assert "  ○ SSH to remote server" in events[0].content


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


def test_task_complete_blocks_until_plan_subtasks_are_done() -> None:
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="P1",
        goal="Implement feature",
        steps=[
            PlanStep(step_id="S1", title="Patch code"),
            PlanStep(step_id="S2", title="Verify"),
        ],
    )
    harness = _harness(state)
    harness.subtask_ledger.import_plan_if_needed(replace_synthetic_root=True)

    blocked = asyncio.run(control.task_complete("done", state, harness))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "plan_subtasks_incomplete"
    assert blocked["metadata"]["next_required_subtask"]["subtask_id"] == "S1"

    assert harness.subtask_ledger.mark_done_if_verified("S1", {"verdict": "pass"}) is True
    assert harness.subtask_ledger.mark_done_if_verified("S2", {"verdict": "pass"}) is True

    completed = asyncio.run(control.task_complete("done", state, harness))

    assert completed["success"] is True


def test_repeated_subtask_failures_block_and_request_escalation() -> None:
    state = LoopState()
    harness = _harness(state)
    harness.config.subtask_block_after_failures = 2
    service = harness.subtask_ledger
    active = service.infer_or_create_active_subtask()
    failure = FailureEvent(
        event_id="F1",
        timestamp=1.0,
        failure_class="tool_execution_failed",
        severity="recoverable",
        source="tool",
        message="shell_exec failed",
    )

    service.attach_failure(active.subtask_id, failure)
    assert active.status == "active"

    service.attach_failure(active.subtask_id, failure)

    assert active.status == "blocked"
    assert "escalate_to_bigger_model" in active.next_action


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
