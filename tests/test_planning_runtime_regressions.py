from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.planning_outcomes import apply_planning_tool_outcomes
from smallctl.graph.routing import LoopRoute
from smallctl.graph.runtime_auto import AutoGraphRuntime
from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.harness import Harness
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def _plan_approval_interrupt() -> dict[str, object]:
    return {
        "kind": "plan_execute_approval",
        "question": "Plan ready. Execute it now?",
        "plan_id": "plan-test",
        "approved": False,
        "response_mode": "yes/no/revise",
        "current_phase": "plan",
        "thread_id": "thread-test",
    }


def test_run_task_with_events_resumes_pending_plan_approval_reply() -> None:
    resume_calls: list[tuple[str, object]] = []

    async def _resume_task_with_events(human_input: str, event_handler=None) -> dict[str, object]:
        resume_calls.append((human_input, event_handler))
        return {"status": "needs_human", "message": "resumed"}

    harness = SimpleNamespace(
        get_pending_interrupt=_plan_approval_interrupt,
        resume_task_with_events=_resume_task_with_events,
    )

    result = asyncio.run(Harness.run_task_with_events(harness, "yes"))

    assert result == {"status": "needs_human", "message": "resumed"}
    assert resume_calls == [("yes", None)]


def test_run_auto_with_events_resumes_pending_plan_approval_reply() -> None:
    resume_calls: list[tuple[str, object]] = []

    async def _resume_task_with_events(human_input: str, event_handler=None) -> dict[str, object]:
        resume_calls.append((human_input, event_handler))
        return {"status": "needs_human", "message": "resumed"}

    harness = SimpleNamespace(
        get_pending_interrupt=_plan_approval_interrupt,
        resume_task_with_events=_resume_task_with_events,
    )

    result = asyncio.run(Harness.run_auto_with_events(harness, "yes"))

    assert result == {"status": "needs_human", "message": "resumed"}
    assert resume_calls == [("yes", None)]


def test_auto_runtime_replaces_pending_interrupt_for_unrelated_new_task(tmp_path, monkeypatch) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.pending_interrupt = _plan_approval_interrupt()
    state.planner_interrupt = object()
    decisions: list[str] = []
    chat_runs: list[str] = []
    runlog_events: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def _decide_run_mode(task: str) -> str:
        decisions.append(task)
        return "chat"

    class _StubChatRuntime:
        async def run(self, task: str) -> dict[str, object]:
            chat_runs.append(task)
            return {"status": "chat_completed", "message": "new task routed"}

    monkeypatch.setattr(
        "smallctl.graph.runtime.ChatGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubChatRuntime()),
    )

    harness = SimpleNamespace(
        state=state,
        has_pending_interrupt=lambda: bool(state.pending_interrupt),
        get_pending_interrupt=lambda: dict(state.pending_interrupt or {}),
        decide_run_mode=_decide_run_mode,
        _runlog=lambda *args, **kwargs: runlog_events.append((args, kwargs)),
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("write a local README"))

    assert decisions == ["write a local README"]
    assert chat_runs == ["write a local README"]
    assert state.pending_interrupt is None
    assert state.planner_interrupt is None
    assert result == {"status": "chat_completed", "message": "new task routed"}
    replacement_logs = [
        kwargs for args, kwargs in runlog_events
        if args[:2] == ("runtime_route", "replacing pending interrupt with new task")
    ]
    assert replacement_logs
    assert replacement_logs[-1]["interrupt_kind"] == "plan_execute_approval"
    assert replacement_logs[-1]["interrupt_plan_id"] == "plan-test"
    assert replacement_logs[-1]["replacement_task"] == "write a local README"


def test_auto_runtime_plan_approval_reply_replays_as_resume_not_new_task(tmp_path, monkeypatch) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.pending_interrupt = _plan_approval_interrupt()
    state.active_tool_profiles = ["core", "network"]
    state.run_brief.original_task = (
        'ssh root@192.168.1.89 with password "@S02v1735" and go to '
        "/var/www/demo-site/index.html, read the file and remove the google branding"
    )
    decisions: list[str] = []
    resumes: list[str] = []
    chat_runs: list[str] = []

    async def _decide_run_mode(task: str) -> str:
        decisions.append(task)
        return "chat"

    class _StubPlanningRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            resumes.append(human_input)
            return {"status": "resumed", "message": "plan approved"}

    class _StubChatRuntime:
        async def run(self, task: str) -> dict[str, object]:
            chat_runs.append(task)
            return {"status": "chat_completed", "message": "wrong path"}

    monkeypatch.setattr(
        "smallctl.graph.runtime_specialized.PlanningGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubPlanningRuntime()),
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime.ChatGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubChatRuntime()),
    )

    harness = SimpleNamespace(
        state=state,
        has_pending_interrupt=lambda: bool(state.pending_interrupt),
        get_pending_interrupt=lambda: dict(state.pending_interrupt or {}),
        decide_run_mode=_decide_run_mode,
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("yes"))

    assert result == {"status": "resumed", "message": "plan approved"}
    assert resumes == ["yes"]
    assert decisions == []
    assert chat_runs == []
    assert state.pending_interrupt == _plan_approval_interrupt()
    assert state.active_tool_profiles == ["core", "network"]


def test_auto_runtime_plan_approval_reply_uses_planner_interrupt_fallback(tmp_path, monkeypatch) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.pending_interrupt = None
    state.planner_interrupt = SimpleNamespace(
        kind="plan_execute_approval",
        question="Plan ready. Execute it now?",
        plan_id="plan-test",
        approved=False,
        response_mode="yes/no/revise",
    )
    resumes: list[str] = []
    decisions: list[str] = []

    async def _decide_run_mode(task: str) -> str:
        decisions.append(task)
        return "chat"

    class _StubPlanningRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            resumes.append(human_input)
            return {"status": "resumed", "message": "plan approved"}

    monkeypatch.setattr(
        "smallctl.graph.runtime_specialized.PlanningGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubPlanningRuntime()),
    )

    harness = SimpleNamespace(
        state=state,
        has_pending_interrupt=lambda: state.planner_interrupt is not None,
        get_pending_interrupt=lambda: {
            "kind": state.planner_interrupt.kind,
            "question": state.planner_interrupt.question,
            "plan_id": state.planner_interrupt.plan_id,
            "approved": state.planner_interrupt.approved,
            "response_mode": state.planner_interrupt.response_mode,
        },
        decide_run_mode=_decide_run_mode,
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("yes"))

    assert result == {"status": "resumed", "message": "plan approved"}
    assert resumes == ["yes"]
    assert decisions == []


def test_auto_runtime_ignores_stale_planner_interrupt_for_approved_plan(tmp_path, monkeypatch) -> None:
    from smallctl.state import ExecutionPlan

    state = LoopState(cwd=str(tmp_path))
    state.pending_interrupt = None
    state.active_plan = ExecutionPlan(
        plan_id="plan-test",
        goal="do the thing",
        status="approved",
        approved=True,
    )
    state.planner_interrupt = SimpleNamespace(
        kind="plan_execute_approval",
        question="Plan ready. Execute it now?",
        plan_id="plan-test",
        approved=False,
        response_mode="yes/no/revise",
    )
    loop_runs: list[str] = []
    planning_resumes: list[str] = []

    class _StubPlanningRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            planning_resumes.append(human_input)
            return {"status": "wrong", "message": "stale resume"}

    class _StubLoopRuntime:
        async def run(self, task: str) -> dict[str, object]:
            loop_runs.append(task)
            return {"status": "loop", "message": "continued"}

    async def _decide_run_mode(task: str) -> str:
        return "loop"

    monkeypatch.setattr(
        "smallctl.graph.runtime_specialized.PlanningGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubPlanningRuntime()),
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubLoopRuntime()),
    )

    harness = SimpleNamespace(
        state=state,
        has_pending_interrupt=lambda: False,
        get_pending_interrupt=lambda: None,
        decide_run_mode=_decide_run_mode,
        config=SimpleNamespace(
            run_mode="auto",
            tool_plan_runtime_enabled=False,
            tool_plan_auto_select=False,
        ),
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("approve"))

    assert result == {"status": "loop", "message": "continued"}
    assert planning_resumes == []
    assert loop_runs == ["approve"]


def test_apply_planning_tool_outcomes_finalizes_successful_task_complete_without_plan_approval() -> None:
    emitted: list[object] = []

    async def _emit(_handler, event) -> None:
        emitted.append(event)

    state = SimpleNamespace(
        scratchpad={},
        recent_errors=[],
        planning_mode_enabled=True,
        active_plan=None,
        draft_plan=None,
        append_message=lambda _message: None,
        sync_plan_mirror=lambda: None,
        touch=lambda: None,
    )
    harness = SimpleNamespace(
        state=state,
        log=logging.getLogger("test.planning_runtime_regressions"),
        _emit=_emit,
        _runlog=lambda *args, **kwargs: None,
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-planning-complete",
        run_mode="planning",
        last_assistant_text="Task complete from prior execution.",
        last_tool_results=[
            ToolExecutionRecord(
                operation_id="op-task-complete",
                tool_name="task_complete",
                args={"message": "done"},
                tool_call_id="tool-task-complete",
                result=ToolEnvelope(
                    success=True,
                    output={"status": "complete", "message": "Remote file updated successfully."},
                ),
            )
        ],
    )

    route = asyncio.run(apply_planning_tool_outcomes(graph_state, deps))

    assert route == LoopRoute.FINALIZE
    assert graph_state.final_result is not None
    assert graph_state.final_result["status"] == "completed"
    assert state.planning_mode_enabled is False
    assert emitted


def test_planning_runtime_resume_approval_starts_execution_runtime(monkeypatch) -> None:
    from smallctl.graph.runtime_planning import PlanningGraphRuntime
    from smallctl.state import ExecutionPlan, RunBrief

    executions: list[str] = []
    persistence_calls: list[str] = []

    class _StubLoopRuntime:
        async def run(self, task: str) -> dict[str, object]:
            executions.append(task)
            return {"status": "ok", "message": "executed"}

    async def _drain_persistence() -> None:
        persistence_calls.append("drain")

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubLoopRuntime()),
    )

    state = SimpleNamespace(
        active_plan=ExecutionPlan(plan_id="plan-test", goal="do the thing"),
        draft_plan=None,
        pending_interrupt=_plan_approval_interrupt(),
        planner_interrupt=SimpleNamespace(kind="plan_execute_approval"),
        run_brief=RunBrief(original_task="original"),
        thread_id="t1",
        artifacts={},
        planning_mode_enabled=True,
        current_phase="plan",
        sync_plan_mirror=lambda: None,
        touch=lambda: None,
        step_count=0,
        inactive_steps=0,
        token_usage={},
    )

    harness = SimpleNamespace(
        state=state,
        get_pending_interrupt=_plan_approval_interrupt,
        conversation_id="c1",
        config=SimpleNamespace(staged_execution_enabled=False),
        _runlog=lambda *args, **kwargs: None,
        _autosave_chat_session_state=lambda: persistence_calls.append("autosave"),
        _drain_background_persistence_tasks=_drain_persistence,
        _failure=lambda message, error_type="runtime", details=None: {
            "status": "failed",
            "reason": message,
            "error": {"type": error_type, "message": message, "details": details or {}},
        },
    )

    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    runtime = PlanningGraphRuntime(deps)

    async def _mock_resume_planning_run(graph_state, deps, *, human_input):
        state.pending_interrupt = None

    monkeypatch.setattr(
        "smallctl.graph.runtime_planning.resume_planning_run",
        _mock_resume_planning_run,
    )

    result = asyncio.run(runtime.resume("yes"))

    assert result == {"status": "ok", "message": "executed"}
    assert executions == ["do the thing"]
    assert state.planning_mode_enabled is False
    assert state.current_phase == "execute"
    assert state.pending_interrupt is None
    assert state.planner_interrupt is None
    assert state.active_plan.approved is True
    assert persistence_calls == ["autosave", "drain"]


def test_planning_runtime_resume_missing_plan_returns_failure(monkeypatch) -> None:
    from smallctl.graph.runtime_planning import PlanningGraphRuntime

    state = SimpleNamespace(
        active_plan=None,
        draft_plan=None,
        pending_interrupt=_plan_approval_interrupt(),
        run_brief=SimpleNamespace(original_task="original"),
        thread_id="t1",
        artifacts={},
        planning_mode_enabled=True,
        current_phase="plan",
        sync_plan_mirror=lambda: None,
        touch=lambda: None,
        step_count=0,
        inactive_steps=0,
        token_usage={},
    )

    harness = SimpleNamespace(
        state=state,
        get_pending_interrupt=_plan_approval_interrupt,
        conversation_id="c1",
        config=SimpleNamespace(staged_execution_enabled=False),
        _runlog=lambda *args, **kwargs: None,
        _failure=lambda message, error_type="runtime", details=None: {
            "status": "failed",
            "reason": message,
            "error": {"type": error_type, "message": message, "details": details or {}},
        },
    )

    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    runtime = PlanningGraphRuntime(deps)

    async def _mock_resume_planning_run(graph_state, deps, *, human_input):
        state.pending_interrupt = None

    monkeypatch.setattr(
        "smallctl.graph.runtime_planning.resume_planning_run",
        _mock_resume_planning_run,
    )

    result = asyncio.run(runtime.resume("yes"))

    assert result["status"] == "failed"
    assert result["error"]["type"] == "interrupt"
    assert result["error"]["details"]["reason"] == "approved_plan_missing"


def test_planning_runtime_resume_rehydrates_plan_from_artifact(monkeypatch) -> None:
    from smallctl.graph.runtime_planning import PlanningGraphRuntime
    from smallctl.state_schema import ArtifactRecord

    executions: list[str] = []

    class _StubLoopRuntime:
        async def run(self, task: str) -> dict[str, object]:
            executions.append(task)
            return {"status": "ok", "message": "executed"}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubLoopRuntime()),
    )

    artifact = ArtifactRecord(
        artifact_id="A0001",
        kind="plan_playbook",
        source="do the thing",
        created_at="2024-01-01T00:00:00",
        size_bytes=100,
        summary="plan summary",
        metadata={"plan_id": "plan-test", "plan_status": "draft"},
        inline_content=json.dumps(
            {
                "status": "plan_set",
                "plan": {
                    "plan_id": "plan-test",
                    "goal": "do the thing",
                    "status": "draft",
                    "implementation_plan": ["execute the thing"],
                },
            }
        ),
    )

    state = SimpleNamespace(
        active_plan=None,
        draft_plan=None,
        pending_interrupt=_plan_approval_interrupt(),
        run_brief=SimpleNamespace(original_task="original"),
        thread_id="t1",
        artifacts={"A0001": artifact},
        planning_mode_enabled=True,
        current_phase="plan",
        sync_plan_mirror=lambda: None,
        touch=lambda: None,
        step_count=0,
        inactive_steps=0,
        token_usage={},
    )

    harness = SimpleNamespace(
        state=state,
        get_pending_interrupt=_plan_approval_interrupt,
        conversation_id="c1",
        config=SimpleNamespace(staged_execution_enabled=False),
        _runlog=lambda *args, **kwargs: None,
        _failure=lambda message, error_type="runtime", details=None: {
            "status": "failed",
            "reason": message,
            "error": {"type": error_type, "message": message, "details": details or {}},
        },
    )

    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    runtime = PlanningGraphRuntime(deps)

    async def _mock_resume_planning_run(graph_state, deps, *, human_input):
        state.pending_interrupt = None

    monkeypatch.setattr(
        "smallctl.graph.runtime_planning.resume_planning_run",
        _mock_resume_planning_run,
    )

    result = asyncio.run(runtime.resume("yes"))

    assert result == {"status": "ok", "message": "executed"}
    assert executions == ["do the thing"]
    assert state.active_plan is not None
    assert state.active_plan.plan_id == "plan-test"
    assert state.active_plan.implementation_plan == ["execute the thing"]
    assert state.active_plan.approved is True
