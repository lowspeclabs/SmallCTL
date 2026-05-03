from __future__ import annotations

import asyncio
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
