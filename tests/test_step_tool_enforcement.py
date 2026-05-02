from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.graph.lifecycle_prompt import select_staged_tools
from smallctl.state import ExecutionPlan, LoopState, PlanStep
from smallctl.tools import control
from smallctl.tools.base import ToolSpec
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.registry import ToolRegistry
from smallctl.tools.register import build_registry


def _make_graph_state(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(loop_state=state)


def _make_deps(state: LoopState) -> SimpleNamespace:
    harness = SimpleNamespace(
        state=state,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
        _indexer=False,
    )
    harness.registry = build_registry(harness, registry_profiles={"core"})
    return SimpleNamespace(harness=harness)


def _tool_names(tools: list[dict[str, object]]) -> list[str]:
    return [
        str(t["function"]["name"])
        for t in tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    ]


def test_select_staged_tools_filters_to_active_step_allowlist() -> None:
    state = LoopState()
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="Read only", tool_allowlist=["file_read", "dir_list"]),
        ],
    )
    state.active_step_id = "S1"
    state.active_step_run_id = "run-1"

    graph_state = _make_graph_state(state)
    deps = _make_deps(state)

    tools = select_staged_tools(graph_state, deps)
    names = _tool_names(tools)

    assert "file_read" in names
    assert "dir_list" in names
    assert "step_complete" in names
    assert "step_fail" in names
    assert "task_complete" not in names


def test_select_staged_tools_hides_all_non_control_tools_when_allowlist_empty() -> None:
    state = LoopState()
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[PlanStep(step_id="S1", title="No tools")],
    )
    state.active_step_id = "S1"
    state.active_step_run_id = "run-1"

    graph_state = _make_graph_state(state)
    deps = _make_deps(state)

    tools = select_staged_tools(graph_state, deps)
    names = _tool_names(tools)

    assert "step_complete" in names
    assert "step_fail" in names
    assert "loop_status" in names
    assert "file_read" not in names
    assert "dir_list" not in names
    assert "task_complete" not in names


def test_dispatcher_allows_tool_on_active_step_allowlist() -> None:
    async def handler() -> dict:
        return {"success": True, "output": "ran", "error": None, "metadata": {}}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="file_read",
            description="read",
            schema={"type": "object", "properties": {}, "required": []},
            handler=handler,
        )
    )
    state = LoopState(
        active_plan=ExecutionPlan(
            plan_id="plan-1",
            goal="goal",
            steps=[PlanStep(step_id="S1", title="Read", tool_allowlist=["file_read"])],
        ),
        plan_execution_mode=True,
        active_step_id="S1",
        active_step_run_id="run-1",
    )

    result = asyncio.run(ToolDispatcher(registry, state=state).dispatch("file_read", {}))

    assert result.success is True


def test_task_complete_cannot_finalize_while_staged_execution_is_active() -> None:
    state = LoopState(
        plan_execution_mode=True,
        active_step_id="S1",
        active_step_run_id="run-1",
    )

    result = asyncio.run(control.task_complete("done", state, SimpleNamespace()))

    assert result["success"] is False
    assert "staged execution" in str(result.get("error") or "").lower()


def test_step_complete_signals_step_completion_without_global_finalization() -> None:
    state = LoopState(
        plan_execution_mode=True,
        active_step_id="S1",
        active_step_run_id="run-1",
    )

    result = asyncio.run(control.step_complete("step done", state, SimpleNamespace(log=SimpleNamespace(info=lambda *a, **k: None))))

    assert result["success"] is True
    assert result["output"]["status"] == "step_completion_requested"
    assert result["output"]["step_id"] == "S1"
    assert "_task_complete" not in state.scratchpad
