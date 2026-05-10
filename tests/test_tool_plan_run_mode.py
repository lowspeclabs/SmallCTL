from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.config import resolve_config
from smallctl.graph.runtime_auto import AutoGraphRuntime
from smallctl.harness import Harness
from smallctl.state import LoopState


def test_resolve_config_accepts_tool_plan_run_mode() -> None:
    config = resolve_config({"run_mode": "tool_plan"})

    assert config.run_mode == "tool_plan"
    assert config.tool_plan_max_steps == 6


def test_harness_preserves_tool_plan_config_fields(tmp_path) -> None:
    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        run_mode="tool_plan",
        tool_plan_max_steps=4,
        tool_plan_max_repair_attempts=2,
        tool_plan_observation_token_limit=123,
        tool_plan_solver_fresh_output_limit=456,
    )
    harness.state.cwd = str(tmp_path)

    assert harness.config.run_mode == "tool_plan"
    assert harness.config.tool_plan_max_steps == 4
    assert harness.config.tool_plan_max_repair_attempts == 2
    assert harness.config.tool_plan_observation_token_limit == 123
    assert harness.config.tool_plan_solver_fresh_output_limit == 456


def test_auto_runtime_routes_explicit_tool_plan(monkeypatch, tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    runs: list[str] = []

    class _StubToolPlanRuntime:
        async def run(self, task: str) -> dict[str, object]:
            runs.append(task)
            return {"status": "tool_plan"}

    monkeypatch.setattr(
        "smallctl.graph.runtime_specialized.ToolPlanRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubToolPlanRuntime()),
    )
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(run_mode="tool_plan"),
        has_pending_interrupt=lambda: False,
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("find where dispatch happens"))

    assert result == {"status": "tool_plan"}
    assert runs == ["find where dispatch happens"]


def test_auto_runtime_can_auto_select_tool_plan_when_enabled(monkeypatch, tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    runs: list[str] = []
    decisions: list[str] = []

    class _StubToolPlanRuntime:
        async def run(self, task: str) -> dict[str, object]:
            runs.append(task)
            return {"status": "tool_plan"}

    async def _decide_run_mode(task: str) -> str:
        decisions.append(task)
        return "loop"

    monkeypatch.setattr(
        "smallctl.graph.runtime_specialized.ToolPlanRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubToolPlanRuntime()),
    )
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            run_mode="auto",
            tool_plan_runtime_enabled=True,
            tool_plan_auto_select=True,
        ),
        has_pending_interrupt=lambda: False,
        decide_run_mode=_decide_run_mode,
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("read through the repo and find where dispatch happens"))

    assert result == {"status": "tool_plan"}
    assert runs == ["read through the repo and find where dispatch happens"]
    assert decisions == []


def test_auto_runtime_does_not_auto_select_tool_plan_for_execution(monkeypatch, tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    decisions: list[str] = []
    loop_runs: list[str] = []

    async def _decide_run_mode(task: str) -> str:
        decisions.append(task)
        return "loop"

    class _StubLoopRuntime:
        async def run(self, task: str) -> dict[str, object]:
            loop_runs.append(task)
            return {"status": "loop"}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        staticmethod(lambda harness, event_handler=None: _StubLoopRuntime()),
    )
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            run_mode="auto",
            tool_plan_runtime_enabled=True,
            tool_plan_auto_select=True,
        ),
        has_pending_interrupt=lambda: False,
        decide_run_mode=_decide_run_mode,
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("run pytest and fix src/app.py"))

    assert result == {"status": "loop"}
    assert decisions == ["run pytest and fix src/app.py"]
    assert loop_runs == ["run pytest and fix src/app.py"]
