from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.config import resolve_config
from smallctl.graph.runtime_auto import AutoGraphRuntime
from smallctl.harness import Harness
from smallctl.main import cli
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
        tool_dag_enabled=True,
        tool_dag_max_parallel=7,
        tool_dag_timeout_sec=11,
        tool_dag_preserve_result_order=False,
    )
    harness.state.cwd = str(tmp_path)

    assert harness.config.run_mode == "tool_plan"
    assert harness.config.tool_plan_max_steps == 4
    assert harness.config.tool_plan_max_repair_attempts == 2
    assert harness.config.tool_plan_observation_token_limit == 123
    assert harness.config.tool_plan_solver_fresh_output_limit == 456
    assert harness.config.tool_dag_enabled is True
    assert harness.config.tool_dag_max_parallel == 7
    assert harness.config.tool_dag_timeout_sec == 11
    assert harness.config.tool_dag_preserve_result_order is False


def test_cli_passes_tool_dag_config_to_harness(monkeypatch, tmp_path, capsys) -> None:
    captured_kwargs: dict[str, object] = {}

    class _Harness:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            self.state = SimpleNamespace(thread_id="thread-1")
            self.conversation_id = "thread-1"

        async def run_auto(self, task: str) -> dict[str, object]:
            return {"status": "completed", "task": task}

        async def teardown(self) -> None:
            return None

        def note_task_shutdown(self, reason: str) -> None:
            del reason

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_TOOL_DAG_ENABLED", "true")
    monkeypatch.setenv("SMALLCTL_TOOL_DAG_MAX_PARALLEL", "9")
    monkeypatch.setenv("SMALLCTL_TOOL_DAG_TIMEOUT_SEC", "13")
    monkeypatch.setenv("SMALLCTL_TOOL_DAG_PRESERVE_RESULT_ORDER", "false")
    monkeypatch.setattr("smallctl.main.Harness", _Harness)

    exit_code = cli(["--task", "inspect files", "--endpoint", "http://example.test/v1", "--model", "wrench-9b"])

    assert exit_code == 0
    assert captured_kwargs["tool_dag_enabled"] is True
    assert captured_kwargs["tool_dag_max_parallel"] == 9
    assert captured_kwargs["tool_dag_timeout_sec"] == 13
    assert captured_kwargs["tool_dag_preserve_result_order"] is False
    capsys.readouterr()


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
