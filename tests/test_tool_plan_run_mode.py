from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.config import resolve_config
from smallctl.graph.runtime_auto import AutoGraphRuntime
from smallctl.harness import Harness, HarnessConfig
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
        tool_plan_allow_git=True,
        tool_dag_enabled=True,
        tool_dag_max_parallel=7,
        tool_dag_timeout_sec=11,
        tool_dag_preserve_result_order=False,
        solver_refine_enabled=True,
        solver_refine_max_passes=3,
        solver_refine_token_budget=888,
    )
    harness.state.cwd = str(tmp_path)

    assert harness.config.run_mode == "tool_plan"
    assert harness.config.tool_plan_max_steps == 4
    assert harness.config.tool_plan_max_repair_attempts == 2
    assert harness.config.tool_plan_observation_token_limit == 123
    assert harness.config.tool_plan_solver_fresh_output_limit == 456
    assert harness.config.tool_plan_allow_git is True
    assert harness.config.tool_dag_enabled is True
    assert harness.config.tool_dag_max_parallel == 7
    assert harness.config.tool_dag_timeout_sec == 11
    assert harness.config.tool_dag_preserve_result_order is False
    assert harness.config.solver_refine_enabled is True
    assert harness.config.solver_refine_max_passes == 3
    assert harness.config.solver_refine_token_budget == 888


def test_harness_preserves_test_time_scaling_config_fields(tmp_path) -> None:
    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        staged_execution_enabled=True,
        staged_step_prompt_tokens=2048,
        test_time_scaling_enabled=True,
        test_time_scaling_runtimes=["staged_execution"],
        test_time_scaling_trigger="explicit",
        test_time_scaling_max_candidates=4,
        test_time_scaling_min_candidates=2,
        test_time_scaling_policy="sequential_branch",
        test_time_scaling_score_threshold=0.75,
        test_time_scaling_parallel_max=3,
        test_time_scaling_timeout_sec=45,
        test_time_scaling_mutating_parallel_enabled=True,
        test_time_scaling_all_fail_action="fail_step",
    )
    harness.state.cwd = str(tmp_path)

    assert harness.config.staged_execution_enabled is True
    assert harness.config.staged_step_prompt_tokens == 2048
    assert harness.config.test_time_scaling_enabled is True
    assert harness.config.test_time_scaling_runtimes == ["staged_execution"]
    assert harness.config.test_time_scaling_trigger == "explicit"
    assert harness.config.test_time_scaling_max_candidates == 4
    assert harness.config.test_time_scaling_min_candidates == 2
    assert harness.config.test_time_scaling_policy == "sequential_branch"
    assert harness.config.test_time_scaling_score_threshold == 0.75
    assert harness.config.test_time_scaling_parallel_max == 3
    assert harness.config.test_time_scaling_timeout_sec == 45
    assert harness.config.test_time_scaling_mutating_parallel_enabled is True
    assert harness.config.test_time_scaling_all_fail_action == "fail_step"


def test_harness_preserves_runtime_policy_config_fields(tmp_path) -> None:
    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        graph_node_timeout_sec=17,
        graph_model_call_timeout_sec=19,
        graph_dispatch_tools_timeout_sec=23,
        graph_idle_watchdog_sec=29,
        needs_human_timeout_sec=31,
        min_exploration_steps=2,
        artifact_summarization_threshold=333,
        small_model_soft_write_chars=444,
        small_model_hard_write_chars=555,
        new_file_chunk_mode_line_estimate=66,
        allow_multi_section_turns_for_small_edits=False,
        failed_local_patch_limit=7,
        enable_write_intent_recovery=False,
        enable_assistant_code_write_recovery=False,
        write_recovery_min_confidence="medium",
        write_recovery_allow_raw_text_targets=False,
    )
    harness.state.cwd = str(tmp_path)

    assert harness.config.graph_node_timeout_sec == 17
    assert harness.config.graph_model_call_timeout_sec == 19
    assert harness.config.graph_dispatch_tools_timeout_sec == 23
    assert harness.config.graph_idle_watchdog_sec == 29
    assert harness.config.needs_human_timeout_sec == 31
    assert harness.config.min_exploration_steps == 2
    assert harness.context_policy.artifact_summarization_threshold == 333
    assert harness.config.small_model_soft_write_chars == 444
    assert harness.config.small_model_hard_write_chars == 555
    assert harness.config.new_file_chunk_mode_line_estimate == 66
    assert harness.config.allow_multi_section_turns_for_small_edits is False
    assert harness.config.failed_local_patch_limit == 7
    assert harness.config.enable_write_intent_recovery is False
    assert harness.config.enable_assistant_code_write_recovery is False
    assert harness.config.write_recovery_min_confidence == "medium"
    assert harness.config.write_recovery_allow_raw_text_targets is False


def test_cli_passes_tool_dag_config_to_harness(monkeypatch, tmp_path, capsys) -> None:
    captured_kwargs: dict[str, object] = {}

    class _Harness:
        def __init__(self, config: HarnessConfig) -> None:
            captured_kwargs.update(config.__dict__)
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
    monkeypatch.setenv("SMALLCTL_TOOL_PLAN_ALLOW_GIT", "true")
    monkeypatch.setenv("SMALLCTL_SOLVER_REFINE_ENABLED", "true")
    monkeypatch.setenv("SMALLCTL_SOLVER_REFINE_MAX_PASSES", "2")
    monkeypatch.setattr("smallctl.main.Harness", _Harness)

    exit_code = cli(["--task", "inspect files", "--endpoint", "http://example.test/v1", "--model", "wrench-9b"])

    assert exit_code == 0
    assert captured_kwargs["tool_dag_enabled"] is True
    assert captured_kwargs["tool_dag_max_parallel"] == 9
    assert captured_kwargs["tool_dag_timeout_sec"] == 13
    assert captured_kwargs["tool_dag_preserve_result_order"] is False
    assert captured_kwargs["tool_plan_allow_git"] is True
    assert captured_kwargs["solver_refine_enabled"] is True
    assert captured_kwargs["solver_refine_max_passes"] == 2
    capsys.readouterr()


def test_cli_passes_test_time_scaling_config_to_harness(monkeypatch, tmp_path, capsys) -> None:
    captured_kwargs: dict[str, object] = {}

    class _Harness:
        def __init__(self, config: HarnessConfig) -> None:
            captured_kwargs.update(config.__dict__)
            self.state = SimpleNamespace(thread_id="thread-1")
            self.conversation_id = "thread-1"

        async def run_auto(self, task: str) -> dict[str, object]:
            return {"status": "completed", "task": task}

        async def teardown(self) -> None:
            return None

        def note_task_shutdown(self, reason: str) -> None:
            del reason

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_ENABLED", "true")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_TRIGGER", "explicit")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_POLICY", "sequential_branch")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_MAX_CANDIDATES", "4")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_PARALLEL_MAX", "3")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_ALL_FAIL_ACTION", "fail_step")
    monkeypatch.setattr("smallctl.main.Harness", _Harness)

    exit_code = cli(
        [
            "--task",
            "inspect files",
            "--endpoint",
            "http://example.test/v1",
            "--model",
            "wrench-9b",
            "--staged-execution",
            "--staged-step-prompt-tokens",
            "2048",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["staged_execution_enabled"] is True
    assert captured_kwargs["staged_step_prompt_tokens"] == 2048
    assert captured_kwargs["test_time_scaling_enabled"] is True
    assert captured_kwargs["test_time_scaling_trigger"] == "explicit"
    assert captured_kwargs["test_time_scaling_policy"] == "sequential_branch"
    assert captured_kwargs["test_time_scaling_max_candidates"] == 4
    assert captured_kwargs["test_time_scaling_parallel_max"] == 3
    assert captured_kwargs["test_time_scaling_all_fail_action"] == "fail_step"
    capsys.readouterr()


def test_tui_passes_staged_and_scaling_config_to_app(monkeypatch, tmp_path, capsys) -> None:
    captured_kwargs: dict[str, object] = {}

    class _App:
        closed_by_ctrl_c = False

        def __init__(self, harness_kwargs: dict[str, object]) -> None:
            captured_kwargs.update(harness_kwargs)
            self.harness = None

        def run(self) -> None:
            return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_ENABLED", "true")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_POLICY", "sequential_branch")
    monkeypatch.setattr("smallctl.ui.SmallctlApp", _App)

    exit_code = cli(
        [
            "--tui",
            "--endpoint",
            "http://example.test/v1",
            "--model",
            "wrench-9b",
            "--staged-execution",
            "--staged-step-prompt-tokens",
            "2048",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["staged_execution_enabled"] is True
    assert captured_kwargs["staged_step_prompt_tokens"] == 2048
    assert captured_kwargs["test_time_scaling_enabled"] is True
    assert captured_kwargs["test_time_scaling_policy"] == "sequential_branch"
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
