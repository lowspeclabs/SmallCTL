from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.config import SmallctlConfig, resolve_config
from smallctl.main import build_harness_config_kwargs
from smallctl.graph.progress_guard import _maybe_inject_verifier_success_nudge
from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.harness.run_mode import _approved_plan_matches_plan_interrupt
from smallctl.harness.runtime_facade import _approved_plan_matches_interrupt
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools import network, shell
from smallctl.tools.process_lifecycle import truncate_output
from smallctl.tools.shell_support import guard_fail
from smallctl.tools.web import _resolve_fetch_selector
from smallctl.tools.fs_loop_guard import LoopGuardDecision, _emit_block


class _EmptyStream:
    async def read(self, _chunk_size: int) -> bytes:
        return b""

    def close(self) -> None:
        pass


class _TimeoutProbeProcess:
    def __init__(self) -> None:
        self.stdout = _EmptyStream()
        self.stderr = _EmptyStream()
        self.stdin = _EmptyStream()
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            raise asyncio.TimeoutError
        return self.returncode


def test_local_preflight_timeout_kills_probe(monkeypatch) -> None:
    proc = _TimeoutProbeProcess()

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "_create_process", _fake_create_process)

    result = asyncio.run(
        shell._run_local_installer_preflight_probes(
            "./install.sh",
            state=LoopState(cwd="/tmp"),
        )
    )

    assert result["probe_error"] == "Preflight probes timed out after 30s"
    assert proc.terminate_calls == 1
    assert proc.wait_calls >= 2


def test_remote_preflight_timeout_kills_and_unregisters_probe(monkeypatch) -> None:
    proc = _TimeoutProbeProcess()
    harness = SimpleNamespace(_active_processes={proc})

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(network, "create_process", _fake_create_process)
    monkeypatch.setattr(network, "_build_ssh_command", lambda **_kwargs: ("ssh example", {}))

    result = asyncio.run(
        network._run_remote_installer_preflight_probes(
            host="example.test",
            command="./install.sh",
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )

    assert result["probe_error"] == "Preflight probes timed out after 30s"
    assert proc.terminate_calls == 1
    assert proc not in harness._active_processes


def test_verifier_success_nudge_key_is_stable_for_same_context() -> None:
    state = LoopState(cwd="/tmp")
    state.files_changed_this_cycle = ["src/app.py"]
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="loop")
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="shell_exec",
            args={"command": "python -m py_compile src/app.py"},
            tool_call_id=None,
            result=ToolEnvelope(success=True, output={"exit_code": 0}),
        )
    ]

    _maybe_inject_verifier_success_nudge(state, graph_state)
    first_keys = [key for key in state.scratchpad if key.startswith("_verifier_nudge_")]
    _maybe_inject_verifier_success_nudge(state, graph_state)

    assert len(first_keys) == 1
    assert [key for key in state.scratchpad if key.startswith("_verifier_nudge_")] == first_keys
    assert len([message for message in state.recent_messages if message.metadata.get("recovery_kind") == "verifier_success_completion_prompt"]) == 1


def test_loop_guard_env_values_are_converted_like_cli(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_ENABLED", "false")
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_CUMULATIVE_WRITE_GATE", "0")
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_STAGNATION_THRESHOLD", "7")
    monkeypatch.setenv("SMALLCTL_LOOP_GUARD_SIMILARITY_THRESHOLD", "0.75")
    monkeypatch.setenv("SMALLCTL_STAGED_REASONING", "false")
    monkeypatch.setenv("SMALLCTL_TEST_TIME_SCALING_RUNTIMES", "alpha, beta")

    config = resolve_config({})

    assert config.loop_guard_enabled is False
    assert config.loop_guard_cumulative_write_gate is False
    assert config.loop_guard_stagnation_threshold == 7
    assert config.loop_guard_similarity_threshold == 0.75
    assert config.staged_reasoning is False
    assert config.staged_execution_enabled is False
    assert config.test_time_scaling_runtimes == ["alpha", "beta"]


def test_cli_and_env_numeric_conversion_use_same_schema(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_MAX_PROMPT_TOKENS", "1234")
    monkeypatch.setenv("SMALLCTL_SUMMARIZE_AT_RATIO", "0.65")

    env_config = resolve_config({})
    cli_config = resolve_config({"max_prompt_tokens": "1234", "summarize_at_ratio": "0.65"})

    assert env_config.max_prompt_tokens == cli_config.max_prompt_tokens == 1234
    assert env_config.summarize_at_ratio == cli_config.summarize_at_ratio == 0.65


def test_interactive_ssh_cleanup_unregisters_process_and_cancels_collectors() -> None:
    async def _run() -> None:
        proc = _TimeoutProbeProcess()
        harness = SimpleNamespace(_active_processes={proc})

        async def _collector() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_collector())
        session_id = "sshint-test"
        session = {"proc": proc, "tasks": [task], "stdout": [], "stderr": []}
        network._SSH_INTERACTIVE_SESSIONS[session_id] = session

        await network._cleanup_interactive_session(
            session_id,
            session,
            harness=harness,
            terminate=True,
        )

        assert session_id not in network._SSH_INTERACTIVE_SESSIONS
        assert proc not in harness._active_processes
        assert task.cancelled()
        assert proc.terminate_calls == 1

    asyncio.run(_run())


def test_truthy_plan_approval_metadata_matches_interrupt() -> None:
    state = SimpleNamespace(
        active_plan=SimpleNamespace(approved=1, plan_id="plan-1"),
        draft_plan=None,
    )
    harness = SimpleNamespace(state=state)
    interrupt = {"kind": "plan_execute_approval", "plan_id": "plan-1"}

    assert _approved_plan_matches_interrupt(harness, interrupt) is True
    assert _approved_plan_matches_plan_interrupt(state, interrupt) is True


def test_truncate_output_does_not_modify_short_text() -> None:
    assert truncate_output("hello") == "hello"


def test_truncate_output_truncates_long_text() -> None:
    long_text = "x" * (300 * 1024)
    result = truncate_output(long_text)
    assert len(result) < len(long_text)
    assert result.endswith("[OUTPUT TRUNCATED - TOO LARGE]")


def test_tui_and_cli_harness_kwargs_parity() -> None:
    import logging
    run_logger = logging.getLogger("test")
    config = SmallctlConfig(task="test task")
    tui_kwargs = build_harness_config_kwargs(config, run_logger=run_logger, task=config.task)
    cli_kwargs = build_harness_config_kwargs(config, run_logger=run_logger)
    # TUI intentionally includes task; CLI omits it (passed to run_auto separately)
    assert tui_kwargs.get("task") == "test task"
    assert "task" not in cli_kwargs
    # Verify core fields are identical when task is removed from TUI
    tui_without_task = {k: v for k, v in tui_kwargs.items() if k != "task"}
    assert tui_without_task == cli_kwargs


def test_guard_fail_builds_consistent_metadata() -> None:
    result = guard_fail(
        "blocked",
        reason="test_reason",
        command="rm -rf /",
        error_kind="destructive_delete",
        next_required_tool={"tool_name": "file_read"},
        next_required_action="ask_human",
        extra_metadata={"blocked_targets": ["/"]},
    )
    assert result["success"] is False
    meta = result["metadata"]
    assert meta["reason"] == "test_reason"
    assert meta["error_kind"] == "destructive_delete"
    assert meta["next_required_tool"]["tool_name"] == "file_read"
    assert meta["next_required_action"] == "ask_human"
    assert meta["blocked_targets"] == ["/"]


def test_loop_guard_decision_and_emit_block() -> None:
    state = LoopState(cwd="/tmp")
    path_state: dict[str, Any] = {"blocked_attempts": 0, "escalation_level": 0}
    result = _emit_block(
        state,
        path_state,
        LoopGuardDecision(
            action="block",
            message="test block",
            error_kind="test_error",
        ),
        resolved_path="/tmp/test.py",
        session_id="s1",
        section_name="sec1",
        next_section_name="sec2",
        score=3,
        signals={"hash_stagnation": True},
        tail_excerpt="tail",
        level=2,
        outline_required=True,
    )
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "test_error"
    assert path_state["escalation_level"] == 2
    assert path_state["blocked_attempts"] == 1
    assert path_state["outline_required"] is True


def test_task_classification_rules_table_preserves_precedence() -> None:
    from smallctl.harness.task_classifier import _TASK_CLASSIFICATION_RULES, classify_task_mode

    # Precedence: local_execute > chat > plan_only > remote_execute > debug_inspect > analysis
    assert _TASK_CLASSIFICATION_RULES[0].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[1].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[2].mode == "chat"
    assert _TASK_CLASSIFICATION_RULES[3].mode == "plan_only"
    assert _TASK_CLASSIFICATION_RULES[4].mode == "remote_execute"
    assert _TASK_CLASSIFICATION_RULES[5].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[6].mode == "debug_inspect"
    assert _TASK_CLASSIFICATION_RULES[7].mode == "local_execute"
    assert _TASK_CLASSIFICATION_RULES[8].mode == "analysis"
    # Default fallback
    assert classify_task_mode("") == "chat"
    assert classify_task_mode("hi") == "chat"


def test_resolve_fetch_selector_prefers_single_argument() -> None:
    url, rid, fid, warnings = _resolve_fetch_selector(None, url="http://example.com", result_id=None, fetch_id=None)
    assert url == "http://example.com"
    assert not warnings


def test_resolve_fetch_selector_resolves_multiple_arguments() -> None:
    url, rid, fid, warnings = _resolve_fetch_selector(None, url="http://a.com", result_id="r1", fetch_id="f1")
    assert fid == "f1"
    assert url is None
    assert rid is None
    assert any("fetch_id" in w for w in warnings)
