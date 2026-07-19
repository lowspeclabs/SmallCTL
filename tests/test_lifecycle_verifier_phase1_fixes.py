from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.errors import GraphRecursionError

from smallctl.graph.lifecycle_guard_recovery import _dispatch_stagnation_recovery
from smallctl.graph.progress_guard import _check_progress_stagnation
from smallctl.graph.runtime_payloads import execute_streaming_graph
from smallctl.harness.tool_result_verification_semantic import (
    _passing_verifier_is_weaker_than_prior_failure,
    _prior_failed_verifier_command,
    _semantic_verifier_failure,
)
from smallctl.harness.tool_result_verification_store import _store_verifier_verdict
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


class _FamaConfig:
    fama_enabled = True
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 4
    fama_signal_window = 8
    fama_done_gate_on_failure = True
    loop_guard_stagnation_threshold = 3


def _recovery_harness(state: LoopState) -> Any:
    events: list[str] = []

    def _runlog(event: str, message: str, **data: Any) -> None:
        events.append(event)

    harness = SimpleNamespace(state=state, config=_FamaConfig(), _runlog=_runlog)
    harness.runlog_events = events
    return harness


def test_h6_stagnation_recovery_resets_tripping_counter_and_retains_tool_history() -> None:
    state = LoopState(step_count=7)
    state.current_phase = "act"
    state.stagnation_counters = {
        "no_actionable_progress": 5,
        "no_progress": 2,
        "repeat_command": 1,
        "repeat_patch": 1,
    }
    state.tool_history = [
        "shell_exec|{\"command\": \"docker ps\"}|ok",
        "shell_exec|{\"command\": \"docker ps\"}|ok",
    ]
    harness = _recovery_harness(state)
    guard_error = (
        "Progress stagnation guard tripped: no actionable progress made in 5 steps. "
        "The model is repeating analysis or read-only operations without moving the task forward."
    )

    _dispatch_stagnation_recovery(harness, guard_error)

    assert state.stagnation_counters["no_actionable_progress"] == 0
    assert state.stagnation_counters["no_progress"] == 0
    assert state.stagnation_counters["repeat_command"] == 0
    assert state.stagnation_counters["repeat_patch"] == 0
    # tool_history must be retained so the repeated-action guard stays armed.
    assert state.tool_history == [
        "shell_exec|{\"command\": \"docker ps\"}|ok",
        "shell_exec|{\"command\": \"docker ps\"}|ok",
    ]
    # The FAMA guard observation must fire even though the caller clears
    # guard_error right after dispatch.
    fama = state.scratchpad.get("_fama")
    assert fama is not None
    assert fama["signals"]
    assert fama["signals"][0]["source"] == "guard_trip"
    # The next stagnation check must not re-trip.
    graph_state = SimpleNamespace()
    assert _check_progress_stagnation(harness, graph_state) is None


class _GraphHarness:
    def __init__(self) -> None:
        self.state = SimpleNamespace(thread_id="thread-h7")
        self.conversation_id = "conversation-h7"
        self.config = SimpleNamespace(graph_idle_watchdog_sec=0)
        self.finalized: list[dict[str, Any]] = []
        self.runlog_events: list[str] = []

    def _runlog(self, event: str, message: str, **data: Any) -> None:
        self.runlog_events.append(event)

    def _failure(self, message: str, error_type: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": "failed",
            "reason": message,
            "error": {"type": error_type, "details": details or {}},
        }

    def _finalize(self, result: dict[str, Any]) -> dict[str, Any]:
        # The real harness writes the task summary/checkpoint here.
        self.finalized.append(dict(result))
        return result


class _ExplodingCompiledGraph:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.get_state_calls = 0

    async def astream(self, payload, config):  # type: ignore[no-untyped-def]
        del payload, config
        raise self.exc
        yield {}

    def get_state(self, config):  # type: ignore[no-untyped-def]
        del config
        self.get_state_calls += 1
        raise AssertionError("get_state() should not run after a graph exception")


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("router exploded"),
        GraphRecursionError("Recursion limit of 3 reached"),
    ],
    ids=["node_runtime_error", "langgraph_recursion_error"],
)
def test_h7_unexpected_graph_exception_finalizes_exactly_once(exc: Exception) -> None:
    harness = _GraphHarness()
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))
    compiled = _ExplodingCompiledGraph(exc)

    result = asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=lambda: compiled,
            empty_result_message="unused",
            recursion_limit=3,
        )
    )

    assert result["status"] == "failed"
    assert result["error"]["type"] == "runtime_graph_error"
    assert result["error"]["details"]["exception_type"] == type(exc).__name__
    # Finalize (which writes the task summary) ran exactly once.
    assert len(harness.finalized) == 1
    assert harness.finalized[0]["status"] == "failed"
    assert "runtime_graph_error" in harness.runlog_events
    assert compiled.get_state_calls == 0


def _shell_result(*, success: bool, exit_code: int | None, stdout: str = "", stderr: str = "", error: str = "") -> ToolEnvelope:
    return ToolEnvelope(
        success=success,
        output={"exit_code": exit_code, "stdout": stdout, "stderr": stderr},
        error=error,
    )


def test_h8_corrected_rerun_same_family_clears_prior_failure() -> None:
    state = LoopState()
    state.run_brief.original_task = "Enumerate the LXC containers on the proxmox node."
    failed = _shell_result(
        success=False,
        exit_code=4,
        stderr="hostname lookup 'pve1' failed",
        error="hostname lookup 'pve1' failed",
    )
    seeded = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=failed,
        arguments={"command": "python scripts/Proxmox-cli.py lxcs list --node pve1 2>&1"},
    )
    assert seeded is not None and seeded["verdict"] == "fail"

    passing = _shell_result(success=True, exit_code=0, stdout="## LXCs on pve\n\n- 100: alpha\n")
    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=passing,
        arguments={"command": "python scripts/Proxmox-cli.py lxcs list --node pve 2>&1"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict.get("insufficient_verifier") is not True
    # An accepted pass clears the failed-verifier baseline.
    assert "_last_failed_verifier" not in state.scratchpad
    assert _prior_failed_verifier_command(state) == ""


def test_h8_family_gate_helper_directly() -> None:
    state = LoopState()
    state.run_brief.original_task = "Enumerate the LXC containers on the proxmox node."
    state.scratchpad["_last_failed_verifier"] = {
        "tool_name": "shell_exec",
        "command": "cd /repo && python scripts/Proxmox-cli.py lxcs list --node pve1 2>&1",
        "summary": ["hostname lookup 'pve1' failed"],
        "raw_output": "hostname lookup 'pve1' failed",
    }
    assert _passing_verifier_is_weaker_than_prior_failure(
        state,
        current_command="python scripts/Proxmox-cli.py lxcs list --node pve 2>&1 | head -50",
        current_kind="run_target",
    ) is False
    # A different family of equal strength still may not overwrite the failure.
    assert _passing_verifier_is_weaker_than_prior_failure(
        state,
        current_command="python scripts/Proxmox-cli.py vms list --node pve 2>&1",
        current_kind="run_target",
    ) is True


def test_h8_insufficient_verifier_rejection_does_not_reseed_baseline() -> None:
    state = LoopState()
    state.run_brief.original_task = "Fix the parser and make the test suite pass."
    state.scratchpad["_last_failed_verifier"] = {
        "tool_name": "shell_exec",
        "command": "cd /repo && pytest -q",
        "summary": ["FAILED tests/test_parser.py"],
        "raw_output": "FAILED tests/test_parser.py",
    }

    first = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=_shell_result(success=True, exit_code=0, stdout="ruff: clean"),
        arguments={"command": "cd /repo && ruff check ."},
    )
    assert first is not None
    assert first["verdict"] == "fail"
    assert first["failure_mode"] == "insufficient_verifier"
    # The rejection must not re-seed the baseline with the rejected command.
    assert state.scratchpad["_last_failed_verifier"]["command"] == "cd /repo && pytest -q"
    assert _prior_failed_verifier_command(state) == "cd /repo && pytest -q"

    second = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=_shell_result(success=True, exit_code=0, stdout="mypy: clean"),
        arguments={"command": "cd /repo && mypy ."},
    )
    assert second is not None
    assert second["verdict"] == "fail"
    assert second["failure_mode"] == "insufficient_verifier"
    # The goalpost stays on the genuine failure rather than moving to `ruff`.
    assert "pytest -q" in second["acceptance_delta"]["notes"][0]
    assert _prior_failed_verifier_command(state) == "cd /repo && pytest -q"


def test_m13_app_level_error_block_with_exit_zero_fails() -> None:
    state = LoopState()
    state.run_brief.original_task = "Enumerate the LXC containers on the proxmox node."
    result = _shell_result(
        success=True,
        exit_code=0,
        stdout=(
            "## Error\n\n"
            "- Type: api_error\n"
            "- Message: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed\n"
            "---EXIT:4\n"
        ),
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=result,
        arguments={"command": "python scripts/Proxmox-cli.py lxcs list --node pve 2>&1 | head -50"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert "Error" in verdict["acceptance_delta"]["notes"][0] or "EXIT" in verdict["acceptance_delta"]["notes"][0]


def test_m13_semantic_detector_anchoring() -> None:
    # Leading structured block trips.
    assert _semantic_verifier_failure(
        command="python tool.py list",
        stdout="## Error\n\n- Type: api_error\n---EXIT:4\n",
        stderr="",
    )
    # Trailing nonzero exit marker alone trips.
    assert _semantic_verifier_failure(
        command="python tool.py list",
        stdout="doing work\n---EXIT:4\n",
        stderr="",
    )
    # Mid-output '## Error' (e.g. file contents) must not trip.
    assert not _semantic_verifier_failure(
        command="cat /tmp/app.log",
        stdout="2026-07-16 INFO startup ok\n## Error\nthis line is quoted log content\n",
        stderr="",
    )
    # A mid-output exit marker that is not trailing must not trip.
    assert not _semantic_verifier_failure(
        command="cat /tmp/app.log",
        stdout="job one\n---EXIT:4\njob two recovered\n",
        stderr="",
    )
    # A zero exit marker does not contradict success.
    assert not _semantic_verifier_failure(
        command="python tool.py list",
        stdout="all good\n---EXIT:0\n",
        stderr="",
    )


def test_m13_benign_mid_output_occurrences_pass_verdict() -> None:
    state = LoopState()
    state.run_brief.original_task = "Enumerate the log files on the node."
    result = _shell_result(
        success=True,
        exit_code=0,
        stdout="2026-07-16 INFO startup ok\n## Error\nquoted log content, not a failure\n",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=result,
        arguments={"command": "cat /tmp/app.log"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
