from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from smallctl.fama.runtime import observe_guard_trip
from smallctl.fama.state import active_mitigation_names
from smallctl.graph.lifecycle_nodes import _guard_trip_reason, _guard_trip_recovery_action
from smallctl.graph.lifecycle_nodes_support import (
    _apply_guard_trip_resteer_or_escalate,
    _MAX_GUARD_TRIP_RESTEER_ATTEMPTS,
)
from smallctl.state import LoopState


class _Config:
    fama_enabled = True
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 4
    fama_signal_window = 8
    fama_done_gate_on_failure = True
    loop_guard_stagnation_threshold = 3


def _harness(state: LoopState) -> Any:
    final_result: dict[str, Any] | None = None

    def _failure(message: str, *, error_type: str = "guard", details: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": "failed",
            "error": message,
            "error_type": error_type,
            "details": details or {},
        }

    def _runlog(event: str, message: str, **data: Any) -> None:
        pass

    return SimpleNamespace(
        state=state,
        config=_Config(),
        _failure=_failure,
        _runlog=_runlog,
    )


def _graph_state() -> Any:
    return SimpleNamespace(
        thread_id="thread-1",
        pending_interrupt=None,
        interrupt_payload=None,
        final_result=None,
        error=None,
    )


def test_observe_guard_trip_routes_looping_signal() -> None:
    state = LoopState(step_count=5)
    harness = _harness(state)
    asyncio.run(
        observe_guard_trip(
            harness,
            guard_error="Guard tripped: repeated tool call loop detected",
            tool_history_tail=[
                "shell_exec|{\"command\": \"docker ps\"}|error:daemon not running",
                "shell_exec|{\"command\": \"docker ps\"}|error:daemon not running",
            ],
        )
    )
    fama = state.scratchpad.get("_fama")
    assert fama is not None
    assert len(fama["signals"]) == 1
    assert fama["signals"][0]["kind"] == "looping"
    assert fama["signals"][0]["source"] == "guard_trip"
    assert "tool_exposure_narrowing" in active_mitigation_names(state)


def test_observe_guard_trip_routes_context_drift_for_max_errors() -> None:
    state = LoopState(step_count=5)
    harness = _harness(state)
    asyncio.run(
        observe_guard_trip(
            harness,
            guard_error="Guard tripped: max_consecutive_errors (5)",
            grouped_errors=[
                {"signature": "ssh_exec: Remote command failed", "count": 3},
            ],
        )
    )
    fama = state.scratchpad.get("_fama")
    assert fama is not None
    assert fama["signals"][0]["kind"] == "context_drift"
    assert "micro_plan_capsule" in active_mitigation_names(state)


def test_observe_guard_trip_extracts_repeated_tool() -> None:
    state = LoopState(step_count=5)
    harness = _harness(state)
    asyncio.run(
        observe_guard_trip(
            harness,
            guard_error="Guard tripped: stagnation limit reached",
            grouped_errors=[
                {"signature": "ssh_exec: Remote command failed", "count": 3},
            ],
        )
    )
    fama = state.scratchpad.get("_fama")
    assert fama["signals"][0]["tool_name"] == "ssh_exec"
    assert "repeated_tool=ssh_exec" in fama["signals"][0]["evidence"]


def test_guard_trip_reason_and_recovery_action() -> None:
    grouped = [{"signature": "ssh_exec: Remote command failed", "count": 3}]
    reason = _guard_trip_reason("Guard tripped: max_consecutive_errors (5)", grouped)
    assert "max_consecutive_errors" in reason
    assert "ssh_exec" in reason

    action = _guard_trip_recovery_action("Guard tripped: max_consecutive_errors (5)", grouped)
    assert "ssh_exec" in action
    assert "Stop repeating" in action


def test_guard_trip_resteer_allowed_within_limit() -> None:
    state = LoopState(step_count=5)
    state.recent_errors.append("Guard tripped: max_consecutive_errors (5)")
    harness = _harness(state)
    graph_state = _graph_state()

    for attempt in range(1, _MAX_GUARD_TRIP_RESTEER_ATTEMPTS + 1):
        result = _apply_guard_trip_resteer_or_escalate(
            harness, graph_state, task="continue", resolved_task="continue"
        )
        assert result is True
        assert state.scratchpad["_guard_trip_resteer_count"] == attempt
        # Simulate the next continuation by re-adding a guard error.
        state.recent_errors.append("Guard tripped: max_consecutive_errors (5)")


def test_guard_trip_resteer_escalates_after_limit() -> None:
    state = LoopState(step_count=5)
    state.recent_errors.append("Guard tripped: max_consecutive_errors (5)")
    harness = _harness(state)
    graph_state = _graph_state()

    for _ in range(_MAX_GUARD_TRIP_RESTEER_ATTEMPTS):
        _apply_guard_trip_resteer_or_escalate(
            harness, graph_state, task="continue", resolved_task="continue"
        )
        state.recent_errors.append("Guard tripped: max_consecutive_errors (5)")

    result = _apply_guard_trip_resteer_or_escalate(
        harness, graph_state, task="continue", resolved_task="continue"
    )
    assert result is False
    assert graph_state.interrupt_payload is not None
    assert graph_state.interrupt_payload["kind"] == "ask_human"
    assert graph_state.final_result is not None
    assert graph_state.final_result["status"] == "failed"


def test_non_guard_continue_does_not_increment_counter() -> None:
    state = LoopState(step_count=5)
    harness = _harness(state)
    graph_state = _graph_state()

    result = _apply_guard_trip_resteer_or_escalate(
        harness, graph_state, task="continue", resolved_task="continue"
    )
    assert result is True
    assert "_guard_trip_resteer_count" not in state.scratchpad
