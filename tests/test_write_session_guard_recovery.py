from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from smallctl.graph.lifecycle_guard_recovery import (
    _dispatch_write_session_guard_recovery,
    _extract_write_session_guard_target_path,
    _is_write_session_guard_trip,
)
from smallctl.state import LoopState, WriteSession


class _Harness:
    def __init__(self, state: LoopState) -> None:
        self.state = state
        self.logs: list[dict[str, Any]] = []

    def _runlog(self, event: str, message: str, **data: Any) -> None:
        self.logs.append({"event": event, "message": message, **data})


def _graph_state() -> Any:
    return SimpleNamespace(
        pending_tool_calls=[],
        pending_interrupt=None,
        interrupt_payload=None,
        final_result=None,
        error=None,
    )


def test_extract_target_path_from_write_session_errors() -> None:
    errors = [
        "file_write: patch_over_rewrite_guard: file_write to `./temp/report.html` was rejected ...",
        "file_write: patch_existing_requires_explicit_replace_strategy: file_write to `./temp/report.html` ...",
    ]
    assert _extract_write_session_guard_target_path("Guard tripped: ...", errors) == "./temp/report.html"


def test_is_write_session_guard_trip_requires_write_session_errors() -> None:
    guard = "Guard tripped: max_consecutive_errors (5)"
    assert _is_write_session_guard_trip(guard, ["ssh_exec: failed"] * 5) is False
    assert _is_write_session_guard_trip(
        guard,
        [
            "file_write: patch_over_rewrite_guard: file_write to `./temp/report.html` ...",
            "file_write: patch_existing_requires_explicit_replace_strategy: file_write to `./temp/report.html` ...",
            "file_write: Patch-existing write sessions need an explicit first-chunk choice ...",
        ],
    ) is True


def test_dispatch_write_session_guard_recovery_aborts_session_and_schedules_read() -> None:
    state = LoopState(step_count=18)
    state.recent_errors = [
        "file_write: patch_over_rewrite_guard: file_write to `./temp/report.html` was rejected because ...",
        "file_write: patch_existing_requires_explicit_replace_strategy: file_write to `./temp/report.html` ...",
        "file_write: Patch-existing write sessions need an explicit first-chunk choice ...",
        "file_write: patch_existing_requires_explicit_replace_strategy: file_write to `./temp/report.html` ...",
        "file_write: Patch-existing write sessions need an explicit first-chunk choice ...",
    ]
    session = WriteSession(
        write_session_id="ws-123",
        write_target_path="./temp/report.html",
        write_session_intent="patch_existing",
        write_staging_path="/tmp/stage_report.html",
        status="open",
    )
    state.write_session = session
    harness = _Harness(state)
    graph_state = _graph_state()

    recovered = _dispatch_write_session_guard_recovery(
        harness,
        graph_state,
        "Guard tripped: max_consecutive_errors (5)",
    )

    assert recovered is True
    assert state.recent_errors == []
    assert state.tool_history == []
    assert state.stagnation_counters == {}
    assert len(graph_state.pending_tool_calls) == 1
    assert graph_state.pending_tool_calls[0].tool_name == "file_read"
    assert graph_state.pending_tool_calls[0].args == {"path": "./temp/report.html"}
    assert len(state.recent_messages) == 1
    content = state.recent_messages[0].content or ""
    assert "Recovery:" in content
    assert "ws-123" in content
    assert state.write_session is None or getattr(state.write_session, "status", "") == "aborted"


def test_dispatch_skips_non_write_session_guard_trip() -> None:
    state = LoopState(step_count=5)
    state.recent_errors = ["ssh_exec: failed"] * 5
    harness = _Harness(state)
    graph_state = _graph_state()

    recovered = _dispatch_write_session_guard_recovery(
        harness,
        graph_state,
        "Guard tripped: max_consecutive_errors (5)",
    )

    assert recovered is False
    assert len(graph_state.pending_tool_calls) == 0
    assert len(state.recent_messages) == 0


def test_dispatch_skips_non_max_consecutive_errors_guard() -> None:
    state = LoopState(step_count=5)
    state.recent_errors = [
        "file_write: patch_over_rewrite_guard: file_write to `./temp/report.html` ...",
    ] * 5
    harness = _Harness(state)
    graph_state = _graph_state()

    recovered = _dispatch_write_session_guard_recovery(
        harness,
        graph_state,
        "Guard tripped: max_steps (50)",
    )

    assert recovered is False
    assert len(graph_state.pending_tool_calls) == 0
