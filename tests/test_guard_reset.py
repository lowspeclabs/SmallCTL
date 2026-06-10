from __future__ import annotations

from types import SimpleNamespace
from smallctl.graph.tool_loop_guard_progress import _dir_list_same_path_repeat_is_loop
from smallctl.graph.tool_loop_guards import _detect_repeated_tool_loop
from smallctl.state import LoopState


def test_dir_list_same_path_does_not_trip_on_second_call() -> None:
    """Fix 3: dir_list should not trip until 3+ repetitions within current task."""
    state = LoopState(cwd="/tmp")
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "dir_list", "fingerprint": '{"args": {"path": "/tmp"}, "tool_name": "dir_list"}'},
    ]
    harness = SimpleNamespace(
        state=state,
        client=None,
    )
    pending = SimpleNamespace(tool_name="dir_list", args={"path": "/tmp"})

    assert _dir_list_same_path_repeat_is_loop(harness, pending) is False


def test_dir_list_same_path_trips_on_third_call() -> None:
    """Fix 3: dir_list should trip on 3rd call to same path."""
    state = LoopState(cwd="/tmp")
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "dir_list", "fingerprint": '{"args": {"path": "/tmp"}, "tool_name": "dir_list"}'},
        {"tool_name": "dir_list", "fingerprint": '{"args": {"path": "/tmp"}, "tool_name": "dir_list"}'},
    ]
    harness = SimpleNamespace(
        state=state,
        client=None,
    )
    pending = SimpleNamespace(tool_name="dir_list", args={"path": "/tmp"})

    assert _dir_list_same_path_repeat_is_loop(harness, pending) is True


def test_repeated_tool_loop_detects_file_read_repeat() -> None:
    """Fix 3: _detect_repeated_tool_loop should catch file_read repeated 3 times."""
    state = LoopState(cwd="/tmp")
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "file_read", "fingerprint": '{"args": {"path": "/tmp/foo.py"}, "tool_name": "file_read"}'},
        {"tool_name": "file_read", "fingerprint": '{"args": {"path": "/tmp/foo.py"}, "tool_name": "file_read"}'},
    ]
    harness = SimpleNamespace(
        state=state,
        client=None,
        config=SimpleNamespace(loop_guard_stagnation_threshold=3),
    )
    pending = SimpleNamespace(tool_name="file_read", args={"path": "/tmp/foo.py"})

    result = _detect_repeated_tool_loop(harness, pending)
    assert result is not None
    assert "repeated" in result


def test_tool_attempt_history_reset_on_task_boundary() -> None:
    """Fix 3: _tool_attempt_history should be cleared on task boundary reset."""
    from smallctl.harness.task_boundary_lifecycle_mixin import TaskBoundaryLifecycleMixin
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "prior task"
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "dir_list", "fingerprint": '{"args": {"path": "/tmp"}, "tool_name": "dir_list"}'},
    ]
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(fama_enabled=True, fama_mode="lite"),
        _runlog=lambda *args, **kwargs: None,
        log=None,
        _initial_phase="explore",
        _configured_planning_mode=False,
    )
    mixin = TaskBoundaryLifecycleMixin()
    mixin.harness = harness

    mixin.reset_task_boundary_state(
        reason="new_task",
        new_task="new task",
        previous_task="prior task",
        preserve_memory=False,
        preserve_summaries=False,
        preserve_recent_tail=False,
        preserve_guard_context=True,
    )

    assert "_tool_attempt_history" not in state.scratchpad
