from __future__ import annotations

from types import SimpleNamespace
from smallctl.graph.lifecycle_nodes_support import _apply_continue_task_state_reset
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


def test_continue_after_guard_preserves_capsule_and_clears_bloated_context() -> None:
    """Continue/proceed after a terminal outcome must reset state that bloats
    the prompt (repair phase, conversation history, evidence lanes, exposed-tool
    accumulation) while preserving the guard-trip recovery capsule."""
    state = LoopState(cwd="/tmp")
    state.step_count = 8
    state.inactive_steps = 3
    state.current_phase = "repair"
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.task_exposed_tools = {"ssh_exec", "artifact_read"}
    state.recent_errors = [
        "ssh_exec: Unit postgresql.service could not be found.",
        "ssh_exec: Remote SSH command exited with code 1",
        "ssh_exec: error: no such object: netbox",
        "Guard tripped: max_consecutive_errors (5)",
    ]
    state.tool_history = ["ssh_exec|inspect|fail"]
    state.stagnation_counters["no_progress"] = 2
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "ssh_exec", "fingerprint": "inspect-netbox"},
    ]
    from smallctl.models.conversation import ConversationMessage
    state.recent_messages = [
        ConversationMessage(role="user", content="inspect netbox"),
        ConversationMessage(role="assistant", content="failed"),
        ConversationMessage(role="user", content="continue"),
    ]
    state.reasoning_graph.evidence_records = [{"tool": "ssh_exec", "result": "fail"}]
    state.context_briefs = [{"brief_id": "B1", "text": "brief"}]
    state.episodic_summaries = [{"summary_id": "S1", "text": "summary"}]
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _initial_phase="explore",
    )

    _apply_continue_task_state_reset(harness, task="continue", resolved_task="continue install")

    assert state.step_count == 0
    assert state.inactive_steps == 0
    assert state.recent_errors == []
    assert state.tool_history == []
    assert state.stagnation_counters == {}
    assert "_tool_attempt_history" not in state.scratchpad
    assert state.scratchpad["_continued_after_guard_trip"] is True
    capsule = state.scratchpad["_guard_trip_recovery_capsule"]
    assert capsule["reason"] == "Guard tripped: max_consecutive_errors (5)"
    assert capsule["continued_after_guard"] is True
    assert state.current_phase == "explore"
    assert state.task_exposed_tools == set()
    assert len(state.recent_messages) == 2
    assert state.recent_messages[-1].content == "continue"
    assert state.reasoning_graph.evidence_records == []
    assert state.context_briefs == []
    assert state.episodic_summaries == []
