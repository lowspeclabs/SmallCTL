from __future__ import annotations

from types import SimpleNamespace
from smallctl.harness.task_boundary_lifecycle_mixin import TaskBoundaryLifecycleMixin
from smallctl.recovery_schema import FailureEvent
from smallctl.state import LoopState


def test_task_boundary_reset_does_not_crash_when_active_is_none() -> None:
    """Fix 2: Null-check failure_classes in task_boundary_reset / subtask ledger path.

    When there is no active subtask (ledger.active() returns None), the
    _record_same_scope_resteer method must not crash on active.failure_classes.
    """
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "prior task"
    state.subtask_ledger = SimpleNamespace(
        subtasks=[],
        active_subtask_id=None,
        active=lambda: None,
        handle_human_resteer=lambda text: None,
    )
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(fama_enabled=True, fama_mode="lite"),
        _runlog=lambda *args, **kwargs: None,
        log=None,
    )
    mixin = TaskBoundaryLifecycleMixin()
    mixin.harness = harness

    # This should not raise AttributeError
    mixin._record_same_scope_resteer(
        raw_task="continue",
        effective_task="continue",
        turn_type="ITERATION",
    )

    assert state.last_failure_class == "human_resteer"


def test_task_boundary_reset_preserves_guard_history_within_task() -> None:
    """Fix 3: Guard action window should be reset on task boundary."""
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
        preserve_guard_context=False,
    )

    assert "_tool_attempt_history" not in state.scratchpad


def test_task_boundary_reset_generates_context_brief() -> None:
    """Fix 4: Enable summarization at task boundaries."""
    from smallctl.models.conversation import ConversationMessage
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "prior task"
    state.working_memory.known_facts = ["fact1"]
    state.working_memory.decisions = ["decision1"]
    state.recent_messages = [
        ConversationMessage(role="user", content="read the file"),
        ConversationMessage(role="assistant", content="ok"),
    ]
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(fama_enabled=True, fama_mode="lite"),
        _runlog=lambda *args, **kwargs: None,
        log=None,
        context_policy=None,
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
        preserve_guard_context=False,
    )

    assert len(state.context_briefs) >= 1
    assert state.context_briefs[0].task_goal == "prior task"
