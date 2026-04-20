from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.retrieval import build_retrieval_query
from smallctl.harness import Harness
from smallctl.models.conversation import ConversationMessage
from smallctl.state import ExecutionPlan, LoopState
from smallctl.task_targets import primary_task_target_path


def _make_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="explore",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
    )


def test_new_task_replaces_old_goal_in_run_brief_and_retrieval_query() -> None:
    state = LoopState(cwd="/tmp")
    old_task = "run nmap against 192.168.1.0/24 and list all hosts found in scan"
    new_task = (
        "Build a Python script at `./temp/dead_letter_queue.py` that simulates a tiny "
        "message processor with retry, backoff, and dead-letter queue behavior."
    )
    state.run_brief.original_task = old_task
    state.working_memory.current_goal = old_task

    harness = _make_harness(state)

    Harness._maybe_reset_for_new_task(harness, new_task)
    Harness._initialize_run_brief(harness, new_task, raw_task=new_task)

    assert state.run_brief.original_task == new_task
    assert state.working_memory.current_goal == new_task

    query = build_retrieval_query(state)
    assert new_task in query
    assert old_task not in query


def test_continue_collapses_legacy_followup_chain_to_current_task() -> None:
    state = LoopState(cwd="/tmp")
    old_task = "run nmap against 192.168.1.0/24 and list all hosts found in scan"
    new_task = (
        "Build a Python script at `./temp/dead_letter_queue.py` that simulates a tiny "
        "message processor with retry, backoff, and dead-letter queue behavior."
    )
    contaminated = f"{old_task}\nFollow-up: {new_task}"
    state.run_brief.original_task = contaminated
    state.working_memory.current_goal = contaminated
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": new_task,
        "effective_task": contaminated,
        "current_goal": contaminated,
    }

    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(harness, "continue")
    Harness._initialize_run_brief(harness, resolved, raw_task="continue")

    assert resolved == new_task
    assert state.run_brief.original_task == new_task
    assert state.working_memory.current_goal == new_task
    assert "Follow-up:" not in state.run_brief.original_task

    query = build_retrieval_query(state)
    assert new_task in query
    assert old_task not in query


def test_continue_preserves_active_plan_goal() -> None:
    state = LoopState(cwd="/tmp")
    task = (
        "Build a Python script at `./temp/dead_letter_queue.py` that simulates a tiny "
        "message processor with retry, backoff, and dead-letter queue behavior."
    )
    plan_goal = "Implement the retry loop and dead-letter queue tests."
    state.run_brief.original_task = task
    state.working_memory.current_goal = plan_goal
    state.active_plan = ExecutionPlan(plan_id="plan-1", goal=plan_goal)

    harness = _make_harness(state)

    Harness._initialize_run_brief(harness, task, raw_task="continue")

    assert state.run_brief.original_task == task
    assert state.working_memory.current_goal == plan_goal


def test_affirmative_followup_resolves_to_prior_task_after_action_confirmation() -> None:
    state = LoopState(cwd="/tmp")
    task = "read ./portainer_cli.py and recommend updates to that script"
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="Would you like me to create an updated version incorporating these recommendations?",
        )
    ]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": task,
        "effective_task": task,
        "current_goal": task,
    }

    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(harness, "yes")
    Harness._initialize_run_brief(harness, resolved, raw_task="yes")

    assert resolved == task
    assert state.run_brief.original_task == task
    assert state.working_memory.current_goal == task

    query = build_retrieval_query(state)
    assert task in query


def test_primary_task_target_path_falls_back_to_last_task_handoff() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "run and confirm functionality"
    state.working_memory.current_goal = "run and confirm functionality"
    state.scratchpad["_task_target_paths"] = []
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": "Build a Python script at `./temp/mini_indexer.py`.",
        "effective_task": "Build a Python script at `./temp/mini_indexer.py`.",
        "current_goal": "Verify `./temp/mini_indexer.py`.",
        "target_paths": ["./temp/mini_indexer.py"],
    }

    harness = _make_harness(state)

    assert primary_task_target_path(harness) == "./temp/mini_indexer.py"
