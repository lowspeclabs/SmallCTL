from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.graph.lifecycle_nodes import prepare_loop_step
from smallctl.graph.state import GraphRunState
from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.guards import GuardConfig
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState


def _make_harness(state: LoopState) -> SimpleNamespace:
    harness = SimpleNamespace(
        state=state,
        _cancel_requested=False,
        _runlog=lambda *args, **kwargs: None,
        log=SimpleNamespace(warning=lambda *args, **kwargs: None),
        _emit=lambda *args, **kwargs: asyncio.Future(),
        _ensure_context_limit=lambda: asyncio.Future(),
        _initialize_run_brief=lambda *args, **kwargs: None,
        _activate_tool_profiles=lambda *args, **kwargs: None,
        _log_conversation_state=lambda *args, **kwargs: None,
        dispatcher=SimpleNamespace(phase="explore"),
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        guards=GuardConfig(),
    )
    return harness


def test_prepare_loop_step_refreshes_stale_phase_objective() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "explore"
    state.run_brief.current_phase_objective = "explore: redesign the remote site"
    state.run_brief.original_task = "redesign the remote site"
    state.recent_messages = [ConversationMessage(role="user", content="redesign the remote site")]

    harness = _make_harness(state)
    harness.state.contract_phase = lambda: "execute"

    graph_state = GraphRunState(loop_state=state, thread_id="test-1", run_mode="loop")
    deps = GraphRuntimeDeps(harness=harness)

    asyncio.run(prepare_loop_step(graph_state, deps))

    assert state.current_phase == "execute"
    assert state.run_brief.current_phase_objective == "execute: redesign the remote site"


def test_prepare_loop_step_leaves_matching_phase_objective_unchanged() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.run_brief.current_phase_objective = "execute: patch the config"
    state.run_brief.original_task = "patch the config"
    state.recent_messages = [ConversationMessage(role="user", content="patch the config")]

    harness = _make_harness(state)
    harness.state.contract_phase = lambda: "execute"

    graph_state = GraphRunState(loop_state=state, thread_id="test-2", run_mode="loop")
    deps = GraphRuntimeDeps(harness=harness)

    asyncio.run(prepare_loop_step(graph_state, deps))

    assert state.current_phase == "execute"
    assert state.run_brief.current_phase_objective == "execute: patch the config"
