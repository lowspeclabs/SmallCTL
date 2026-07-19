"""Regression tests for Phase 3 hygiene fixes (L1-L7, M4)."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from smallctl.graph import error_hardening, interpret_nodes, node_support
from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.error_hardening import (
    _maybe_nudge_ssh_auth_fallback,
    _maybe_schedule_web_search_for_repeated_error,
)
from smallctl.graph.interpret_nodes import interpret_model_output
from smallctl.graph.lifecycle_step_budget import (
    STEP_BUDGET_NUDGE_THRESHOLD,
    _maybe_inject_step_budget_nudge,
)
from smallctl.graph.model_stream_resolution import (
    _maybe_inject_strategy_switch,
    _reset_empty_write_failure,
    _track_empty_write_failure,
)
from smallctl.graph.node_support import HALLUCINATION_MAP
from smallctl.graph.routing import LoopRoute
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.graph.write_session_outcomes_support import _abort_write_session
from smallctl.harness.tool_dispatch import dispatch_tool_call
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools import build_registry
from smallctl.tools.base import ToolSpec, build_tool_schema
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.fs_write_sessions import _store_active_write_session
from smallctl.tools.registry import ToolRegistry
from smallctl.write_session_fsm import new_write_session


# -----------------------------------------------------------------------------
# L1: step-budget nudge is latched per task/run
# -----------------------------------------------------------------------------


def _make_step_budget_harness() -> SimpleNamespace:
    state = LoopState(cwd="/tmp")
    state.step_count = STEP_BUDGET_NUDGE_THRESHOLD + 1
    state.scratchpad["_model_name"] = "qwen2.5:7b"
    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        client=None,
        _runlog=lambda event, *_args, **_kwargs: runlog_events.append(event),
    )
    harness.runlog_events = runlog_events
    return harness


def test_l1_step_budget_nudge_latched_until_task_boundary() -> None:
    harness = _make_step_budget_harness()
    state = harness.state
    graph_state = SimpleNamespace(final_result=None)

    assert _maybe_inject_step_budget_nudge(harness, graph_state) is True
    assert _maybe_inject_step_budget_nudge(harness, graph_state) is False

    nudges = [
        message
        for message in state.recent_messages
        if message.metadata.get("recovery_kind") == "step_budget_exceeded"
    ]
    assert len(nudges) == 1
    assert harness.runlog_events.count("step_budget_exceeded") == 1

    # A new task/run boundary re-arms the latch.
    state.scratchpad["_task_sequence"] = 2
    assert _maybe_inject_step_budget_nudge(harness, graph_state) is True
    assert harness.runlog_events.count("step_budget_exceeded") == 2


# -----------------------------------------------------------------------------
# L2: SSH auth fallback is per-target, threshold-crossing, reset on success
# -----------------------------------------------------------------------------


def _ssh_record(
    *,
    success: bool,
    error: str = "",
    host: str = "192.0.2.10",
    user: str = "root",
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        operation_id="op-ssh",
        tool_name="ssh_exec",
        args={"host": host, "user": user, "command": "whoami"},
        tool_call_id="tc-ssh",
        result=ToolEnvelope(success=success, error=error),
    )


def _ssh_harness() -> MagicMock:
    harness = MagicMock()
    harness.state.scratchpad = {}
    harness.state.append_message = MagicMock()
    harness._runlog = MagicMock()
    return harness


def test_l2_ssh_auth_fallback_emits_once_per_target_episode() -> None:
    graph_state = MagicMock()
    harness = _ssh_harness()
    failure = _ssh_record(success=False, error="Permission denied (publickey).")

    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, failure) is False
    # Threshold crossing: emit once.
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, failure) is True
    assert harness.state.append_message.call_count == 1
    # Third failure does not re-emit.
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, failure) is False
    assert harness.state.append_message.call_count == 1

    # Successful auth clears the target; the next episode emits once again.
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, _ssh_record(success=True)) is False
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, failure) is False
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, failure) is True
    assert harness.state.append_message.call_count == 2


def test_l2_ssh_auth_fallback_counts_are_target_specific() -> None:
    graph_state = MagicMock()
    harness = _ssh_harness()
    host_a = _ssh_record(success=False, error="Permission denied (publickey).", host="192.0.2.10")
    host_b = _ssh_record(success=False, error="Permission denied (publickey).", host="192.0.2.20")

    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, host_a) is False
    # A different target has its own count and does not trip the first target.
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, host_b) is False
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, host_a) is True
    assert _maybe_nudge_ssh_auth_fallback(graph_state, harness, host_b) is True
    assert harness.state.append_message.call_count == 2
    counts = harness.state.scratchpad["_ssh_auth_failure_counts"]
    assert counts == {"root@192.0.2.10": 2, "root@192.0.2.20": 2}


# -----------------------------------------------------------------------------
# L3: empty-write strategy-switch flag resets per failure episode
# -----------------------------------------------------------------------------


def _stream_harness() -> MagicMock:
    harness = MagicMock()
    harness.state.scratchpad = {"_model_name": "qwen2.5:7b"}
    harness.state.append_message = MagicMock()
    harness._runlog = MagicMock()
    return harness


def test_l3_empty_write_strategy_switch_rearms_per_episode() -> None:
    harness = _stream_harness()

    # First episode on path A: one strategy-switch nudge.
    for _ in range(2):
        _track_empty_write_failure(harness, "src/a.py")
    count = _track_empty_write_failure(harness, "src/a.py")
    _maybe_inject_strategy_switch(harness, "src/a.py", count)
    assert harness.state.append_message.call_count == 1
    # Same episode does not re-nudge.
    _maybe_inject_strategy_switch(harness, "src/a.py", count + 1)
    assert harness.state.append_message.call_count == 1

    # Successful write resets the episode.
    _reset_empty_write_failure(harness)

    # New episode on another path: nudge fires again.
    for _ in range(2):
        _track_empty_write_failure(harness, "src/b.py")
    count = _track_empty_write_failure(harness, "src/b.py")
    _maybe_inject_strategy_switch(harness, "src/b.py", count)
    assert harness.state.append_message.call_count == 2
    kinds = [
        call.args[0].metadata.get("recovery_kind")
        for call in harness.state.append_message.call_args_list
    ]
    assert kinds == ["empty_write_strategy_switch", "empty_write_strategy_switch"]


def test_l3_empty_write_path_change_rearms_strategy_switch() -> None:
    harness = _stream_harness()

    for _ in range(2):
        _track_empty_write_failure(harness, "src/a.py")
    count = _track_empty_write_failure(harness, "src/a.py")
    _maybe_inject_strategy_switch(harness, "src/a.py", count)
    assert harness.state.append_message.call_count == 1

    # Switching paths without an intervening success is a new episode.
    for _ in range(2):
        _track_empty_write_failure(harness, "src/b.py")
    count = _track_empty_write_failure(harness, "src/b.py")
    _maybe_inject_strategy_switch(harness, "src/b.py", count)
    assert harness.state.append_message.call_count == 2


# -----------------------------------------------------------------------------
# L4: non-numeric timeout_sec does not raise before schema validation
# -----------------------------------------------------------------------------


def test_l4_non_numeric_timeout_sec_yields_structured_error() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="shell_exec",
            description="fake shell",
            schema=build_tool_schema(
                required=["command"],
                properties={
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer"},
                },
            ),
            handler=lambda **kwargs: None,
            category="shell",
            risk="high",
        )
    )
    dispatcher = ToolDispatcher(registry, phase="execute")
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(graph_dispatch_tools_timeout_sec=300),
        registry=registry,
        dispatcher=dispatcher,
        _current_user_task=lambda: "run a command",
        _runlog=lambda *args, **kwargs: None,
        artifact_store=SimpleNamespace(
            compact_tool_message=lambda artifact, result, **kwargs: str(result.output or result.error or "")
        ),
        context_policy=SimpleNamespace(tool_result_inline_token_limit=200),
    )

    result = asyncio.run(
        dispatch_tool_call(harness, "shell_exec", {"command": "sleep 1", "timeout_sec": "120s"})
    )

    assert result.success is False
    assert "timeout_sec" in str(result.error)


# -----------------------------------------------------------------------------
# L5: aborting one write session leaves other sessions active and fixes alias
# -----------------------------------------------------------------------------


def test_l5_abort_one_session_keeps_other_active_and_alias(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    session_a = new_write_session(
        session_id="ws-a",
        target_path="src/a.py",
        intent="replace_file",
    )
    session_b = new_write_session(
        session_id="ws-b",
        target_path="src/b.py",
        intent="replace_file",
    )
    _store_active_write_session(state, session_a)
    _store_active_write_session(state, session_b)
    state.write_session = session_a
    session_a.write_failed_local_patches = 5

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    _abort_write_session(harness, session_a)

    active = list(state.active_write_sessions_by_path.values())
    assert session_a not in active
    assert session_b in active
    assert state.write_session is session_b

    archived = state.scratchpad.get("_archived_write_sessions", [])
    assert [entry["write_session_id"] for entry in archived] == ["ws-a"]
    assert archived[0]["reason"].startswith("write_session_aborted_after_")


# -----------------------------------------------------------------------------
# L6: scratchpad counter maps are bounded
# -----------------------------------------------------------------------------


def test_l6_repeated_error_signatures_map_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(error_hardening, "_SCRATCHPAD_COUNTER_MAP_CAP", 4)
    graph_state = MagicMock()
    harness = MagicMock()
    harness.state.scratchpad = {}
    harness.state.append_message = MagicMock()
    harness._runlog = MagicMock()
    harness.registry.names = MagicMock(return_value=[])

    for index in range(6):
        record = ToolExecutionRecord(
            operation_id=f"op-{index}",
            tool_name="shell_exec",
            args={"command": "failing-cmd"},
            tool_call_id=f"tc-{index}",
            result=ToolEnvelope(
                success=False,
                error=f"distinct failure number {index} occurred here",
            ),
        )
        _maybe_schedule_web_search_for_repeated_error(graph_state, harness, record)

    counts = harness.state.scratchpad["_repeated_error_signatures"]
    assert len(counts) == 4
    serialized = json.dumps(counts)
    assert "distinct failure number 0" not in serialized
    assert "distinct failure number 1" not in serialized
    assert "distinct failure number 5" in serialized


def test_l6_non_actionable_prose_counts_map_is_bounded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(interpret_nodes, "_NON_ACTIONABLE_PROSE_COUNTS_CAP", 4)

    state = LoopState(cwd=str(tmp_path))
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "inspect the workspace"
    state.working_memory.current_goal = "inspect the workspace"
    harness = SimpleNamespace(
        state=state,
        registry=None,
        config=SimpleNamespace(min_exploration_steps=0),
        summarizer=None,
        summarizer_client=None,
        _runlog=lambda *args, **kwargs: None,
        _emit=lambda *args, **kwargs: None,
        _extract_planning_request=lambda _task: None,
        _record_experience=lambda *args, **kwargs: None,
        _failure=lambda error, error_type="runtime", details=None: {
            "error": error,
            "error_type": error_type,
            "details": details or {},
        },
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    for index in range(6):
        graph_state = GraphRunState(
            loop_state=state,
            thread_id=f"thread-l6-{index}",
            run_mode="loop",
        )
        graph_state.last_assistant_text = f"I need to check item number {index} next."
        route = asyncio.run(interpret_model_output(graph_state, deps))
        assert route == LoopRoute.NEXT_STEP

    counts = state.scratchpad["_non_actionable_prose_counts"]
    assert len(counts) == 4
    serialized = json.dumps(counts)
    assert "item number 0" not in serialized
    assert "item number 1" not in serialized
    assert "item number 5" in serialized


# -----------------------------------------------------------------------------
# L7: blocked (not exposed) tool calls do not inflate repeat history
# -----------------------------------------------------------------------------


def _l7_tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {},
        },
    }


def _l7_harness(state: LoopState, dispatched: list) -> SimpleNamespace:
    async def _dispatch_tool_call(tool_name: str, args: dict) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="ok")

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    return SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        registry=SimpleNamespace(
            names=lambda: {"file_read", "dir_list"},
            get=lambda _name: None,
            export_openai_tools=lambda **kwargs: [_l7_tool_schema("file_read")],
        ),
        _current_user_task=lambda: "inspect this log and explain what failed",
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
        log=logging.getLogger("test.graph_hygiene_phase3.l7"),
    )


def _l7_graph_state(state: LoopState, tool_name: str, args: dict, thread: str) -> GraphRunState:
    return GraphRunState(
        loop_state=state,
        thread_id=thread,
        run_mode="chat",
        pending_tool_calls=[
            PendingToolCall(tool_name=tool_name, args=args, source="model"),
        ],
    )


def test_l7_hidden_tool_calls_leave_repeat_history_unchanged(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "inspect this log and explain what failed"
    state.working_memory.current_goal = state.run_brief.original_task

    dispatched: list = []
    harness = _l7_harness(state, dispatched)
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    for index in range(5):
        graph_state = _l7_graph_state(
            state, "dir_list", {"path": "."}, f"thread-l7-hidden-{index}"
        )
        asyncio.run(dispatch_tools(graph_state, deps))
        assert len(graph_state.last_tool_results) == 1
        result = graph_state.last_tool_results[0].result
        assert result.success is False
        assert result.metadata["reason"] == "tool_not_exposed_this_turn"

    assert dispatched == []
    assert state.scratchpad.get("_tool_attempt_history", []) == []
    assert not any("Guard tripped" in str(error) for error in state.recent_errors)

    # Control: an admitted call is still recorded exactly once.
    graph_state = _l7_graph_state(state, "file_read", {"path": "README.md"}, "thread-l7-admitted")
    asyncio.run(dispatch_tools(graph_state, deps))
    history = state.scratchpad.get("_tool_attempt_history", [])
    assert [entry["tool_name"] for entry in history] == ["file_read"]


# -----------------------------------------------------------------------------
# M4: dead tool names purged; HALLUCINATION_MAP targets registered tools
# -----------------------------------------------------------------------------


def test_m4_hallucination_map_targets_registered_tools_and_no_dead_names() -> None:
    class _FakeStateProvider:
        def __init__(self) -> None:
            self.state = LoopState(cwd="/tmp")
            self.log = SimpleNamespace(info=lambda *args, **kwargs: None)

    registry = build_registry(_FakeStateProvider())
    registered = {str(name) for name in registry.names()}

    assert registered
    assert set(HALLUCINATION_MAP.values()) <= registered

    dead_names = ("bash_exec", "long_context_lookup", "summarize_report")
    for module in (interpret_nodes, node_support):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for name in dead_names:
            assert name not in source
