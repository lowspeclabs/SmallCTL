from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.nodes import LoopRoute, interpret_model_output
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_outcomes import apply_tool_outcomes
from smallctl.models.tool_result import ToolEnvelope
from smallctl.phases import phase_contract
from smallctl.state import LoopState

_CONCLUSION_TEXT = (
    "The dashboard shows steady latency across every checked host. "
    "All probes passed and the report is ready for review."
)


def _make_state() -> LoopState:
    state = LoopState(cwd=".")
    state.active_tool_profiles = ["core"]
    state.artifacts = {}
    return state


def _make_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(min_exploration_steps=0),
        summarizer=None,
        _extract_planning_request=lambda task: None,
        _record_experience=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
        _emit=lambda *args, **kwargs: None,
        _failure=lambda error, error_type="runtime", details=None: {
            "error": error,
            "error_type": error_type,
            "details": details or {},
        },
    )


def _run_interpret(
    harness: SimpleNamespace,
    state: LoopState,
    *,
    pending_tool_calls: list[PendingToolCall] | None = None,
    assistant_text: str = "",
    thread_id: str = "thread-phase2",
) -> tuple[LoopRoute, GraphRunState]:
    graph_state = GraphRunState(
        loop_state=state,
        thread_id=thread_id,
        run_mode="loop",
        pending_tool_calls=list(pending_tool_calls or []),
    )
    graph_state.last_assistant_text = assistant_text
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    route = asyncio.run(interpret_model_output(graph_state, deps))
    return route, graph_state


def _no_tool_nudges(state: LoopState) -> int:
    return int(state.scratchpad.get("_no_tool_nudges", 0))


def test_phase_contract_blocks_do_not_arm_no_tool_auto_finalize() -> None:
    state = _make_state()
    state.strategy = {"thought_architecture": "staged_reasoning"}
    state.current_phase = "author"
    state.working_memory.known_facts = ["latency probes all passed"]
    harness = _make_harness(state)

    for _ in range(4):
        route, graph_state = _run_interpret(
            harness,
            state,
            pending_tool_calls=[PendingToolCall(tool_name="task_complete", args={"message": "done"})],
        )
        assert route == LoopRoute.NEXT_STEP
        assert graph_state.pending_tool_calls == []
        assert graph_state.final_result is None
        assert state.recent_messages[-1].metadata["recovery_kind"] == "phase_contract_all_tools_blocked"

    assert _no_tool_nudges(state) == 0

    route, graph_state = _run_interpret(harness, state, assistant_text=_CONCLUSION_TEXT)

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.final_result is None
    assert _no_tool_nudges(state) == 1
    assert state.recent_messages[-1].metadata["recovery_kind"] == "missing_task_complete"


def test_missing_task_complete_nudges_still_force_complete_at_threshold() -> None:
    state = _make_state()
    state.working_memory.known_facts = ["latency probes all passed"]
    harness = _make_harness(state)

    for expected_count in range(1, 5):
        route, graph_state = _run_interpret(harness, state, assistant_text=_CONCLUSION_TEXT)
        assert route == LoopRoute.NEXT_STEP
        assert graph_state.final_result is None
        assert _no_tool_nudges(state) == expected_count
        assert state.recent_messages[-1].metadata["recovery_kind"] == "missing_task_complete"

    route, graph_state = _run_interpret(harness, state, assistant_text=_CONCLUSION_TEXT)

    assert route == LoopRoute.FINALIZE
    assert graph_state.final_result is not None
    assert graph_state.final_result["status"] == "completed"
    assert _CONCLUSION_TEXT[:200] in graph_state.final_result["message"]["message"]


def _apply_records(
    harness: SimpleNamespace,
    state: LoopState,
    records: list[ToolExecutionRecord],
) -> None:
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-phase2",
        run_mode="loop",
        last_tool_results=records,
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    asyncio.run(apply_tool_outcomes(graph_state, deps))


def _record(tool_name: str, *, success: bool) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        operation_id=f"op-{tool_name}",
        tool_name=tool_name,
        args={},
        tool_call_id=f"call-{tool_name}",
        result=ToolEnvelope(
            success=success,
            output={"status": "ok"} if success else None,
            error=None if success else "dispatch failed",
        ),
    )


def test_no_tool_nudge_counter_resets_only_after_successful_tool_dispatch() -> None:
    state = _make_state()
    harness = _make_harness(state)

    route, _graph_state = _run_interpret(harness, state, assistant_text=_CONCLUSION_TEXT)
    assert route == LoopRoute.NEXT_STEP
    assert _no_tool_nudges(state) == 1

    # Routing a call to dispatch must not reset the counter by itself.
    route, graph_state = _run_interpret(
        harness,
        state,
        pending_tool_calls=[PendingToolCall(tool_name="file_read", args={"path": "README.md"})],
    )
    assert route == LoopRoute.DISPATCH_TOOLS
    assert [call.tool_name for call in graph_state.pending_tool_calls] == ["file_read"]
    assert _no_tool_nudges(state) == 1

    # A failed dispatch preserves the counter.
    _apply_records(harness, state, [_record("file_read", success=False)])
    assert _no_tool_nudges(state) == 1

    # A successful dispatch resets it.
    _apply_records(harness, state, [_record("file_read", success=True)])
    assert _no_tool_nudges(state) == 0

    route, _graph_state = _run_interpret(harness, state, assistant_text=_CONCLUSION_TEXT)
    assert route == LoopRoute.NEXT_STEP
    assert _no_tool_nudges(state) == 1
    assert state.recent_messages[-1].metadata["recovery_kind"] == "missing_task_complete"


def test_verify_phase_blocks_contract_declared_tools_and_names_them() -> None:
    contract = phase_contract("verify")
    assert contract.blocks("task_complete")
    assert contract.blocks("file_append")

    for tool_name in ("task_complete", "file_append"):
        state = _make_state()
        state.strategy = {"thought_architecture": "staged_reasoning"}
        state.current_phase = "verify"
        harness = _make_harness(state)

        route, graph_state = _run_interpret(
            harness,
            state,
            pending_tool_calls=[PendingToolCall(tool_name=tool_name, args={})],
        )

        assert route == LoopRoute.NEXT_STEP
        assert graph_state.pending_tool_calls == []
        nudge = state.recent_messages[-1]
        assert nudge.metadata["is_recovery_nudge"] is True
        assert tool_name in nudge.content
        assert _no_tool_nudges(state) == 0


def test_verify_phase_allows_contract_read_tools() -> None:
    contract = phase_contract("verify")
    assert not contract.blocks("artifact_read")
    assert not contract.blocks("file_read")

    state = _make_state()
    state.strategy = {"thought_architecture": "staged_reasoning"}
    state.current_phase = "verify"
    harness = _make_harness(state)

    route, graph_state = _run_interpret(
        harness,
        state,
        pending_tool_calls=[PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0001"})],
    )

    assert route == LoopRoute.DISPATCH_TOOLS
    assert [call.tool_name for call in graph_state.pending_tool_calls] == ["artifact_read"]


def test_verify_phase_block_list_is_single_sourced_from_phase_contract() -> None:
    contract = phase_contract("verify")
    state = _make_state()
    state.strategy = {"thought_architecture": "staged_reasoning"}
    state.current_phase = "verify"
    harness = _make_harness(state)

    route, graph_state = _run_interpret(
        harness,
        state,
        pending_tool_calls=[
            PendingToolCall(tool_name="task_complete", args={"message": "done"}),
            PendingToolCall(tool_name="file_read", args={"path": "README.md"}),
        ],
    )

    assert route == LoopRoute.NEXT_STEP
    assert [call.tool_name for call in graph_state.pending_tool_calls] == ["file_read"]
    nudge = state.recent_messages[-1]
    assert nudge.metadata["recovery_kind"] == "phase_contract_partial_tools_blocked"
    assert "task_complete" in nudge.content
    assert contract.blocks("task_complete")
    assert not contract.blocks("file_read")
