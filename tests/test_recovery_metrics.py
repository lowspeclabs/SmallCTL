from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.graph.tool_outcome_resolution import maybe_apply_terminal_tool_outcome
from smallctl.models.tool_result import ToolEnvelope
from smallctl.recovery_schema import FailureEvent
from smallctl.state import LoopState


def test_terminal_success_records_recovery_success_metrics() -> None:
    state = LoopState(step_count=10)
    state.tool_execution_records = {"op-1": {}, "op-2": {}}
    state.failure_events.append(
        FailureEvent(
            event_id="F-resteer",
            timestamp=1.0,
            failure_class="human_resteer",
            severity="warning",
            source="task_boundary",
            message="human_resteer: user corrected direction",
            metadata={"step": 4},
        )
    )

    async def _emit(_handler, _event) -> None:
        return None

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="loop",
        last_assistant_text="done",
    )
    record = ToolExecutionRecord(
        operation_id="op-complete",
        tool_name="task_complete",
        args={"message": "done"},
        tool_call_id="call-complete",
        result=ToolEnvelope(success=True, output={"status": "complete", "message": "done"}),
    )

    terminal = asyncio.run(
        maybe_apply_terminal_tool_outcome(
            graph_state,
            GraphRuntimeDeps(harness=harness, event_handler=None),
            record,
            chat_mode=False,
        )
    )
    asyncio.run(
        maybe_apply_terminal_tool_outcome(
            graph_state,
            GraphRuntimeDeps(harness=harness, event_handler=None),
            record,
            chat_mode=False,
        )
    )

    metrics = state.scratchpad["_recovery_metrics"]
    assert terminal is True
    assert metrics["recovery_success_count"] == 1
    assert metrics["resteer_recovery_success"] == 1
    assert metrics["tool_calls_until_success"] == [2]
    assert metrics["turns_until_success"] == [6]

