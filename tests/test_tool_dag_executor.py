from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.graph.tool_dag_executor import dispatch_tool_dag, _dispatch_single_tool
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.models.tool_result import ToolEnvelope


def _make_harness(*, results: dict[str, ToolEnvelope] | None = None, dispatch_fn=None) -> Any:
    results = results or {}

    async def default_dispatch(tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
        await asyncio.sleep(0.01)
        return results.get(tool_name, ToolEnvelope(success=True, output="ok"))

    registry = SimpleNamespace(
        names=lambda: list(results.keys()) or ["file_read", "grep"],
    )
    return SimpleNamespace(
        registry=registry,
        state=SimpleNamespace(step_count=1, scratchpad={}),
        log=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
        _runlog=lambda *a, **k: None,
        _emit=lambda *a, **k: asyncio.sleep(0),
        _dispatch_tool_call=dispatch_fn or default_dispatch,
        _active_dispatch_task=None,
    )


def _make_deps(harness: Any) -> Any:
    return SimpleNamespace(harness=harness, event_handler=None)


def _make_graph_state() -> GraphRunState:
    return GraphRunState(
        loop_state=SimpleNamespace(step_count=1, scratchpad={}, thread_id="t-1"),
        thread_id="t-1",
        run_mode="tool_plan",
    )


@pytest.mark.asyncio
async def test_dispatch_tool_dag_runs_independent_steps_concurrently() -> None:
    call_order: list[str] = []

    async def slow_dispatch(tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
        call_order.append(tool_name)
        await asyncio.sleep(0.05)
        return ToolEnvelope(success=True, output=f"{tool_name}-result")

    harness = _make_harness(dispatch_fn=slow_dispatch)
    deps = _make_deps(harness)
    graph_state = _make_graph_state()
    batch = [
        PendingToolCall(tool_name="file_read", args={"path": "a.py"}, tool_call_id="tc1", source="tool_plan"),
        PendingToolCall(tool_name="file_read", args={"path": "b.py"}, tool_call_id="tc2", source="tool_plan"),
    ]
    started = asyncio.get_event_loop().time()
    records = await dispatch_tool_dag(graph_state, deps, [batch], max_parallel=4, timeout_sec=10)
    elapsed = asyncio.get_event_loop().time() - started

    assert len(records) == 2
    # Both should have started before either finished (concurrent)
    assert elapsed < 0.15  # serial would be ~0.10; concurrent should be ~0.05 + overhead
    assert all(record.result.metadata["dag_batch_index"] == 0 for record in records)
    assert all(record.result.metadata["dag_batch_size"] == 2 for record in records)
    assert all(record.result.metadata["dag_latency_ms"] > 0 for record in records)


@pytest.mark.asyncio
async def test_dispatch_tool_dag_respects_timeout() -> None:
    async def slow_dispatch(tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
        await asyncio.sleep(1.0)
        return ToolEnvelope(success=True, output="ok")

    harness = _make_harness(dispatch_fn=slow_dispatch)
    deps = _make_deps(harness)
    graph_state = _make_graph_state()
    batch = [
        PendingToolCall(tool_name="file_read", args={"path": "a.py"}, tool_call_id="tc1", source="tool_plan"),
    ]
    records = await dispatch_tool_dag(graph_state, deps, [batch], timeout_sec=0.05)
    assert len(records) == 1
    assert records[0].result.success is False
    assert "timed out" in records[0].result.error


@pytest.mark.asyncio
async def test_dispatch_tool_dag_partial_failure_still_returns_others() -> None:
    async def flaky_dispatch(tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
        if tool_name == "grep":
            raise RuntimeError("boom")
        return ToolEnvelope(success=True, output="ok")

    harness = _make_harness(dispatch_fn=flaky_dispatch)
    deps = _make_deps(harness)
    graph_state = _make_graph_state()
    batch = [
        PendingToolCall(tool_name="file_read", args={"path": "a.py"}, tool_call_id="tc1", source="tool_plan"),
        PendingToolCall(tool_name="grep", args={"pattern": "x"}, tool_call_id="tc2", source="tool_plan"),
    ]
    records = await dispatch_tool_dag(graph_state, deps, [batch], timeout_sec=10)
    assert len(records) == 2
    successes = [r.result.success for r in records]
    assert sorted(successes) == [False, True]


@pytest.mark.asyncio
async def test_dispatch_tool_dag_preserves_result_order_when_enabled() -> None:
    async def ordered_dispatch(tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
        await asyncio.sleep(0.01)
        return ToolEnvelope(success=True, output=tool_name)

    harness = _make_harness(dispatch_fn=ordered_dispatch)
    deps = _make_deps(harness)
    graph_state = _make_graph_state()
    batch = [
        PendingToolCall(tool_name="a", args={}, tool_call_id="tc1", source="tool_plan"),
        PendingToolCall(tool_name="b", args={}, tool_call_id="tc2", source="tool_plan"),
        PendingToolCall(tool_name="c", args={}, tool_call_id="tc3", source="tool_plan"),
    ]
    records = await dispatch_tool_dag(graph_state, deps, [batch], preserve_result_order=True)
    assert [r.tool_name for r in records] == ["a", "b", "c"]
