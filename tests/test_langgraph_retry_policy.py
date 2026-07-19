"""Phase 2: LangGraph native safe, opt-in retry policy tests."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langgraph.errors import GraphInterrupt, NodeCancelledError, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from smallctl.graph.runtime import LoopGraphRuntime
from smallctl.graph.runtime_base import (
    begin_graph_retry_attempts,
    end_graph_retry_attempts,
    graph_retry_policy,
)
from smallctl.graph.runtime_specialized import (
    IndexerGraphRuntime,
    PlanningGraphRuntime,
    ToolPlanRuntime,
)
from smallctl.graph.runtime_staged import StagedExecutionRuntime


class _RetryState(TypedDict, total=False):
    value: int


def _make_harness(
    *, langgraph_native_retries_enabled: bool = False, **overrides: Any
) -> SimpleNamespace:
    max_attempts = overrides.pop("langgraph_native_retry_max_attempts", 2)
    config = SimpleNamespace(
        langgraph_native_retries_enabled=langgraph_native_retries_enabled,
        langgraph_native_retry_max_attempts=max_attempts,
        **overrides,
    )
    events: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(
        config=config,
        _runlog=lambda event, message, **data: events.append((event, data)),
        graph_checkpointer="memory",
        graph_checkpoint_path=None,
    )
    harness.events = events
    return harness


def _build_graph(
    node: Any,
    *,
    node_name: str,
    harness: SimpleNamespace,
) -> StateGraph:
    builder = StateGraph(_RetryState)
    builder.add_node(
        node_name,
        node,
        retry_policy=graph_retry_policy(harness, node_name),
    )
    builder.add_edge(START, node_name)
    builder.add_edge(node_name, END)
    return builder.compile()


async def _ainvoke_with_retry(
    node: Any,
    *,
    node_name: str = "prepare_prompt",
    harness: SimpleNamespace | None = None,
) -> tuple[Any, list[tuple[str, dict[str, object]]]]:
    harness = harness or _make_harness(langgraph_native_retries_enabled=True)
    compiled = _build_graph(node, node_name=node_name, harness=harness)
    result = await compiled.ainvoke({"value": 0})
    return result, harness.events


def test_safe_node_retries_once_when_enabled() -> None:
    calls: list[int] = []

    async def node(state: _RetryState) -> _RetryState:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise ConnectionError("transient failure")
        return {"value": state["value"] + 1}

    harness = _make_harness(langgraph_native_retries_enabled=True)
    result, events = asyncio.run(_ainvoke_with_retry(node, harness=harness))

    assert len(calls) == 2
    assert result["value"] == 1
    retry_events = [e for e in events if e[0] == "graph_retry_attempt"]
    assert len(retry_events) == 1
    assert retry_events[0][1]["node"] == "prepare_prompt"
    assert retry_events[0][1]["attempt"] == 2
    assert retry_events[0][1]["exception_type"] == "ConnectionError"
    assert "delay_sec" in retry_events[0][1]


def test_safe_node_fails_once_when_disabled() -> None:
    calls: list[int] = []

    async def node(state: _RetryState) -> _RetryState:
        calls.append(len(calls) + 1)
        raise ConnectionError("transient failure")

    harness = _make_harness(langgraph_native_retries_enabled=False)

    with pytest.raises(ConnectionError):
        asyncio.run(_ainvoke_with_retry(node, harness=harness))

    assert len(calls) == 1
    assert not any(e[0] == "graph_retry_attempt" for e in harness.events)


@pytest.mark.parametrize(
    "node_name",
    [
        "dispatch_tools",
        "persist_tool_results",
        "apply_tool_outcomes",
        "apply_chat_tool_outcomes",
        "apply_indexer_tool_outcomes",
        "apply_planning_tool_outcomes",
        "interrupt_for_human",
        "model_call",
        "initialize_run",
        "prepare_step",
        "interpret_model_output",
        "activate_or_finalize_step",
        "parse_and_validate_tool_plan",
        "compress_observations",
        "verify_step_completion",
        "route__prepare_prompt",
    ],
)
def test_excluded_nodes_never_get_retry_policy(node_name: str) -> None:
    harness = _make_harness(langgraph_native_retries_enabled=True)
    assert graph_retry_policy(harness, node_name) is None


@pytest.mark.parametrize(
    "runtime_cls",
    [
        LoopGraphRuntime,
        PlanningGraphRuntime,
        IndexerGraphRuntime,
        ToolPlanRuntime,
        StagedExecutionRuntime,
    ],
)
def test_excluded_nodes_have_no_retry_policy_in_compiled_graph(
    runtime_cls: type,
) -> None:
    harness = _make_harness(langgraph_native_retries_enabled=True)
    runtime = runtime_cls.from_harness(harness)
    compiled = runtime._build_compiled_graph()

    for node_name, spec in compiled.builder.nodes.items():
        if node_name.startswith("route__"):
            assert spec.retry_policy is None, f"route node {node_name} should not retry"
            continue
        if node_name in {
            "dispatch_tools",
            "persist_tool_results",
            "interrupt_for_human",
        } or node_name.endswith("_tool_outcomes"):
            assert spec.retry_policy is None, f"{node_name} should not have a retry policy"


def test_graph_interrupt_not_retried() -> None:
    calls: list[int] = []

    async def node(state: _RetryState) -> _RetryState:
        calls.append(len(calls) + 1)
        raise GraphInterrupt("human pause")

    harness = _make_harness(langgraph_native_retries_enabled=True)
    compiled = _build_graph(node, node_name="prepare_prompt", harness=harness)

    async def _consume() -> bool:
        async for chunk in compiled.astream({"value": 0}):
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                return True
        return False

    saw_interrupt = asyncio.run(_consume())
    assert saw_interrupt
    assert len(calls) == 1
    assert not any(e[0] == "graph_retry_attempt" for e in harness.events)


def test_cancelled_error_not_retried() -> None:
    calls: list[int] = []

    async def node(state: _RetryState) -> _RetryState:
        calls.append(len(calls) + 1)
        raise asyncio.CancelledError()

    harness = _make_harness(langgraph_native_retries_enabled=True)

    with pytest.raises(NodeCancelledError):
        asyncio.run(_ainvoke_with_retry(node, harness=harness))

    assert len(calls) == 1
    assert not any(e[0] == "graph_retry_attempt" for e in harness.events)


def test_node_timeout_error_not_retried() -> None:
    calls: list[int] = []

    async def node(state: _RetryState) -> _RetryState:
        calls.append(len(calls) + 1)
        raise NodeTimeoutError("prepare_prompt", 0.05, kind="run", run_timeout=0.05)

    harness = _make_harness(langgraph_native_retries_enabled=True)

    with pytest.raises(NodeTimeoutError):
        asyncio.run(_ainvoke_with_retry(node, harness=harness))

    assert len(calls) == 1
    assert not any(e[0] == "graph_retry_attempt" for e in harness.events)


def test_httpx_5xx_status_error_is_retried() -> None:
    policy = graph_retry_policy(
        _make_harness(langgraph_native_retries_enabled=True), "prepare_prompt"
    )
    assert policy is not None
    response = httpx.Response(status_code=500, text="boom")
    exc = httpx.HTTPStatusError(
        "server error",
        request=httpx.Request("POST", "http://example.test/v1"),
        response=response,
    )
    assert policy.retry_on(exc) is True


def test_generic_runtime_error_is_not_retried() -> None:
    policy = graph_retry_policy(
        _make_harness(langgraph_native_retries_enabled=True), "prepare_prompt"
    )
    assert policy is not None
    assert policy.retry_on(RuntimeError("not transient")) is False


def test_retry_max_attempts_is_capped_at_two() -> None:
    harness = _make_harness(
        langgraph_native_retries_enabled=True,
        langgraph_native_retry_max_attempts=10,
    )
    policy = graph_retry_policy(harness, "prepare_prompt")
    assert policy is not None
    assert policy.max_attempts == 2


def test_retry_attempt_accounting_is_reset_per_graph_invocation() -> None:
    harness = _make_harness(langgraph_native_retries_enabled=True)
    policy = graph_retry_policy(harness, "prepare_prompt")
    assert policy is not None

    first_token = begin_graph_retry_attempts()
    try:
        assert policy.retry_on(ConnectionError("first")) is True
    finally:
        end_graph_retry_attempts(first_token)

    second_token = begin_graph_retry_attempts()
    try:
        assert policy.retry_on(ConnectionError("second")) is True
    finally:
        end_graph_retry_attempts(second_token)

    attempts = [data["attempt"] for event, data in harness.events if event == "graph_retry_attempt"]
    assert attempts == [2, 2]
