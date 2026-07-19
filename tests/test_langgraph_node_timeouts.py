"""Phase 1: LangGraph native per-node timeout behavior."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.errors import GraphInterrupt, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from smallctl.graph.runtime import LoopGraphRuntime
from smallctl.graph.runtime_base import (
    GraphNodeTimeoutError,
    graph_node_timeout_policy,
    graph_node_timeout_sec,
)
from smallctl.graph.runtime_payloads import execute_streaming_graph
from smallctl.graph.runtime_specialized import (
    IndexerGraphRuntime,
    PlanningGraphRuntime,
    ToolPlanRuntime,
)
from smallctl.graph.runtime_staged import StagedExecutionRuntime


class _TimeoutState(TypedDict, total=False):
    x: int


def _make_harness(**config_overrides: Any) -> SimpleNamespace:
    config = SimpleNamespace(
        graph_node_timeout_sec=1.0,
        graph_model_call_timeout_sec=2.0,
        graph_dispatch_tools_timeout_sec=3.0,
        graph_idle_watchdog_sec=0,
        langgraph_native_timeouts_enabled=True,
    )
    for key, value in config_overrides.items():
        setattr(config, key, value)
    harness = SimpleNamespace(
        config=config,
        state=SimpleNamespace(thread_id="thread-123"),
        conversation_id="conversation-123",
        graph_checkpointer="memory",
        graph_checkpoint_path=None,
        _runlog=lambda *args, **kwargs: None,
    )
    harness._finalize = lambda result: result
    harness._failure = lambda message, error_type, details=None: {
        "status": "failed",
        "reason": message,
        "error": {"type": error_type, "details": details or {}},
    }
    return harness


def test_graph_node_timeout_policy_uses_existing_resolution_for_safe_nodes() -> None:
    harness = _make_harness()
    assert graph_node_timeout_policy(harness, "prepare_prompt") == graph_node_timeout_sec(
        harness, "prepare_prompt"
    )
    assert graph_node_timeout_policy(harness, "model_call") == graph_node_timeout_sec(
        harness, "model_call"
    )
    assert graph_node_timeout_policy(harness, "interpret_model_output") == 1.0


def test_graph_node_timeout_policy_disabled_flag_returns_none() -> None:
    harness = _make_harness(langgraph_native_timeouts_enabled=False)
    assert graph_node_timeout_policy(harness, "prepare_prompt") is None
    assert graph_node_timeout_policy(harness, "model_call") is None
    assert graph_node_timeout_policy(harness, "route__prepare_prompt") is None


def test_graph_node_timeout_policy_is_opt_in_when_flag_is_absent() -> None:
    harness = _make_harness()
    del harness.config.langgraph_native_timeouts_enabled
    assert graph_node_timeout_policy(harness, "prepare_prompt") is None


def test_graph_node_timeout_policy_returns_none_for_mutating_nodes() -> None:
    harness = _make_harness()
    for node_name in (
        "dispatch_tools",
        "persist_tool_results",
        "apply_tool_outcomes",
        "apply_chat_tool_outcomes",
        "apply_indexer_tool_outcomes",
        "apply_planning_tool_outcomes",
        "interrupt_for_human",
    ):
        assert graph_node_timeout_policy(harness, node_name) is None


def test_native_timeout_fires_for_non_mutating_node() -> None:
    async def slow_node(state: _TimeoutState) -> _TimeoutState:
        del state
        await asyncio.sleep(10)
        return {"x": 1}

    builder = StateGraph(_TimeoutState)
    builder.add_node("prepare_prompt", slow_node, timeout=0.05)
    builder.add_edge(START, "prepare_prompt")
    builder.add_edge("prepare_prompt", END)
    compiled = builder.compile()

    with pytest.raises(NodeTimeoutError) as exc_info:
        asyncio.run(compiled.ainvoke({"x": 0}))

    assert exc_info.value.node == "prepare_prompt"
    assert exc_info.value.run_timeout == 0.05


def test_execute_streaming_graph_translates_node_timeout_error() -> None:
    class _TimeoutCompiledGraph:
        async def astream(self, payload: Any, config: Any) -> Any:
            del payload, config
            raise NodeTimeoutError("prepare_prompt", 0.05, kind="run", run_timeout=0.05)
            yield {}

        def get_state(self, config: Any) -> Any:
            del config
            raise AssertionError("get_state() should not run after node timeout")

    harness = _make_harness(graph_node_timeout_sec=0.05)
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))

    result = asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=_TimeoutCompiledGraph,
            empty_result_message="unused",
            recursion_limit=8,
        )
    )

    assert result["status"] == "failed"
    assert result["error"]["type"] == "graph_node_timeout"
    assert result["error"]["details"]["node"] == "prepare_prompt"
    assert result["error"]["details"]["timeout_sec"] == 0.05


def test_execute_streaming_graph_still_handles_graph_node_timeout_error() -> None:
    class _LegacyTimeoutCompiledGraph:
        async def astream(self, payload: Any, config: Any) -> Any:
            del payload, config
            raise GraphNodeTimeoutError("model_call", 0.05)
            yield {}

        def get_state(self, config: Any) -> Any:
            del config
            raise AssertionError("get_state() should not run after node timeout")

    harness = _make_harness()
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))

    result = asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=_LegacyTimeoutCompiledGraph,
            empty_result_message="unused",
            recursion_limit=8,
        )
    )

    assert result["status"] == "failed"
    assert result["error"]["type"] == "graph_node_timeout"
    assert result["error"]["details"]["node"] == "model_call"


def test_graph_node_timeout_policy_returns_timeout_for_route_nodes() -> None:
    harness = _make_harness()
    # Route nodes inherit the per-node timeout resolution from graph_node_timeout_sec.
    # "dispatch_tools" is in the route node name, so the dispatch-tools default applies.
    assert graph_node_timeout_policy(harness, "route__dispatch_tools") == 3.0
    assert graph_node_timeout_policy(harness, "route__apply_tool_outcomes") == 1.0


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
def test_mutating_nodes_not_given_native_timeout(runtime_cls: type) -> None:
    harness = _make_harness()
    runtime = runtime_cls.from_harness(harness)
    compiled = runtime._build_compiled_graph()

    for node_name, spec in compiled.builder.nodes.items():
        if node_name.startswith("route__"):
            continue
        if (
            node_name in {"dispatch_tools", "persist_tool_results", "interrupt_for_human"}
            or node_name.endswith("_tool_outcomes")
        ):
            assert spec.timeout is None, f"{node_name} should not have a native timeout"


def test_graph_interrupt_escapes_without_timeout_conversion() -> None:
    class _InterruptCompiledGraph:
        async def astream(self, payload: Any, config: Any) -> Any:
            del payload, config
            yield {"__interrupt__": {"question": "continue?"}}

        def get_state(self, config: Any) -> Any:
            del config
            return SimpleNamespace(values={})

    harness = _make_harness()
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))

    result = asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=_InterruptCompiledGraph,
            empty_result_message="unused",
            recursion_limit=8,
        )
    )

    assert result["status"] == "needs_human"
    assert result["message"]["status"] == "human_input_required"


def test_graph_interrupt_exception_not_converted_to_timeout() -> None:
    class _InterruptCompiledGraph:
        async def astream(self, payload: Any, config: Any) -> Any:
            del payload, config
            raise GraphInterrupt()
            yield {}

        def get_state(self, config: Any) -> Any:
            del config
            return SimpleNamespace(values={})

    harness = _make_harness()
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))

    with pytest.raises(GraphInterrupt):
        asyncio.run(
            execute_streaming_graph(
                runtime,
                {"input_task": "demo"},
                build_graph=_InterruptCompiledGraph,
                empty_result_message="unused",
                recursion_limit=8,
            )
        )


@pytest.mark.asyncio
async def test_indexer_runtime_native_timeout_returns_public_shape() -> None:
    """IndexerGraphRuntime must use the shared timeout-result translation path."""
    from smallctl.harness import Harness

    harness = Harness(
        endpoint="http://example.test/v1",
        model="test-model",
        provider_profile="generic",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        graph_node_timeout_sec=0.05,
        langgraph_native_timeouts_enabled=True,
    )
    harness.config.graph_idle_watchdog_sec = None
    runtime = IndexerGraphRuntime.from_harness(harness)

    async def patched_prepare_indexer_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        await asyncio.sleep(10)
        return {}

    runtime._prepare_indexer_prompt_node = patched_prepare_indexer_prompt

    result = await runtime.run("indexer timeout test")

    assert result["status"] == "failed"
    assert result["error"]["type"] == "graph_node_timeout"
    assert result["error"]["details"]["node"] == "prepare_indexer_prompt"
    assert result["error"]["details"]["timeout_sec"] == 0.05
