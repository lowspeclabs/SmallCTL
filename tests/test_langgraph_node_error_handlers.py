"""Phase 3: LangGraph native node error handlers."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.errors import NodeError
from langgraph.graph import END

from smallctl.graph.runtime import LoopGraphRuntime
from smallctl.graph.runtime_base import (
    _ERROR_RECOVERY_ALLOWED_NODES,
    _ERROR_RECOVERY_INTERPRET_NODES,
    _ERROR_RECOVERY_PREPARE_NODES,
    _is_interpret_error_recoverable,
    _prepare_node_for_runtime,
    _sanitize_error_message_for_handler,
    graph_error_handler_policy,
    make_graph_error_handler,
)
from smallctl.graph.runtime_payloads import load_runtime_state, serialize_runtime_state
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.harness import Harness


def _make_harness(*, langgraph_error_handlers_enabled: bool = False) -> Harness:
    harness = Harness(
        endpoint="http://example.test/v1",
        model="test-model",
        provider_profile="generic",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        langgraph_error_handlers_enabled=langgraph_error_handlers_enabled,
    )
    harness.config.graph_idle_watchdog_sec = None
    harness.config.tool_call_repair_enabled = True
    runlog_events: list[tuple[str, dict[str, Any]]] = []
    original_runlog = harness._runlog

    def _capture_runlog(event, _message, **data):
        runlog_events.append((event, data))
        original_runlog(event, _message, **data)

    harness._runlog = _capture_runlog
    harness.runlog_events = runlog_events
    return harness


async def _run_loop_graph(runtime: LoopGraphRuntime) -> dict[str, Any]:
    return await runtime.run("error handler test")


def test_allowed_nodes_set_is_explicit_and_safe() -> None:
    allowed = _ERROR_RECOVERY_ALLOWED_NODES
    prepare = _ERROR_RECOVERY_PREPARE_NODES
    interpret = _ERROR_RECOVERY_INTERPRET_NODES

    for node in prepare:
        assert node in allowed
    for node in interpret:
        assert node in allowed

    # Mutation-capable and approval nodes must not be allowed.
    for node in (
        "dispatch_tools",
        "persist_tool_results",
        "apply_tool_outcomes",
        "apply_chat_tool_outcomes",
        "apply_indexer_tool_outcomes",
        "apply_planning_tool_outcomes",
        "interrupt_for_human",
        "model_call",
    ):
        assert node not in allowed


def test_sanitize_error_message_truncates_and_returns_summary() -> None:
    long_message = "x" * 1000
    assert len(_sanitize_error_message_for_handler(ValueError(long_message))) == 500
    assert _sanitize_error_message_for_handler(ValueError("boom")) == "boom"


def test_prepare_node_for_runtime_picks_context_specific_node() -> None:
    runtime = SimpleNamespace(
        GRAPH_SPEC=SimpleNamespace(
            node_map={
                "prepare_prompt": "_prepare_prompt_node",
                "prepare_solver_prompt": "_prepare_solver_prompt_node",
            }
        )
    )
    assert _prepare_node_for_runtime(runtime, "prepare_prompt") == "prepare_prompt"
    assert _prepare_node_for_runtime(runtime, "interpret_model_output") == "prepare_prompt"
    assert _prepare_node_for_runtime(runtime, "interpret_solver_output") == "prepare_solver_prompt"


def test_make_graph_error_handler_returns_none_for_unallowed_nodes() -> None:
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=SimpleNamespace()))
    assert make_graph_error_handler(runtime, "dispatch_tools") is None
    assert make_graph_error_handler(runtime, "model_call") is None
    assert make_graph_error_handler(runtime, "interrupt_for_human") is None


def test_graph_error_handler_policy_respects_feature_flag() -> None:
    class _Harness:
        config = SimpleNamespace(langgraph_error_handlers_enabled=False)

    runtime = SimpleNamespace(deps=SimpleNamespace(harness=_Harness()))
    assert graph_error_handler_policy(runtime, "prepare_prompt") is None

    runtime.deps.harness.config.langgraph_error_handlers_enabled = True
    assert graph_error_handler_policy(runtime, "prepare_prompt") is not None


def test_is_interpret_error_recoverable_requires_tool_call_repair() -> None:
    harness_enabled = SimpleNamespace(config=SimpleNamespace(tool_call_repair_enabled=True))
    harness_disabled = SimpleNamespace(config=SimpleNamespace(tool_call_repair_enabled=False))

    assert _is_interpret_error_recoverable(harness_enabled, ValueError("parse failure")) is True
    assert _is_interpret_error_recoverable(harness_disabled, ValueError("parse failure")) is False


@pytest.mark.asyncio
async def test_disabled_handlers_convert_prepare_prompt_failure_to_finalized_failure() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=False)
    runtime = LoopGraphRuntime.from_harness(harness)

    calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(len(calls) + 1)
        raise ValueError("prepare prompt failure")

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_interpret(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret

    result = await _run_loop_graph(runtime)

    assert result["status"] == "failed"
    assert result["error"]["type"] == "runtime_graph_error"
    assert calls == [1]
    assert not any(
        event == "graph_error_handler"
        for event, _data in harness.runlog_events
    )
    assert any(
        event == "runtime_graph_error"
        for event, _data in harness.runlog_events
    )


@pytest.mark.asyncio
async def test_enabled_handlers_recover_prepare_prompt_failure_and_continue() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    runtime = LoopGraphRuntime.from_harness(harness)

    prepare_calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        prepare_calls.append(len(prepare_calls) + 1)
        graph_state = load_runtime_state(runtime, payload)
        if len(prepare_calls) == 1:
            raise ValueError("transient prepare failure")
        # Second attempt: set a final result so the graph can end cleanly.
        graph_state.loop_state.scratchpad["_completed_after_recovery"] = True
        graph_state.final_result = {
            "status": "completed",
            "assistant": "recovered and completed",
        }
        return serialize_runtime_state(graph_state)

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        # Should not be reached because prepare_prompt sets the final result.
        return payload

    async def patched_interpret(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret

    result = await _run_loop_graph(runtime)

    assert result["status"] == "completed"
    assert result["assistant"] == "recovered and completed"
    assert len(prepare_calls) == 2
    assert harness.state.scratchpad.get("_graph_error_recovery_hint", "").startswith(
        "prepare_prompt failed:"
    )
    assert harness.state.scratchpad.get("_completed_after_recovery") is True

    handler_events = [
        (event, data)
        for event, data in harness.runlog_events
        if event == "graph_error_handler"
    ]
    assert len(handler_events) == 1
    _event, data = handler_events[0]
    assert data["node"] == "prepare_prompt"
    assert data["error_type"] == "ValueError"
    assert data["recovery_action"] == "recover"
    assert "error_message" in data


@pytest.mark.asyncio
async def test_enabled_handlers_bound_permanent_prepare_failure() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    runtime = LoopGraphRuntime.from_harness(harness)
    calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(len(calls) + 1)
        raise ValueError("permanent prepare failure")

    runtime._prepare_prompt_node = patched_prepare_prompt

    result = await _run_loop_graph(runtime)

    assert result["status"] == "failed"
    assert result["error"]["type"] == "graph_node_error"
    assert len(calls) == 2
    handler_actions = [
        data["recovery_action"]
        for event, data in harness.runlog_events
        if event == "graph_error_handler"
    ]
    assert handler_actions == ["recover", "terminal"]


@pytest.mark.asyncio
async def test_enabled_handlers_recover_interpret_failure_when_repair_supported() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    harness.config.tool_call_repair_enabled = True
    runtime = LoopGraphRuntime.from_harness(harness)

    interpret_calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_interpret_model_output(payload: dict[str, Any]) -> dict[str, Any]:
        interpret_calls.append(len(interpret_calls) + 1)
        graph_state = load_runtime_state(runtime, payload)
        if len(interpret_calls) == 1:
            raise ValueError("model output parse failure")
        graph_state.final_result = {
            "status": "completed",
            "assistant": "recovered after interpret failure",
        }
        return serialize_runtime_state(graph_state)

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret_model_output

    result = await _run_loop_graph(runtime)

    assert result["status"] == "completed"
    assert len(interpret_calls) == 2
    assert harness.state.scratchpad.get("_graph_error_recovery_hint", "").startswith(
        "interpret_model_output failed:"
    )
    handler_events = [
        (event, data)
        for event, data in harness.runlog_events
        if event == "graph_error_handler"
    ]
    assert len(handler_events) == 1
    assert handler_events[0][1]["recovery_action"] == "recover"


@pytest.mark.asyncio
async def test_enabled_handlers_make_interpret_failure_terminal_when_repair_disabled() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    harness.config.tool_call_repair_enabled = False
    runtime = LoopGraphRuntime.from_harness(harness)

    interpret_calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_interpret_model_output(payload: dict[str, Any]) -> dict[str, Any]:
        interpret_calls.append(len(interpret_calls) + 1)
        raise ValueError("model output parse failure")

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret_model_output

    result = await _run_loop_graph(runtime)

    assert result["status"] == "failed"
    assert result["error"]["type"] == "graph_node_error"
    assert len(interpret_calls) == 1
    assert harness.state.scratchpad.get("_graph_error_recovery_hint") is None

    handler_events = [
        (event, data)
        for event, data in harness.runlog_events
        if event == "graph_error_handler"
    ]
    assert len(handler_events) == 1
    assert handler_events[0][1]["recovery_action"] == "terminal"


@pytest.mark.asyncio
async def test_dispatch_tools_failure_not_recovered_by_node_handler() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    runtime = LoopGraphRuntime.from_harness(harness)

    dispatch_calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        graph_state = load_runtime_state(runtime, payload)
        graph_state.pending_tool_calls = [
            PendingToolCall(
                tool_call_id="tc-1",
                tool_name="ask_human",
                args={"question": "test?"},
                source="internal",
            )
        ]
        return serialize_runtime_state(graph_state)

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_interpret_model_output(payload: dict[str, Any]) -> dict[str, Any]:
        graph_state = load_runtime_state(runtime, payload)
        graph_state.pending_tool_calls = [
            PendingToolCall(
                tool_call_id="tc-1",
                tool_name="ask_human",
                args={"question": "test?"},
                source="internal",
            )
        ]
        return serialize_runtime_state(graph_state)

    async def patched_dispatch_tools(payload: dict[str, Any]) -> dict[str, Any]:
        dispatch_calls.append(len(dispatch_calls) + 1)
        raise ValueError("dispatch failure")

    async def patched_persist_tool_results(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_apply_tool_outcomes(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret_model_output
    runtime._dispatch_tools_node = patched_dispatch_tools
    runtime._persist_tool_results_node = patched_persist_tool_results
    runtime._apply_tool_outcomes_node = patched_apply_tool_outcomes

    result = await _run_loop_graph(runtime)

    assert result["status"] == "failed"
    assert result["error"]["type"] == "runtime_graph_error"
    assert len(dispatch_calls) == 1
    assert not any(
        event == "graph_error_handler" and data.get("node") == "dispatch_tools"
        for event, data in harness.runlog_events
    )
    assert any(
        event == "runtime_graph_error"
        for event, _data in harness.runlog_events
    )


@pytest.mark.asyncio
async def test_error_handler_runlog_does_not_include_state_or_secrets() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    harness.config.api_key = "super-secret-key"
    runtime = LoopGraphRuntime.from_harness(harness)

    calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise ValueError("prepare failure")
        graph_state = load_runtime_state(runtime, payload)
        graph_state.final_result = {
            "status": "completed",
            "assistant": "done",
        }
        return serialize_runtime_state(graph_state)

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_interpret(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret

    await _run_loop_graph(runtime)

    assert len(calls) == 2
    handler_events = [
        data
        for event, data in harness.runlog_events
        if event == "graph_error_handler"
    ]
    assert len(handler_events) == 1
    data = handler_events[0]
    assert "node" in data
    assert "error_type" in data
    assert "recovery_action" in data
    assert "error_message" in data
    assert "loop_state" not in data
    assert "api_key" not in data
    assert "super-secret-key" not in str(data)


@pytest.mark.asyncio
async def test_error_handler_runlog_redacts_credential_shaped_exception_message() -> None:
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    runtime = LoopGraphRuntime.from_harness(harness)
    secret = "sk-testsecret123"
    calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise ValueError(f"provider request failed: api_key={secret}")
        graph_state = load_runtime_state(runtime, payload)
        graph_state.final_result = {
            "status": "completed",
            "assistant": "recovered after secret-bearing error",
        }
        return serialize_runtime_state(graph_state)

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_interpret(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret

    result = await _run_loop_graph(runtime)

    assert result["status"] == "completed"
    result_text = str(result)
    assert secret not in result_text
    assert "api_key=sk-testsecret123" not in result_text

    assert len(calls) == 2

    handler_events = [
        data
        for event, data in harness.runlog_events
        if event == "graph_error_handler"
    ]
    assert len(handler_events) == 1
    data = handler_events[0]
    assert "error_message" in data
    error_message = data["error_message"]
    assert secret not in error_message
    assert "api_key=sk-testsecret123" not in error_message
    assert "REDACTED" in error_message

    runlog_text = str(harness.runlog_events)
    assert secret not in runlog_text
    assert "api_key=sk-testsecret123" not in runlog_text


@pytest.mark.asyncio
async def test_error_handler_preserves_existing_fama_nudge() -> None:
    from smallctl.models.conversation import ConversationMessage

    harness = _make_harness(langgraph_error_handlers_enabled=True)
    runtime = LoopGraphRuntime.from_harness(harness)

    fama_message = ConversationMessage(
        role="system",
        content="FAMA test nudge",
        metadata={"is_recovery_nudge": True, "recovery_kind": "fama_test"},
    )
    harness.state.append_message(fama_message)

    call_count: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        call_count.append(len(call_count) + 1)
        graph_state = load_runtime_state(runtime, payload)
        if len(call_count) == 1:
            raise ValueError("prepare failure")
        graph_state.final_result = {
            "status": "completed",
            "assistant": "recovered",
        }
        return serialize_runtime_state(graph_state)

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    async def patched_interpret(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret

    result = await _run_loop_graph(runtime)

    assert result["status"] == "completed"
    assert any(
        msg.metadata.get("recovery_kind") == "fama_test"
        for msg in harness.state.recent_messages
    )
    assert harness.state.scratchpad.get("_graph_error_recovery_hint")


@pytest.mark.asyncio
async def test_error_handler_passes_langgraph_node_error_shape() -> None:
    """Verify the handler accepts the NodeError shape LangGraph provides."""
    harness = _make_harness(langgraph_error_handlers_enabled=True)
    runtime = LoopGraphRuntime.from_harness(harness)
    handler = make_graph_error_handler(runtime, "prepare_prompt")
    assert handler is not None

    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id=harness.state.thread_id,
        run_mode="loop",
    )
    payload = serialize_runtime_state(graph_state)
    error = NodeError(node="prepare_prompt", error=ValueError("unit test"))

    command = await handler(payload, error)
    assert command.goto == "prepare_prompt"
    assert "_graph_error_recovery_hint" in command.update["loop_state"]["scratchpad"]


@pytest.mark.asyncio
async def test_error_handler_terminal_for_timeout_and_interrupt() -> None:
    from langgraph.errors import NodeTimeoutError
    from langgraph.errors import GraphInterrupt

    harness = _make_harness(langgraph_error_handlers_enabled=True)
    harness.config.tool_call_repair_enabled = True
    runtime = LoopGraphRuntime.from_harness(harness)
    handler = make_graph_error_handler(runtime, "interpret_model_output")
    assert handler is not None

    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id=harness.state.thread_id,
        run_mode="loop",
    )
    payload = serialize_runtime_state(graph_state)

    timeout_error = NodeTimeoutError(
        "interpret_model_output", 0.05, kind="run", run_timeout=0.05
    )
    command = await handler(payload, NodeError(node="interpret_model_output", error=timeout_error))
    assert command.goto == END
    assert command.update["final_result"]["status"] == "failed"

    payload = serialize_runtime_state(
        GraphRunState(
            loop_state=harness.state,
            thread_id=harness.state.thread_id,
            run_mode="loop",
        )
    )
    with pytest.raises(GraphInterrupt):
        await handler(payload, NodeError(node="interpret_model_output", error=GraphInterrupt("human")))


@pytest.mark.asyncio
async def test_graph_interrupt_preserved_through_enabled_handler() -> None:
    """GraphInterrupt in an error-handler node must still yield needs_human and resume."""
    from langgraph.types import interrupt

    harness = _make_harness(langgraph_error_handlers_enabled=True)
    runtime = LoopGraphRuntime.from_harness(harness)

    prepare_calls: list[int] = []

    async def patched_prepare_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        prepare_calls.append(len(prepare_calls) + 1)
        if len(prepare_calls) == 1:
            interrupt({"question": "continue?"})
        return payload

    async def patched_model_call(payload: dict[str, Any]) -> dict[str, Any]:
        graph_state = load_runtime_state(runtime, payload)
        graph_state.final_result = {
            "status": "completed",
            "assistant": "resumed after interrupt",
        }
        return serialize_runtime_state(graph_state)

    async def patched_interpret(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    runtime._prepare_prompt_node = patched_prepare_prompt
    runtime._model_call_node = patched_model_call
    runtime._interpret_model_output_node = patched_interpret

    result = await _run_loop_graph(runtime)

    assert result["status"] == "needs_human"
    assert result["interrupt"]["question"] == "continue?"
    harness.state.pending_interrupt = result["interrupt"]
    assert harness.has_pending_interrupt()

    resume_result = await runtime.resume("yes")

    assert resume_result["status"] == "completed"
    assert resume_result["assistant"] == "resumed after interrupt"
    assert len(prepare_calls) == 2
