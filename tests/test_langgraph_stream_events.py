"""Phase 4: LangGraph task/checkpoint stream event exposure."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.graph.runtime_payloads import (
    _extract_interrupt_from_updates_payload,
    _forward_langgraph_event,
    _normalize_langgraph_stream_event,
    _sanitize_error_summary,
    execute_streaming_graph,
    load_runtime_state,
    serialize_runtime_state,
)
from smallctl.models.events import UIEvent, UIEventType


class _FakeHarness:
    def __init__(self, *, langgraph_stream_events_enabled: bool = False) -> None:
        self.state = SimpleNamespace(thread_id="thread-123")
        self.conversation_id = "conversation-123"
        self.config = SimpleNamespace(
            langgraph_stream_events_enabled=langgraph_stream_events_enabled,
            graph_idle_watchdog_sec=0,
        )
        self.finalized: list[dict[str, object]] = []
        self._runlog = lambda *args, **kwargs: None

    def _finalize(self, result: dict[str, object]) -> dict[str, object]:
        self.finalized.append(dict(result))
        return result

    def _failure(
        self, message: str, *, error_type: str = "runtime", details: dict[str, object] | None = None
    ) -> dict[str, object]:
        return {
            "status": "failed",
            "reason": message,
            "error": {"type": error_type, "details": details or {}},
        }


class _CompiledGraphStreamDisabled:
    def __init__(self) -> None:
        self.astream_calls: list[tuple[Any, ...]] = []

    async def astream(self, payload: Any, config: Any) -> Any:  # type: ignore[no-untyped-def]
        self.astream_calls.append((payload, config))
        yield {}

    def get_state(self, config: Any) -> Any:  # type: ignore[no-untyped-def]
        del config
        return SimpleNamespace(values={})


class _CompiledGraphStreamEnabled:
    def __init__(self, *, events: list[tuple[str, Any]] | None = None) -> None:
        self.events = events or []
        self.astream_calls: list[tuple[Any, ...]] = []

    async def astream(
        self, payload: Any, config: Any, *, stream_mode: Any = None
    ) -> Any:  # type: ignore[no-untyped-def]
        self.astream_calls.append((payload, config, stream_mode))
        for event in self.events:
            yield event

    def get_state(self, config: Any) -> Any:  # type: ignore[no-untyped-def]
        del config
        return SimpleNamespace(values={})


def test_disabled_calls_astream_with_exact_signature() -> None:
    harness = _FakeHarness(langgraph_stream_events_enabled=False)
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness, event_handler=None))
    compiled = _CompiledGraphStreamDisabled()

    result = asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=lambda: compiled,
            empty_result_message="unused",
            recursion_limit=8,
        )
    )

    assert result["status"] == "failed"
    assert len(compiled.astream_calls) == 1
    payload, config = compiled.astream_calls[0]
    assert payload == {"input_task": "demo"}
    assert isinstance(config, dict)


def test_enabled_calls_astream_with_stream_mode() -> None:
    harness = _FakeHarness(langgraph_stream_events_enabled=True)
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness, event_handler=None))
    compiled = _CompiledGraphStreamEnabled()

    asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=lambda: compiled,
            empty_result_message="unused",
            recursion_limit=8,
        )
    )

    assert len(compiled.astream_calls) == 1
    payload, config, stream_mode = compiled.astream_calls[0]
    assert payload == {"input_task": "demo"}
    assert isinstance(config, dict)
    assert stream_mode == ["updates", "tasks", "checkpoints"]


def test_normalize_task_started_event() -> None:
    event = _normalize_langgraph_stream_event(
        "tasks",
        {
            "id": "task-1",
            "name": "prepare_prompt",
            "input": {"messages": "secret prompt"},
            "triggers": ["start"],
        },
    )
    assert event is not None
    assert event.event_type == UIEventType.SYSTEM
    assert event.data["category"] == "task_started"
    assert event.data["node"] == "prepare_prompt"
    assert event.data["task_id"] == "task-1"
    assert event.data["status"] == "success"
    assert "input" not in event.data


def test_normalize_task_finished_with_error_sanitizes_and_omits_result() -> None:
    event = _normalize_langgraph_stream_event(
        "tasks",
        {
            "id": "task-2",
            "name": "interpret_model_output",
            "error": ConnectionError("backend failed"),
            "result": {"tool_calls": [{"name": "shell_exec", "args": {"command": "rm -rf /"}}]},
        },
    )
    assert event is not None
    assert event.data["category"] == "task_finished"
    assert event.data["node"] == "interpret_model_output"
    assert event.data["task_id"] == "task-2"
    assert event.data["status"] == "error"
    assert "error_summary" in event.data
    assert "ConnectionError" in event.data["error_summary"]
    assert "result" not in event.data
    assert "tool_calls" not in event.data


def test_normalize_checkpoint_event_has_no_values_or_secrets() -> None:
    event = _normalize_langgraph_stream_event(
        "checkpoints",
        {
            "config": {
                "configurable": {
                    "thread_id": "thread-123",
                    "checkpoint_id": "chk-1",
                    "checkpoint_ns": "ns-a",
                }
            },
            "parent_config": {"configurable": {"checkpoint_id": "chk-0"}},
            "values": {
                "loop_state": {"api_key": "super-secret", "messages": ["secret prompt"]},
                "pending_tool_calls": [{"args": {"command": "secret command"}}],
            },
            "metadata": {"source": "loop"},
            "next": ["dispatch_tools"],
            "tasks": [
                {"id": "task-1", "name": "prepare_prompt"},
            ],
        },
    )
    assert event is not None
    assert event.data["category"] == "checkpoint"
    assert event.data["node"] == "prepare_prompt"
    assert event.data["task_id"] == "chk-1"
    assert event.data["status"] == "checkpoint"
    assert "values" not in event.data
    assert "parent_config" not in event.data
    assert "config" not in event.data
    assert "super-secret" not in str(event.to_dict())
    assert "secret command" not in str(event.to_dict())


def test_normalize_interrupt_event_from_tasks() -> None:
    event = _normalize_langgraph_stream_event(
        "tasks",
        {
            "id": "task-3",
            "name": "interrupt_for_human",
            "result": {"interrupt": [{"value": {"question": "continue?"}}]},
            "interrupts": [{"value": {"question": "continue?"}}],
        },
    )
    assert event is not None
    assert event.data["category"] == "interrupt"
    assert event.data["node"] == "interrupt_for_human"
    assert event.data["status"] == "needs_human"


def test_extract_interrupt_from_nested_updates_payload() -> None:
    assert _extract_interrupt_from_updates_payload({"__interrupt__": ["x"]}) == ["x"]
    assert _extract_interrupt_from_updates_payload(
        {"prepare_prompt": {"__interrupt__": ["y"]}}
    ) == ["y"]
    assert _extract_interrupt_from_updates_payload({"prepare_prompt": {"messages": []}}) is None


def test_sanitize_error_summary_for_exception_and_dict() -> None:
    assert "ConnectionError" in _sanitize_error_summary(ConnectionError("boom"))
    assert _sanitize_error_summary({"type": "ValidationError", "message": "bad"}) == "ValidationError: bad"
    assert _sanitize_error_summary("plain error") == "plain error"


@pytest.mark.asyncio
async def test_interrupt_still_produces_needs_human_when_streaming_enabled() -> None:
    harness = _FakeHarness(langgraph_stream_events_enabled=True)
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness, event_handler=None))
    compiled = _CompiledGraphStreamEnabled(
        events=[
            ("updates", {"__interrupt__": {"question": "continue?"}}),
        ]
    )

    result = await execute_streaming_graph(
        runtime,
        {"input_task": "demo"},
        build_graph=lambda: compiled,
        empty_result_message="unused",
        recursion_limit=8,
    )

    assert result["status"] == "needs_human"
    assert result["message"]["status"] == "human_input_required"


@pytest.mark.asyncio
async def test_interrupt_from_nested_update_payload_produces_needs_human() -> None:
    harness = _FakeHarness(langgraph_stream_events_enabled=True)
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness, event_handler=None))
    compiled = _CompiledGraphStreamEnabled(
        events=[
            ("updates", {"interrupt_for_human": {"__interrupt__": [{"value": {"question": "continue?"}}]}}),
        ]
    )

    result = await execute_streaming_graph(
        runtime,
        {"input_task": "demo"},
        build_graph=lambda: compiled,
        empty_result_message="unused",
        recursion_limit=8,
    )

    assert result["status"] == "needs_human"


@pytest.mark.asyncio
async def test_forward_langgraph_event_calls_async_and_sync_handlers() -> None:
    async_events: list[UIEvent] = []
    sync_events: list[UIEvent] = []

    async def async_handler(event: UIEvent) -> None:
        async_events.append(event)

    def sync_handler(event: UIEvent) -> None:
        sync_events.append(event)

    event = UIEvent(event_type=UIEventType.SYSTEM, content="test")
    await _forward_langgraph_event(
        SimpleNamespace(deps=SimpleNamespace(event_handler=async_handler)), event
    )
    await _forward_langgraph_event(
        SimpleNamespace(deps=SimpleNamespace(event_handler=sync_handler)), event
    )

    assert async_events == [event]
    assert sync_events == [event]


@pytest.mark.asyncio
async def test_forward_langgraph_event_emits_metadata_only_runlog_record() -> None:
    runlog_events: list[tuple[str, dict[str, Any]]] = []
    harness = SimpleNamespace(
        _runlog=lambda event, _message, **data: runlog_events.append((event, data))
    )
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        content="Graph task started: prepare_prompt",
        data={"graph_event": True, "category": "task_started", "node": "prepare_prompt"},
    )

    await _forward_langgraph_event(
        SimpleNamespace(deps=SimpleNamespace(harness=harness, event_handler=None)), event
    )

    assert runlog_events == [
        ("langgraph_stream_event", {"graph_event": True, "category": "task_started", "node": "prepare_prompt"})
    ]


def test_sanitize_error_summary_redacts_credential_shaped_exception_message() -> None:
    secret = "sk-testsecret123"
    summary = _sanitize_error_summary(ValueError(f"provider error: api_key={secret}"))
    assert secret not in summary
    assert "api_key=sk-testsecret123" not in summary
    assert "REDACTED" in summary


def test_normalize_task_error_redacts_credential_shaped_message() -> None:
    secret = "sk-testsecret123"
    event = _normalize_langgraph_stream_event(
        "tasks",
        {
            "id": "task-2",
            "name": "model_call",
            "error": ValueError(f"provider rejected request: api_key={secret}"),
        },
    )
    assert event is not None
    assert event.data["category"] == "task_finished"
    assert event.data["status"] == "error"
    assert "error_summary" in event.data
    error_summary = event.data["error_summary"]
    assert secret not in error_summary
    assert "api_key=sk-testsecret123" not in error_summary
    assert "REDACTED" in error_summary


def test_ui_event_handler_receives_normalized_event_without_leaks() -> None:
    harness = _FakeHarness(langgraph_stream_events_enabled=True)
    received: list[UIEvent] = []

    def handler(event: UIEvent) -> None:
        received.append(event)

    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness, event_handler=handler))
    compiled = _CompiledGraphStreamEnabled(
        events=[
            ("tasks", {"id": "task-1", "name": "prepare_prompt", "input": {"messages": "api_key=secret"}}),
            ("checkpoints", {
                "config": {"configurable": {"checkpoint_id": "chk-1", "checkpoint_ns": "ns"}},
                "values": {"api_key": "secret"},
                "metadata": {"source": "loop"},
                "next": ["model_call"],
                "tasks": [],
            }),
        ]
    )

    asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=lambda: compiled,
            empty_result_message="unused",
            recursion_limit=8,
        )
    )

    assert len(received) == 2
    for event in received:
        assert event.event_type == UIEventType.SYSTEM
        data = event.data
        assert "values" not in data
        assert "input" not in data
        assert "result" not in data
        assert "api_key" not in data
    assert received[0].data["category"] == "task_started"
    assert received[1].data["category"] == "checkpoint"
    assert "secret" not in str([event.to_dict() for event in received])


@pytest.mark.asyncio
async def test_indexer_runtime_emits_stream_events_when_enabled() -> None:
    """IndexerGraphRuntime must route through the shared opt-in streaming path."""
    from smallctl.harness import Harness
    from smallctl.graph.runtime_specialized import IndexerGraphRuntime

    harness = Harness(
        endpoint="http://example.test/v1",
        model="test-model",
        provider_profile="generic",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        langgraph_stream_events_enabled=True,
    )
    harness.config.graph_idle_watchdog_sec = None
    events: list[UIEvent] = []

    def handler(event: UIEvent) -> None:
        events.append(event)

    runtime = IndexerGraphRuntime.from_harness(harness, event_handler=handler)

    async def patched_initialize_run(payload: dict[str, Any]) -> dict[str, Any]:
        graph_state = load_runtime_state(runtime, payload)
        graph_state.final_result = {
            "status": "completed",
            "assistant": "indexer done",
        }
        return serialize_runtime_state(graph_state)

    runtime._initialize_run_node = patched_initialize_run

    result = await runtime.run("indexer event test")

    assert result["status"] == "completed"
    assert result["assistant"] == "indexer done"
    assert any(event.data.get("graph_event") for event in events)
    for event in events:
        assert "values" not in event.data
        assert "input" not in event.data
        assert "result" not in event.data
