from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.graph.runtime_base import GraphNodeTimeoutError, RuntimeGraphBuilder
from smallctl.graph.runtime_payloads import execute_streaming_graph


class _Harness:
    def __init__(self) -> None:
        self.state = SimpleNamespace(thread_id="thread-123")
        self.conversation_id = "conversation-123"
        self._cancel_requested = True
        self._active_dispatch_task = object()
        self._pending_task_shutdown_reason = "cancel_requested"
        self.finalized: list[dict[str, object]] = []

    def _finalize(self, result: dict[str, object]) -> dict[str, object]:
        self.finalized.append(dict(result))
        self._cancel_requested = False
        self._active_dispatch_task = None
        return result


class _CancelledCompiledGraph:
    def __init__(self) -> None:
        self.get_state_calls = 0

    async def astream(self, payload, config):  # type: ignore[no-untyped-def]
        del payload, config
        raise asyncio.CancelledError
        yield {}

    def get_state(self, config):  # type: ignore[no-untyped-def]
        del config
        self.get_state_calls += 1
        raise AssertionError("get_state() should not run after cancellation")


def test_execute_streaming_graph_finalizes_cancelled_runs_before_reraising() -> None:
    harness = _Harness()
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))
    compiled = _CancelledCompiledGraph()

    async def _run() -> None:
        try:
            await execute_streaming_graph(
                runtime,
                {"input_task": "demo"},
                build_graph=lambda: compiled,
                empty_result_message="unused",
                recursion_limit=8,
            )
        except asyncio.CancelledError:
            return
        raise AssertionError("execute_streaming_graph should re-raise CancelledError")

    asyncio.run(_run())

    assert harness.finalized == [
        {
            "status": "cancelled",
            "reason": "cancel_requested",
        }
    ]
    assert harness._cancel_requested is False
    assert harness._active_dispatch_task is None
    assert compiled.get_state_calls == 0


def test_runtime_graph_node_wrapper_logs_entry_exit() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(config=SimpleNamespace(graph_node_timeout_sec=1))
    harness._runlog = lambda event, message, **data: events.append((event, data))
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))

    async def node(payload):  # type: ignore[no-untyped-def]
        return {"ok": payload["ok"]}

    wrapped = RuntimeGraphBuilder._wrap_node(runtime, "interpret_model_output", node)
    result = asyncio.run(wrapped({"ok": True}))

    assert result == {"ok": True}
    assert [event for event, _ in events] == [
        "interpret_model_output_start",
        "interpret_model_output_end",
    ]
    assert events[0][1]["node"] == "interpret_model_output"


def test_runtime_graph_node_wrapper_does_not_enforce_timeout_locally() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(config=SimpleNamespace(graph_node_timeout_sec=0.01))
    harness._runlog = lambda event, message, **data: events.append((event, data))
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))

    async def node(payload):  # type: ignore[no-untyped-def]
        return {"ok": payload["ok"]}

    wrapped = RuntimeGraphBuilder._wrap_node(runtime, "interpret_model_output", node)
    result = asyncio.run(wrapped({"ok": True}))

    assert result == {"ok": True}
    assert [event for event, _ in events] == [
        "interpret_model_output_start",
        "interpret_model_output_end",
    ]


def test_execute_streaming_graph_finalizes_node_timeout() -> None:
    class _TimeoutCompiledGraph:
        async def astream(self, payload, config):  # type: ignore[no-untyped-def]
            del payload, config
            raise GraphNodeTimeoutError("dispatch_tools", 0.01)
            yield {}

        def get_state(self, config):  # type: ignore[no-untyped-def]
            del config
            raise AssertionError("get_state() should not run after node timeout")

    harness = _Harness()
    harness.config = SimpleNamespace(graph_idle_watchdog_sec=0)
    harness._failure = lambda message, error_type, details=None: {
        "status": "failed",
        "reason": message,
        "error": {"type": error_type, "details": details or {}},
    }
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
    assert result["error"]["details"] == {"node": "dispatch_tools", "timeout_sec": 0.01}


def test_execute_streaming_graph_logs_idle_watchdog_when_graph_is_silent() -> None:
    class _SilentCompiledGraph:
        async def astream(self, payload, config):  # type: ignore[no-untyped-def]
            del payload, config
            await asyncio.sleep(0.03)
            yield {"__interrupt__": {"question": "continue?"}}

        def get_state(self, config):  # type: ignore[no-untyped-def]
            del config
            return SimpleNamespace(values={})

    harness = _Harness()
    harness.config = SimpleNamespace(graph_idle_watchdog_sec=0.01)
    events: list[str] = []
    harness._runlog = lambda event, message, **data: events.append(event)
    runtime = SimpleNamespace(deps=SimpleNamespace(harness=harness))

    result = asyncio.run(
        execute_streaming_graph(
            runtime,
            {"input_task": "demo"},
            build_graph=_SilentCompiledGraph,
            empty_result_message="unused",
            recursion_limit=8,
        )
    )

    assert result["status"] == "needs_human"
    assert "harness_idle_watchdog" in events
