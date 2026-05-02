from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
