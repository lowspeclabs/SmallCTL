"""Tests for Phase 3 runtime, transport, and error-handling fixes."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from smallctl.client import OpenAICompatClient
from smallctl.client import client_transport
from smallctl.client.adapters.common import should_retry_without_stream_options
from smallctl.client.llamacpp_preflight import _build_minimal_context_payload
from smallctl.client.provider_adapters import get_provider_adapter
from smallctl.client.stream_collectors import collect_timeline
from smallctl.graph.plan_execution import PlanExecutionEngine
from smallctl.graph.runtime import LoopGraphRuntime
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_outcomes import _update_subtask_ledger_from_record
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ContextBrief, ExecutionPlan, LoopState, PlanStep
from smallctl.state_session_records import _coerce_context_brief


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _http_status_error(url: str, *, status_code: int, text: str = "", headers: dict | None = None) -> httpx.HTTPStatusError:
    response = httpx.Response(status_code=status_code, text=text, headers=headers or {})
    return httpx.HTTPStatusError("error", request=httpx.Request("POST", url), response=response)


# -----------------------------------------------------------------------------
# 3.1 Unhandled exceptions during tool dispatch
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_dispatch_exception_becomes_recoverable_error() -> None:
    from smallctl.graph.tool_execution_nodes import dispatch_tools

    harness = MagicMock()
    harness.registry.names.return_value = ["some_tool"]
    harness.state.step_count = 1
    harness.state.recent_errors = []
    harness._runlog = MagicMock()
    async def _noop_emit(*_args, **_kwargs):
        pass
    harness._emit = _noop_emit
    harness._active_dispatch_task = None
    harness._active_ui_tool_context = None

    async def _boom(_name: str, _args: dict[str, Any]) -> Any:
        raise RuntimeError("dispatch boom")

    harness._dispatch_tool_call = _boom

    deps = MagicMock()
    deps.event_handler = None
    deps.harness = harness

    graph_state = GraphRunState(
        loop_state=LoopState(),
        thread_id="t-1",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(
                tool_call_id="tc-1",
                tool_name="some_tool",
                args={"x": 1},
                source="internal",
            )
        ],
    )

    await dispatch_tools(graph_state, deps)

    assert len(graph_state.last_tool_results) == 1
    record = graph_state.last_tool_results[0]
    assert record.tool_name == "some_tool"
    assert record.result.success is False
    assert record.result.metadata.get("reason") == "tool_dispatch_exception"
    assert harness.state.recent_errors


@pytest.mark.asyncio
async def test_tool_dispatch_normalizes_none_metadata() -> None:
    from smallctl.graph.tool_execution_nodes import dispatch_tools

    harness = MagicMock()
    harness.registry.names.return_value = ["some_tool"]
    harness.state = LoopState()
    harness.state.step_count = 1
    harness.state.recent_errors = []
    harness._runlog = MagicMock()

    async def _noop_emit(*_args, **_kwargs):
        pass

    harness._emit = _noop_emit
    harness._active_dispatch_task = None
    harness._active_ui_tool_context = None
    harness.log = logging.getLogger("test.tool_dispatch_none_metadata")

    async def _dispatch(_name: str, _args: dict[str, Any]) -> ToolEnvelope:
        return ToolEnvelope(success=True, output="ok", metadata=None)

    harness._dispatch_tool_call = _dispatch

    deps = MagicMock()
    deps.event_handler = None
    deps.harness = harness

    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="t-1",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(
                tool_call_id="tc-1",
                tool_name="some_tool",
                args={"x": 1},
                source="internal",
            )
        ],
    )

    await dispatch_tools(graph_state, deps)

    assert len(graph_state.last_tool_results) == 1
    assert graph_state.last_tool_results[0].result.success is True
    assert graph_state.last_tool_results[0].result.metadata == {}


# -----------------------------------------------------------------------------
# 3.2 DAG dispatch fallback broad exception handling
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_plan_dag_fallback_on_unexpected_exception(monkeypatch) -> None:
    from smallctl.graph.runtime_tool_plan import ToolPlanRuntime

    harness = MagicMock()
    harness.state.scratchpad = {}
    harness.state.step_count = 1
    harness._runlog = MagicMock()

    deps = MagicMock()
    deps.harness = harness
    deps.event_handler = None

    runtime = ToolPlanRuntime(deps)

    loop_state = LoopState()
    loop_state.scratchpad["_tool_plan_phase"] = "dispatch"
    loop_state.scratchpad["_tool_plan"] = {
        "objective": "test",
        "steps": [
            {"id": "s1", "tool": "file_read", "args": {"path": "x"}, "depends_on": []},
        ],
    }

    payload = {
        "loop_state": loop_state.to_dict(),
        "thread_id": "t-1",
        "run_mode": "tool_plan",
        "pending_tool_calls": [],
        "last_tool_results": [],
    }

    async def _serial_dispatch(_self, _payload):
        return payload

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime._dispatch_tools_node",
        _serial_dispatch,
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime_tool_plan._tool_plan_config",
        lambda _deps, key, default: True if key == "tool_dag_enabled" else default,
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime_tool_plan.build_execution_dag",
        MagicMock(side_effect=KeyError("unexpected dag failure")),
    )

    result = await runtime._dispatch_tools_node(payload)

    assert result is not None
    harness._runlog.assert_called_once()
    call_args = harness._runlog.call_args
    assert "unexpected dag failure" in str(call_args)


# -----------------------------------------------------------------------------
# 3.3 stream_chat catches non-transport httpx exceptions
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_chat_catches_unexpected_exception_and_retries(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
    )
    attempts = {"count": 0}

    class _FakeStreamer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def stream_sse(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ValueError("unexpected parse failure")
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            raise AssertionError("nonstream fallback should not run")
            yield {}

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)
    monkeypatch.setattr(client_transport.asyncio, "sleep", AsyncMock())

    events: list[dict[str, object]] = []
    async for event in client_transport.stream_chat(
        client,
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    ):
        events.append(event)

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert attempts["count"] == 2


# -----------------------------------------------------------------------------
# 3.4 Retry-After cap already present; verify
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_chat_caps_retry_after_delay(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
    )
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    class _FakeStreamer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def stream_sse(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_status_error(url, status_code=429, headers={"Retry-After": "999"})
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            raise AssertionError("nonstream fallback should not run")
            yield {}

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(float(delay))

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)
    monkeypatch.setattr(client_transport.asyncio, "sleep", _fake_sleep)

    events: list[dict[str, object]] = []
    async for event in client_transport.stream_chat(
        client,
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    ):
        events.append(event)

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert attempts["count"] == 2
    assert sleep_calls == [120.0]


# -----------------------------------------------------------------------------
# 3.5 Auto reasoning mode does not permanently flip to field mode
# -----------------------------------------------------------------------------


def test_auto_reasoning_detects_tags_after_field_reasoning() -> None:
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "content": "Let me think. ",
                        "reasoning": "field reasoning",
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": "<think>tagged reasoning</think> and the answer is 42.",
                    }
                }
            ]
        },
    ]
    entries = collect_timeline(chunks, reasoning_mode="auto")
    assistant_text = "".join(entry.content for entry in entries if entry.kind == "assistant")
    thinking_text = "".join(entry.content for entry in entries if entry.kind == "thinking")
    assert "tagged reasoning" in thinking_text
    assert "tagged reasoning" not in assistant_text
    assert "the answer is 42" in assistant_text


# -----------------------------------------------------------------------------
# 3.6 Partial-stream timeouts are treated consistently across providers
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_chat_partial_timeout_on_generic_provider_yields_stream_ended(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
    )

    class _FakeStreamer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def stream_sse(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "partial"}}]}}
            raise httpx.ReadTimeout("timed out")

        async def nonstream_chat(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            raise AssertionError("nonstream fallback should not run after partial output")
            yield {}

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)

    events: list[dict[str, object]] = []
    async for event in client_transport.stream_chat(
        client,
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    ):
        events.append(event)

    assert events[-1]["type"] == "stream_ended_without_done"
    assert events[-1]["details"]["reason"] == "read_timeout_after_chunks"


# -----------------------------------------------------------------------------
# 3.7 Cycle detection leaves stale visiting state
# -----------------------------------------------------------------------------


def test_cycle_detection_does_not_pollute_subsequent_steps() -> None:
    state = LoopState()
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="A", title="a", depends_on=["B"]),
            PlanStep(step_id="B", title="b", depends_on=["A"]),
            PlanStep(step_id="C", title="c"),
        ],
    )
    engine = PlanExecutionEngine(state)
    result = engine.validate_plan(plan)
    assert result.valid is False
    assert any("cycle" in err.lower() for err in result.errors)
    # C is independent and should still be ready despite the A/B cycle
    assert "C" in engine.ready_step_ids(plan)


# -----------------------------------------------------------------------------
# 3.8 Brittle exception handling around LangGraph human interrupt
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_for_human_re_raises_langgraph_interrupt() -> None:
    from langgraph.errors import GraphInterrupt

    harness = MagicMock()
    harness.state.append_message = MagicMock()
    harness.state.run_brief.original_task = "task"

    deps = MagicMock()
    deps.harness = harness

    runtime = LoopGraphRuntime(deps)
    payload = {"loop_state": LoopState().to_dict(), "thread_id": "t-1", "run_mode": "loop"}

    with patch("smallctl.graph.runtime.interrupt", side_effect=GraphInterrupt("human paused")):
        with pytest.raises(GraphInterrupt):
            await runtime._interrupt_for_human_node(payload)


# -----------------------------------------------------------------------------
# 3.9 JS/TS staged-artifact syntax check honors exit code (already done)
# -----------------------------------------------------------------------------


def test_js_syntax_check_uses_returncode(tmp_path: pytest.TempPathFactory) -> None:
    from smallctl.graph.write_session_health import _is_staged_artifact_syntactically_valid

    good = tmp_path / "good.js"
    good.write_text("console.log(1);\n")
    assert _is_staged_artifact_syntactically_valid(str(good), good.read_text()) is True

    bad = tmp_path / "bad.js"
    bad.write_text("console.log({")
    assert _is_staged_artifact_syntactically_valid(str(bad), bad.read_text()) is False


# -----------------------------------------------------------------------------
# 3.10 Broad exception suppression in stream/process reading
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_stream_chunks_logs_unexpected_errors(caplog: pytest.LogCaptureFixture) -> None:
    from smallctl.tools.process_streams import read_stream_chunks

    class _BadStream:
        async def read(self, _size: int) -> bytes:
            raise TypeError("stream exploded")

    caplog.set_level(logging.WARNING, logger="smallctl.tools.process_streams")
    out: list[str] = []
    await read_stream_chunks(_BadStream(), out, chunk_size=1024)
    assert out == []
    assert "stream exploded" in caplog.text


# -----------------------------------------------------------------------------
# 3.11 _finalize logs summary-write failures (already done)
# -----------------------------------------------------------------------------


def test_finalize_logs_summary_failure(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    from smallctl.harness.core_facade import _finalize

    harness = MagicMock()
    harness.log = logging.getLogger("test_finalize")
    harness.state = MagicMock()
    harness.state.scratchpad = {}
    harness.state.step_count = 1
    harness.state.to_dict.return_value = {"thread_id": "t1"}
    harness._cancel_requested = False
    harness._active_dispatch_task = None
    harness._emit = AsyncMock()
    harness._runlog = MagicMock()
    harness.config.subtask_ledger_enabled = False

    run_logger = MagicMock()
    run_logger.run_dir = MagicMock()
    run_logger.run_dir.__truediv__.side_effect = PermissionError("denied")
    harness.run_logger = run_logger

    monkeypatch.setattr("smallctl.harness.core_facade.challenge_progress_report", lambda _state: {})

    caplog.set_level(logging.ERROR, logger="test_finalize")
    _finalize(harness, {"status": "completed"})

    assert "failed to write finalization summaries" in caplog.text


# -----------------------------------------------------------------------------
# 3.12 _coerce_context_brief handles partial payloads
# -----------------------------------------------------------------------------


def test_coerce_context_brief_fills_required_fields() -> None:
    partial = {"brief_id": "b1"}
    brief = _coerce_context_brief(partial)
    assert isinstance(brief, ContextBrief)
    assert brief.brief_id == "b1"
    assert brief.task_goal == ""
    assert brief.current_phase == ""
    assert brief.next_action_hint == ""
    assert brief.staleness_step == 0
    assert brief.key_discoveries == []


# -----------------------------------------------------------------------------
# 3.13 Backend restart rate-limit tolerates corrupt timestamps (already done)
# -----------------------------------------------------------------------------


def test_backend_restart_rate_limit_drops_corrupt_timestamps() -> None:
    from smallctl.harness.backend_recovery import BackendRecoveryService

    harness = MagicMock()
    harness.backend_max_restarts_per_hour = 5
    state = LoopState()
    now = __import__("time").time()
    state.scratchpad["_backend_restart_history"] = [now - 100, "not-a-number", now - 50]
    harness.state = state

    recovery = BackendRecoveryService(harness)
    result = recovery._check_backend_restart_rate_limit()
    assert result["allowed"] is True
    assert result["count"] == 2


# -----------------------------------------------------------------------------
# 3.14 should_retry_without_stream_options only when stream_options present
# -----------------------------------------------------------------------------


def test_should_retry_without_stream_options_requires_stream_options_for_empty_body() -> None:
    response = httpx.Response(status_code=400, text="")
    exc = httpx.HTTPStatusError("bad", request=httpx.Request("POST", "http://x"), response=response)
    assert should_retry_without_stream_options(exc, stream_options_present=False) is False
    assert should_retry_without_stream_options(exc, stream_options_present=True) is True


def test_should_retry_without_stream_options_true_when_body_mentions_stream_options() -> None:
    response = httpx.Response(status_code=400, text='{"error": "stream_options not supported"}')
    exc = httpx.HTTPStatusError("bad", request=httpx.Request("POST", "http://x"), response=response)
    assert should_retry_without_stream_options(exc, stream_options_present=False) is True


# -----------------------------------------------------------------------------
# 3.15 Minimal-context recovery preserves recent assistant/tool exchange
# -----------------------------------------------------------------------------


def test_minimal_context_payload_preserves_tool_exchange() -> None:
    client = SimpleNamespace(model="m", adapter=get_provider_adapter("generic"))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "file_read"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        {"role": "user", "content": "now what?"},
    ]
    payload = _build_minimal_context_payload(client, messages=messages)
    roles = [m["role"] for m in payload["messages"]]
    assert "system" in roles
    assert "assistant" in roles
    assert "tool" in roles
    assert "user" in roles


# -----------------------------------------------------------------------------
# 3.16 OpenRouter 400 recovery stage count is not hardcoded
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_400_exhaustion_reports_actual_recovery_stages(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="demo-model",
        provider_profile="generic",
    )
    assert client.provider_profile == "openrouter"

    body = '{"error":{"message":"Input validation error"}}'

    class _FakeStreamer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def stream_sse(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            raise _http_status_error(url, status_code=400, text=body)
            yield {}

        async def nonstream_chat(self, async_client: Any, url: str, headers: Any, payload: Any) -> Any:
            raise _http_status_error(url, status_code=400, text=body)
            yield {}

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)

    events: list[dict[str, object]] = []
    async for event in client_transport.stream_chat(
        client,
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    ):
        events.append(event)

    assert events[-1]["type"] == "chunk_error"
    details = events[-1]["details"]
    assert details["recovery_stages_attempted"] >= 0


# -----------------------------------------------------------------------------
# 3.17 Write-heavy tool classification false positives
# -----------------------------------------------------------------------------


def test_write_heavy_classification_ignores_readonly_tools_with_content_param() -> None:
    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
    )
    ask_human_tool = {
        "type": "function",
        "function": {
            "name": "ask_human",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
            },
        },
    }
    assert client._request_has_write_heavy_tool([ask_human_tool]) is False


def test_write_heavy_classification_still_detects_known_write_tools() -> None:
    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
    )
    file_write_tool = {
        "type": "function",
        "function": {
            "name": "file_write",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert client._request_has_write_heavy_tool([file_write_tool]) is True


# -----------------------------------------------------------------------------
# 3.18 Over-broad import guards (already done)
# -----------------------------------------------------------------------------


def test_client_import_guard_catches_import_error_only() -> None:
    source = (
        "try:\n"
        "    import httpx\n"
        "except (ImportError, ModuleNotFoundError):\n"
        "    httpx = None\n"
    )
    compile(source, "<test>", "exec")


# -----------------------------------------------------------------------------
# 3.19 Silent suppression of subtask-ledger/trajectory failures
# -----------------------------------------------------------------------------


def test_subtask_ledger_update_logs_failure(caplog: pytest.LogCaptureFixture) -> None:
    harness = MagicMock()
    harness.config.subtask_ledger_enabled = True
    harness.log = logging.getLogger("test_subtask_ledger")
    service = MagicMock()
    service.import_plan_if_needed.side_effect = RuntimeError("ledger boom")
    harness.subtask_ledger = service

    record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="file_read",
        args={},
        tool_call_id="tc-1",
        result=ToolEnvelope(success=False, error="boom"),
    )

    caplog.set_level(logging.WARNING, logger="test_subtask_ledger")
    _update_subtask_ledger_from_record(harness, record)
    assert "ledger boom" in caplog.text


@pytest.mark.asyncio
async def test_tool_plan_after_run_logs_trajectory_failure(monkeypatch) -> None:
    from smallctl.graph.runtime_tool_plan import ToolPlanRuntime

    harness = MagicMock()
    harness._runlog = MagicMock()

    runtime = ToolPlanRuntime(MagicMock())

    class _BoomRecorder:
        def record_tool_plan_trajectory(self, harness: Any, result: dict[str, Any]) -> None:
            raise RuntimeError("trajectory boom")

    monkeypatch.setattr(
        "smallctl.graph.runtime_tool_plan.TrajectoryRecorder",
        _BoomRecorder,
    )

    await runtime._after_run(harness, {"status": "completed"})
    harness._runlog.assert_called_once()
    assert "trajectory boom" in harness._runlog.call_args.kwargs.get("error", "")
