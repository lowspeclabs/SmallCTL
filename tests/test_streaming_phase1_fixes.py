from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx

from smallctl.client import OpenAICompatClient
from smallctl.client import client_transport
from smallctl.client.streaming import SSEStreamer
from smallctl.client.transport_error_classification import (
    _llamacpp_context_overflow_chunk_error_details,
    _provider_400_chunk_error_details,
)
from smallctl.graph.model_stream_loop import (
    _parse_context_window_overflow,
    run_model_stream_loop,
)
from smallctl.graph.state import GraphRunState
from smallctl.state import LoopState


class _StreamResponse:
    def __init__(self, lines, raise_after_lines=None):
        self._lines = lines
        self._raise_after_lines = raise_after_lines

    def raise_for_status(self):
        pass

    def aiter_lines(self):
        async def _gen():
            for line in self._lines:
                yield line
            if self._raise_after_lines is not None:
                raise self._raise_after_lines

        return _gen()


class _StreamContext:
    def __init__(self, lines, raise_on_enter=None, raise_after_lines=None):
        self._response = _StreamResponse(lines, raise_after_lines=raise_after_lines)
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self):
        if self._raise_on_enter is not None:
            raise self._raise_on_enter
        return self._response

    async def __aexit__(self, *args):
        return False


class _StreamClient:
    def __init__(self, contexts):
        self._contexts = contexts
        self.call_count = 0

    def stream(self, method, url, **kwargs):
        idx = self.call_count
        self.call_count += 1
        if idx >= len(self._contexts):
            raise AssertionError("stream must not be retried")
        return self._contexts[idx]


def _collect(streamer, client):
    async def _run():
        return [event async for event in streamer.stream_sse(client, "http://test", {}, {})]

    return asyncio.run(_run())


def test_remote_protocol_error_after_chunks_salvages_without_retry() -> None:
    client = _StreamClient(
        [
            _StreamContext(
                ['data: {"choices":[{"delta":{"content":"Hello "}}]}'],
                raise_after_lines=httpx.RemoteProtocolError("Server disconnected"),
            ),
            _StreamContext(['data: {"choices":[{"delta":{"content":"Hello cruel world"}}]}', "data: [DONE]"]),
        ]
    )
    streamer = SSEStreamer(provider_profile="generic")

    events = _collect(streamer, client)

    assert [event["type"] for event in events] == ["chunk", "stream_ended_without_done"]
    assert events[1]["details"]["reason"] == "remote_protocol_error_after_chunks"
    assert client.call_count == 1
    streamed_text = "".join(
        str(event["data"]["choices"][0]["delta"].get("content") or "")
        for event in events
        if event["type"] == "chunk"
    )
    assert streamed_text == "Hello "


def test_closed_client_runtime_error_pre_chunk_refetches_and_retries_once(monkeypatch) -> None:
    reset_calls = []
    get_calls = []

    async def _fake_reset(client):
        reset_calls.append(client)

    fresh_client = _StreamClient([_StreamContext(["data: [DONE]"])])

    def _fake_get(client):
        get_calls.append(client)
        return fresh_client

    monkeypatch.setattr("smallctl.client.streaming._reset_async_client", _fake_reset)
    monkeypatch.setattr("smallctl.client.streaming._get_async_client", _fake_get)

    api_client = SimpleNamespace(
        base_url="http://example.test/v1",
        api_key="test-key",
        _shared_clients={},
    )
    closed_client = _StreamClient(
        [
            _StreamContext(
                [],
                raise_on_enter=RuntimeError("Cannot send a request, as the client has been closed."),
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic", api_client=api_client)

    events = _collect(streamer, closed_client)

    assert [event["type"] for event in events] == ["done"]
    assert closed_client.call_count == 1
    assert fresh_client.call_count == 1
    assert reset_calls == [api_client]
    assert get_calls == [api_client]


def test_closed_client_runtime_error_after_chunks_salvages_without_retry(monkeypatch) -> None:
    reset_calls = []

    async def _fake_reset(client):
        reset_calls.append(client)

    monkeypatch.setattr("smallctl.client.streaming._reset_async_client", _fake_reset)

    api_client = SimpleNamespace(
        base_url="http://example.test/v1",
        api_key="test-key",
        _shared_clients={},
    )
    client = _StreamClient(
        [
            _StreamContext(
                ['data: {"choices":[{"delta":{"content":"Hello "}}]}'],
                raise_after_lines=RuntimeError("Cannot send a request, as the client has been closed."),
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic", api_client=api_client)

    events = _collect(streamer, client)

    assert [event["type"] for event in events] == ["chunk", "stream_ended_without_done"]
    assert events[1]["details"]["reason"] == "closed_client_after_chunks"
    assert client.call_count == 1
    assert reset_calls == []


def test_streaming_400_body_is_read_for_context_overflow_classification(monkeypatch) -> None:
    from httpx._content import AsyncIteratorByteStream

    overflow_body = (
        b"request (16385 tokens) exceeds the available context size (16384 tokens), "
        b"try increasing it"
    )

    class _StreamingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            async def _body():
                yield overflow_body

            return httpx.Response(
                400,
                stream=AsyncIteratorByteStream(_body()),
                request=request,
            )

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    client.STREAM_RETRY_ATTEMPTS = 1
    async_client = httpx.AsyncClient(transport=_StreamingTransport(), timeout=None)

    async def _fake_reset(_client):
        return None

    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: async_client)
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)

    async def _run():
        events = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "continue"}],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk_error"]
    details = events[0]["details"]
    assert details["reason"] == "context_overflow"
    assert details["request_tokens"] == 16385
    assert details["context_limit"] == 16384


def test_llamacpp_context_overflow_builder_round_trips_through_parser() -> None:
    details = _llamacpp_context_overflow_chunk_error_details(
        SimpleNamespace(provider_profile="llamacpp"),
        payload=None,
        body_summary={"request_tokens": 9214, "context_limit": 8192},
        status_code=400,
        attempt=1,
    )

    assert _parse_context_window_overflow("llamacpp context window exceeded", details) == (9214, 8192)


class _NonRecoverableChunkErrorClient:
    model = "qwen3.5:9b"
    provider_profile = "openrouter"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        yield {
            "type": "chunk_error",
            "error": "Provider content policy violation: the request was blocked",
            "details": {
                "type": "content_policy_violation",
                "reason": "provider_content_policy_block",
                "provider_profile": "openrouter",
                "status_code": 403,
                "recoverable": False,
            },
        }


class _LoopHarness:
    def __init__(self, state: LoopState) -> None:
        self.state = state
        self.reasoning_mode = "off"
        self.thinking_start_tag = "<think>"
        self.thinking_end_tag = "</think>"
        self.thinking_visibility = False
        self.runlog_events = []
        self._cancel_requested = False
        self.client = _NonRecoverableChunkErrorClient()

    def _runlog(self, *args, **kwargs) -> None:
        self.runlog_events.append((args, kwargs))

    async def _emit(self, *args, **kwargs) -> None:
        del args, kwargs

    def _failure(self, message: str, *, error_type: str = "stream", details=None):
        del details
        return {
            "status": "failed",
            "message": message,
            "error": {"message": message, "type": error_type},
        }


def test_non_recoverable_chunk_error_is_not_retried() -> None:
    state = LoopState(cwd="/tmp")
    harness = _LoopHarness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = SimpleNamespace(event_handler=None, harness=harness)

    result = asyncio.run(
        run_model_stream_loop(
            graph_state,
            deps,
            harness=harness,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            echo_to_stdout=False,
            start_tag="<think>",
            end_tag="</think>",
            start_time=time.perf_counter(),
        )
    )

    assert len(harness.client.calls) == 1
    assert result["stream_completed_cleanly"] is False
    assert result["last_chunk_error_details"]["recoverable"] is False


def test_provider_400_exhausted_details_marked_not_recoverable() -> None:
    client = SimpleNamespace(provider_profile="openrouter", model="Gemma 4 12b")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        text='{"error":{"message":"Gemma 4 12b is not a valid model ID"}}',
    )
    exc = httpx.HTTPStatusError("http 400", request=request, response=response)

    details = _provider_400_chunk_error_details(
        client,
        payload={"messages": [{"role": "user", "content": "hi"}]},
        exc=exc,
        attempt=3,
        recovery_stages_attempted=3,
    )

    assert details["recoverable"] is False
    assert details["error_kind"] == "provider_400_exhausted"


def test_stream_chat_generic_error_after_chunks_salvages_without_retry(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
    )
    attempts = {"count": 0}
    reset_calls = []

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            attempts["count"] += 1
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "partial"}}]}}
            raise ValueError("boom")

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    async def _fake_reset(_client):
        reset_calls.append(_client)

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)

    async def _run():
        events = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "stream_ended_without_done"]
    assert events[1]["details"]["reason"] == "unexpected_error_after_chunks"
    assert attempts["count"] == 1
    assert reset_calls == []
