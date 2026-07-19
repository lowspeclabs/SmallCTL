"""Regression tests for the 2026-07-17 client-plane gap fixes (C3, H17, D4/L21, D5/L22, L25)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest

from smallctl.client import OpenAICompatClient
from smallctl.client import client_transport
from smallctl.client import client_transport_client_lifecycle as client_lifecycle
from smallctl.client import client_transport_helpers
from smallctl.client import llamacpp_preflight
from smallctl.client import openrouter_preflight
from smallctl.client import transport_constants
from smallctl.client.streaming import SSEStreamer


def _http_status_error(url: str, *, status_code: int, text: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status_code, request=request, text=text)
    return httpx.HTTPStatusError(f"http {status_code}", request=request, response=response)


def _tool_call_chunk() -> dict[str, object]:
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "file_write", "arguments": '{"path": "x"}'},
                        }
                    ]
                }
            }
        ]
    }


def _text_chunk(text: str = "partial") -> dict[str, object]:
    return {"choices": [{"delta": {"content": text}}]}


class _RunLogger:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def log(self, channel: str, event: str, message: str, **data) -> None:
        self.entries.append(
            {
                "channel": channel,
                "event": event,
                "message": message,
                "data": data,
            }
        )


def _run_stream_chat(
    monkeypatch,
    *,
    stream_events: list[object],
    nonstream_events: list[object] | None = None,
    nonstream_exc: Exception | None = None,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
    )
    attempts = {"stream": 0, "nonstream": 0}

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            attempts["stream"] += 1
            for event in stream_events:
                if isinstance(event, Exception):
                    raise event
                yield event

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            attempts["nonstream"] += 1
            for event in nonstream_events or []:
                yield event
            if nonstream_exc is not None:
                raise nonstream_exc
            yield {"type": "done"}

    async def _fake_reset(_client: object) -> None:
        return None

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def _run() -> list[dict[str, object]]:
        return [
            event
            async for event in client_transport.stream_chat(
                client,
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
            )
        ]

    return attempts, asyncio.run(_run())


# ---------------------------------------------------------------------------
# C3: once any chunk has been yielded in the current attempt, a transport or
# unexpected failure surfaces as a non-retryable salvage event (never a
# retryable chunk_error, never a re-POST).
# ---------------------------------------------------------------------------


def test_c3_transport_error_after_tool_call_chunk_salvages_without_retry(monkeypatch) -> None:
    attempts, events = _run_stream_chat(
        monkeypatch,
        stream_events=[
            {"type": "chunk", "data": _tool_call_chunk()},
            httpx.ReadTimeout("timed out waiting for tool call continuation"),
        ],
    )

    assert attempts == {"stream": 1, "nonstream": 0}
    assert [event["type"] for event in events] == ["chunk", "stream_ended_without_done"]
    details = events[1]["details"]
    assert details["reason"] == "transport_error_after_chunks"
    assert details["tool_call_stream_active"] is True
    assert all(event["type"] != "chunk_error" for event in events)


def test_c3_generic_error_after_tool_call_chunk_salvages_without_retry(monkeypatch) -> None:
    attempts, events = _run_stream_chat(
        monkeypatch,
        stream_events=[
            {"type": "chunk", "data": _tool_call_chunk()},
            ValueError("boom"),
        ],
    )

    assert attempts == {"stream": 1, "nonstream": 0}
    assert [event["type"] for event in events] == ["chunk", "stream_ended_without_done"]
    details = events[1]["details"]
    assert details["reason"] == "unexpected_error_after_chunks"
    assert details["tool_call_stream_active"] is True


def test_c3_retryable_http_status_after_chunks_salvages_without_retry(monkeypatch) -> None:
    attempts, events = _run_stream_chat(
        monkeypatch,
        stream_events=[
            {"type": "chunk", "data": _tool_call_chunk()},
            _http_status_error("http://test", status_code=503, text="provider overloaded"),
        ],
    )

    assert attempts == {"stream": 1, "nonstream": 0}
    assert [event["type"] for event in events] == ["chunk", "stream_ended_without_done"]
    details = events[1]["details"]
    assert details["reason"] == "http_error_after_chunks"
    assert details["status_code"] == 503
    assert details["tool_call_stream_active"] is True


def test_c3_nonstream_fallback_yields_then_fails_emits_salvage(monkeypatch) -> None:
    attempts, events = _run_stream_chat(
        monkeypatch,
        stream_events=[httpx.ConnectError("connection reset")],
        nonstream_events=[{"type": "chunk", "data": _text_chunk()}],
        nonstream_exc=httpx.ReadTimeout("read timeout"),
    )

    assert attempts == {"stream": 1, "nonstream": 1}
    assert [event["type"] for event in events] == ["chunk", "stream_ended_without_done"]
    details = events[1]["details"]
    assert details["reason"] == "transport_error_after_chunks"
    assert details["message"] == "read timeout"


# ---------------------------------------------------------------------------
# H17: shared-client teardown defers closing a client with active stream
# leases until the leases drain; the in-flight stream completes first.
# ---------------------------------------------------------------------------


def _lease_streamer_class(chunk_seen: asyncio.Event, proceed: asyncio.Event):
    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            yield {"type": "chunk", "data": _text_chunk()}
            chunk_seen.set()
            await proceed.wait()
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    return _FakeStreamer


def test_h17_reset_defers_close_until_stream_lease_released(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
    )
    real_async_client = httpx.AsyncClient(timeout=None)
    key = (client.base_url, client.api_key)
    client._shared_clients = {key: real_async_client}
    chunk_seen = asyncio.Event()
    proceed = asyncio.Event()

    monkeypatch.setattr(client_transport, "SSEStreamer", _lease_streamer_class(chunk_seen, proceed))
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: real_async_client)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []

        async def _consume() -> None:
            async for event in client_transport.stream_chat(
                client,
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
            ):
                events.append(event)

        task = asyncio.create_task(_consume())
        await asyncio.wait_for(chunk_seen.wait(), timeout=5)
        await client_lifecycle._reset_async_client(client)
        assert key not in client._shared_clients
        assert not real_async_client.is_closed
        proceed.set()
        await asyncio.wait_for(task, timeout=5)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert real_async_client.is_closed


def test_h17_aclose_shared_clients_defers_leased_client_until_stream_completes(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
    )
    real_async_client = httpx.AsyncClient(timeout=None)
    key = (client.base_url, client.api_key)
    shared_clients: dict[tuple[str, str], object] = {key: real_async_client}
    monkeypatch.setattr(OpenAICompatClient, "_shared_clients", shared_clients)
    chunk_seen = asyncio.Event()
    proceed = asyncio.Event()

    monkeypatch.setattr(client_transport, "SSEStreamer", _lease_streamer_class(chunk_seen, proceed))
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: real_async_client)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []

        async def _consume() -> None:
            async for event in client_transport.stream_chat(
                client,
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
            ):
                events.append(event)

        task = asyncio.create_task(_consume())
        await asyncio.wait_for(chunk_seen.wait(), timeout=5)
        await OpenAICompatClient.aclose_shared_clients()
        assert shared_clients == {}
        assert not real_async_client.is_closed
        proceed.set()
        await asyncio.wait_for(task, timeout=5)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert real_async_client.is_closed


def test_h17_unleased_shared_client_resets_and_closes_immediately() -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
    )
    real_async_client = httpx.AsyncClient(timeout=None)
    key = (client.base_url, client.api_key)
    client._shared_clients = {key: real_async_client}

    asyncio.run(client_lifecycle._reset_async_client(client))

    assert key not in client._shared_clients
    assert real_async_client.is_closed


# ---------------------------------------------------------------------------
# D4/L21: SSEStreamer delegates continuation-timeout resolution to the shared
# resolver in transport_constants; client and streamer resolve identically.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_profile,model,expected_continuation",
    [
        ("generic", "wrench-9b", 30.0),
        ("generic", "qwen3.5:4b", 12.0),
        ("lmstudio", "wrench-9b", 90.0),
        ("lmstudio", "qwen3.5:4b", 135.0),
        ("llamacpp", "demo-model", 90.0),
        ("openrouter", "qwen/qwen3.5-9b", 30.0),
    ],
)
def test_d4_client_and_streamer_resolve_continuation_identically(
    provider_profile: str,
    model: str,
    expected_continuation: float,
) -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model=model,
        provider_profile=provider_profile,
    )
    streamer = SSEStreamer(
        provider_profile=provider_profile,
        aggressive_tool_call_timeout=client.is_small_model,
    )

    shared = transport_constants.resolve_tool_call_continuation_timeout_sec(
        None,
        client.adapter,
        provider_profile,
        client.is_small_model,
    )
    assert shared == client.tool_call_continuation_timeout_sec == expected_continuation
    assert streamer.tool_call_continuation_timeout_sec == shared
    assert (
        transport_constants.resolve_first_token_timeout_sec(
            None,
            client.adapter,
            provider_profile,
        )
        == client.first_token_timeout_sec
        == streamer.first_token_timeout_sec
    )

    bare_streamer = SSEStreamer(provider_profile=provider_profile)
    assert bare_streamer.tool_call_continuation_timeout_sec == (
        transport_constants.resolve_tool_call_continuation_timeout_sec(
            None,
            bare_streamer.adapter,
            provider_profile,
            False,
        )
    )


# ---------------------------------------------------------------------------
# D5/L22: the llama.cpp preflight logs the shared context-pressure/context-
# limit diagnostics so both preflights report identical values for identical
# payloads.
# ---------------------------------------------------------------------------


def _preflight_payload() -> dict[str, object]:
    return {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "x" * 40000}],
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ssh_exec",
                    "description": "Run SSH command",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def _llamacpp_test_client(run_logger: _RunLogger | None) -> SimpleNamespace:
    return SimpleNamespace(
        provider_profile="llamacpp",
        model="demo-model",
        log=logging.getLogger("test"),
        run_logger=run_logger,
    )


def test_d5_llamacpp_preflight_logs_shared_context_pressure_diagnostics() -> None:
    payload = _preflight_payload()
    shared = client_transport_helpers.context_pressure_diagnostics(payload, context_limit=8192)
    assert shared["likely_provider_rejection"] == "context_overflow"

    run_logger = _RunLogger()
    client = _llamacpp_test_client(run_logger)
    llamacpp_preflight._llamacpp_budget_preflight(
        client,
        payload=payload,
        stage="test",
        context_limit=8192,
    )

    entry = next(e for e in run_logger.entries if e["event"] == "payload_preflight_budget")
    for key, value in shared.items():
        assert entry["data"][key] == value

    openrouter_diag = openrouter_preflight._tool_schema_diagnostics(payload, context_limit=8192)
    for key, value in shared.items():
        assert openrouter_diag[key] == entry["data"][key] == value


def test_d5_llamacpp_preflight_unknown_limit_logs_shared_estimates() -> None:
    payload = _preflight_payload()
    shared = client_transport_helpers.context_pressure_diagnostics(payload, context_limit=None)

    run_logger = _RunLogger()
    client = _llamacpp_test_client(run_logger)
    result = llamacpp_preflight._llamacpp_budget_preflight(
        client,
        payload=payload,
        stage="test",
        context_limit=None,
    )

    assert result is None
    entry = next(e for e in run_logger.entries if e["event"] == "payload_preflight_budget")
    for key, value in shared.items():
        assert entry["data"][key] == value

    openrouter_diag = openrouter_preflight._tool_schema_diagnostics(payload, context_limit=None)
    for key, value in shared.items():
        assert openrouter_diag[key] == entry["data"][key] == value


# ---------------------------------------------------------------------------
# L25: SSE data lines accumulate per event and dispatch once at the blank
# event boundary; eager per-line dispatch only survives as a flush of
# unterminated buffered data at stream end (or before a mid-stream failure).
# ---------------------------------------------------------------------------


class _StreamResponse:
    def __init__(self, lines: list[str], raise_after_lines: Exception | None = None) -> None:
        self._lines = lines
        self._raise_after_lines = raise_after_lines

    def raise_for_status(self) -> None:
        pass

    def aiter_lines(self):
        async def _gen():
            for line in self._lines:
                yield line
            if self._raise_after_lines is not None:
                raise self._raise_after_lines

        return _gen()


class _StreamContext:
    def __init__(self, lines: list[str], raise_after_lines: Exception | None = None) -> None:
        self._response = _StreamResponse(lines, raise_after_lines=raise_after_lines)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class _StreamClient:
    def __init__(self, contexts: list[_StreamContext]) -> None:
        self._contexts = contexts
        self.call_count = 0

    def stream(self, method, url, **kwargs):
        del method, url, kwargs
        idx = self.call_count
        self.call_count += 1
        if idx >= len(self._contexts):
            raise AssertionError("stream must not be retried")
        return self._contexts[idx]


def _collect(streamer: SSEStreamer, client: _StreamClient) -> list[dict[str, object]]:
    async def _run() -> list[dict[str, object]]:
        return [event async for event in streamer.stream_sse(client, "http://test", {}, {})]

    return asyncio.run(_run())


def test_l25_valid_first_data_line_event_dispatches_once_at_boundary() -> None:
    client = _StreamClient(
        [
            _StreamContext(
                [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    "data: ",
                    "",
                    "data: [DONE]",
                    "",
                ]
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic")

    events = _collect(streamer, client)

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert events[0]["data"] == {"choices": [{"delta": {"content": "Hello"}}]}


def test_l25_multi_json_event_at_blank_boundary_is_not_split() -> None:
    client = _StreamClient(
        [
            _StreamContext(
                [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    'data: {"choices": [{"delta": {"content": " world"}}]}',
                    "",
                    "data: [DONE]",
                    "",
                ]
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic")

    events = _collect(streamer, client)

    # The joined payload of a boundary-terminated event is dispatched as one
    # event per SSE rules; it is never split into per-line chunks.
    assert [event["type"] for event in events] == ["done"]


def test_l25_unterminated_data_lines_flush_per_line_at_stream_end() -> None:
    client = _StreamClient(
        [
            _StreamContext(
                [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    'data: {"choices": [{"delta": {"content": " world"}}]}',
                ]
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic")

    events = _collect(streamer, client)

    assert [event["type"] for event in events] == ["chunk", "chunk", "stream_ended_without_done"]
    assert events[0]["data"] == {"choices": [{"delta": {"content": "Hello"}}]}
    assert events[1]["data"] == {"choices": [{"delta": {"content": " world"}}]}


def test_l25_unterminated_done_sentinel_flushes_at_stream_end() -> None:
    client = _StreamClient(
        [
            _StreamContext(
                [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    "data: [DONE]",
                ]
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic")

    events = _collect(streamer, client)

    assert [event["type"] for event in events] == ["chunk", "done"]


def test_l25_disconnect_flushes_unterminated_buffer_before_salvage() -> None:
    client = _StreamClient(
        [
            _StreamContext(
                [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    'data: {"choices": [{"delta": {"content": " world"}}]}',
                ],
                raise_after_lines=httpx.RemoteProtocolError("Server disconnected"),
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic")

    events = _collect(streamer, client)

    assert [event["type"] for event in events] == ["chunk", "chunk", "stream_ended_without_done"]
    assert events[2]["details"]["reason"] == "remote_protocol_error_after_chunks"
    assert client.call_count == 1
