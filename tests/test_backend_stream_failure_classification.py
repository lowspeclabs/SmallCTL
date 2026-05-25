from __future__ import annotations

import httpx
import pytest

from smallctl.client.streaming import SSEStreamer
from smallctl.graph.model_stream_resolution import _chunk_error_failure_type


def test_backend_stream_failure_reason() -> None:
    assert _chunk_error_failure_type({"reason": "backend_stream_failure"}) == "backend_stream_failure"


def test_model_unloaded_reason() -> None:
    assert _chunk_error_failure_type({"reason": "model_unloaded"}) == "provider"


def test_generic_stream_returns_stream() -> None:
    assert _chunk_error_failure_type({"reason": "unknown"}) == "stream"
    assert _chunk_error_failure_type({}) == "stream"


class _MockResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def aiter_lines(self):
        async def _gen():
            for line in self._lines:
                yield line
        return _gen()


class _MockStreamContext:
    def __init__(self, lines, raise_on_enter=None):
        self._response = _MockResponse(lines)
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self):
        if self._raise_on_enter is not None:
            raise self._raise_on_enter
        return self._response

    async def __aexit__(self, *args):
        pass


class _MockClient:
    def __init__(self, responses):
        self._responses = responses
        self._call_count = 0

    def stream(self, method, url, **kwargs):
        idx = self._call_count
        self._call_count += 1
        return self._responses[idx]


@pytest.mark.asyncio
async def test_stream_sse_retries_remote_protocol_error_once() -> None:
    """RemoteProtocolError on first stream, success on retry yields done event."""
    client = _MockClient([
        _MockStreamContext([], raise_on_enter=httpx.RemoteProtocolError("Server disconnected")),
        _MockStreamContext(["data: [DONE]"]),
    ])
    streamer = SSEStreamer(provider_profile="generic")
    events = []
    async for ev in streamer.stream_sse(client, "http://test", {}, {}):
        events.append(ev)
    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert client._call_count == 2


@pytest.mark.asyncio
async def test_stream_sse_classifies_backend_failure_after_two_remote_protocol_errors() -> None:
    """Two RemoteProtocolError failures yield backend_stream_failure classification."""
    client = _MockClient([
        _MockStreamContext([], raise_on_enter=httpx.RemoteProtocolError("Server disconnected")),
        _MockStreamContext([], raise_on_enter=httpx.RemoteProtocolError("Server disconnected again")),
    ])
    streamer = SSEStreamer(provider_profile="generic")
    events = []
    async for ev in streamer.stream_sse(client, "http://test", {}, {}):
        events.append(ev)
    assert len(events) == 1
    assert events[0]["type"] == "chunk_error"
    assert events[0]["details"]["reason"] == "backend_stream_failure"
    assert events[0]["details"]["exception_type"] == "httpx.RemoteProtocolError"
    assert client._call_count == 2
