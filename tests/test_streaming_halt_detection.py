from __future__ import annotations

import asyncio

from smallctl.client.streaming import SSEStreamer


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class _FakeClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def stream(self, *args, **kwargs):
        del args, kwargs
        return _FakeStreamContext(_FakeResponse(self._lines))


def test_stream_sse_emits_stream_ended_without_done_event() -> None:
    async def _run() -> list[dict[str, object]]:
        streamer = SSEStreamer()
        client = _FakeClient(
            [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            ]
        )
        events: list[dict[str, object]] = []
        async for event in streamer.stream_sse(client, "http://example.test", {}, {"model": "x"}):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert events[-1]["type"] == "stream_ended_without_done"
    assert events[-1]["details"]["chunk_count"] == 1
