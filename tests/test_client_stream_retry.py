from __future__ import annotations

import asyncio

import httpx

from smallctl.client import OpenAICompatClient
from smallctl.client import client_transport


def _http_status_error(
    url: str,
    *,
    status_code: int,
    headers: dict[str, str] | None = None,
    text: str = "",
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status_code, request=request, headers=headers, text=text)
    return httpx.HTTPStatusError(f"http {status_code}", request=request, response=response)


def test_stream_chat_retries_http_429(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
    )
    attempts = {"count": 0}
    reset_calls = {"count": 0}
    sleep_calls: list[float] = []

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_status_error(url, status_code=429, headers={"Retry-After": "0"})
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run for retryable 429")
            yield {}

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(float(delay))

    async def _fake_reset(_client: object) -> None:
        reset_calls["count"] += 1

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)
    monkeypatch.setattr(client_transport.asyncio, "sleep", _fake_sleep)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert attempts["count"] == 2
    assert reset_calls["count"] == 1
    assert sleep_calls == [1.0]


def test_stream_chat_uses_retry_after_delay_when_present(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
    )
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_status_error(url, status_code=429, headers={"Retry-After": "2.5"})
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run for retryable 429")
            yield {}

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(float(delay))

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)
    monkeypatch.setattr(client_transport.asyncio, "sleep", _fake_sleep)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert attempts["count"] == 2
    assert sleep_calls == [2.5]


def test_stream_chat_openrouter_400_recovers_via_reduced_payload_and_nonstream(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="demo-model",
        provider_profile="generic",
    )
    assert client.provider_profile == "openrouter"

    stream_attempts = {"count": 0}
    stream_payloads: list[dict[str, object]] = []
    nonstream_payloads: list[dict[str, object]] = []

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers
            stream_attempts["count"] += 1
            stream_payloads.append(dict(payload))
            raise _http_status_error(url, status_code=400, text="bad payload")
            yield {}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers
            nonstream_payloads.append(dict(payload))
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "fallback ok"}}]}}
            yield {"type": "done"}

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[
                {
                    "role": "assistant",
                    "content": "prep",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell_exec", "arguments": "{\"command\":\"pwd\"}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "shell_exec",
                    "tool_call_id": "call_1",
                    "content": "ok",
                },
                {"role": "user", "content": "continue"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "description": "Run shell command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert stream_attempts["count"] == 3
    assert len(nonstream_payloads) == 1
    assert "tools" in stream_payloads[0]
    assert "tools" in stream_payloads[1]
    assert "tools" not in stream_payloads[2]
    assert "tools" not in nonstream_payloads[0]


def test_openrouter_payload_preflight_repairs_invalid_shape() -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="demo-model",
        provider_profile="generic",
    )
    payload = {
        "model": "demo-model",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "bad-role", "content": "hello"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "shell_exec", "arguments": {"command": "pwd"}},
                    }
                ],
            },
            {"role": "tool", "name": "shell_exec", "tool_call_id": "call_1", "content": "ok"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "description": "Run shell command",
                    "parameters": [],
                },
            }
        ],
    }

    repaired = client_transport._preflight_openrouter_payload(client, payload, stage="test")

    assert "stream_options" not in repaired
    assert repaired["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert repaired["messages"][0] == {"role": "user", "content": "hello"}
    assert repaired["messages"][1]["role"] == "assistant"
    assert repaired["messages"][1]["content"] is None
    assert repaired["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"command": "pwd"}'


def test_stream_chat_openrouter_400_exhaustion_yields_actionable_chunk_error(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="demo-model",
        provider_profile="generic",
    )
    assert client.provider_profile == "openrouter"

    body = '{"error":{"metadata":{"provider_name":"Together"},"message":"Input validation error"}}'

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            raise _http_status_error(url, status_code=400, text=body)
            yield {}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, headers, payload
            raise _http_status_error(url, status_code=400, text=body)
            yield {}

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert len(events) == 1
    assert events[0]["type"] == "chunk_error"
    assert "openrouter/Together input validation failed" in str(events[0]["error"])
    details = events[0]["details"]
    assert details["status_code"] == 400
    assert details["provider_profile"] == "openrouter"
    assert details["upstream_provider"] == "Together"
    assert details["provider_error"] == "Input validation error"
    assert details["model"] == "demo-model"
    assert details["role_counts"] == {"user": 1}
