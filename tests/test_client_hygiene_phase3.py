"""Phase 3 client hygiene regressions (D3-D6, L24-L26, M19)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest

from smallctl.client import OpenAICompatClient
from smallctl.client import client_transport
from smallctl.client import client_transport_helpers
from smallctl.client import llamacpp_preflight
from smallctl.client import openrouter_preflight
from smallctl.client import transport_constants
from smallctl.client import usage
from smallctl.client.streaming import SSEStreamer


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


def _read_tool(name: str = "file_read") -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }


def _write_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "file_write",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
        },
    }


# ---------------------------------------------------------------------------
# D3/L20: dead duplicated first-token timeout method removed from the client;
# timeout selection flows through stream_chat via the transport helper.
# ---------------------------------------------------------------------------


def _stream_chat_streamer_kwargs(
    monkeypatch,
    *,
    provider_profile: str,
    model: str,
    tools: list[dict[str, object]],
) -> tuple[OpenAICompatClient, dict[str, object]]:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model=model,
        provider_profile=provider_profile,
    )
    captured: dict[str, object] = {}

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    async def _run() -> list[dict[str, object]]:
        return [
            event
            async for event in client_transport.stream_chat(
                client,
                messages=[{"role": "user", "content": "hello"}],
                tools=tools,
            )
        ]

    events = asyncio.run(_run())
    assert [event["type"] for event in events] == ["done"]
    return client, captured


@pytest.mark.parametrize(
    "provider_profile,model,tools,expected_first_token",
    [
        ("llamacpp", "demo-model", [], 60.0),
        ("lmstudio", "wrench-9b", [], 45.0),
        ("generic", "wrench-9b", [], 30.0),
        ("lmstudio", "wrench-9b", [_read_tool(f"tool_{i}") for i in range(12)], 60.0),
        ("lmstudio", "gemma-3-4b-it", [_read_tool()], 60.0),
        ("generic", "wrench-9b", [_write_tool()], 60.0),
    ],
)
def test_d3_timeout_selection_flows_through_stream_chat(
    monkeypatch,
    provider_profile: str,
    model: str,
    tools: list[dict[str, object]],
    expected_first_token: float,
) -> None:
    assert not hasattr(OpenAICompatClient, "_request_first_token_timeout_sec")

    client, captured = _stream_chat_streamer_kwargs(
        monkeypatch,
        provider_profile=provider_profile,
        model=model,
        tools=tools,
    )

    assert captured["first_token_timeout_sec"] == expected_first_token
    # The transport helper is the authoritative implementation.
    assert (
        client_transport_helpers.request_first_token_timeout_sec(client, tools)
        == expected_first_token
    )


# ---------------------------------------------------------------------------
# D4/L21: timeout constants and resolvers live in transport_constants and are
# shared by OpenAICompatClient and SSEStreamer.
# ---------------------------------------------------------------------------

_SHARED_CONSTANT_NAMES = (
    "STREAM_CONNECT_TIMEOUT_SEC",
    "STREAM_WRITE_TIMEOUT_SEC",
    "STREAM_READ_TIMEOUT_SEC",
    "STREAM_FIRST_TOKEN_TIMEOUT_SEC",
    "LMSTUDIO_FIRST_TOKEN_TIMEOUT_SEC",
    "STREAM_POOL_TIMEOUT_SEC",
    "STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC",
    "SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC",
)


@pytest.mark.parametrize("name", _SHARED_CONSTANT_NAMES)
def test_d4_timeout_constants_have_single_home(name: str) -> None:
    shared = getattr(transport_constants, name)
    assert getattr(OpenAICompatClient, name) == shared
    assert getattr(SSEStreamer, name) == shared
    assert isinstance(shared, float)


def test_d4_lmstudio_continuation_variants_preserved() -> None:
    assert (
        OpenAICompatClient.LMSTUDIO_SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC
        == transport_constants.LMSTUDIO_SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC
        == 135.0
    )
    assert (
        OpenAICompatClient.LMSTUDIO_TOOL_CALL_CONTINUATION_TIMEOUT_SEC
        == transport_constants.LMSTUDIO_TOOL_CALL_CONTINUATION_TIMEOUT_SEC
        == 90.0
    )


@pytest.mark.parametrize(
    "provider_profile,model",
    [
        ("generic", "wrench-9b"),
        ("lmstudio", "wrench-9b"),
        ("lmstudio", "qwen3.5:4b"),
        ("llamacpp", "demo-model"),
        ("openrouter", "qwen/qwen3.5-9b"),
    ],
)
def test_d4_client_and_streamer_resolve_identically(provider_profile: str, model: str) -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model=model,
        provider_profile=provider_profile,
    )
    streamer = SSEStreamer(provider_profile=provider_profile)

    assert (
        transport_constants.resolve_first_token_timeout_sec(
            None,
            client.adapter,
            provider_profile,
        )
        == client.first_token_timeout_sec
        == streamer.first_token_timeout_sec
    )
    assert (
        transport_constants.resolve_tool_call_continuation_timeout_sec(
            None,
            client.adapter,
            provider_profile,
            client.is_small_model,
        )
        == client.tool_call_continuation_timeout_sec
    )
    bridged = SSEStreamer(
        provider_profile=provider_profile,
        tool_call_continuation_timeout_sec=client.tool_call_continuation_timeout_sec,
    )
    assert bridged.tool_call_continuation_timeout_sec == client.tool_call_continuation_timeout_sec


# ---------------------------------------------------------------------------
# D5/L22: tool-name and context-pressure diagnostics live in
# client_transport_helpers and both preflights use them.
# ---------------------------------------------------------------------------


def test_d5_preflights_share_tool_name_and_context_diagnostics() -> None:
    assert openrouter_preflight._tool_name is client_transport_helpers.tool_name
    assert llamacpp_preflight._tool_name is client_transport_helpers.tool_name
    assert (
        openrouter_preflight._context_pressure_diagnostics
        is client_transport_helpers.context_pressure_diagnostics
    )

    payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "inspect remote host"}],
        "tools": [
            _read_tool(),
            _write_tool(),
            {
                "type": "function",
                "function": {
                    "name": "ssh_exec",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            },
        ],
    }

    openrouter_diag = openrouter_preflight._tool_schema_diagnostics(payload, context_limit=8192)
    helper_diag = client_transport_helpers.context_pressure_diagnostics(payload, context_limit=8192)
    for key, value in helper_diag.items():
        assert openrouter_diag[key] == value
    assert openrouter_diag["tool_names"] == ["file_read", "file_write", "ssh_exec"]
    assert [
        client_transport_helpers.tool_name(tool) for tool in payload["tools"]
    ] == openrouter_diag["tool_names"]

    # The llama.cpp preflight produces the same tool names via the shared helper.
    client = SimpleNamespace(
        provider_profile="llamacpp",
        model="demo-model",
        log=logging.getLogger("test"),
        run_logger=None,
    )
    result = llamacpp_preflight._llamacpp_budget_preflight(
        client,
        payload=payload,
        stage="test",
        context_limit=None,
    )
    assert result is None  # unknown-limit path only logs; names checked below
    # kept_tool_names for the unknown-limit path is derived from the shared helper.
    assert [
        llamacpp_preflight._tool_name(tool) for tool in payload["tools"]
    ] == openrouter_diag["tool_names"]


# ---------------------------------------------------------------------------
# D6/L23: _normalize_parameter_name lives in usage.py; the client imports it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("maxTokens", "max_tokens"),
        ("max-tokens", "max_tokens"),
        ("max_tokens", "max_tokens"),
        ("", ""),
        ("topK", "top_k"),
        ("XMLParser", "x_m_l_parser"),
    ],
)
def test_d6_normalize_parameter_name_shared(name: str, expected: str) -> None:
    assert not hasattr(OpenAICompatClient, "_normalize_request_parameter_name")
    assert usage._normalize_parameter_name(name) == expected

    # Public path 1: usage metadata extraction.
    extracted = usage.extract_supported_parameters({"supported_parameters": [name]})
    assert extracted == ([name] if expected else None)

    # Public path 2: client request-parameter support checks.
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="generic",
    )
    probe = name or "maxTokens"
    client.model_supported_parameters = [probe]
    assert client._request_supports_parameter(probe) is True
    assert client._request_supports_parameter(f"{probe}Extra") is False


# ---------------------------------------------------------------------------
# L25: SSE data lines accumulate per event and parse at event boundaries.
# ---------------------------------------------------------------------------


class _StreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        pass

    def aiter_lines(self):
        async def _gen():
            for line in self._lines:
                yield line

        return _gen()


class _StreamContext:
    def __init__(self, lines: list[str]) -> None:
        self._response = _StreamResponse(lines)

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


def test_l25_split_sse_event_parses_as_single_chunk() -> None:
    client = _StreamClient(
        [
            _StreamContext(
                [
                    ": keepalive",
                    'data: {"choices": [{"delta":',
                    'data:  {"content": "Hello"}}]}',
                    "",
                    'data: {"choices": [{"delta": {"content": " world"}}]}',
                    "",
                    "data: [DONE]",
                    "",
                ]
            )
        ]
    )
    streamer = SSEStreamer(provider_profile="generic")

    async def _run() -> list[dict[str, object]]:
        return [event async for event in streamer.stream_sse(client, "http://test", {}, {})]

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "chunk", "done"]
    # One JSON event split across two data lines yields exactly one chunk.
    assert events[0]["data"] == {"choices": [{"delta": {"content": "Hello"}}]}
    # Single-line event and comment-line coverage.
    assert events[1]["data"] == {"choices": [{"delta": {"content": " world"}}]}


# ---------------------------------------------------------------------------
# L26: one bounded pre-chunk retry for unclassified transient HTTP 500s;
# 404/422 stay terminal.
# ---------------------------------------------------------------------------


def _run_status_error_stream(
    monkeypatch,
    *,
    statuses: list[int],
) -> tuple[int, list[dict[str, object]]]:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
    )
    attempts = {"count": 0}

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            attempts["count"] += 1
            idx = attempts["count"] - 1
            if idx < len(statuses):
                raise _http_status_error(
                    url,
                    status_code=statuses[idx],
                    text="provider error body",
                )
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    async def _no_sleep(_delay: float) -> None:
        return None

    async def _fake_reset(_client: object) -> None:
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

    events = asyncio.run(_run())
    return attempts["count"], events


def test_l26_transient_500_then_success_retries_once(monkeypatch) -> None:
    attempts, events = _run_status_error_stream(monkeypatch, statuses=[500])

    assert attempts == 2
    assert [event["type"] for event in events] == ["chunk", "done"]
    assert events[0]["data"]["choices"][0]["delta"]["content"] == "ok"


def test_l26_terminal_404_and_422_are_not_retried(monkeypatch) -> None:
    for status in (404, 422):
        client = OpenAICompatClient(
            base_url="http://127.0.0.1:8080/v1",
            model="demo-model",
            provider_profile="generic",
        )
        attempts = {"count": 0}

        class _FakeStreamer:
            def __init__(self, **kwargs) -> None:
                del kwargs

            async def stream_sse(self, async_client, url, headers, payload):
                del async_client, headers, payload
                attempts["count"] += 1
                raise _http_status_error(
                    url,
                    status_code=status,
                    text="model not found",
                )
                yield {}

            async def nonstream_chat(self, async_client, url, headers, payload):
                del async_client, url, headers, payload
                raise AssertionError("nonstream fallback should not run")
                yield {}

        monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
        monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

        async def _run() -> list[dict[str, object]]:
            return [
                event
                async for event in client_transport.stream_chat(
                    client,
                    messages=[{"role": "user", "content": "hello"}],
                    tools=[],
                )
            ]

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_run())
        assert attempts["count"] == 1, f"status {status} must not be retried"


def test_l26_exhausted_transient_500_is_bounded(monkeypatch) -> None:
    attempts = {"count": 0}
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
    )

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            attempts["count"] += 1
            raise _http_status_error(url, status_code=500, text="boom")
            yield {}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    async def _no_sleep(_delay: float) -> None:
        return None

    async def _fake_reset(_client: object) -> None:
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

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_run())
    assert attempts["count"] == 2


# ---------------------------------------------------------------------------
# M19: context probe returns None when the configured model is absent from
# the provider's model list; exact and same-family matches still resolve.
# ---------------------------------------------------------------------------


class _ProbeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _ProbeAsyncClient:
    def __init__(self, list_payload: dict[str, object]) -> None:
        self._list_payload = list_payload

    async def get(self, url, headers, timeout=None):
        del headers, timeout
        if url.endswith("/props") or url.endswith("/slots"):
            return _ProbeResponse(404, {})
        if url.endswith("/models"):
            return _ProbeResponse(200, self._list_payload)
        if "/models/" in url:
            return _ProbeResponse(404, {})
        raise AssertionError(f"unexpected url: {url}")


def _probe_limit(
    monkeypatch,
    *,
    model: str,
    list_payload: dict[str, object],
) -> tuple[int | None, OpenAICompatClient]:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model=model,
        provider_profile="openrouter",
        api_key="test-key",
    )
    monkeypatch.setattr(
        "smallctl.client.client_transport_client_lifecycle._get_async_client",
        lambda _client: _ProbeAsyncClient(list_payload),
    )
    limit = asyncio.run(client.fetch_model_context_limit())
    return limit if limit is None else int(limit), client


def test_m19_absent_configured_model_returns_none(monkeypatch) -> None:
    limit, client = _probe_limit(
        monkeypatch,
        model="acme/absent-9b",
        list_payload={
            "data": [
                {"id": "other/model-a", "context_length": 1000000},
                {"id": "other/model-b", "context_length": 32000},
            ]
        },
    )

    assert limit is None
    assert getattr(client, "runtime_context_limit", None) is None


def test_m19_exact_match_returns_that_limit(monkeypatch) -> None:
    limit, client = _probe_limit(
        monkeypatch,
        model="acme/absent-9b",
        list_payload={
            "data": [
                {"id": "other/model-a", "context_length": 1000000},
                {"id": "acme/absent-9b", "context_length": 128000},
            ]
        },
    )

    assert limit == 128000
    assert client.runtime_context_limit == 128000


def test_m19_same_family_match_returns_that_limit(monkeypatch) -> None:
    limit, client = _probe_limit(
        monkeypatch,
        model="acme/absent-9b",
        list_payload={
            "data": [
                {"id": "other/model-a", "context_length": 1000000},
                {"id": "absent-9b-awq", "context_length": 64000},
            ]
        },
    )

    assert limit == 64000
    assert client.runtime_context_limit == 64000
