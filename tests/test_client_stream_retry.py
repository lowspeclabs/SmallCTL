from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest

from smallctl.client import OpenAICompatClient
from smallctl.client import client_transport
from smallctl.client.streaming import SSEStreamer


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


def test_nonstream_chat_fallback_handles_plain_httpx_response() -> None:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(
        200,
        request=request,
        json={"choices": [{"message": {"content": "fallback ok"}}]},
    )

    class _FakeAsyncClient:
        async def post(self, *args, **kwargs):
            del args, kwargs
            return response

    async def _collect() -> list[dict[str, object]]:
        streamer = SSEStreamer(run_logger=_RunLogger())
        return [
            event
            async for event in streamer.nonstream_chat(
                _FakeAsyncClient(),
                "https://example.test/v1/chat/completions",
                {},
                {"messages": [], "stream": True, "stream_options": {"include_usage": True}},
            )
        ]

    events = asyncio.run(_collect())

    assert events == [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {"message": {"content": "fallback ok"}, "delta": {"content": "fallback ok"}}
                ]
            },
        },
        {"type": "done"},
    ]
    assert response.is_closed


def test_llamacpp_transport_guard_repairs_late_system_messages() -> None:
    client = SimpleNamespace(
        provider_profile="llamacpp",
        log=logging.getLogger("test"),
        run_logger=None,
    )
    messages = [
        {"role": "user", "content": "Continue the task."},
        {"role": "System", "content": "Recovery nudge."},
    ]

    repaired = client_transport._repair_llamacpp_system_messages_for_transport(client, messages)

    assert repaired == [
        {"role": "system", "content": "Recovery nudge."},
        {"role": "user", "content": "Continue the task."},
    ]


def test_llamacpp_transport_guard_normalizes_leading_system_role_case() -> None:
    client = SimpleNamespace(
        provider_profile="llamacpp",
        log=logging.getLogger("test"),
        run_logger=None,
    )
    messages = [
        {"role": "System", "content": "Base prompt."},
        {"role": "user", "content": "Continue the task."},
    ]

    repaired = client_transport._repair_llamacpp_system_messages_for_transport(client, messages)

    assert repaired == [
        {"role": "system", "content": "Base prompt."},
        {"role": "user", "content": "Continue the task."},
    ]


def test_stream_chat_llamacpp_omits_stream_options_and_uses_local_timeouts(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    payloads: list[dict[str, object]] = []
    streamer_kwargs: dict[str, object] = {}

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            streamer_kwargs.update(kwargs)

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers
            payloads.append(dict(payload))
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

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

    assert [event["type"] for event in events] == ["done"]
    assert "stream_options" not in payloads[0]
    assert streamer_kwargs["first_token_timeout_sec"] == 60.0
    assert streamer_kwargs["tool_call_continuation_timeout_sec"] == 90.0


def test_stream_chat_llamacpp_repairs_system_messages_before_first_request(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    payloads: list[dict[str, object]] = []

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers
            payloads.append(dict(payload))
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[
                {"role": "System", "content": "Base prompt."},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "Late recovery nudge."},
            ],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["done"]
    messages = payloads[0]["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages].count("system") == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Base prompt.\n\nLate recovery nudge."
    assert all(message["role"] != "system" for message in messages[1:])


def test_stream_chat_openrouter_auth_preflight_returns_actionable_chunk_error(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-35b-a3b",
        provider_profile="openrouter",
        api_key="bad-key",
    )

    class _Response:
        status_code = 401
        text = '{"error":{"message":"User not found.","code":401}}'

    class _FakeAsyncClient:
        async def get(self, url, headers, timeout=None):
            assert url == "https://openrouter.ai/api/v1/credits"
            assert headers["Authorization"] == "Bearer bad-key"
            return _Response()

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise AssertionError("streamer should not be constructed after auth preflight failure")

    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: _FakeAsyncClient())
    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)

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
    assert "OpenRouter authentication failed" in str(events[0]["error"])
    details = events[0]["details"]
    assert isinstance(details, dict)
    assert details["reason"] == "openrouter_authentication_failed"
    assert details["provider_error"] == "User not found."
    assert details["recoverable"] is False


def test_stream_chat_openrouter_401_chat_error_returns_actionable_chunk_error(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-35b-a3b",
        provider_profile="openrouter",
        api_key="bad-key",
    )

    class _AuthResponse:
        status_code = 200
        text = '{"data":{"total_credits":1}}'

    class _FakeAsyncClient:
        async def get(self, url, headers, timeout=None):
            del url, headers, timeout
            return _AuthResponse()

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            raise _http_status_error(
                url,
                status_code=401,
                text='{"error":{"message":"User not found.","code":401}}',
            )
            yield {}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run for auth failures")
            yield {}

    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: _FakeAsyncClient())
    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)

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
    assert "Update SMALLCTL_API_KEY" in str(events[0]["error"])
    details = events[0]["details"]
    assert isinstance(details, dict)
    assert details["phase"] == "chat_completion"
    assert details["provider_error"] == "User not found."


def test_stream_chat_openrouter_caps_auto_max_tokens_with_large_context_limit(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-35b-a3b",
        provider_profile="openrouter",
        api_key="test-key",
    )
    client.runtime_context_limit = 2_000_000
    payloads: list[dict[str, object]] = []

    class _AuthResponse:
        status_code = 200
        text = '{"data":{"total_credits":1}}'

    class _FakeAsyncClient:
        async def get(self, url, headers, timeout=None):
            del url, headers, timeout
            return _AuthResponse()

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers
            payloads.append(dict(payload))
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: _FakeAsyncClient())
    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "read ./temp/pong.py"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_write",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["done"]
    assert payloads[0]["max_completion_tokens"] == 4096


def test_openrouter_context_probe_remembers_model_capabilities(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-35b-a3b",
        provider_profile="openrouter",
        api_key="test-key",
    )

    class _Response:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeAsyncClient:
        async def get(self, url, headers, timeout=None):
            del headers, timeout
            if url.endswith("/props") or url.endswith("/slots"):
                return _Response(404, {})
            if url.endswith("/models/qwen%2Fqwen3.6-35b-a3b"):
                return _Response(404, {})
            if url.endswith("/models"):
                return _Response(
                    200,
                    {
                        "data": [
                            {
                                "id": "qwen/qwen3.6-35b-a3b",
                                "context_length": 128000,
                                "supported_parameters": ["temperature", "max_tokens"],
                                "top_provider": {"max_completion_tokens": 8192},
                            }
                        ]
                    },
                )
            raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(
        "smallctl.client.client_transport_client_lifecycle._get_async_client",
        lambda _client: _FakeAsyncClient(),
    )

    limit = asyncio.run(client.fetch_model_context_limit())

    assert limit == 128000
    assert client.runtime_context_limit == 128000
    assert client.model_max_completion_tokens == 8192
    assert client.model_supported_parameters == ["temperature", "max_tokens"]
    assert client._request_max_completion_tokens([]) == 8192


def test_openrouter_metadata_completion_tokens_equal_context_length_is_capped(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.5-9b",
        provider_profile="openrouter",
        api_key="test-key",
    )

    class _Response:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeAsyncClient:
        async def get(self, url, headers, timeout=None):
            del headers, timeout
            if url.endswith("/props") or url.endswith("/slots"):
                return _Response(404, {})
            if url.endswith("/models/qwen%2Fqwen3.5-9b"):
                return _Response(404, {})
            if url.endswith("/models"):
                return _Response(
                    200,
                    {
                        "data": [
                            {
                                "id": "qwen/qwen3.5-9b",
                                "context_length": 262144,
                                "supported_parameters": ["temperature", "max_tokens"],
                                "top_provider": {"max_completion_tokens": 262140},
                            }
                        ]
                    },
                )
            raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(
        "smallctl.client.client_transport_client_lifecycle._get_async_client",
        lambda _client: _FakeAsyncClient(),
    )

    limit = asyncio.run(client.fetch_model_context_limit())

    assert limit == 262144
    assert client.model_max_completion_tokens == 262140
    # Without messages the metadata value is ignored (OpenRouter fallback is 2048).
    assert client._request_max_completion_tokens([]) == 2048
    # With messages the cap is also prompt-aware and stays under the context window.
    requested_max_tokens = client._request_max_completion_tokens(
        [], messages=[{"role": "system", "content": "hi"}, {"role": "user", "content": "hi"}]
    )
    assert requested_max_tokens < 262144
    assert requested_max_tokens > 0


def test_stream_chat_openrouter_omits_unsupported_max_tokens_from_metadata(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-35b-a3b",
        provider_profile="openrouter",
        api_key="test-key",
    )
    client.model_supported_parameters = ["temperature"]
    client.model_max_completion_tokens = 8192
    payloads: list[dict[str, object]] = []

    class _AuthResponse:
        status_code = 200
        text = '{"data":{"total_credits":1}}'

    class _FakeAsyncClient:
        async def get(self, url, headers, timeout=None):
            del url, headers, timeout
            return _AuthResponse()

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers
            payloads.append(dict(payload))
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: _FakeAsyncClient())
    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)

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

    assert [event["type"] for event in events] == ["done"]
    assert "max_tokens" not in payloads[0]


def test_stream_chat_llamacpp_500_jinja_system_message_retries_once(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    payloads: list[dict[str, object]] = []
    attempts = {"count": 0}

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers
            attempts["count"] += 1
            payloads.append(dict(payload))
            if attempts["count"] == 1:
                raise _http_status_error(
                    url,
                    status_code=500,
                    text="Jinja Exception: System message must be at the beginning",
                )
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[
                {"role": "system", "content": "Base prompt."},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "Late nudge."},
            ],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["done"]
    assert attempts["count"] == 2
    assert [message["role"] for message in payloads[1]["messages"]].count("system") == 1
    assert payloads[1]["messages"][0]["role"] == "system"
    assert payloads[1]["messages"][0]["content"] == "Base prompt.\n\nLate nudge."


def test_stream_chat_llamacpp_500_malformed_tool_json_becomes_recoverable_chunk_error(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="Qwen3.5-4B.Q3_K_M.gguf",
        provider_profile="llamacpp",
    )

    def _tool(name: str) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    malformed_body = (
        '{"error":{"code":500,"message":"Failed to parse tool call arguments as JSON: '
        '[json.exception.parse_error.101] parse error at line 1, column 9698: syntax error '
        'while parsing value - invalid string: missing closing quote; last read: '
        '\'\\"import sys\\\\nclass CronMatcher:\\\\n    pass\\\\n    dt\'","type":"server_error"}}'
    )

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            raise _http_status_error(url, status_code=500, text=malformed_body)
            yield {}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "write temp/cron_matcher.py"}],
            tools=[_tool("file_write"), _tool("file_read")],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk_error"]
    details = events[0]["details"]
    assert details["type"] == "malformed_tool_call_json"
    assert details["reason"] == "tool_call_continuation_timeout"
    assert details["recoverable"] is True
    assert details["tool_name_hint"] == "file_write"
    assert "CronMatcher" in details["partial_tool_call_arguments_preview"]


def test_stream_chat_llamacpp_400_retries_with_reduced_tools(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    payloads: list[dict[str, object]] = []
    attempts = {"count": 0}

    def _tool(name: str) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers
            attempts["count"] += 1
            payloads.append(dict(payload))
            if attempts["count"] == 1:
                raise _http_status_error(url, status_code=400)
            yield {
                "type": "chunk",
                "data": {"choices": [{"delta": {"content": "ok"}}]},
            }
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    tools = [
        _tool(name)
        for name in [
            "artifact_grep",
            "artifact_print",
            "artifact_read",
            "ask_human",
            "ast_patch",
            "dir_list",
            "file_download",
            "file_patch",
            "file_read",
            "file_write",
            "find_files",
            "grep",
            "http_get",
            "http_post",
            "log_note",
            "loop_status",
            "memory_update",
            "shell_exec",
            "ssh_exec",
            "ssh_file_patch",
            "ssh_file_read",
            "ssh_file_replace_between",
            "ssh_file_write",
            "step_complete",
            "step_fail",
            "task_complete",
            "task_fail",
        ]
    ]
    original_names = [
        item["function"]["name"]
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    ]

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "inspect remote host"}],
            tools=tools,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert attempts["count"] == 2
    assert len(payloads[0]["tools"]) == 27
    reduced_names = [
        item["function"]["name"]
        for item in payloads[1]["tools"]
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    ]
    assert reduced_names == [
        "ask_human",
        "log_note",
        "loop_status",
        "memory_update",
        "ssh_exec",
        "ssh_file_read",
        "step_complete",
        "step_fail",
        "task_complete",
        "task_fail",
    ]
    assert "stream_options" not in payloads[1]


def test_stream_chat_llamacpp_model_unloaded_recovers_and_reuses_reduced_payload(monkeypatch) -> None:
    run_logger = _RunLogger()
    recovery_calls: list[dict[str, object]] = []

    async def _recover(payload: dict[str, object]) -> dict[str, object]:
        recovery_calls.append(payload)
        return {"status": "recovered", "action": "restart_command"}

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
        run_logger=run_logger,
        backend_recovery_handler=_recover,
    )
    payloads: list[dict[str, object]] = []
    attempts = {"count": 0}

    def _tool(name: str) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers
            attempts["count"] += 1
            payloads.append(dict(payload))
            if attempts["count"] == 1:
                raise _http_status_error(url, status_code=400)
            if attempts["count"] == 2:
                yield {
                    "type": "chunk_error",
                    "error": "Model is unloaded.",
                    "details": {"message": "Model is unloaded."},
                }
                return
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    tools = [
        _tool(name)
        for name in [
            "artifact_grep",
            "artifact_print",
            "artifact_read",
            "ask_human",
            "ast_patch",
            "dir_list",
            "file_download",
            "file_patch",
            "file_read",
            "file_write",
            "find_files",
            "grep",
            "http_get",
            "http_post",
            "log_note",
            "loop_status",
            "memory_update",
            "shell_exec",
            "ssh_exec",
            "ssh_file_patch",
            "ssh_file_read",
            "ssh_file_replace_between",
            "ssh_file_write",
            "step_complete",
            "step_fail",
            "task_complete",
            "task_fail",
        ]
    ]

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "inspect remote host"}],
            tools=tools,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk", "done"]
    assert attempts["count"] == 3
    assert len(recovery_calls) == 1
    assert recovery_calls[0]["details"]["reason"] == "model_unloaded"
    assert len(payloads[0]["tools"]) == 27
    assert len(payloads[1]["tools"]) == len(payloads[2]["tools"])
    assert payloads[1]["tools"] == payloads[2]["tools"]

    retry_budget_entries = [
        entry
        for entry in run_logger.entries
        if entry["event"] == "payload_preflight_budget"
        and entry["data"]["stage"] == "http_400_reduced_tools_retry"
    ]
    assert retry_budget_entries
    assert retry_budget_entries[-1]["data"]["context_limit"] is None
    assert retry_budget_entries[-1]["data"]["context_limit_source"] == "unknown"
    assert retry_budget_entries[-1]["data"]["reduction_reason"] == "http_400_recovery"


def test_stream_chat_backend_stream_failure_invokes_recovery_and_retries(monkeypatch) -> None:
    recovery_calls: list[dict[str, object]] = []

    async def _recover(payload: dict[str, object]) -> dict[str, object]:
        recovery_calls.append(payload)
        return {"status": "recovered", "action": "restart_command"}

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="Gemma 4 e4b",
        provider_profile="llamacpp",
        run_logger=_RunLogger(),
        backend_recovery_handler=_recover,
    )
    attempts = {"count": 0}

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            attempts["count"] += 1
            if attempts["count"] == 1:
                yield {
                    "type": "chunk_error",
                    "error": "Remote protocol error after retry",
                    "details": {
                        "reason": "backend_stream_failure",
                        "provider_profile": "llamacpp",
                        "exception_type": "httpx.RemoteProtocolError",
                    },
                }
                return
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

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
    assert len(recovery_calls) == 1
    assert recovery_calls[0]["details"]["reason"] == "backend_stream_failure"


def test_stream_chat_preserves_password_in_provider_payload_without_mutating_live_messages(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
        run_logger=_RunLogger(),
    )
    payloads: list[dict[str, object]] = []

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers
            payloads.append(dict(payload))
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    messages = [{"role": "user", "content": "ssh password is hunter2"}]
    original_messages = [dict(message) for message in messages]

    async def _run() -> None:
        async for _event in client_transport.stream_chat(client, messages=messages, tools=[]):
            pass

    asyncio.run(_run())

    assert messages == original_messages
    assert payloads
    assert payloads[0]["messages"][0]["content"] == "ssh password is hunter2"


def test_stream_chat_llamacpp_model_unloaded_yields_provider_chunk_error(monkeypatch) -> None:
    async def _recover(payload: dict[str, object]) -> dict[str, object]:
        del payload
        return {"status": "unrecovered", "action": "none"}

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
        backend_recovery_handler=_recover,
    )

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            yield {
                "type": "chunk_error",
                "error": "Model is unloaded.",
                "details": {"message": "Model is unloaded."},
            }

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

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

    assert [event["type"] for event in events] == ["chunk_error"]
    assert events[0]["error"] == "llama.cpp model is unloaded"
    assert events[0]["details"]["type"] == "model_unloaded"
    assert events[0]["details"]["reason"] == "model_unloaded"
    assert events[0]["details"]["recovery"]["status"] == "unrecovered"


def test_stream_chat_llamacpp_preflight_reduces_before_first_request(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    client.runtime_context_limit = 8192
    payloads: list[dict[str, object]] = []

    def _tool(name: str) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool " + ("x" * 700),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "y" * 700},
                    },
                },
            },
        }

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers
            payloads.append(dict(payload))
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

    tools = [
        _tool(name)
        for name in [
            "artifact_grep",
            "artifact_print",
            "artifact_read",
            "ask_human",
            "ast_patch",
            "dir_list",
            "file_patch",
            "file_read",
            "file_write",
            "http_get",
            "http_post",
            "log_note",
            "loop_status",
            "memory_update",
            "shell_exec",
            "ssh_exec",
            "ssh_file_patch",
            "ssh_file_read",
            "ssh_file_replace_between",
            "ssh_file_write",
            "step_complete",
            "step_fail",
            "task_complete",
            "task_fail",
            "web_fetch",
            "web_search",
            "process_kill",
        ]
    ]
    original_names = [
        item["function"]["name"]
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    ]

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "inspect remote host"}],
            tools=tools,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["done"]
    sent_names = [
        item["function"]["name"]
        for item in payloads[0]["tools"]
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    ]
    assert len(sent_names) < 27
    assert {"ssh_exec", "ssh_file_read", "task_complete", "task_fail"} <= set(sent_names)
    assert {"web_search", "web_fetch", "process_kill"} & (set(original_names) - set(sent_names))


def test_llamacpp_preflight_preserves_file_write_for_build_script_intent() -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    client.runtime_context_limit = 4096

    def _tool(name: str, *, size: int = 4000) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool " + ("x" * size),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "y" * size},
                    },
                },
            },
        }

    tools = [
        _tool("artifact_read"),
        _tool("dir_list"),
        _tool("file_read"),
        _tool("file_write", size=100),
        _tool("file_patch", size=100),
        _tool("ask_human", size=100),
        _tool("loop_status", size=100),
        _tool("task_complete", size=100),
        _tool("task_fail", size=100),
    ]
    payload = {
        "model": client.model,
        "messages": [
            {
                "role": "user",
                "content": "Build a self-contained Python script at ./temp/task_queue.py.",
            }
        ],
        "stream": True,
        "tools": tools,
    }

    result = client_transport._llamacpp_budget_preflight(client, payload=payload, stage="test")

    assert result is not None
    assert result.action == "reduced_tools"
    assert "file_write" in result.kept_tool_names
    assert "file_read" in result.kept_tool_names
    assert {"artifact_read", "dir_list"} & set(result.dropped_tool_names)


def test_llamacpp_preflight_warns_and_caps_context_for_small_gemma4() -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="gemma-4-12b",
        provider_profile="llamacpp",
    )
    client.runtime_context_limit = 64768
    run_logger = _RunLogger()
    client.run_logger = run_logger

    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": "fix the backup script"}],
        "stream": True,
        "tools": [],
    }

    result = client_transport._llamacpp_budget_preflight(client, payload=payload, stage="test")

    assert result is not None
    assert result.action == "unchanged"
    warning = next(
        (entry for entry in run_logger.entries if entry["event"] == "small_gemma4_context_warning"),
        None,
    )
    assert warning is not None
    assert warning["data"]["context_limit"] == 64768
    assert warning["data"]["recommended_context_limit"] == 32768
    preflight_log = next(
        (entry for entry in run_logger.entries if entry["event"] == "payload_preflight_budget"),
        None,
    )
    assert preflight_log is not None
    assert preflight_log["data"]["context_limit"] == 32768
    assert preflight_log["data"]["effective_prompt_budget"] == 24576


def test_llamacpp_preflight_small_gemma4_uses_enforced_budget_in_log() -> None:
    """The logged effective_prompt_budget must reflect the 24576 override."""
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="gemma-4-12b",
        provider_profile="llamacpp",
    )
    client.runtime_context_limit = 32768
    run_logger = _RunLogger()
    client.run_logger = run_logger

    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": "fix the backup script"}],
        "stream": True,
        "tools": [],
    }

    result = client_transport._llamacpp_budget_preflight(client, payload=payload, stage="test")

    assert result is not None
    preflight_log = next(
        (entry for entry in run_logger.entries if entry["event"] == "payload_preflight_budget"),
        None,
    )
    assert preflight_log is not None
    assert preflight_log["data"]["effective_prompt_budget"] == 24576


def test_stream_chat_llamacpp_context_budget_error_is_not_retryable(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    client.runtime_context_limit = 2048

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise AssertionError("streamer should not be created for preflight budget failure")

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)

    async def _run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "x" * 30000}],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk_error"]
    assert events[0]["error"] == "llamacpp context budget exceeded before request"
    assert events[0]["details"]["type"] == "context_budget_exceeded"
    assert events[0]["details"]["recoverable"] is False


def test_llamacpp_400_diagnostics_parse_context_overflow_body() -> None:
    summary = client_transport._summarize_http_error_body(
        "request (9214 tokens) exceeds the available context size (8192 tokens), try increasing it"
    )

    assert summary["provider_error"] == "Context overflow"
    assert summary["context_overflow"] is True
    assert summary["request_tokens"] == 9214
    assert summary["context_limit"] == 8192
    assert summary["context_overflow_tokens"] == 1022


def test_stream_chat_llamacpp_400_context_overflow_returns_chunk_error(monkeypatch) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    client.STREAM_RETRY_ATTEMPTS = 1

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, headers, payload
            raise _http_status_error(
                url,
                status_code=400,
                text="request (16385 tokens) exceeds the available context size (16384 tokens), try increasing it",
            )
            yield {}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run for context overflow")
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
            messages=[{"role": "user", "content": "continue"}],
            tools=[],
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["chunk_error"]
    assert events[0]["error"] == "llamacpp context window exceeded"
    assert events[0]["details"]["type"] == "context_budget_exceeded"
    assert events[0]["details"]["reason"] == "context_overflow"
    assert events[0]["details"]["request_tokens"] == 16385
    assert events[0]["details"]["context_limit"] == 16384
    assert events[0]["details"]["recoverable"] is True
    assert client.runtime_context_limit == 16384


def test_llamacpp_400_payload_diagnostics_include_context_pressure_estimate() -> None:
    payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "x" * 40000}],
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

    diagnostics = client_transport._summarize_400_payload(payload, context_limit=8192)

    assert diagnostics["known_context_limit"] == 8192
    assert diagnostics["estimated_payload_tokens"] > 8192
    assert diagnostics["estimated_context_tokens_remaining"] < 0
    assert diagnostics["likely_provider_rejection"] == "context_overflow"
    assert diagnostics["estimated_tool_schema_tokens"] > 0


def test_stream_chat_logs_transport_exhaustion_with_endpoint_details(monkeypatch, caplog) -> None:
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="llamacpp",
    )
    client.STREAM_RETRY_ATTEMPTS = 1
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise httpx.ConnectError("connection refused", request=request)
            yield {}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise httpx.ConnectError("connection refused", request=request)
            yield {}

    async def _fake_reset(_client: object) -> None:
        return None

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())
    monkeypatch.setattr(client_transport, "_reset_async_client", _fake_reset)

    async def _run() -> None:
        async for _event in client_transport.stream_chat(
            client,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        ):
            pass

    with caplog.at_level(logging.ERROR, logger="smallctl.client"):
        with pytest.raises(httpx.ConnectError):
            asyncio.run(_run())

    exhausted_records = [
        record for record in caplog.records if "chat_transport_exhausted" in record.getMessage()
    ]
    assert exhausted_records
    message = exhausted_records[-1].getMessage()
    assert '"url": "http://127.0.0.1:8080/v1/chat/completions"' in message
    assert '"provider_profile": "llamacpp"' in message
    assert '"exception_type": "ConnectError"' in message


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


def test_extract_cached_tokens_reads_prompt_tokens_details() -> None:
    from smallctl.client.llamacpp_preflight import _extract_cached_tokens

    assert _extract_cached_tokens({"prompt_tokens_details": {"cached_tokens": 42}}) == 42
    assert _extract_cached_tokens({"cached_tokens": 7}) == 7
    assert _extract_cached_tokens({"prompt_tokens": 100}) is None


def test_swa_cache_observation_tracks_zero_cached_streak() -> None:
    from smallctl.client.llamacpp_preflight import _record_swa_cache_observation

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e4b-it", provider_profile="llamacpp"),
        state=SimpleNamespace(scratchpad={}),
    )
    usage = {"prompt_tokens_details": {"cached_tokens": 0}}
    assert not _record_swa_cache_observation(harness, usage)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 1
    assert _record_swa_cache_observation(harness, usage)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 2

    usage_cached = {"prompt_tokens_details": {"cached_tokens": 10}}
    assert not _record_swa_cache_observation(harness, usage_cached)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 0


def test_swa_cache_observation_ignores_non_swa_model() -> None:
    from smallctl.client.llamacpp_preflight import _record_swa_cache_observation

    harness = SimpleNamespace(
        client=SimpleNamespace(model="qwen2.5-7b", provider_profile="llamacpp"),
        state=SimpleNamespace(scratchpad={}),
    )
    usage = {"prompt_tokens_details": {"cached_tokens": 0}}
    assert not _record_swa_cache_observation(harness, usage)
    assert "_swa_zero_cached_streak" not in harness.state.scratchpad


def test_swa_cache_observation_tracks_gemma_4_12b() -> None:
    from smallctl.client.llamacpp_preflight import _record_swa_cache_observation

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-12b", provider_profile="llamacpp"),
        state=SimpleNamespace(scratchpad={}),
    )
    usage = {"prompt_tokens_details": {"cached_tokens": 0}}
    assert not _record_swa_cache_observation(harness, usage)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 1
    assert _record_swa_cache_observation(harness, usage)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 2


def test_is_swa_model_matches_gemma_4_variants_on_llamacpp() -> None:
    from smallctl.client.llamacpp_preflight import _is_swa_model

    assert _is_swa_model("gemma-4-12b", "llamacpp") is True
    assert _is_swa_model("Gemma 4 12b", "llamacpp") is True
    assert _is_swa_model("gemma-4-e4b-it", "llamacpp") is True
    assert _is_swa_model("gemma-4-27b-it", "llamacpp") is True
    assert _is_swa_model("gemma-4-12b", "lmstudio") is False
    assert _is_swa_model("gemma-3-4b-it", "llamacpp") is False
    assert _is_swa_model("qwen2.5-7b", "llamacpp") is False


def test_swa_cache_observation_treats_missing_cached_as_zero_for_gemma4_llamacpp() -> None:
    from smallctl.client.llamacpp_preflight import _record_swa_cache_observation

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-12b", provider_profile="llamacpp"),
        state=SimpleNamespace(scratchpad={}),
    )
    usage_no_cached = {"prompt_tokens": 1000}
    assert not _record_swa_cache_observation(harness, usage_no_cached)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 1
    assert _record_swa_cache_observation(harness, usage_no_cached)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 2

    usage_cached = {"prompt_tokens_details": {"cached_tokens": 10}}
    assert not _record_swa_cache_observation(harness, usage_cached)
    assert harness.state.scratchpad["_swa_zero_cached_streak"] == 0


def test_swa_cache_observation_ignores_missing_cached_for_non_swa() -> None:
    from smallctl.client.llamacpp_preflight import _record_swa_cache_observation

    harness = SimpleNamespace(
        client=SimpleNamespace(model="qwen2.5-7b", provider_profile="llamacpp"),
        state=SimpleNamespace(scratchpad={}),
    )
    usage = {"prompt_tokens": 1000}
    assert not _record_swa_cache_observation(harness, usage)
    assert "_swa_zero_cached_streak" not in harness.state.scratchpad


def test_maybe_emit_swa_cache_warning_logs_for_gemma4_llamacpp() -> None:
    from smallctl.client.llamacpp_preflight import _maybe_emit_swa_cache_warning

    log_calls: list[dict[str, object]] = []
    runlog_calls: list[dict[str, object]] = []

    class _FakeLog:
        def warning(self, *args, **kwargs):
            log_calls.append({"args": args, "kwargs": kwargs})

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-12b", provider_profile="llamacpp"),
        state=SimpleNamespace(scratchpad={}),
        log=_FakeLog(),
        _runlog=lambda event, message, **data: runlog_calls.append({"event": event, "message": message, **data}),
    )
    usage = {"prompt_tokens": 1000}
    # First call should start the streak but not yet warn.
    _maybe_emit_swa_cache_warning(harness, usage)
    assert not any(c.get("event") == "swa_cache_inactive" for c in runlog_calls)

    # Second call should trigger the warning.
    _maybe_emit_swa_cache_warning(harness, usage)
    assert any(c.get("event") == "swa_cache_inactive" for c in runlog_calls)
    assert any("--swa-full" in str(c.get("recommendation", "")) for c in runlog_calls)


def test_stream_chat_non_backend_chunk_error_is_not_retried(monkeypatch) -> None:
    """Generic provider stream errors are surfaced, not silently retried."""
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
            del async_client, url, headers, payload
            attempts["count"] += 1
            yield {
                "type": "chunk_error",
                "error": "list index out of range",
                "details": {"message": "list index out of range"},
            }
            return

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run for chunk_error")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

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

    assert [event["type"] for event in events] == ["chunk_error"]
    assert attempts["count"] == 1


def test_stream_chat_backend_stream_failure_is_retried(monkeypatch) -> None:
    """Only backend_stream_failure chunk errors trigger the retry path."""
    recovery_calls: list[dict[str, object]] = []

    async def _recover(payload: dict[str, object]) -> dict[str, object]:
        recovery_calls.append(payload)
        return {"status": "recovered", "action": "restart_command"}

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:8080/v1",
        model="demo-model",
        provider_profile="generic",
        backend_recovery_handler=_recover,
    )
    attempts = {"count": 0}

    class _FakeStreamer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def stream_sse(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            attempts["count"] += 1
            if attempts["count"] == 1:
                yield {
                    "type": "chunk_error",
                    "error": "Remote protocol error after retry",
                    "details": {
                        "reason": "backend_stream_failure",
                        "provider_profile": "generic",
                        "exception_type": "httpx.RemoteProtocolError",
                    },
                }
                return
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "ok"}}]}}
            yield {"type": "done"}

        async def nonstream_chat(self, async_client, url, headers, payload):
            del async_client, url, headers, payload
            raise AssertionError("nonstream fallback should not run")
            yield {}

    monkeypatch.setattr(client_transport, "SSEStreamer", _FakeStreamer)
    monkeypatch.setattr(client_transport, "_get_async_client", lambda _client: object())

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
    assert len(recovery_calls) == 1
    assert recovery_calls[0]["details"]["reason"] == "backend_stream_failure"
