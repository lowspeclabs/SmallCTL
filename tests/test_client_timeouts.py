from __future__ import annotations

from smallctl.client import OpenAICompatClient
from smallctl.client.streaming import SSEStreamer


def test_client_resolves_lmstudio_tool_call_continuation_timeouts() -> None:
    small_model_client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="qwen3.5:4b",
        provider_profile="lmstudio",
    )
    large_model_client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
    )
    generic_client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="generic",
    )
    override_client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        tool_call_continuation_timeout_sec=33.0,
    )

    assert small_model_client.tool_call_continuation_timeout_sec == 135.0
    assert large_model_client.tool_call_continuation_timeout_sec == 90.0
    assert generic_client.tool_call_continuation_timeout_sec == 30.0
    assert override_client.tool_call_continuation_timeout_sec == 33.0


def test_client_increases_lmstudio_timeout_for_write_heavy_tools() -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
    )

    timeout = client._request_tool_call_continuation_timeout_sec(
        [
            {
                "type": "function",
                "function": {
                    "name": "ssh_file_write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]
    )

    assert timeout == 180.0


def test_client_doubles_write_heavy_timeout_for_generic_provider() -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="generic",
    )

    timeout = client._request_tool_call_continuation_timeout_sec(
        [
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
        ]
    )

    assert timeout == 60.0


def test_client_doubles_write_heavy_timeout_from_override() -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="openrouter",
        tool_call_continuation_timeout_sec=33.0,
    )

    timeout = client._request_tool_call_continuation_timeout_sec(
        [
            {
                "type": "function",
                "function": {
                    "name": "ssh_file_patch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "target_text": {"type": "string"},
                            "replacement_text": {"type": "string"},
                        },
                    },
                },
            }
        ]
    )

    assert timeout == 66.0


def test_streamer_defaults_lmstudio_tool_call_continuation_timeout() -> None:
    streamer = SSEStreamer(provider_profile="lmstudio")

    assert streamer.tool_call_continuation_timeout_sec == 90.0
