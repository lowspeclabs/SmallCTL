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

    assert small_model_client.tool_call_continuation_timeout_sec == 45.0
    assert large_model_client.tool_call_continuation_timeout_sec == 60.0
    assert generic_client.tool_call_continuation_timeout_sec == 30.0
    assert override_client.tool_call_continuation_timeout_sec == 33.0


def test_streamer_defaults_lmstudio_tool_call_continuation_timeout() -> None:
    streamer = SSEStreamer(provider_profile="lmstudio")

    assert streamer.tool_call_continuation_timeout_sec == 60.0
