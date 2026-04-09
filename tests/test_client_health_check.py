from __future__ import annotations

from smallctl.client import OpenAICompatClient


def test_client_initializes_backend_recovery_defaults() -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="qwen3.5:4b",
        provider_profile="lmstudio",
    )
    assert client.provider_profile == "lmstudio"
    assert client.first_token_timeout_sec > 0
    assert client.tool_call_continuation_timeout_sec > 0
