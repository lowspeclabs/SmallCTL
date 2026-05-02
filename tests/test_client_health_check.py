from __future__ import annotations

import inspect
from types import SimpleNamespace

from smallctl.client import OpenAICompatClient
from smallctl.harness.prompt_builder import PromptBuilderService


def test_client_initializes_backend_recovery_defaults() -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="qwen3.5:4b",
        provider_profile="lmstudio",
    )
    assert client.provider_profile == "lmstudio"
    assert client.first_token_timeout_sec > 0
    assert client.tool_call_continuation_timeout_sec > 0


def test_client_stream_chat_returns_async_iterator() -> None:
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
    )

    stream = client.stream_chat(messages=[], tools=[])

    assert hasattr(stream, "__aiter__")
    assert inspect.iscoroutine(stream) is False


def test_client_forces_openrouter_profile_for_openrouter_endpoint() -> None:
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="demo-model",
        provider_profile="generic",
    )
    assert client.provider_profile == "openrouter"
    assert client.adapter.name == "openrouter"


def test_prompt_builder_skips_context_probe_when_runtime_probe_disabled() -> None:
    async def fail_fetch_model_context_limit() -> int | None:
        raise AssertionError("fetch_model_context_limit should not run when runtime_context_probe is disabled")

    harness = SimpleNamespace(
        client=SimpleNamespace(
            runtime_context_probe=False,
            fetch_model_context_limit=fail_fetch_model_context_limit,
        ),
        server_context_limit=None,
        _runtime_context_probe_attempted=False,
        _apply_server_context_limit=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("_apply_server_context_limit should not run when probe is disabled")
        ),
    )

    import asyncio

    asyncio.run(PromptBuilderService(harness).ensure_context_limit())

    assert harness._runtime_context_probe_attempted is False
