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


def test_client_forces_llamacpp_profile_for_local_gguf_model() -> None:
    client = OpenAICompatClient(
        base_url="http://192.168.1.9:8080/v1",
        model="Qwen3.5-Coder-4B-Instruct-GGUF",
        provider_profile="generic",
    )
    assert client.provider_profile == "llamacpp"
    assert client.adapter.name == "llamacpp"


def test_client_does_not_treat_remote_gguf_model_as_llamacpp() -> None:
    client = OpenAICompatClient(
        base_url="https://api.example.com/v1",
        model="vendor/model-gguf-test",
        provider_profile="generic",
    )
    assert client.provider_profile == "generic"
    assert client.adapter.name == "generic"


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


def test_prompt_builder_reprobes_llamacpp_context_limit_after_initial_probe() -> None:
    calls: list[str] = []

    async def fetch_model_context_limit() -> int | None:
        calls.append("probe")
        return 16384

    applied: list[tuple[int, str]] = []

    def apply_server_context_limit(limit: int, *, source: str) -> None:
        applied.append((limit, source))
        harness.server_context_limit = limit

    harness = SimpleNamespace(
        client=SimpleNamespace(
            runtime_context_probe=True,
            fetch_model_context_limit=fetch_model_context_limit,
        ),
        provider_profile="llamacpp",
        server_context_limit=8192,
        _runtime_context_probe_attempted=True,
        _apply_server_context_limit=apply_server_context_limit,
    )

    import asyncio

    asyncio.run(PromptBuilderService(harness).ensure_context_limit())

    assert calls == ["probe"]
    assert applied == [(16384, "runtime_probe")]
    assert harness._runtime_context_probe_attempted is True


def test_prompt_builder_does_not_reprobe_non_llamacpp_after_initial_probe() -> None:
    async def fail_fetch_model_context_limit() -> int | None:
        raise AssertionError("fetch_model_context_limit should not run after probe")

    harness = SimpleNamespace(
        client=SimpleNamespace(
            runtime_context_probe=True,
            fetch_model_context_limit=fail_fetch_model_context_limit,
        ),
        provider_profile="generic",
        server_context_limit=8192,
        _runtime_context_probe_attempted=True,
        _apply_server_context_limit=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("_apply_server_context_limit should not run")
        ),
    )

    import asyncio

    asyncio.run(PromptBuilderService(harness).ensure_context_limit())

    assert harness._runtime_context_probe_attempted is True
