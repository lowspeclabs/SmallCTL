from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx

from smallctl.harness import Harness


def _make_harness(*, provider_profile: str, backend_unload_command: str | None = None) -> Harness:
    harness = Harness.__new__(Harness)
    harness.provider_profile = provider_profile
    harness.client = SimpleNamespace(
        base_url="http://example.test/v1",
        api_key="test-key",
        model="demo-model",
    )
    harness.state = SimpleNamespace(scratchpad={})
    harness.backend_healthcheck_url = "http://example.test/v1/models"
    harness.backend_unload_command = backend_unload_command
    harness.backend_restart_command = None
    harness.backend_healthcheck_timeout_sec = 5
    harness.backend_restart_grace_sec = 20
    harness.backend_max_restarts_per_hour = 2
    harness._runlog = lambda *args, **kwargs: None
    return harness


def test_backend_unload_command_recovers_when_health_probe_is_ok() -> None:
    harness = _make_harness(provider_profile="lmstudio", backend_unload_command="echo unloading model")
    calls: list[Any] = []

    async def _probe(_url: str, *, timeout_sec: int) -> dict[str, Any]:
        calls.append(("probe", timeout_sec))
        return {"ok": True, "status_code": 200}

    async def _unload(command: str | None) -> dict[str, Any]:
        calls.append(("unload", command))
        return {"ok": True}

    async def _wait(_url: str, *, timeout_sec: int) -> dict[str, Any]:
        calls.append(("wait", timeout_sec))
        return {"ok": True, "status_code": 200}

    async def _restart(command: str) -> dict[str, Any]:
        raise AssertionError(f"restart should not run: {command}")

    harness._probe_backend_health = _probe
    harness._run_backend_unload_command = _unload
    harness._wait_for_backend_health = _wait
    harness._run_backend_restart_command = _restart

    result = asyncio.run(
        harness.recover_backend_wedge({"details": {"reason": "first_token_timeout"}})
    )

    assert result["status"] == "recovered"
    assert result["action"] == "unload_command"
    assert calls == [("probe", 5), ("unload", "echo unloading model"), ("wait", 20)]


def test_ollama_unload_falls_back_to_keep_alive_zero() -> None:
    harness = _make_harness(provider_profile="ollama", backend_unload_command=None)
    calls: list[Any] = []

    async def _probe(_url: str, *, timeout_sec: int) -> dict[str, Any]:
        calls.append(("probe", timeout_sec))
        return {"ok": True, "status_code": 200}

    async def _wait(_url: str, *, timeout_sec: int) -> dict[str, Any]:
        calls.append(("wait", timeout_sec))
        return {"ok": True, "status_code": 200}

    class _FakeResponse:
        status_code = 200
        text = "{\"done\": true}"

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("client", kwargs.get("timeout")))

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
            calls.append(("post", url, headers["Content-Type"], json))
            return _FakeResponse()

    original_async_client = httpx.AsyncClient
    httpx.AsyncClient = _FakeAsyncClient  # type: ignore[assignment]
    try:
        harness._probe_backend_health = _probe
        harness._wait_for_backend_health = _wait

        result = asyncio.run(
            harness.recover_backend_wedge({"details": {"reason": "first_token_timeout"}})
        )
    finally:
        httpx.AsyncClient = original_async_client  # type: ignore[assignment]

    assert result["status"] == "recovered"
    assert result["action"] == "ollama_keep_alive_zero"
    assert calls[0] == ("probe", 5)
    assert calls[1][0] == "client"
    assert calls[2][0] == "post"
    assert calls[2][1] == "http://example.test/api/generate"
    assert calls[2][3]["keep_alive"] == 0
    assert calls[3] == ("wait", 20)
