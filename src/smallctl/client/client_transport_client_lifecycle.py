from __future__ import annotations

import logging
import threading
from typing import Any

from .usage import extract_context_limit, extract_max_completion_tokens, extract_supported_parameters

_logger = logging.getLogger("smallctl.client.lifecycle")
_shared_client_lock = threading.Lock()
_shared_client_leases: dict[tuple[str, str], int] = {}
_shared_client_pending_close: dict[tuple[str, str], list[Any]] = {}


def _client_key(client: Any) -> tuple[str, str]:
    return (client.base_url, client.api_key)


def _get_async_client(client: Any) -> Any:
    import httpx
    if httpx is None:
        raise RuntimeError("Dependency missing: httpx")
    key = _client_key(client)
    shared_clients = client._shared_clients
    with _shared_client_lock:
        async_client = shared_clients.get(key)
        if async_client is None:
            async_client = httpx.AsyncClient(timeout=None)
            shared_clients[key] = async_client
    return async_client


def acquire_stream_lease(client: Any) -> None:
    key = _client_key(client)
    with _shared_client_lock:
        _shared_client_leases[key] = _shared_client_leases.get(key, 0) + 1


async def release_stream_lease(client: Any) -> None:
    key = _client_key(client)
    with _shared_client_lock:
        remaining = _shared_client_leases.get(key, 0) - 1
        if remaining > 0:
            _shared_client_leases[key] = remaining
            return
        _shared_client_leases.pop(key, None)
        drained = _shared_client_pending_close.pop(key, [])
    for async_client in drained:
        await _close_shared_client(client, async_client)


async def _close_shared_client(client: Any, async_client: Any) -> None:
    try:
        await async_client.aclose()
    except Exception as exc:
        _logger.warning(
            "Failed to close shared async client for %s: %s",
            client.base_url,
            exc,
        )


def _defer_close_if_leased(key: tuple[str, str], async_client: Any) -> Any | None:
    if _shared_client_leases.get(key, 0) > 0:
        _shared_client_pending_close.setdefault(key, []).append(async_client)
        return None
    return async_client


def _drain_shared_clients(shared_clients: dict[tuple[str, str], Any]) -> list[Any]:
    with _shared_client_lock:
        entries = list(shared_clients.items())
        shared_clients.clear()
        immediate: list[Any] = []
        for key, async_client in entries:
            closable = _defer_close_if_leased(key, async_client)
            if closable is not None:
                immediate.append(closable)
    return immediate


async def _reset_async_client(client: Any) -> None:
    key = _client_key(client)
    with _shared_client_lock:
        async_client = client._shared_clients.pop(key, None)
        if async_client is not None:
            async_client = _defer_close_if_leased(key, async_client)
    if async_client is None:
        return
    await _close_shared_client(client, async_client)


def _remember_context_limit(client: Any, limit: int | None) -> int | None:
    if limit is None:
        return None
    try:
        normalized = int(limit)
    except Exception:
        return None
    if normalized <= 0:
        return None
    try:
        client.runtime_context_limit = normalized
    except Exception:
        pass
    return normalized


def _remember_model_metadata(client: Any, payload: Any, *, source: str) -> int | None:
    if isinstance(payload, dict):
        try:
            client.model_metadata = dict(payload)
        except Exception:
            pass
        try:
            client.model_metadata_source = source
        except Exception:
            pass

        max_completion_tokens = extract_max_completion_tokens(payload)
        if max_completion_tokens is not None:
            try:
                client.model_max_completion_tokens = int(max_completion_tokens)
            except Exception:
                pass

        supported_parameters = extract_supported_parameters(payload)
        if supported_parameters is not None:
            try:
                client.model_supported_parameters = list(supported_parameters)
            except Exception:
                pass

    return _remember_context_limit(client, extract_context_limit(payload))
