from __future__ import annotations

from typing import Any


def _client_key(client: Any) -> tuple[str, str]:
    return (client.base_url, client.api_key)


def _get_async_client(client: Any) -> Any:
    import httpx
    if httpx is None:
        raise RuntimeError("Dependency missing: httpx")
    key = _client_key(client)
    shared_clients = client._shared_clients
    async_client = shared_clients.get(key)
    if async_client is None:
        async_client = httpx.AsyncClient(timeout=None)
        shared_clients[key] = async_client
    return async_client


async def _reset_async_client(client: Any) -> None:
    key = _client_key(client)
    async_client = client._shared_clients.pop(key, None)
    if async_client is None:
        return
    try:
        await async_client.aclose()
    except Exception:
        pass
