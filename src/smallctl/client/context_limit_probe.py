from __future__ import annotations

from typing import Any

from ..logging_utils import log_kv
from .usage import extract_runtime_context_limit


async def fetch_model_context_limit(client: Any) -> int | None:
    if not client.runtime_context_probe:
        return None
    try:
        import httpx
    except Exception:  # pragma: no cover
        raise RuntimeError("Dependency missing: httpx")

    from urllib.parse import quote

    headers = {"Authorization": f"Bearer {client.api_key}"}
    model_id = client.model
    model_url = f"{client.base_url}/models/{quote(model_id, safe='')}"  
    list_url = f"{client.base_url}/models"

    # Reuse the shared async client from client_transport
    from .client_transport import _get_async_client, _remember_context_limit, _remember_model_metadata
    async_client = _get_async_client(client)

    runtime_urls = [f"{client.base_url}/props", f"{client.base_url}/slots"]
    if client.base_url.endswith("/v1"):
        root = client.base_url[: -len("/v1")]
        runtime_urls.extend([f"{root}/props", f"{root}/slots"])

    log_kv(client.log, log_level=10, event="context_probe_start", model=model_id, base_url=client.base_url)
    for runtime_url in runtime_urls:
        try:
            response = await async_client.get(runtime_url, headers=headers, timeout=10.0)
            if response.status_code < 400:
                runtime_payload = response.json()
                runtime_limit = extract_runtime_context_limit(runtime_payload)
                if runtime_limit:
                    log_kv(client.log, log_level=20, event="context_probe_success", source="runtime", limit=runtime_limit)
                    return _remember_context_limit(client, runtime_limit)
        except Exception:
            pass
    try:
        response = await async_client.get(model_url, headers=headers, timeout=10.0)
        if response.status_code < 400:
            payload = response.json()
            limit = _remember_model_metadata(client, payload, source="model_metadata")
            if limit:
                log_kv(client.log, log_level=20, event="context_probe_success", source="model_metadata", limit=limit)
                return limit
    except Exception:
        pass
    try:
        response = await async_client.get(list_url, headers=headers, timeout=10.0)
        if response.status_code >= 400:
            return None
        payload = response.json()
    except Exception:
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return _remember_model_metadata(client, payload, source="model_list")

    selected: dict[str, Any] | None = None
    for item in data:
        if isinstance(item, dict) and str(item.get("id", "")) == model_id:
            selected = item
            break
    if selected is None:
        for item in data:
            if isinstance(item, dict) and model_id in str(item.get("id", "")):
                selected = item
                break
    if selected is not None:
        return _remember_model_metadata(client, selected, source="model_list")
    return _remember_model_metadata(client, payload, source="model_list")
