from __future__ import annotations

import logging
from typing import Any

from ..logging_utils import log_kv
from .client_transport_helpers import provider_root as _provider_root
from .transport_error_classification import _openrouter_auth_failure_details


async def _preflight_openrouter_auth(client: Any, async_client: Any) -> dict[str, Any] | None:
    if client.provider_profile != "openrouter":
        return None
    normalized_base = str(client.base_url or "").strip().rstrip("/")
    if normalized_base.endswith("/api/v1"):
        credits_url = f"{normalized_base}/credits"
    else:
        credits_url = f"{_provider_root(normalized_base)}/api/v1/credits"
    headers = client.adapter.mutate_headers(
        {
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json",
        }
    )
    try:
        response = await async_client.get(credits_url, headers=headers)
    except Exception as exc:
        log_kv(
            client.log,
            logging.WARNING,
            "openrouter_auth_preflight_error",
            error=str(exc),
            url=credits_url,
        )
        if client.run_logger:
            client.run_logger.log(
                "chat",
                "openrouter_auth_preflight_error",
                "OpenRouter auth preflight could not be completed",
                error=str(exc),
                url=credits_url,
            )
        return None
    if int(response.status_code) != 401:
        return None
    body = str(getattr(response, "text", "") or "")[:1000]
    details = _openrouter_auth_failure_details(
        client,
        status_code=401,
        body=body,
        phase="auth_preflight",
    )
    log_kv(client.log, logging.ERROR, "openrouter_auth_failed", **details)
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "openrouter_auth_failed",
            "OpenRouter authentication failed during preflight",
            **details,
        )
    return details
