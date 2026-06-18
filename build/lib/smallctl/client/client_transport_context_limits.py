from __future__ import annotations

from typing import Any

from .request_budget import (
    approx_token_count as _budget_approx_token_count,
    client_context_limit as _budget_client_context_limit,
    json_size_bytes as _budget_json_size_bytes,
)


def _json_size_bytes(value: Any) -> int:
    return _budget_json_size_bytes(value)


def _approx_token_count(value: Any) -> int:
    return _budget_approx_token_count(value)


def _client_context_limit(client: Any) -> int | None:
    return _budget_client_context_limit(client)


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
