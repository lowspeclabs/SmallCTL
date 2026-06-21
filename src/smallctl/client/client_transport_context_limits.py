from __future__ import annotations

from typing import Any

from .client_transport_client_lifecycle import _remember_context_limit
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


__all__ = ["_json_size_bytes", "_approx_token_count", "_client_context_limit", "_remember_context_limit"]
