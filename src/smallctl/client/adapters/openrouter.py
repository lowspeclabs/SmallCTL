"""OpenRouter provider adapter."""

from __future__ import annotations

from typing import Any

from .base import StreamPolicy
from .common import sanitize_messages_for_openrouter
from .common import should_retry_without_stream_options as _retry_without_stream_options


class OpenRouterAdapter:
    name = "openrouter"
    stream_policy = StreamPolicy(
        supports_stream_options=False,
        first_token_timeout_sec=30.0,
        tool_call_continuation_timeout_sec=30.0,
    )

    def sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sanitize_messages_for_openrouter(messages)

    def mutate_headers(self, headers: dict[str, str]) -> dict[str, str]:
        mutated = dict(headers)
        mutated["HTTP-Referer"] = "https://github.com/hocus-pocus/smallctl"
        mutated["X-Title"] = "smallctl"
        return mutated

    def mutate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    def should_retry_without_stream_options(self, exc: Any) -> bool:
        return _retry_without_stream_options(exc)
