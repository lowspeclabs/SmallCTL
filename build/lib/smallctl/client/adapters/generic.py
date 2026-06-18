"""Generic provider adapter."""

from __future__ import annotations

from typing import Any

from .base import StreamPolicy
from .common import sanitize_message_for_transport
from .common import should_retry_without_stream_options as _retry_without_stream_options


class GenericAdapter:
    name = "generic"
    stream_policy = StreamPolicy(
        supports_stream_options=True,
        first_token_timeout_sec=30.0,
        tool_call_continuation_timeout_sec=30.0,
    )

    def sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [sanitize_message_for_transport(message) for message in messages]

    def mutate_headers(self, headers: dict[str, str]) -> dict[str, str]:
        return dict(headers)

    def mutate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    def should_retry_without_stream_options(self, exc: Any) -> bool:
        return _retry_without_stream_options(exc)
