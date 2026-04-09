"""Provider adapter contract and shared policy types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StreamPolicy:
    """Provider-specific stream policy knobs."""

    supports_stream_options: bool = True
    first_token_timeout_sec: float = 30.0
    tool_call_continuation_timeout_sec: float = 30.0


class ProviderAdapter(Protocol):
    """Protocol implemented by all provider adapters."""

    name: str
    stream_policy: StreamPolicy

    def sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ...

    def mutate_headers(self, headers: dict[str, str]) -> dict[str, str]:
        ...

    def mutate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def should_retry_without_stream_options(self, exc: Any) -> bool:
        ...
