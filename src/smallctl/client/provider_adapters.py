"""Compatibility layer for provider adapter helpers."""

from __future__ import annotations

from typing import Any

from .adapters import get_provider_adapter, registered_provider_adapters
from .adapters.common import sanitize_message_for_transport
from .adapters.common import sanitize_messages_for_lmstudio
from .adapters.common import sanitize_messages_for_openrouter
from .adapters.common import sanitize_messages_with_pending_tool_cleanup
from .adapters.common import should_retry_without_stream_options


def adapter_for_profile(profile: str | None) -> Any:
    """Return the registered provider adapter for a profile."""
    return get_provider_adapter(profile)


__all__ = [
    "adapter_for_profile",
    "get_provider_adapter",
    "registered_provider_adapters",
    "sanitize_message_for_transport",
    "sanitize_messages_with_pending_tool_cleanup",
    "sanitize_messages_for_openrouter",
    "sanitize_messages_for_lmstudio",
    "should_retry_without_stream_options",
]
