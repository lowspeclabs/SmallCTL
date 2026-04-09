"""Legacy compatibility shim for `smallctl.client`.

The canonical implementation now lives under `smallctl/client/`.
"""

from __future__ import annotations

from .client import (  # noqa: F401
    OpenAICompatClient,
    SSEStreamer,
    StreamResult,
    TimelineEntry,
    adapter_for_profile,
    chunk_contains_tool_call_delta,
    detect_provider_profile,
    extract_content_fragments,
    extract_context_limit,
    extract_runtime_context_limit,
    extract_thinking_from_tags,
    format_tool_call_text,
    get_provider_adapter,
    maybe_parse_tool_args,
    registered_provider_adapters,
    sanitize_message_for_transport,
    sanitize_messages_for_lmstudio,
    sanitize_messages_for_openrouter,
    sanitize_messages_with_pending_tool_cleanup,
    should_retry_without_stream_options,
)

__all__ = [
    "OpenAICompatClient",
    "SSEStreamer",
    "StreamResult",
    "TimelineEntry",
    "adapter_for_profile",
    "chunk_contains_tool_call_delta",
    "detect_provider_profile",
    "extract_content_fragments",
    "extract_context_limit",
    "extract_runtime_context_limit",
    "extract_thinking_from_tags",
    "format_tool_call_text",
    "get_provider_adapter",
    "maybe_parse_tool_args",
    "registered_provider_adapters",
    "sanitize_message_for_transport",
    "sanitize_messages_for_lmstudio",
    "sanitize_messages_for_openrouter",
    "sanitize_messages_with_pending_tool_cleanup",
    "should_retry_without_stream_options",
]
