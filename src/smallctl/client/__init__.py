"""Client module for model API interactions.

This module provides:
- OpenAICompatClient: Main client for OpenAI-compatible API endpoints
- Provider adapters: Message sanitization for specific providers
- Chunk parsing: Content and tool call extraction from stream chunks
- Streaming: SSE streaming and non-stream fallback
- Usage: Context limit extraction utilities
"""

from __future__ import annotations

# Data classes (imported from client.py to avoid duplication)
from .client import StreamResult, TimelineEntry, OpenAICompatClient

# Import from submodules
from .chunk_parser import (
    extract_thinking_from_tags,
    extract_content_fragments,
    chunk_contains_tool_call_delta,
    format_tool_call_text,
    maybe_parse_tool_args,
)
from .provider_adapters import (
    adapter_for_profile,
    get_provider_adapter,
    registered_provider_adapters,
    sanitize_message_for_transport,
    sanitize_messages_with_pending_tool_cleanup,
    sanitize_messages_for_openrouter,
    sanitize_messages_for_lmstudio,
    should_retry_without_stream_options,
)
from .streaming import SSEStreamer
from .usage import extract_context_limit, extract_runtime_context_limit
from .usage import detect_provider_profile
from .model_listing import (
    ModelListResult,
    ProviderModel,
    fetch_available_models,
    parse_lmstudio_models,
    parse_ollama_models,
    parse_openai_models,
)

__all__ = [
    # Data classes
    "StreamResult",
    "TimelineEntry",
    # Client
    "OpenAICompatClient",
    # Chunk parsing
    "extract_thinking_from_tags",
    "extract_content_fragments",
    "chunk_contains_tool_call_delta",
    "format_tool_call_text",
    "maybe_parse_tool_args",
    # Provider adapters
    "adapter_for_profile",
    "get_provider_adapter",
    "registered_provider_adapters",
    "sanitize_message_for_transport",
    "sanitize_messages_with_pending_tool_cleanup",
    "sanitize_messages_for_openrouter",
    "sanitize_messages_for_lmstudio",
    "should_retry_without_stream_options",
    # Streaming
    "SSEStreamer",
    # Usage
    "extract_context_limit",
    "extract_runtime_context_limit",
    "detect_provider_profile",
    # Model listing
    "ModelListResult",
    "ProviderModel",
    "fetch_available_models",
    "parse_lmstudio_models",
    "parse_ollama_models",
    "parse_openai_models",
]
