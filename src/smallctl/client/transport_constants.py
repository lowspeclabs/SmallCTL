from __future__ import annotations

import re
from typing import Any

_LLAMACPP_CONTEXT_OVERFLOW_RE = re.compile(
    r"request\s*\((?P<request_tokens>\d+)\s+tokens?\)\s+exceeds\s+the\s+available\s+context\s+size\s*\((?P<context_tokens>\d+)\s+tokens?\)",
    re.IGNORECASE,
)
_LOCAL_WRITE_INTENT_RE = re.compile(
    r"\b(build|create|implement|write|generate|add|make)\b.*\b(file|script|module|\.py|\.js|\.ts|\.md|\.txt)\b"
    r"|\b(file|script|module)\b.*\b(build|create|implement|write|generate|add|make)\b",
    re.IGNORECASE | re.DOTALL,
)
_LOCAL_PATCH_INTENT_RE = re.compile(
    r"\b(fix|patch|update|modify|edit|change|repair|refactor)\b",
    re.IGNORECASE,
)
_UNSET = object()
_DEFAULT_MAX_COMPLETION_TOKENS = 2048

STREAM_CONNECT_TIMEOUT_SEC = 10.0
STREAM_WRITE_TIMEOUT_SEC = 30.0
STREAM_READ_TIMEOUT_SEC = 120.0
STREAM_FIRST_TOKEN_TIMEOUT_SEC = 30.0
LMSTUDIO_FIRST_TOKEN_TIMEOUT_SEC = 45.0
STREAM_POOL_TIMEOUT_SEC = 30.0
STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 30.0
SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 12.0
LMSTUDIO_SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 135.0
LMSTUDIO_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 90.0


def resolve_first_token_timeout_sec(
    override: float | None,
    adapter: Any,
    provider_profile: str,
) -> float:
    if override is not None:
        return max(1.0, float(override))
    adapter_timeout = float(adapter.stream_policy.first_token_timeout_sec)
    if adapter_timeout > 0:
        return adapter_timeout
    if provider_profile == "lmstudio":
        return float(LMSTUDIO_FIRST_TOKEN_TIMEOUT_SEC)
    return float(STREAM_FIRST_TOKEN_TIMEOUT_SEC)


def resolve_tool_call_continuation_timeout_sec(
    override: float | None,
    adapter: Any,
    provider_profile: str,
    is_small_model: bool,
) -> float:
    if override is not None:
        return max(1.0, float(override))

    if provider_profile == "lmstudio":
        if is_small_model:
            return float(LMSTUDIO_SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
        return float(LMSTUDIO_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)

    adapter_timeout = float(adapter.stream_policy.tool_call_continuation_timeout_sec)
    if adapter.name != "generic" and adapter_timeout > 0:
        return adapter_timeout

    timeout = float(STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
    if is_small_model:
        return min(timeout, float(SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC))
    return timeout
