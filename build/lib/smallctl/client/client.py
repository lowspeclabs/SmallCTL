"""Main client for OpenAI-compatible API endpoints."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Awaitable, Callable, ClassVar, Literal

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from ..logging_utils import RunLogger, log_kv
from ..guards import is_four_b_or_under_model_name
from ..provider_profiles import resolve_provider_profile
from .provider_adapters import get_provider_adapter
from .stream_collectors import StreamResult, TimelineEntry, collect_stream, collect_timeline
from .usage import extract_context_limit, extract_runtime_context_limit
from .client_transport import _DEFAULT_MAX_COMPLETION_TOKENS
from .client_transport import fetch_model_context_limit, stream_chat
from .request_budget import client_context_limit as _request_budget_client_context_limit


class OpenAICompatClient:
    """Client for OpenAI-compatible chat completion APIs."""

    STREAM_RETRY_ATTEMPTS = 3
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
    WRITE_HEAVY_TOOL_CALL_CONTINUATION_TIMEOUT_MULTIPLIER = 2.0
    WRITE_HEAVY_FIRST_TOKEN_TIMEOUT_MULTIPLIER = 2.0
    WRITE_HEAVY_MAX_COMPLETION_TOKENS_MULTIPLIER = 2.0
    OPENROUTER_AUTO_MAX_COMPLETION_TOKENS = _DEFAULT_MAX_COMPLETION_TOKENS
    _WRITE_HEAVY_TOOL_NAMES: ClassVar[set[str]] = {
        "file_write",
        "file_append",
        "file_patch",
        "ast_patch",
        "ssh_file_write",
        "ssh_file_patch",
        "ssh_file_replace_between",
    }
    _WRITE_HEAVY_ARGUMENT_FIELDS: ClassVar[set[str]] = {
        "content",
        "replacement_text",
        "target_text",
        "start_text",
        "end_text",
        "patch",
        "diff",
    }
    _shared_clients: ClassVar[dict[tuple[str, str], Any]] = {}

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        chat_endpoint: str = "/chat/completions",
        provider_profile: str = "generic",
        first_token_timeout_sec: float | None = None,
        tool_call_continuation_timeout_sec: float | None = None,
        runtime_context_probe: bool = True,
        run_logger: RunLogger | None = None,
        backend_recovery_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.log = logging.getLogger("smallctl.client")
        self.run_logger = run_logger
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "none"
        self.chat_endpoint = chat_endpoint if chat_endpoint.startswith("/") else f"/{chat_endpoint}"
        resolved_provider, provider_profile_warnings = resolve_provider_profile(
            self.base_url,
            self.model,
            provider_profile,
        )
        for warning in provider_profile_warnings:
            log_kv(
                self.log,
                logging.WARNING,
                "provider_profile_resolved",
                warning=warning,
                endpoint=self.base_url,
                model=self.model,
            )
            if self.run_logger:
                self.run_logger.log(
                    "chat",
                    "provider_profile_resolved",
                    "provider profile normalization/update applied",
                    warning=warning,
                    endpoint=self.base_url,
                    model=self.model,
                )
        self.provider_profile = resolved_provider
        self.adapter = get_provider_adapter(self.provider_profile)
        self.is_small_model = is_four_b_or_under_model_name(self.model)
        self.first_token_timeout_sec = self._resolve_first_token_timeout_sec(first_token_timeout_sec)
        self.tool_call_continuation_timeout_sec = self._resolve_tool_call_continuation_timeout_sec(
            tool_call_continuation_timeout_sec
        )
        self.runtime_context_probe = runtime_context_probe
        self.backend_recovery_handler = backend_recovery_handler

    def _resolve_first_token_timeout_sec(self, override: float | None) -> float:
        if override is not None:
            return max(1.0, float(override))
        adapter_timeout = float(self.adapter.stream_policy.first_token_timeout_sec)
        if adapter_timeout > 0:
            return adapter_timeout
        if self.provider_profile == "lmstudio":
            return float(self.LMSTUDIO_FIRST_TOKEN_TIMEOUT_SEC)
        return float(self.STREAM_FIRST_TOKEN_TIMEOUT_SEC)

    def _request_first_token_timeout_sec(self, tools: list[dict[str, Any]]) -> float:
        """Increase the first-token watchdog for heavy LM Studio tool requests.

        Larger local models can take noticeably longer to emit the first stream
        chunk once the prompt includes the full tool schema. The baseline LM
        Studio timeout remains intentionally short for plain chat, but we relax
        it for tool-bearing requests on non-small models to avoid false
        "backend wedged" failures.
        """
        timeout = float(self.first_token_timeout_sec)
        if self._request_has_write_heavy_tool(tools):
            return max(timeout, timeout * float(self.WRITE_HEAVY_FIRST_TOKEN_TIMEOUT_MULTIPLIER))
        if self.provider_profile != "lmstudio":
            return timeout
        # Tool-bearing requests on LM Studio often need more time for prompt ingestion,
        # even for small models, due to the complexity of the schema and system prompt.
        # We skip the early return for small models if tools are present.
        if self.is_small_model and not tools:
            return timeout

        tool_count = len(tools)
        if tool_count >= 12:
            return max(timeout, 60.0)
        if tool_count > 0:
            normalized_model = str(self.model or "").strip().lower()
            if "gemma" in normalized_model:
                return max(timeout, 60.0)
        return timeout

    def _request_has_write_heavy_tool(self, tools: list[dict[str, Any]]) -> bool:
        if not tools:
            return False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            tool_name = str(function.get("name") or "").strip()
            parameters = function.get("parameters")
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            property_names = {str(name).strip() for name in properties} if isinstance(properties, dict) else set()
            if tool_name in self._WRITE_HEAVY_TOOL_NAMES:
                return True
            if property_names & self._WRITE_HEAVY_ARGUMENT_FIELDS:
                return True
        return False

    def _resolve_tool_call_continuation_timeout_sec(self, override: float | None) -> float:
        if override is not None:
            return max(1.0, float(override))

        if self.provider_profile == "lmstudio":
            if self.is_small_model:
                return float(self.LMSTUDIO_SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
            return float(self.LMSTUDIO_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)

        adapter_timeout = float(self.adapter.stream_policy.tool_call_continuation_timeout_sec)
        if self.adapter.name != "generic" and adapter_timeout > 0:
            return adapter_timeout

        timeout = float(self.STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
        if self.is_small_model:
            return min(timeout, float(self.SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC))
        return timeout

    def _request_tool_call_continuation_timeout_sec(self, tools: list[dict[str, Any]]) -> float:
        timeout = float(self.tool_call_continuation_timeout_sec)
        if not self._request_has_write_heavy_tool(tools):
            return timeout
        return max(timeout, timeout * float(self.WRITE_HEAVY_TOOL_CALL_CONTINUATION_TIMEOUT_MULTIPLIER))

    def _resolve_base_max_completion_tokens(self) -> int:
        explicit = getattr(self, "max_completion_tokens", None)
        try:
            explicit_tokens = int(explicit)
        except (TypeError, ValueError):
            explicit_tokens = 0
        if explicit_tokens > 0:
            return explicit_tokens
        if self.provider_profile == "openrouter":
            return int(self.OPENROUTER_AUTO_MAX_COMPLETION_TOKENS)
        limit = _request_budget_client_context_limit(self)
        if limit:
            return max(1, limit // 2)
        return _DEFAULT_MAX_COMPLETION_TOKENS

    def _request_supports_parameter(self, parameter_name: str) -> bool:
        supported = getattr(self, "model_supported_parameters", None)
        if not isinstance(supported, list):
            return True
        expected = self._normalize_request_parameter_name(parameter_name)
        normalized = {
            self._normalize_request_parameter_name(str(item or ""))
            for item in supported
            if str(item or "").strip()
        }
        return expected in normalized

    @staticmethod
    def _normalize_request_parameter_name(value: str) -> str:
        text = str(value or "").strip().replace("-", "_")
        normalized: list[str] = []
        for index, char in enumerate(text):
            if char.isupper() and index > 0 and normalized and normalized[-1] != "_":
                normalized.append("_")
            normalized.append(char.lower())
        return "".join(normalized)

    def _metadata_max_completion_tokens(self) -> int | None:
        try:
            value = int(getattr(self, "model_max_completion_tokens", None) or 0)
        except (TypeError, ValueError):
            return None
        if value > 0:
            return value
        return None

    def _request_max_completion_tokens(self, tools: list[dict[str, Any]]) -> int | None:
        if not self._request_supports_parameter("max_tokens"):
            return None
        limit = _request_budget_client_context_limit(self)
        metadata_limit = self._metadata_max_completion_tokens()
        base = metadata_limit or self._resolve_base_max_completion_tokens()
        if not tools:
            return base

        write_heavy = False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            tool_name = str(function.get("name") or "").strip()
            parameters = function.get("parameters")
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            property_names = {str(name).strip() for name in properties} if isinstance(properties, dict) else set()
            if tool_name in self._WRITE_HEAVY_TOOL_NAMES:
                write_heavy = True
                break
            if property_names & self._WRITE_HEAVY_ARGUMENT_FIELDS:
                write_heavy = True
                break

        if not write_heavy:
            return base

        candidate = int(base * float(self.WRITE_HEAVY_MAX_COMPLETION_TOKENS_MULTIPLIER))
        if metadata_limit:
            ceiling = metadata_limit
        else:
            ceiling = limit // 2 if limit else _DEFAULT_MAX_COMPLETION_TOKENS * 2
        return min(max(base, candidate), ceiling)

    @classmethod
    async def aclose_shared_clients(cls) -> None:
        clients = list(cls._shared_clients.values())
        cls._shared_clients.clear()
        for client in clients:
            try:
                await client.aclose()
            except Exception:
                pass

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        return stream_chat(self, messages, tools)

    async def fetch_model_context_limit(self) -> int | None:
        return await fetch_model_context_limit(self)

    @staticmethod
    def collect_stream(
        chunks: list[dict[str, Any]],
        *,
        reasoning_mode: Literal["auto", "tags", "field", "off"] = "auto",
        thinking_start_tag: str = "<think>",
        thinking_end_tag: str = "</think>",
    ) -> StreamResult:
        return collect_stream(
            chunks,
            reasoning_mode=reasoning_mode,
            thinking_start_tag=thinking_start_tag,
            thinking_end_tag=thinking_end_tag,
        )

    @staticmethod
    def collect_timeline(
        chunks: list[dict[str, Any]],
        *,
        reasoning_mode: Literal["auto", "tags", "field", "off"] = "auto",
        thinking_start_tag: str = "<think>",
        thinking_end_tag: str = "</think>",
    ) -> list[TimelineEntry]:
        return collect_timeline(
            chunks,
            reasoning_mode=reasoning_mode,
            thinking_start_tag=thinking_start_tag,
            thinking_end_tag=thinking_end_tag,
        )
