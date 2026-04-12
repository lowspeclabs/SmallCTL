"""Main client for OpenAI-compatible API endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, ClassVar, Literal
from urllib.parse import quote

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from ..logging_utils import RunLogger, log_kv
from ..guards import is_four_b_or_under_model_name
from .chunk_parser import (
    chunk_contains_tool_call_delta,
    extract_thinking_from_tags,
    extract_content_fragments,
    format_tool_call_text,
    merge_reasoning_text,
    maybe_parse_tool_args,
)
from .provider_adapters import get_provider_adapter
from .streaming import SSEStreamer
from .usage import detect_provider_profile, extract_context_limit, extract_runtime_context_limit


@dataclass
class StreamResult:
    """Result of collecting a stream."""
    assistant_text: str = ""
    thinking_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEntry:
    """Single entry in a collected timeline."""
    kind: Literal["assistant", "thinking", "tool_call"]
    content: str
    data: dict[str, Any] = field(default_factory=dict)


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
    LMSTUDIO_SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 45.0
    LMSTUDIO_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 60.0
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
        resolved_provider = str(provider_profile or "auto").strip().lower()
        if resolved_provider == "auto":
            resolved_provider = detect_provider_profile(self.base_url, self.model)
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

    def _resolve_tool_call_continuation_timeout_sec(self, override: float | None) -> float:
        if override is not None:
            return max(1.0, float(override))

        if self.provider_profile == "lmstudio":
            if self.is_small_model:
                return float(self.LMSTUDIO_SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
            return float(self.LMSTUDIO_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)

        timeout = float(self.STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
        if self.is_small_model:
            return min(timeout, float(self.SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC))
        return timeout

    def _client_key(self) -> tuple[str, str]:
        return (self.base_url, self.api_key)

    def _get_async_client(self) -> Any:
        if httpx is None:
            raise RuntimeError("Dependency missing: httpx")
        key = self._client_key()
        client = self._shared_clients.get(key)
        if client is None:
            client = httpx.AsyncClient(timeout=None)
            self._shared_clients[key] = client
        return client

    @classmethod
    async def aclose_shared_clients(cls) -> None:
        clients = list(cls._shared_clients.values())
        cls._shared_clients.clear()
        for client in clients:
            try:
                await client.aclose()
            except Exception:
                pass

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completions with retry logic and provider-specific handling."""
        if httpx is None:
            raise RuntimeError("Dependency missing: httpx")

        # Apply provider-specific message sanitization
        messages = self.adapter.sanitize_messages(messages)

        url = f"{self.base_url}{self.chat_endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers = self.adapter.mutate_headers(headers)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            
        # Only set stream_options when the active adapter supports usage options.
        if self.adapter.stream_policy.supports_stream_options:
            payload["stream_options"] = {"include_usage": True}
        payload = self.adapter.mutate_payload(payload)

        log_kv(
            self.log,
            logging.INFO,
            "chat_request",
            url=url,
            model=self.model,
            message_count=len(messages),
            tool_count=len(tools),
        )
        log_kv(
            self.log,
            logging.INFO,
            "backend_health_check",
            url=url,
            model=self.model,
            provider_profile=self.provider_profile,
            stream=True,
        )
        if self.run_logger:
            self.run_logger.log(
                "chat",
                "request",
                "chat request started",
                url=url,
                model=self.model,
                message_count=len(messages),
                tool_count=len(tools),
            )

        last_error: Exception | None = None
        current_payload = dict(payload)
        client = self._get_async_client()
        request_first_token_timeout_sec = self._request_first_token_timeout_sec(tools)
        streamer = SSEStreamer(
            provider_profile=self.provider_profile,
            first_token_timeout_sec=request_first_token_timeout_sec,
            tool_call_continuation_timeout_sec=self.tool_call_continuation_timeout_sec,
            aggressive_tool_call_timeout=self.is_small_model,
            run_logger=self.run_logger,
            log=self.log,
        )

        for attempt in range(1, self.STREAM_RETRY_ATTEMPTS + 1):
            saw_chunk = False
            saw_tool_call_chunk = False
            retry_after_backend_recovery = False
            try:
                async for event in streamer.stream_sse(client, url, headers, current_payload):
                    if event.get("type") == "backend_first_token_timeout":
                        details = dict(event.get("details") or {})
                        log_kv(
                            self.log,
                            logging.WARNING,
                            "chat_backend_first_token_timeout",
                            attempt=attempt,
                            provider_profile=self.provider_profile,
                            timeout_sec=details.get("timeout_sec"),
                        )
                        if self.run_logger:
                            self.run_logger.log(
                                "chat",
                                "backend_first_token_timeout",
                                "backend stalled before first token",
                                attempt=attempt,
                                provider_profile=self.provider_profile,
                                timeout_sec=details.get("timeout_sec"),
                            )
                        recovery: dict[str, Any] | None = None
                        if self.backend_recovery_handler is not None:
                            recovery = await self.backend_recovery_handler(
                                {
                                    "attempt": attempt,
                                    "provider_profile": self.provider_profile,
                                    "base_url": self.base_url,
                                    "model": self.model,
                                    "details": details,
                                }
                            )
                        if isinstance(recovery, dict) and recovery.get("status") == "recovered":
                            retry_after_backend_recovery = True
                            log_kv(
                                self.log,
                                logging.WARNING,
                                "chat_backend_recovery_succeeded",
                                attempt=attempt,
                                provider_profile=self.provider_profile,
                                action=recovery.get("action"),
                            )
                            if self.run_logger:
                                self.run_logger.log(
                                    "chat",
                                    "backend_recovery_succeeded",
                                    "backend recovery succeeded after first-token timeout",
                                    attempt=attempt,
                                    provider_profile=self.provider_profile,
                                    action=recovery.get("action"),
                                )
                            await self._reset_async_client()
                            client = self._get_async_client()
                            break
                        wedge_details = dict(details)
                        if isinstance(recovery, dict) and recovery:
                            wedge_details["recovery"] = recovery
                        yield {
                            "type": "backend_wedged",
                            "error": "Backend did not emit a first token before timeout",
                            "details": wedge_details,
                        }
                        return
                    if event.get("type") == "chunk":
                        saw_chunk = True
                        if chunk_contains_tool_call_delta(event.get("data", {})):
                            saw_tool_call_chunk = True
                    yield event
                if retry_after_backend_recovery:
                    continue
                return
            except httpx.HTTPStatusError as exc:
                self._log_http_error("chat_stream_http_error", exc)
                if exc.response.status_code == 400:
                    log_kv(self.log, logging.DEBUG, "chat_stream_400_payload", payload=current_payload)
                    if self.run_logger:
                        self.run_logger.log("chat", "400_payload", "request causing 400 error", payload=current_payload)
                # Retry transient 5xx errors
                if exc.response.status_code in {502, 503, 504, 530}:
                    last_error = exc
                    # Clear client to re-establish connection pool if needed
                    await self._reset_async_client()
                    client = self._get_async_client()
                elif self.adapter.should_retry_without_stream_options(exc):
                    if "stream_options" not in current_payload:
                        raise
                    current_payload = dict(current_payload)
                    current_payload.pop("stream_options", None)
                    log_kv(
                        self.log,
                        logging.WARNING,
                        "chat_stream_options_unsupported",
                        status=exc.response.status_code,
                    )
                    if self.run_logger:
                        self.run_logger.log(
                            "chat",
                            "stream_options_unsupported",
                            "retrying stream without usage options",
                            status=exc.response.status_code,
                        )
                    continue
                else:
                    raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if saw_chunk:
                    if self.provider_profile == "lmstudio" or self.is_small_model:
                        if saw_tool_call_chunk or self._is_tool_call_continuation_timeout(exc):
                            log_kv(
                                self.log,
                                logging.WARNING,
                                "chat_stream_incomplete_tool_call",
                                error=str(exc),
                                attempt=attempt,
                            )
                            if self.run_logger:
                                self.run_logger.log(
                                    "chat",
                                    "stream_incomplete_tool_call",
                                    "treating stalled tool call as retryable chunk error",
                                    error=str(exc),
                                    attempt=attempt,
                                )
                            yield {
                                "type": "chunk_error",
                                "error": "Incomplete tool call from provider stream",
                                "details": {
                                    "reason": "tool_call_continuation_timeout",
                                    "attempt": attempt,
                                    "provider_profile": self.provider_profile,
                                    "message": str(exc),
                                },
                            }
                            return
                        log_kv(
                            self.log,
                            logging.WARNING,
                            "chat_stream_stalled_after_chunks",
                            error=str(exc),
                            attempt=attempt,
                        )
                        if self.run_logger:
                            self.run_logger.log(
                                "chat",
                                "stream_stalled_after_chunks",
                                "treating stalled lmstudio stream as complete after partial output",
                                error=str(exc),
                                attempt=attempt,
                            )
                        yield {
                            "type": "stream_ended_without_done",
                            "details": {
                                "reason": "read_timeout_after_chunks",
                                "attempt": attempt,
                                "provider_profile": self.provider_profile,
                                "message": str(exc),
                                "tool_call_stream_active": saw_tool_call_chunk,
                            },
                        }
                        return
                    raise
                last_error = exc
                log_kv(
                    self.log,
                    logging.WARNING,
                    "chat_stream_transport_retry_nonstream",
                    error=str(exc),
                    attempt=attempt,
                )
                if self.run_logger:
                    self.run_logger.log(
                        "chat",
                        "stream_transport_retry_nonstream",
                        "retrying as non-stream chat request",
                        error=str(exc),
                        attempt=attempt,
                    )
                try:
                    async for event in streamer.nonstream_chat(client, url, headers, current_payload):
                        yield event
                    return
                except (httpx.TimeoutException, httpx.TransportError) as fallback_exc:
                    last_error = fallback_exc
                    await self._reset_async_client()
                    client = self._get_async_client()
            if attempt < self.STREAM_RETRY_ATTEMPTS:
                backoff = float(attempt)
                log_kv(
                    self.log,
                    logging.WARNING,
                    "chat_retry_scheduled",
                    attempt=attempt + 1,
                    delay_sec=backoff,
                )
                await asyncio.sleep(backoff)
        if last_error is not None:
            raise last_error

    @staticmethod
    def _is_tool_call_continuation_timeout(exc: Exception) -> bool:
        """Check if exception indicates a tool call continuation timeout."""
        return "tool call continuation" in str(exc).lower()

    def _log_http_error(self, event: str, exc: "httpx.HTTPStatusError") -> None:
        try:
            body = exc.response.text[:1000]
        except Exception:
            body = ""
        log_kv(
            self.log,
            logging.WARNING,
            event,
            status=exc.response.status_code,
            body=body,
        )
        if self.run_logger:
            self.run_logger.log(
                "chat",
                event,
                "chat http error",
                status=exc.response.status_code,
                body=body,
            )

    async def fetch_model_context_limit(self) -> int | None:
        """Fetch the model's context limit from provider APIs."""
        if httpx is None:
            raise RuntimeError("Dependency missing: httpx")
        if not self.runtime_context_probe:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        model_id = self.model
        model_url = f"{self.base_url}/models/{quote(model_id, safe='')}"
        list_url = f"{self.base_url}/models"
        client = self._get_async_client()
        runtime_urls = [f"{self.base_url}/props", f"{self.base_url}/slots"]
        if self.base_url.endswith("/v1"):
            root = self.base_url[: -len("/v1")]
            runtime_urls.extend([f"{root}/props", f"{root}/slots"])
        log_kv(self.log, logging.DEBUG, "context_probe_start", model=model_id, base_url=self.base_url)
        # Prefer runtime server context from llama.cpp-compatible endpoints.
        for runtime_url in runtime_urls:
            try:
                response = await client.get(runtime_url, headers=headers, timeout=10.0)
                if response.status_code < 400:
                    runtime_payload = response.json()
                    runtime_limit = extract_runtime_context_limit(runtime_payload)
                    if runtime_limit:
                        log_kv(self.log, logging.INFO, "context_probe_success", source="runtime", limit=runtime_limit)
                        return runtime_limit
            except Exception:
                pass
        # Prefer model-specific metadata.
        try:
            response = await client.get(model_url, headers=headers, timeout=10.0)
            if response.status_code < 400:
                payload = response.json()
                limit = extract_context_limit(payload)
                if limit:
                    log_kv(self.log, logging.INFO, "context_probe_success", source="model_metadata", limit=limit)
                    return limit
        except Exception:
            pass
        # Fallback to model listing and select by id.
        try:
            response = await client.get(list_url, headers=headers, timeout=10.0)
            if response.status_code >= 400:
                return None
            payload = response.json()
        except Exception:
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return extract_context_limit(payload)

        selected: dict[str, Any] | None = None
        for item in data:
            if isinstance(item, dict) and str(item.get("id", "")) == model_id:
                selected = item
                break
        if selected is None:
            for item in data:
                if isinstance(item, dict) and model_id in str(item.get("id", "")):
                    selected = item
                    break
        if selected is not None:
            return extract_context_limit(selected)
        return extract_context_limit(payload)

    async def _reset_async_client(self) -> None:
        key = self._client_key()
        client = self._shared_clients.pop(key, None)
        if client is None:
            return
        try:
            await client.aclose()
        except Exception:
            pass

    @staticmethod
    def collect_stream(
        chunks: list[dict[str, Any]],
        *,
        reasoning_mode: Literal["auto", "tags", "field", "off"] = "auto",
        thinking_start_tag: str = "<think>",
        thinking_end_tag: str = "</think>",
    ) -> StreamResult:
        """Collect chunks into a structured result.
        
        Aggregates assistant text, thinking text, tool calls, and usage info.
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}

        for item in chunks:
            if not isinstance(item, dict) or item.get("type") != "chunk":
                continue
            data = item.get("data", {})
            if not isinstance(data, dict):
                continue
            next_usage = data.get("usage")
            if isinstance(next_usage, dict):
                usage = next_usage
            choices = data.get("choices") or []
            if not choices:
                continue
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                continue
            delta = first_choice.get("delta", {})
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            for kind, fragment in extract_content_fragments(content):
                if kind == "thinking":
                    reasoning_parts.append(fragment)
                else:
                    text_parts.append(fragment)
            field_reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(field_reasoning, str) and field_reasoning:
                reasoning_parts.append(field_reasoning)

            tc_deltas = delta.get("tool_calls") or []
            for tc in tc_deltas:
                if not isinstance(tc, dict):
                    continue
                try:
                    idx = int(tc.get("index", 0))
                except (TypeError, ValueError):
                    continue
                fn = tc.get("function") or {}
                if not isinstance(fn, dict):
                    continue
                existing = tool_calls.setdefault(
                    idx,
                    {
                        "id": tc.get("id"),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if "id" in tc:
                    existing["id"] = tc["id"]
                if "name" in fn and isinstance(fn["name"], str):
                    existing["function"]["name"] += fn["name"]
                if "arguments" in fn and isinstance(fn["arguments"], str):
                    existing["function"]["arguments"] += fn["arguments"]

        assistant_text = "".join(text_parts)
        tag_assistant, tag_thinking = extract_thinking_from_tags(
            assistant_text,
            thinking_start_tag=thinking_start_tag,
            thinking_end_tag=thinking_end_tag,
        )
        field_thinking = "".join(reasoning_parts)

        thinking_text = ""
        if reasoning_mode == "tags":
            assistant_text = tag_assistant
            thinking_text = tag_thinking
        elif reasoning_mode == "field":
            thinking_text = field_thinking
        elif reasoning_mode == "auto":
            assistant_text = tag_assistant
            thinking_text = merge_reasoning_text(field_thinking, tag_thinking)
        elif reasoning_mode == "off":
            thinking_text = merge_reasoning_text(field_thinking, tag_thinking)

        ordered_tool_calls = [tool_calls[i] for i in sorted(tool_calls.keys())]
        return StreamResult(
            assistant_text=assistant_text,
            thinking_text=thinking_text,
            tool_calls=ordered_tool_calls,
            usage=usage,
        )

    @staticmethod
    def collect_timeline(
        chunks: list[dict[str, Any]],
        *,
        reasoning_mode: Literal["auto", "tags", "field", "off"] = "auto",
        thinking_start_tag: str = "<think>",
        thinking_end_tag: str = "</think>",
    ) -> list[TimelineEntry]:
        """Collect chunks into a timeline of entries.
        
        Preserves the order of thinking, assistant text, and tool calls.
        """
        collector = _TimelineCollector(
            reasoning_mode=reasoning_mode,
            thinking_start_tag=thinking_start_tag,
            thinking_end_tag=thinking_end_tag,
        )
        for item in chunks:
            collector.feed(item)
        return collector.finalize()


class _TimelineCollector:
    """Collects stream chunks into a timeline of entries."""
    
    def __init__(
        self,
        *,
        reasoning_mode: Literal["auto", "tags", "field", "off"],
        thinking_start_tag: str,
        thinking_end_tag: str,
    ) -> None:
        self.reasoning_mode = reasoning_mode
        self.thinking_start_tag = thinking_start_tag
        self.thinking_end_tag = thinking_end_tag
        self.entries: list[TimelineEntry] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self._tool_entry_index: dict[int, int] = {}
        self._tag_pending = ""
        self._inside_thinking_tag = False
        self._auto_prefers_field = False
        self._auto_used_tag_thinking = False

    def feed(self, item: dict[str, Any]) -> None:
        if not isinstance(item, dict) or item.get("type") != "chunk":
            return
        data = item.get("data", {})
        if not isinstance(data, dict):
            return
        choices = data.get("choices") or []
        if not choices:
            return
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return
        delta = first_choice.get("delta", {})
        if not isinstance(delta, dict):
            return

        content = delta.get("content")
        for kind, fragment in extract_content_fragments(content):
            if kind == "thinking":
                self._flush_pending_text()
                if self.reasoning_mode in {"field", "auto"}:
                    if self.reasoning_mode == "auto" and not self._auto_used_tag_thinking:
                        self._auto_prefers_field = True
                    self._append_text("thinking", fragment)
            else:
                self._feed_content(fragment)

        field_reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(field_reasoning, str) and field_reasoning:
            self._flush_pending_text()
            if self.reasoning_mode in {"field", "auto"}:
                if self.reasoning_mode == "auto" and not self._auto_used_tag_thinking:
                    self._auto_prefers_field = True
                self._append_text("thinking", field_reasoning)

        tc_deltas = delta.get("tool_calls") or []
        if tc_deltas:
            self._flush_pending_text()
        for tc in tc_deltas:
            if not isinstance(tc, dict):
                continue
            try:
                idx = int(tc.get("index", 0))
            except (TypeError, ValueError):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue
            existing = self.tool_calls.setdefault(
                idx,
                {
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {"name": "", "arguments": ""},
                },
            )
            if "id" in tc:
                existing["id"] = tc["id"]
            if "name" in fn and isinstance(fn["name"], str):
                existing["function"]["name"] += fn["name"]
            if "arguments" in fn and isinstance(fn["arguments"], str):
                existing["function"]["arguments"] += fn["arguments"]
            self._upsert_tool_call_entry(idx)

    def finalize(self) -> list[TimelineEntry]:
        self._flush_pending_text()
        return self.entries

    def _feed_content(self, text: str) -> None:
        if self.reasoning_mode in {"field", "off"}:
            self._append_text("assistant", text)
            return
        if self.reasoning_mode == "auto" and self._auto_prefers_field:
            self._append_text("assistant", text)
            return
        self._feed_tagged_content(text)

    def _feed_tagged_content(self, text: str) -> None:
        pending = self._tag_pending + text
        self._tag_pending = ""
        cursor = 0
        while cursor < len(pending):
            if self._inside_thinking_tag:
                end = pending.find(self.thinking_end_tag, cursor)
                if end == -1:
                    safe_end = max(cursor, len(pending) - max(0, len(self.thinking_end_tag) - 1))
                    if safe_end > cursor:
                        self._append_text("thinking", pending[cursor:safe_end])
                        self._auto_used_tag_thinking = True
                        cursor = safe_end
                    self._tag_pending = pending[cursor:]
                    return
                if end > cursor:
                    self._append_text("thinking", pending[cursor:end])
                    self._auto_used_tag_thinking = True
                cursor = end + len(self.thinking_end_tag)
                self._inside_thinking_tag = False
                continue

            start = pending.find(self.thinking_start_tag, cursor)
            if start == -1:
                safe_end = max(cursor, len(pending) - max(0, len(self.thinking_start_tag) - 1))
                if safe_end > cursor:
                    self._append_text("assistant", pending[cursor:safe_end])
                    cursor = safe_end
                self._tag_pending = pending[cursor:]
                return
            if start > cursor:
                self._append_text("assistant", pending[cursor:start])
            cursor = start + len(self.thinking_start_tag)
            self._inside_thinking_tag = True

    def _append_text(self, kind: Literal["assistant", "thinking"], text: str) -> None:
        if not text:
            return
        if self.entries and self.entries[-1].kind == kind:
            self.entries[-1].content += text
            return
        self.entries.append(TimelineEntry(kind=kind, content=text))

    def _upsert_tool_call_entry(self, idx: int) -> None:
        payload = self.tool_calls[idx]
        function = payload.get("function", {})
        if not isinstance(function, dict):
            return
        tool_name = str(function.get("name") or "").strip() or "tool_call"
        args_text = str(function.get("arguments") or "")
        args = maybe_parse_tool_args(args_text)
        data: dict[str, Any] = {
            "tool_call_id": payload.get("id"),
            "args_text": args_text,
            "display_text": format_tool_call_text(tool_name, args_text, args),
            "source": "model",
        }
        if isinstance(args, dict):
            data["args"] = args
        entry = TimelineEntry(kind="tool_call", content=tool_name, data=data)
        existing_index = self._tool_entry_index.get(idx)
        if existing_index is None:
            self._tool_entry_index[idx] = len(self.entries)
            self.entries.append(entry)
            return
        self.entries[existing_index] = entry

    def _flush_pending_text(self) -> None:
        if not self._tag_pending:
            return
        kind = "thinking" if self._inside_thinking_tag and self.reasoning_mode != "off" else "assistant"
        self._append_text(kind, self._tag_pending)
        if kind == "thinking":
            self._auto_used_tag_thinking = True
        self._tag_pending = ""
