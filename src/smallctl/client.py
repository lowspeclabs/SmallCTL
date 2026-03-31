from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar, Literal
from urllib.parse import quote

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from .logging_utils import RunLogger, log_kv


@dataclass
class StreamResult:
    assistant_text: str = ""
    thinking_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEntry:
    kind: Literal["assistant", "thinking", "tool_call"]
    content: str
    data: dict[str, Any] = field(default_factory=dict)


class OpenAICompatClient:
    STREAM_RETRY_ATTEMPTS = 3
    STREAM_CONNECT_TIMEOUT_SEC = 10.0
    STREAM_WRITE_TIMEOUT_SEC = 30.0
    STREAM_READ_TIMEOUT_SEC = 60.0
    STREAM_POOL_TIMEOUT_SEC = 30.0
    STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 8.0
    _shared_clients: ClassVar[dict[tuple[str, str], Any]] = {}

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        chat_endpoint: str = "/chat/completions",
        provider_profile: str = "generic",
        runtime_context_probe: bool = True,
        run_logger: RunLogger | None = None,
    ) -> None:
        self.log = logging.getLogger("smallctl.client")
        self.run_logger = run_logger
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "none"
        self.chat_endpoint = chat_endpoint if chat_endpoint.startswith("/") else f"/{chat_endpoint}"
        self.provider_profile = str(provider_profile or "generic").strip().lower()
        self.runtime_context_probe = runtime_context_probe

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
        if httpx is None:
            raise RuntimeError("Dependency missing: httpx")

        if self.provider_profile == "openrouter":
            messages = self._sanitize_messages_for_openrouter(messages)
        elif self.provider_profile == "lmstudio":
            messages = self._sanitize_messages_for_lmstudio(messages)
        else:
            messages = [self._sanitize_message_for_transport(message) for message in messages]
        url = f"{self.base_url}{self.chat_endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider_profile == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/hocus-pocus/smallctl"
            headers["X-Title"] = "smallctl"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            
        # Optional: Disable stream_options for providers that don't support it.
        # OpenRouter and LM Studio can reject or silently ignore include_usage.
        if self.provider_profile not in {"openrouter", "lmstudio"}:
            payload["stream_options"] = {"include_usage": True}
        log_kv(
            self.log,
            logging.INFO,
            "chat_request",
            url=url,
            model=self.model,
            message_count=len(messages),
            tool_count=len(tools),
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
        for attempt in range(1, self.STREAM_RETRY_ATTEMPTS + 1):
            saw_chunk = False
            saw_tool_call_chunk = False
            try:
                async for event in self._stream_sse(client, url, headers, current_payload):
                    if event.get("type") == "chunk":
                        saw_chunk = True
                        if self._chunk_contains_tool_call_delta(event):
                            saw_tool_call_chunk = True
                    yield event
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
                elif self._should_retry_without_stream_options(exc):
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
                    if self.provider_profile == "lmstudio":
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
                                    "treating stalled lmstudio tool call as retryable chunk error",
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
                    async for event in self._nonstream_chat(client, url, headers, current_payload):
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
    def _chunk_contains_tool_call_delta(event: dict[str, Any]) -> bool:
        data = event.get("data", {})
        if not isinstance(data, dict):
            return False
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return False
        delta = choices[0].get("delta") or {}
        return isinstance(delta, dict) and bool(delta.get("tool_calls"))

    @staticmethod
    def _is_tool_call_continuation_timeout(exc: Exception) -> bool:
        return "tool call continuation" in str(exc).lower()

    @staticmethod
    def _sanitize_message_for_transport(message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role", "user"))
        sanitized: dict[str, Any] = {"role": role, "content": message.get("content")}
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                sanitized["tool_calls"] = tool_calls
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                sanitized["tool_call_id"] = tool_call_id
        else:
            name = message.get("name")
            if name:
                sanitized["name"] = name
        return sanitized

    @staticmethod
    def _sanitize_messages_with_pending_tool_cleanup(
        messages: list[dict[str, Any]],
        *,
        rewrite_orphan_tool_messages: bool = False,
    ) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        pending_tool_call_ids: set[str] = set()
        pending_assistant_index: int | None = None

        def close_pending_tool_block() -> None:
            nonlocal pending_tool_call_ids, pending_assistant_index
            if pending_assistant_index is not None and pending_tool_call_ids:
                sanitized[pending_assistant_index].pop("tool_calls", None)
            pending_tool_call_ids = set()
            pending_assistant_index = None

        for message in messages:
            role = str(message.get("role", "user"))
            if role == "assistant":
                if pending_tool_call_ids:
                    close_pending_tool_block()
                sanitized_message = OpenAICompatClient._sanitize_message_for_transport(message)
                sanitized.append(sanitized_message)
                tool_calls = sanitized_message.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    pending_tool_call_ids = {
                        str(tool_call.get("id"))
                        for tool_call in tool_calls
                        if isinstance(tool_call, dict) and tool_call.get("id")
                    }
                    if pending_tool_call_ids:
                        pending_assistant_index = len(sanitized) - 1
                continue

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "")
                if pending_tool_call_ids and tool_call_id and tool_call_id in pending_tool_call_ids:
                    sanitized.append(OpenAICompatClient._sanitize_message_for_transport(message))
                    pending_tool_call_ids.discard(tool_call_id)
                    if not pending_tool_call_ids:
                        pending_assistant_index = None
                else:
                    if rewrite_orphan_tool_messages:
                        sanitized.append(OpenAICompatClient._rewrite_orphan_tool_message_for_openrouter(message))
                    else:
                        sanitized.append(OpenAICompatClient._sanitize_message_for_transport(message))
                continue

            if pending_tool_call_ids:
                close_pending_tool_block()
            sanitized.append(OpenAICompatClient._sanitize_message_for_transport(message))

        if pending_tool_call_ids:
            close_pending_tool_block()
        return sanitized

    @staticmethod
    def _sanitize_messages_for_openrouter(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return OpenAICompatClient._sanitize_messages_with_pending_tool_cleanup(
            messages,
            rewrite_orphan_tool_messages=True,
        )

    @staticmethod
    def _sanitize_messages_for_lmstudio(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep LM Studio payloads chat-shaped and end with a real user query.

        Some LM Studio prompt templates require a final user query to exist in the
        rendered message list. When a turn ends on assistant/tool context, we append
        a lightweight recap user message so the template has a concrete user query
        to anchor on.
        """
        sanitized = OpenAICompatClient._sanitize_messages_with_pending_tool_cleanup(messages)
        if sanitized:
            last_message = sanitized[-1]
            if (
                str(last_message.get("role", "")).strip() == "user"
                and len(str(last_message.get("content") or "").strip()) >= 20
            ):
                return sanitized
        else:
            return sanitized

        latest_user_content = ""
        for message in reversed(sanitized):
            if str(message.get("role", "")).strip() != "user":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                latest_user_content = content
                break

        if not latest_user_content:
            return sanitized

        recap = (
            "Continue working on the user's request. "
            f"User query recap: {latest_user_content}"
        )
        sanitized.append({"role": "user", "content": recap})
        return sanitized

    @staticmethod
    def _rewrite_orphan_tool_message_for_openrouter(message: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(message.get("name") or "").strip()
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content.strip()
        else:
            content = "" if raw_content is None else str(raw_content)

        prefix = "[OpenRouter compatibility] Orphan tool result"
        if tool_name:
            prefix = f"{prefix} from {tool_name}"
        if tool_call_id:
            prefix = f"{prefix} (tool_call_id={tool_call_id})"
        if content:
            content = f"{prefix}:\n{content}"
        else:
            content = f"{prefix}."
        return {"role": "user", "content": content}

    async def fetch_model_context_limit(self) -> int | None:
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
                    runtime_limit = self._extract_runtime_context_limit(runtime_payload)
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
                limit = self._extract_context_limit(payload)
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
            return self._extract_context_limit(payload)

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
            return self._extract_context_limit(selected)
        return self._extract_context_limit(payload)

    async def _stream_sse(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        chunk_count = 0
        saw_done = False
        tool_call_stream_active = False
        timeout = httpx.Timeout(
            connect=self.STREAM_CONNECT_TIMEOUT_SEC,
            read=self.STREAM_READ_TIMEOUT_SEC,
            write=self.STREAM_WRITE_TIMEOUT_SEC,
            pool=self.STREAM_POOL_TIMEOUT_SEC,
        )
        async with client.stream("POST", url, headers=headers, json=payload, timeout=timeout) as response:
            response.raise_for_status()
            line_iter = response.aiter_lines()
            while True:
                read_timeout = self._next_stream_read_timeout(
                    tool_call_stream_active=tool_call_stream_active,
                )
                try:
                    raw_line = await asyncio.wait_for(line_iter.__anext__(), timeout=read_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    reason = "tool call continuation" if tool_call_stream_active else "stream data"
                    message = f"timed out waiting for {reason}"
                    if tool_call_stream_active:
                        log_kv(
                            self.log,
                            logging.WARNING,
                            "chat_stream_tool_call_watchdog_timeout",
                            chunk_count=chunk_count,
                            timeout_sec=read_timeout,
                        )
                        if self.run_logger:
                            self.run_logger.log(
                                "chat",
                                "tool_call_watchdog_timeout",
                                "tool call stream stalled before arguments completed",
                                chunk_count=chunk_count,
                                timeout_sec=read_timeout,
                            )
                    raise httpx.ReadTimeout(message) from exc
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    saw_done = True
                    log_kv(
                        self.log,
                        logging.INFO,
                        "chat_stream_complete",
                        chunk_count=chunk_count,
                    )
                    if self.run_logger:
                        self.run_logger.log(
                            "chat",
                            "stream_complete",
                            "chat stream completed",
                            chunk_count=chunk_count,
                    )
                    yield {"type": "done"}
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    log_kv(self.log, logging.DEBUG, "chat_stream_decode_error", raw=chunk)
                    if self.run_logger:
                        self.run_logger.log(
                            "chat",
                            "decode_error",
                            "unable to decode chunk",
                            raw=chunk,
                    )
                    continue
                chunk_count += 1
                if _chunk_contains_tool_call_delta(obj):
                    tool_call_stream_active = True
                if isinstance(obj, dict) and "error" in obj:
                    error_info = obj["error"]
                    if isinstance(error_info, dict):
                        message = str(
                            error_info.get("message")
                            or error_info.get("error")
                            or error_info.get("detail")
                            or "Provider returned error"
                        ).strip()
                        details = dict(error_info)
                    else:
                        message = str(error_info).strip()
                        details = {"message": message}
                    if not message:
                        message = "Provider returned error"
                    log_kv(
                        self.log,
                        logging.ERROR,
                        "chat_stream_chunk_error",
                        error=message,
                        details=details,
                    )
                    if self.run_logger:
                        self.run_logger.log(
                            "chat",
                            "chunk_error",
                            "server reported error in stream chunk",
                            error=message,
                            details=details,
                        )
                    # Yield a retryable error event instead of raising immediately.
                    # The outer stream_chat retry loop will re-attempt the full call
                    # rather than crashing the run on transient upstream errors
                    # (e.g. Venice "list index out of range" mid-stream).
                    yield {"type": "chunk_error", "error": message, "details": details}
                    return

                if self.run_logger:
                    self.run_logger.log(
                        "chat",
                        "chunk",
                        "chat stream chunk",
                        index=chunk_count,
                        chunk=obj,
                    )
                yield {"type": "chunk", "data": obj}
        if not saw_done:
            log_kv(
                self.log,
                logging.WARNING,
                "chat_stream_ended_without_done",
                chunk_count=chunk_count,
            )
            if self.run_logger:
                self.run_logger.log(
                    "chat",
                    "stream_ended_without_done",
                    "chat stream ended without [DONE]",
                    chunk_count=chunk_count,
                )

    def _next_stream_read_timeout(self, *, tool_call_stream_active: bool) -> float:
        if self.provider_profile == "lmstudio" and tool_call_stream_active:
            return min(
                float(self.STREAM_READ_TIMEOUT_SEC),
                float(self.STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC),
            )
        return float(self.STREAM_READ_TIMEOUT_SEC)

    async def _nonstream_chat(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        fallback_payload = dict(payload)
        fallback_payload["stream"] = False
        fallback_payload.pop("stream_options", None)
        response = await client.post(url, headers=headers, json=fallback_payload)
        response.raise_for_status()
        obj = response.json()
        if self.run_logger:
            self.run_logger.log(
                "chat",
                "nonstream_response",
                "chat non-stream response received",
                chunk=obj,
            )
        yield {"type": "chunk", "data": self._nonstream_response_to_chunk(obj)}
        yield {"type": "done"}

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
    def _nonstream_response_to_chunk(payload: dict[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return payload
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return payload
        message = first_choice.get("message")
        if not isinstance(message, dict):
            return payload
        delta: dict[str, Any] = {}
        content = message.get("content")
        if content is not None:
            delta["content"] = content
        for key in ("reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value:
                delta[key] = value
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            delta["tool_calls"] = tool_calls
        chunk = dict(payload)
        chunk["choices"] = [{**first_choice, "delta": delta}]
        return chunk

    @staticmethod
    def _should_retry_without_stream_options(exc: "httpx.HTTPStatusError") -> bool:
        response = exc.response
        if response.status_code not in {400, 404, 422}:
            return False
        try:
            body = response.text.lower()
        except Exception:
            return True
        if not body.strip():
            return True
        return (
            "stream_options" in body
            or "include_usage" in body
            or "unknown field" in body
            or "unexpected field" in body
            or "extra fields not permitted" in body
        )

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

    @staticmethod
    def _extract_context_limit(payload: Any) -> int | None:
        keys = (
            "context_length",
            "max_context_length",
            "max_position_embeddings",
            "max_model_len",
            "max_seq_len",
            "context_window",
            "num_ctx",
            "n_ctx",
            "ctx_len",
            "max_input_tokens",
            "input_token_limit",
            "prompt_token_limit",
        )
        found: list[int] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in keys:
                        parsed = OpenAICompatClient._parse_positive_int(value)
                        if parsed is not None:
                            found.append(parsed)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        if not found:
            return None
        # Use the largest discovered token window value.
        return max(found)

    @staticmethod
    def _extract_runtime_context_limit(payload: Any) -> int | None:
        candidates: list[int] = []

        if isinstance(payload, dict):
            settings = payload.get("default_generation_settings")
            if isinstance(settings, dict):
                direct = OpenAICompatClient._parse_positive_int(settings.get("n_ctx"))
                if direct is not None:
                    candidates.append(direct)
                params = settings.get("params")
                if isinstance(params, dict):
                    nested = OpenAICompatClient._parse_positive_int(params.get("n_ctx"))
                    if nested is not None:
                        candidates.append(nested)

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"n_ctx", "num_ctx", "context_length", "max_context_length"}:
                        parsed = OpenAICompatClient._parse_positive_int(value)
                        if parsed is not None:
                            candidates.append(parsed)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        if not candidates:
            return None
        # Runtime slot/context values should be bounded and generally consistent.
        return min(candidates)

    @staticmethod
    def _parse_positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if 0 < parsed < 10_000_000:
            return parsed
        return None

    @staticmethod
    def collect_stream(
        chunks: list[dict[str, Any]],
        *,
        reasoning_mode: Literal["auto", "tags", "field", "off"] = "auto",
        thinking_start_tag: str = "<think>",
        thinking_end_tag: str = "</think>",
    ) -> StreamResult:
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
            for kind, fragment in _extract_content_fragments(content):
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
        tag_assistant, tag_thinking = _extract_thinking_from_tags(
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
            if field_thinking:
                thinking_text = field_thinking
            elif tag_thinking:
                thinking_text = tag_thinking
        elif reasoning_mode == "off":
            thinking_text = field_thinking or tag_thinking

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
        collector = _TimelineCollector(
            reasoning_mode=reasoning_mode,
            thinking_start_tag=thinking_start_tag,
            thinking_end_tag=thinking_end_tag,
        )
        for item in chunks:
            collector.feed(item)
        return collector.finalize()


def _extract_thinking_from_tags(
    text: str,
    *,
    thinking_start_tag: str,
    thinking_end_tag: str,
) -> tuple[str, str]:
    if not text or thinking_start_tag not in text:
        if thinking_start_tag == "<think>":
            text = text.replace("<thinking>", thinking_start_tag)
        if thinking_end_tag == "</think>":
            text = text.replace("</thinking>", thinking_end_tag)
        if thinking_start_tag not in text:
            return text, ""
    normalized = text
    if thinking_start_tag == "<think>":
        normalized = normalized.replace("<thinking>", thinking_start_tag)
    if thinking_end_tag == "</think>":
        normalized = normalized.replace("</thinking>", thinking_end_tag)

    assistant_parts: list[str] = []
    thinking_parts: list[str] = []
    cursor = 0

    while True:
        start = normalized.find(thinking_start_tag, cursor)
        if start == -1:
            assistant_parts.append(normalized[cursor:])
            break
        assistant_parts.append(normalized[cursor:start])
        content_start = start + len(thinking_start_tag)
        end = normalized.find(thinking_end_tag, content_start)
        if end == -1:
            thinking_parts.append(normalized[content_start:])
            break
        thinking_parts.append(normalized[content_start:end])
        cursor = end + len(thinking_end_tag)

    return "".join(assistant_parts), "".join(thinking_parts)


def _chunk_contains_tool_call_delta(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


class _TimelineCollector:
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
        for kind, fragment in _extract_content_fragments(content):
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
        args = _maybe_parse_tool_args(args_text)
        data: dict[str, Any] = {
            "tool_call_id": payload.get("id"),
            "args_text": args_text,
            "display_text": _format_tool_call_text(tool_name, args_text, args),
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


def _maybe_parse_tool_args(arguments: str) -> dict[str, Any] | None:
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_tool_call_text(
    tool_name: str,
    args_text: str,
    args: dict[str, Any] | None,
) -> str:
    if isinstance(args, dict):
        if not args:
            return f"{tool_name}()"
        return f"{tool_name}({json.dumps(args, ensure_ascii=True, sort_keys=True)})"
    stripped = args_text.strip()
    if stripped:
        return f"{tool_name}({stripped})"
    return f"{tool_name}()"


def _extract_content_fragments(content: Any) -> list[tuple[Literal["assistant", "thinking"], str]]:
    fragments: list[tuple[Literal["assistant", "thinking"], str]] = []
    _append_content_fragments(fragments, content, default_kind="assistant")
    return fragments


def _append_content_fragments(
    fragments: list[tuple[Literal["assistant", "thinking"], str]],
    content: Any,
    *,
    default_kind: Literal["assistant", "thinking"],
) -> None:
    if isinstance(content, str):
        _append_content_fragment(fragments, default_kind, content)
        return
    if isinstance(content, list):
        for item in content:
            _append_content_fragments(fragments, item, default_kind=default_kind)
        return
    if not isinstance(content, dict):
        return

    kind = _content_fragment_kind(content.get("type"), default_kind=default_kind)
    handled = False
    for key in ("text", "value"):
        value = content.get(key)
        if isinstance(value, str) and value:
            _append_content_fragment(fragments, kind, value)
            handled = True

    nested_content = content.get("content")
    if isinstance(nested_content, (str, list, dict)):
        _append_content_fragments(fragments, nested_content, default_kind=kind)
        handled = True

    for key in ("summary", "parts", "items"):
        nested = content.get(key)
        if isinstance(nested, (str, list, dict)):
            _append_content_fragments(fragments, nested, default_kind=kind)
            handled = True

    if handled:
        return

    if isinstance(content.get("reasoning"), str):
        _append_content_fragment(fragments, "thinking", content["reasoning"])
        return
    if isinstance(content.get("output_text"), str):
        _append_content_fragment(fragments, "assistant", content["output_text"])


def _content_fragment_kind(
    raw_type: Any,
    *,
    default_kind: Literal["assistant", "thinking"],
) -> Literal["assistant", "thinking"]:
    value = str(raw_type or "").strip().lower()
    if value in {"reasoning", "reasoning_content", "thinking", "summary_text"}:
        return "thinking"
    if value in {"text", "output_text", "input_text", "message", "output_message"}:
        return "assistant"
    return default_kind


def _append_content_fragment(
    fragments: list[tuple[Literal["assistant", "thinking"], str]],
    kind: Literal["assistant", "thinking"],
    text: str,
) -> None:
    if not text:
        return
    if fragments and fragments[-1] == (kind, text):
        return
    fragments.append((kind, text))
