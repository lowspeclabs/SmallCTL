"""SSE streaming and non-stream chat implementations."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, AsyncIterator

try:
    import httpx
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    httpx = None

from .chunk_parser import chunk_contains_tool_call_delta
from .client_transport_client_lifecycle import _get_async_client, _reset_async_client
from .provider_adapters import get_provider_adapter
from ..logging_utils import log_kv


def summarize_stream_chunk(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"type": type(obj).__name__}
    choices = obj.get("choices")
    delta: dict[str, Any] = {}
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    raw_delta = first_choice.get("delta") if isinstance(first_choice, dict) else None
    if isinstance(raw_delta, dict):
        delta = raw_delta
    content = delta.get("content")
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    return {
        "content_delta": str(content or "")[:120],
        "reasoning_delta": str(reasoning or "")[:120],
        "tool_call_delta": chunk_contains_tool_call_delta(obj),
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
    }


class SSEStreamer:
    """SSE stream handler for model chat completions."""

    STREAM_CONNECT_TIMEOUT_SEC = 10.0
    STREAM_WRITE_TIMEOUT_SEC = 30.0
    STREAM_READ_TIMEOUT_SEC = 120.0
    STREAM_FIRST_TOKEN_TIMEOUT_SEC = 30.0
    LMSTUDIO_FIRST_TOKEN_TIMEOUT_SEC = 45.0
    STREAM_POOL_TIMEOUT_SEC = 30.0
    STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 30.0
    SMALL_MODEL_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 12.0

    def __init__(
        self,
        provider_profile: str = "generic",
        first_token_timeout_sec: float | None = None,
        tool_call_continuation_timeout_sec: float | None = None,
        aggressive_tool_call_timeout: bool = False,
        run_logger: Any | None = None,
        log: logging.Logger | None = None,
        api_client: Any | None = None,
        prompt_processing_timeout_sec: float | None = None,
    ) -> None:
        self.provider_profile = provider_profile
        self.adapter = get_provider_adapter(provider_profile)
        self.aggressive_tool_call_timeout = bool(aggressive_tool_call_timeout)
        self.first_token_timeout_sec = self._resolve_first_token_timeout_sec(first_token_timeout_sec)
        if tool_call_continuation_timeout_sec is None:
            default_timeout = float(self.adapter.stream_policy.tool_call_continuation_timeout_sec)
            if default_timeout <= 0:
                default_timeout = float(self.STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
            self.tool_call_continuation_timeout_sec = default_timeout
        else:
            self.tool_call_continuation_timeout_sec = max(1.0, float(tool_call_continuation_timeout_sec))
        if prompt_processing_timeout_sec is None:
            adapter_prompt_timeout = float(self.adapter.stream_policy.prompt_processing_timeout_sec)
            self.prompt_processing_timeout_sec = adapter_prompt_timeout if adapter_prompt_timeout > 0 else 0.0
        else:
            self.prompt_processing_timeout_sec = max(0.0, float(prompt_processing_timeout_sec))
        self.run_logger = run_logger
        self.log = log or logging.getLogger("smallctl.client.streaming")
        self.api_client = api_client

    def _resolve_first_token_timeout_sec(self, override: float | None) -> float:
        if override is not None:
            return max(1.0, float(override))
        adapter_timeout = float(self.adapter.stream_policy.first_token_timeout_sec)
        if adapter_timeout > 0:
            return adapter_timeout
        if self.provider_profile == "lmstudio":
            return float(self.LMSTUDIO_FIRST_TOKEN_TIMEOUT_SEC)
        return float(self.STREAM_FIRST_TOKEN_TIMEOUT_SEC)

    def _next_stream_read_timeout(self, *, chunk_count: int = 1, tool_call_stream_active: bool) -> float:
        """Determine the appropriate read timeout for the next stream chunk."""
        if chunk_count == 0:
            return min(
                float(self.STREAM_READ_TIMEOUT_SEC),
                float(self.first_token_timeout_sec),
            )
        if tool_call_stream_active and (self.provider_profile == "lmstudio" or self.aggressive_tool_call_timeout):
            return max(1.0, float(self.tool_call_continuation_timeout_sec))
        return float(self.STREAM_READ_TIMEOUT_SEC)

    async def stream_sse(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream Server-Sent Events from the model endpoint.

        Yields chunk events with parsed JSON data and done events when complete.
        Also handles error chunks and timeout conditions.
        """
        read_timeout_cap = max(
            float(self.STREAM_READ_TIMEOUT_SEC),
            float(self.tool_call_continuation_timeout_sec),
        )
        timeout = httpx.Timeout(
            connect=self.STREAM_CONNECT_TIMEOUT_SEC,
            read=read_timeout_cap,
            write=self.STREAM_WRITE_TIMEOUT_SEC,
            pool=self.STREAM_POOL_TIMEOUT_SEC,
        )
        attempt = 0
        max_attempts = 2
        while attempt < max_attempts:
            attempt += 1
            try:
                async for event in self._stream_sse_once(client, url, headers, payload, timeout):
                    yield event
                return
            except httpx.RemoteProtocolError as exc:
                if attempt < max_attempts:
                    log_kv(
                        self.log,
                        logging.WARNING,
                        "chat_stream_remote_protocol_error_retry",
                        attempt=attempt,
                        error=str(exc),
                        provider_profile=self.provider_profile,
                    )
                    if self.api_client is not None:
                        await _reset_async_client(self.api_client)
                        client = _get_async_client(self.api_client)
                    continue
                yield {
                    "type": "chunk_error",
                    "error": "Remote protocol error after retry",
                    "details": {
                        "reason": "backend_stream_failure",
                        "provider_profile": self.provider_profile,
                        "attempt_count": attempt,
                        "exception_type": "httpx.RemoteProtocolError",
                        "message": str(exc),
                    },
                }
                return

    async def _stream_sse_once(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout,
    ) -> AsyncIterator[dict[str, Any]]:
        chunk_count = 0
        saw_done = False
        tool_call_stream_active = False
        recent_chunks: deque[dict[str, Any]] = deque(maxlen=5)
        headers_received_at: float | None = None
        async with client.stream("POST", url, headers=headers, json=payload, timeout=timeout) as response:
            response.raise_for_status()
            headers_received_at = time.monotonic()
            line_iter = response.aiter_lines()
            while True:
                read_timeout = self._next_stream_read_timeout(
                    chunk_count=chunk_count,
                    tool_call_stream_active=tool_call_stream_active,
                )
                effective_timeout = read_timeout
                if chunk_count == 0 and self.prompt_processing_timeout_sec > 0:
                    effective_timeout = min(read_timeout, self.prompt_processing_timeout_sec)
                try:
                    raw_line = await asyncio.wait_for(line_iter.__anext__(), timeout=effective_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    if chunk_count == 0:
                        # If the backend has already received the headers but has
                        # not emitted any content chunk within the prompt-processing
                        # budget, report a dedicated timeout so the harness can
                        # shrink context instead of waiting for the full first-token
                        # watchdog (which may be much longer).
                        if (
                            self.prompt_processing_timeout_sec > 0
                            and headers_received_at is not None
                            and (time.monotonic() - headers_received_at) >= self.prompt_processing_timeout_sec
                        ):
                            elapsed = round(time.monotonic() - headers_received_at, 3)
                            log_kv(
                                self.log,
                                logging.WARNING,
                                "chat_backend_prompt_processing_timeout",
                                elapsed_sec=elapsed,
                                prompt_processing_timeout_sec=self.prompt_processing_timeout_sec,
                                provider_profile=self.provider_profile,
                            )
                            if self.run_logger:
                                self.run_logger.log(
                                    "chat",
                                    "backend_prompt_processing_timeout",
                                    "backend spent too long processing the prompt before emitting a token",
                                    elapsed_sec=elapsed,
                                    prompt_processing_timeout_sec=self.prompt_processing_timeout_sec,
                                    provider_profile=self.provider_profile,
                                )
                            yield {
                                "type": "backend_prompt_processing_timeout",
                                "error": "Backend spent too long processing the prompt",
                                "details": {
                                    "reason": "prompt_processing_timeout",
                                    "provider_profile": self.provider_profile,
                                    "elapsed_sec": elapsed,
                                    "prompt_processing_timeout_sec": self.prompt_processing_timeout_sec,
                                    "chunk_count": 0,
                                    "last_chunks": [],
                                },
                            }
                            return
                        message = "timed out waiting for first stream token"
                        log_kv(
                            self.log,
                            logging.WARNING,
                            "chat_stream_first_token_timeout",
                            timeout_sec=read_timeout,
                            provider_profile=self.provider_profile,
                        )
                        if self.run_logger:
                            self.run_logger.log(
                                "chat",
                                "first_token_timeout",
                                "chat stream stalled before first token",
                                timeout_sec=read_timeout,
                                provider_profile=self.provider_profile,
                            )
                        yield {
                            "type": "backend_first_token_timeout",
                            "error": "Backend stalled before first token",
                            "details": {
                                "reason": "first_token_timeout",
                                "provider_profile": self.provider_profile,
                                "message": message,
                                "timeout_sec": read_timeout,
                                "tool_call_stream_active": False,
                                "chunk_count": 0,
                                "last_chunks": [],
                            },
                        }
                        return
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
                        if self.aggressive_tool_call_timeout:
                            yield {
                                "type": "chunk_error",
                                "error": "Incomplete tool call from provider stream",
                                "details": {
                                    "reason": "tool_call_continuation_timeout",
                                    "provider_profile": self.provider_profile,
                                    "message": message,
                                    "timeout_sec": read_timeout,
                                    "tool_call_stream_active": True,
                                    "last_chunks": list(recent_chunks),
                                },
                            }
                            return
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
                recent_chunks.append(summarize_stream_chunk(obj))
                if chunk_contains_tool_call_delta(obj):
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
                    details["last_chunks"] = list(recent_chunks)
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
                    last_chunks=list(recent_chunks),
                )
            yield {
                "type": "stream_ended_without_done",
                "details": {
                    "chunk_count": chunk_count,
                    "tool_call_stream_active": tool_call_stream_active,
                    "last_chunks": list(recent_chunks),
                },
            }

    async def nonstream_chat(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Fallback non-streaming chat request.
        
        Converts the response to chunk format for uniform handling.
        """
        fallback_payload = dict(payload)
        fallback_payload["stream"] = False
        fallback_payload.pop("stream_options", None)
        timeout = httpx.Timeout(
            connect=self.STREAM_CONNECT_TIMEOUT_SEC,
            read=self.STREAM_READ_TIMEOUT_SEC,
            write=self.STREAM_WRITE_TIMEOUT_SEC,
            pool=self.STREAM_POOL_TIMEOUT_SEC,
        )
        response = await client.post(url, headers=headers, json=fallback_payload, timeout=timeout)
        try:
            response.raise_for_status()
            obj = response.json()
            if self.run_logger:
                self.run_logger.log(
                    "chat",
                    "nonstream_response",
                    "chat non-stream response received",
                    chunk=obj,
                )
            yield {"type": "chunk", "data": _nonstream_response_to_chunk(obj)}
            yield {"type": "done"}
        finally:
            await response.aclose()


def _nonstream_response_to_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a non-streaming response to chunk format."""
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
