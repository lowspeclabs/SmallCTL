"""SSE streaming and non-stream chat implementations."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from .chunk_parser import chunk_contains_tool_call_delta
from ..logging_utils import log_kv


class SSEStreamer:
    """SSE stream handler for model chat completions."""
    
    STREAM_CONNECT_TIMEOUT_SEC = 10.0
    STREAM_WRITE_TIMEOUT_SEC = 30.0
    STREAM_READ_TIMEOUT_SEC = 120.0
    STREAM_POOL_TIMEOUT_SEC = 30.0
    STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC = 30.0

    def __init__(
        self,
        provider_profile: str = "generic",
        tool_call_continuation_timeout_sec: float | None = None,
        run_logger: Any | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.provider_profile = provider_profile
        if tool_call_continuation_timeout_sec is None:
            self.tool_call_continuation_timeout_sec = float(self.STREAM_TOOL_CALL_CONTINUATION_TIMEOUT_SEC)
        else:
            self.tool_call_continuation_timeout_sec = max(1.0, float(tool_call_continuation_timeout_sec))
        self.run_logger = run_logger
        self.log = log or logging.getLogger("smallctl.client.streaming")

    def _next_stream_read_timeout(self, *, tool_call_stream_active: bool) -> float:
        """Determine the appropriate read timeout for the next stream chunk."""
        if self.provider_profile == "lmstudio" and tool_call_stream_active:
            return min(
                float(self.STREAM_READ_TIMEOUT_SEC),
                float(self.tool_call_continuation_timeout_sec),
            )
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
        yield {"type": "chunk", "data": _nonstream_response_to_chunk(obj)}
        yield {"type": "done"}


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
