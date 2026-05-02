from __future__ import annotations

import asyncio
import time
from typing import Any

from ..models.events import UIEvent, UIEventType


class BufferedUIEventEmitter:
    def __init__(
        self,
        *,
        harness: Any,
        event_type: UIEventType,
        flush_interval_sec: float = 0.05,
        max_buffer_chars: int = 8192,
        max_chunk_chars: int = 16384,
    ) -> None:
        self._harness = harness
        self._event_type = event_type
        self._flush_interval_sec = max(0.0, float(flush_interval_sec))
        self._max_buffer_chars = max(1, int(max_buffer_chars))
        self._max_chunk_chars = max(1, int(max_chunk_chars))
        self._buffer = ""
        self._last_flush_at = time.monotonic()

    def _enabled(self) -> bool:
        return bool(
            self._harness
            and hasattr(self._harness, "_emit")
            and getattr(self._harness, "event_handler", None)
        )

    def _sanitize_chunk(self, text: str) -> str:
        chunk = str(text or "")
        if len(chunk) > self._max_chunk_chars:
            return chunk[: self._max_chunk_chars] + "\n[UI TRUNCATED - LARGE OUTPUT]"
        return chunk

    async def emit_text(self, text: str) -> None:
        if not self._enabled():
            return
        chunk = self._sanitize_chunk(text)
        if not chunk:
            return
        self._buffer += chunk
        now = time.monotonic()
        if len(self._buffer) >= self._max_buffer_chars or (
            now - self._last_flush_at
        ) >= self._flush_interval_sec:
            await self.flush()

    async def emit_event(self, event: UIEvent) -> None:
        if not self._enabled():
            return
        await self.flush()
        await self._emit(event)

    async def flush(self) -> None:
        if not self._enabled() or not self._buffer:
            return
        content = self._buffer
        self._buffer = ""
        await self._emit(UIEvent(event_type=self._event_type, content=content))
        self._last_flush_at = time.monotonic()

    async def _emit(self, event: UIEvent) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        await self._harness._emit(self._harness.event_handler, event)
