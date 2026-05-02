from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


async def read_stream_chunks(
    stream: Any,
    out_list: list[str],
    *,
    chunk_size: int,
    on_chunk: Callable[[str], Awaitable[None] | None] | None = None,
    idle_timeout_sec: float | None = None,
) -> None:
    if not stream or not hasattr(stream, "read"):
        return

    while True:
        try:
            if idle_timeout_sec is not None:
                chunk = await asyncio.wait_for(stream.read(chunk_size), timeout=idle_timeout_sec)
            else:
                chunk = await stream.read(chunk_size)
        except asyncio.TimeoutError:
            break
        except Exception:
            break
        if not chunk:
            break
        chunk_str = chunk.decode("utf-8", errors="replace")
        out_list.append(chunk_str)
        if on_chunk is not None:
            result = on_chunk(chunk_str)
            if hasattr(result, "__await__"):
                await result  # type: ignore[func-returns-value]
