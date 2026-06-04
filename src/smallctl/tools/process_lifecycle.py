from __future__ import annotations

import asyncio
from typing import Any


def unregister_process(harness: Any, proc: Any) -> None:
    active_processes = getattr(harness, "_active_processes", None)
    if active_processes is None or proc is None:
        return
    try:
        active_processes.discard(proc)
    except Exception:
        pass


def close_process_pipes(proc: Any) -> None:
    if proc is None:
        return
    for attr_name in ("stdin", "stdout", "stderr"):
        pipe = getattr(proc, attr_name, None)
        close = getattr(pipe, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def truncate_output(text: str, *, max_bytes: int = 256 * 1024, suffix: str = "\n[OUTPUT TRUNCATED - TOO LARGE]") -> str:
    if len(text) > max_bytes:
        return text[:max_bytes] + suffix
    return text


async def stop_process(proc: Any, *, harness: Any = None, timeout: float = 1.0) -> None:
    if proc is None:
        return
    try:
        if getattr(proc, "returncode", None) is None:
            terminate = getattr(proc, "terminate", None)
            if callable(terminate):
                try:
                    terminate()
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except Exception:
                kill = getattr(proc, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except ProcessLookupError:
                        pass
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except Exception:
                    pass
    finally:
        close_process_pipes(proc)
        unregister_process(harness, proc)
