"""UI performance diagnostics for TUI glitch tracing.

Enable with: SMALLCTL_UI_PERF_LOG=1
"""
from __future__ import annotations

import os
import time
from typing import Any


_PERF_ENABLED = bool(os.environ.get("SMALLCTL_UI_PERF_LOG"))


def perf_log(logger: Any, fmt: str, *args: Any) -> None:
    if _PERF_ENABLED and logger is not None:
        logger.debug("[UI-PERF] " + fmt, *args)


def perf_log_event_queue(
    logger: Any,
    *,
    action: str,
    event_type: str,
    queue_depth: int,
    elapsed_ms: float | None = None,
) -> None:
    if not _PERF_ENABLED:
        return
    msg = f"[UI-PERF] event_queue action={action} type={event_type} depth={queue_depth}"
    if elapsed_ms is not None:
        msg += f" elapsed_ms={elapsed_ms:.2f}"
    if logger is not None:
        logger.debug(msg)


def perf_log_widget_op(
    logger: Any,
    *,
    op: str,
    widget_type: str,
    elapsed_ms: float | None = None,
    extra: str = "",
) -> None:
    if not _PERF_ENABLED:
        return
    msg = f"[UI-PERF] widget op={op} type={widget_type}"
    if elapsed_ms is not None:
        msg += f" elapsed_ms={elapsed_ms:.2f}"
    if extra:
        msg += f" {extra}"
    if logger is not None:
        logger.debug(msg)


def perf_log_refresh(
    logger: Any,
    *,
    widget_type: str,
    text_len: int,
    elapsed_ms: float | None = None,
) -> None:
    if not _PERF_ENABLED:
        return
    msg = f"[UI-PERF] refresh type={widget_type} text_len={text_len}"
    if elapsed_ms is not None:
        msg += f" elapsed_ms={elapsed_ms:.2f}"
    if logger is not None:
        logger.debug(msg)


class PerfTimer:
    """Context manager for timing operations."""

    def __init__(self, logger: Any, name: str, extra: str = "") -> None:
        self.logger = logger
        self.name = name
        self.extra = extra
        self.start: float = 0.0

    def __enter__(self) -> PerfTimer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.perf_counter() - self.start) * 1000
        perf_log(self.logger, "%s elapsed_ms=%.2f %s", self.name, elapsed, self.extra)


def perf_log_buffer_state(
    logger: Any,
    *,
    pending_events: int,
    pending_tool_results: int,
    active_turn: bool,
) -> None:
    if not _PERF_ENABLED:
        return
    if logger is not None:
        logger.debug(
            "[UI-PERF] buffer_state pending_events=%d pending_tool_results=%d active_turn=%s",
            pending_events,
            pending_tool_results,
            active_turn,
        )
