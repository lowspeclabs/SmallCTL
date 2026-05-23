"""
aho/logging_aho.py
------------------
AHO-specific structured logger.

Writes two files per run:
  aho/logs/<run_id>/aho.jsonl   — machine-readable JSONL event stream
  aho/logs/<run_id>/aho.log     — human-readable text log

Also maintains aho/bug_tracker.jsonl — a persistent record of unexpected
exceptions, runner crashes, and parse failures that survive across runs for
post-hoc analysis.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from src.smallctl.logging_utils import RunLogger

_LOG = logging.getLogger("aho")

AHO_LOGS_DIR = Path("aho/logs")
BUG_TRACKER_PATH = Path("aho/bug_tracker.jsonl")


class AHOLogger(RunLogger):
    """
    Per-run structured logger using core RunLogger as engine.
    Tailored to AHO taxonomy (researcher, runner, eval, bug).
    """

    def __init__(self, run_id: str, logs_dir: Path = AHO_LOGS_DIR) -> None:
        run_dir = logs_dir / run_id
        super().__init__(run_dir, channels={"researcher", "runner", "eval", "bug"})
        self.run_id = run_id
        self.extra_fields["run_id"] = run_id
        self.extra_fields["iteration"] = 0

    def set_iteration(self, n: int) -> None:
        self.extra_fields["iteration"] = n

    def debug(self, channel: str, event: str, message: str = "", **data: Any) -> None:
        self.log(channel, event, message, **data)
        _LOG.debug("[aho/%s] %s %s", channel, event, message)

    def info(self, channel: str, event: str, message: str = "", **data: Any) -> None:
        self.log(channel, event, message, **data)
        _LOG.info("[aho/%s] %s %s", channel, event, message)

    def warning(self, channel: str, event: str, message: str = "", **data: Any) -> None:
        self.log(channel, event, message, **data)
        _LOG.warning("[aho/%s] %s %s", channel, event, message)

    def error(self, channel: str, event: str, message: str = "", **data: Any) -> None:
        self.log(channel, event, message, **data)
        _LOG.error("[aho/%s] %s %s", channel, event, message)

    def bug(
        self,
        event: str,
        message: str,
        *,
        exception: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        data = {**(context or {})}
        if exception:
            data["exception"] = exception
        
        self.log("bug", event, message, global_path=BUG_TRACKER_PATH, **data)
        _LOG.error("[aho/bug] %s %s", event, message)

    def metrics(self, iteration: int, strategy_id: str, scores: dict[str, Any]) -> None:
        self.info(
            "eval",
            "metrics_summary",
            f"iter={iteration} strategy={strategy_id} "
            f"score={scores.get('mean_harness_score', 0):.4f} "
            f"pass@N={scores.get('pass_at_n', 0):.2f}",
            iteration=iteration,
            strategy_id=strategy_id,
            **scores,
        )


def create_aho_logger(logs_dir: Path = AHO_LOGS_DIR) -> AHOLogger:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{uuid.uuid4().hex[:8]}-{ts}"
    return AHOLogger(run_id=run_id, logs_dir=logs_dir)


def setup_aho_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
