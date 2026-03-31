from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable


def setup_logging(debug: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )


def log_kv(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    payload = json.dumps(fields, ensure_ascii=True, sort_keys=True, default=str)
    logger.log(level, "%s | %s", message, payload)


class RunLogger:
    def __init__(self, run_dir: Path, channels: set[str] | None = None) -> None:
        self.run_dir = run_dir
        self._lock = Lock()
        self._listener: Callable[[dict[str, Any]], None] | None = None
        self.channels = channels or {"harness", "tools", "chat", "model_output"}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.extra_fields: dict[str, Any] = {}
        for channel in self.channels:
            (self.run_dir / f"{channel}.jsonl").touch(exist_ok=True)
            (self.run_dir / f"{channel}.log").touch(exist_ok=True)

    def log(
        self,
        channel: str,
        event: str,
        message: str = "",
        global_path: Path | None = None,
        **data: Any,
    ) -> None:
        if channel not in self.channels:
            channel = list(self.channels)[0] if self.channels else "harness"
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        row = {
            "timestamp": timestamp,
            "channel": channel,
            "event": event,
            "message": message,
            **self.extra_fields,
            "data": data,
        }
        jsonl_path = self.run_dir / f"{channel}.jsonl"
        text_path = self.run_dir / f"{channel}.log"
        with self._lock:
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
            with text_path.open("a", encoding="utf-8") as tf:
                tf.write(f"{timestamp} {event} {message} {json.dumps(data, ensure_ascii=True, default=str)}\n")
            if global_path:
                global_path.parent.mkdir(parents=True, exist_ok=True)
                with global_path.open("a", encoding="utf-8") as gf:
                    gf.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
        if self._listener:
            self._listener(row)

    def set_listener(self, listener: Callable[[dict[str, Any]], None] | None) -> None:
        self._listener = listener


def create_run_logger(base_dir: str = "logs") -> RunLogger:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(base_dir).resolve() / f"{run_id}-{ts}"
    return RunLogger(run_dir)
