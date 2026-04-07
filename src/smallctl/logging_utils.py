from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from .redaction import redact_sensitive_data


def setup_logging(
    debug: bool = False,
    log_file: str | None = None,
    *,
    stream_to_terminal: bool = True,
) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handlers: list[logging.Handler] = []
    if stream_to_terminal:
        handlers.append(logging.StreamHandler())
    if log_file:
        path = Path(log_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )


def log_kv(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    payload = json.dumps(fields, ensure_ascii=True, sort_keys=True, default=str)
    logger.log(level, "%s | %s", message, payload)


@dataclass
class _TextStreamState:
    kind: str
    endswith_newline: bool = False


class RunLogger:
    def __init__(self, run_dir: Path, channels: set[str] | None = None) -> None:
        self.run_dir = run_dir
        self._lock = Lock()
        self._listener: Callable[[dict[str, Any]], None] | None = None
        self._text_streams: dict[Path, _TextStreamState] = {}
        self.channels = channels or {"harness", "tools", "chat", "model_output"}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.extra_fields: dict[str, Any] = {}
        for channel in self.channels:
            (self.run_dir / f"{channel}.jsonl").touch(exist_ok=True)
            (self.run_dir / f"{channel}.log").touch(exist_ok=True)

    @staticmethod
    def _token_stream_fragment(
        event: str,
        message: str,
        safe_data: dict[str, Any],
    ) -> tuple[str, str] | None:
        if event != "model_token":
            return None
        token = safe_data.get("token")
        if not isinstance(token, str) or not token:
            return None
        lower_message = str(message or "").strip().lower()
        if "thinking" in lower_message:
            kind = "thinking"
        elif "assistant" in lower_message:
            kind = "assistant"
        else:
            kind = "model"
        return kind, token

    def _flush_text_stream(self, text_path: Path, tf: Any) -> None:
        state = self._text_streams.pop(text_path, None)
        if state and not state.endswith_newline:
            tf.write("\n")

    def _write_text_log(
        self,
        text_path: Path,
        timestamp: str,
        event: str,
        message: str,
        raw_data: dict[str, Any],
        safe_data: dict[str, Any],
    ) -> None:
        fragment = self._token_stream_fragment(event, message, raw_data)
        with text_path.open("a", encoding="utf-8") as tf:
            if fragment is not None:
                kind, token = fragment
                state = self._text_streams.get(text_path)
                if state is None or state.kind != kind:
                    self._flush_text_stream(text_path, tf)
                    tf.write(f"{timestamp} {kind}: ")
                tf.write(token)
                self._text_streams[text_path] = _TextStreamState(
                    kind=kind,
                    endswith_newline=token.endswith("\n"),
                )
                return

            self._flush_text_stream(text_path, tf)
            tf.write(f"{timestamp} {event} {message} {json.dumps(safe_data, ensure_ascii=True, default=str)}\n")

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
        safe_data = redact_sensitive_data(data)
        row = {
            "timestamp": timestamp,
            "channel": channel,
            "event": event,
            "message": message,
            **self.extra_fields,
            "data": safe_data,
        }
        jsonl_path = self.run_dir / f"{channel}.jsonl"
        text_path = self.run_dir / f"{channel}.log"
        with self._lock:
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
            self._write_text_log(text_path, timestamp, event, message, data, safe_data)
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
