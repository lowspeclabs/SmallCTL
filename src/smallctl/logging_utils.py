from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from .redaction import compact_tool_arguments_for_metadata, redact_sensitive_data


EVENT_SCHEMA_VERSION = 1

# Subsystem names used for granular debug controls.
DEBUG_SUBSYSTEMS = {
    "client",
    "graph",
    "tools",
    "context",
    "fama",
    "ui",
    "memory",
    "state",
}

# Maps a subsystem to the Python logger prefixes it controls.
SUBSYSTEM_LOGGER_PREFIXES: dict[str, tuple[str, ...]] = {
    "client": ("smallctl.client",),
    "graph": ("smallctl.graph",),
    "tools": ("smallctl.tools", "smallctl.harness.tool_"),
    "context": ("smallctl.context",),
    "fama": ("smallctl.fama",),
    "ui": ("smallctl.ui",),
    "memory": ("smallctl.memory",),
    "state": ("smallctl.state",),
}

# Primary RunLogger channel(s) owned by each subsystem. A debug-level event on a
# channel is dropped unless at least one subsystem that owns the channel is
# enabled (or the global --debug switch is on).
SUBSYSTEM_CHANNELS: dict[str, tuple[str, ...]] = {
    "client": ("chat", "model_output"),
    "graph": ("harness", "model_output"),
    "tools": ("tools",),
    "context": ("harness",),
    "fama": ("harness",),
    "ui": ("chat", "harness"),
    "memory": ("harness",),
    "state": ("harness",),
}

_CHANNEL_DEFAULT_SUBSYSTEM: dict[str, str] = {
    "harness": "graph",
    "tools": "tools",
    "chat": "client",
    "model_output": "graph",
    "recovery": "graph",
}

_LOG_LEVEL_NAMES = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}


def _level_name(level: int) -> str:
    return _LOG_LEVEL_NAMES.get(level, "info")


def _level_from_name(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def setup_logging(
    debug: bool = False,
    log_file: str | None = None,
    *,
    stream_to_terminal: bool = True,
    debug_subsystems: list[str] | None = None,
) -> None:
    """Configure Python logging with optional subsystem-level granularity.

    When ``debug`` is True, every subsystem is set to DEBUG. When
    ``debug_subsystems`` is provided, only the matching logger prefixes are set
    to DEBUG; everything else stays at INFO.
    """
    enabled = set(DEBUG_SUBSYSTEMS) if debug else set()
    if debug_subsystems:
        enabled.update(s.strip().lower() for s in debug_subsystems if s.strip())

    if enabled:
        level = logging.DEBUG
        # Keep root smallctl at DEBUG so debug events can be emitted, but set
        # non-enabled subsystem loggers to INFO.
        for subsystem in DEBUG_SUBSYSTEMS:
            prefix_level = logging.DEBUG if subsystem in enabled else logging.INFO
            for prefix in SUBSYSTEM_LOGGER_PREFIXES.get(subsystem, ()):
                logging.getLogger(prefix).setLevel(prefix_level)
    else:
        level = logging.INFO

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
    log_method = getattr(logger, "log", None)
    if log_method is None:
        return
    payload = json.dumps(fields, ensure_ascii=True, sort_keys=True, default=str)
    log_method(level, "%s | %s", message, payload)


def synthetic_trace_id(state: Any, suffix: str) -> str:
    """Return a synthetic trace id for events that occur outside a model call."""
    session_id = ""
    task_id = ""
    step_count = 0
    if state is not None:
        session_id = str(getattr(state, "thread_id", "") or getattr(state, "session_id", "") or "").strip()
        scratchpad = getattr(state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            task_id = str(scratchpad.get("_active_task_id") or scratchpad.get("_task_sequence") or "").strip()
        step_count = int(getattr(state, "step_count", 0) or 0)
    session_id = session_id or "run"
    task_id = task_id or "task"
    return f"{session_id}:{task_id}:step-{step_count}:{suffix}"


@dataclass
class _TextStreamState:
    kind: str
    endswith_newline: bool = False


class RunLogger:
    def __init__(
        self,
        run_dir: Path,
        channels: set[str] | None = None,
        *,
        debug_subsystems: list[str] | None = None,
        log_max_mb: int | None = None,
        debug_tokens: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self._lock = Lock()
        self._listener: Callable[[dict[str, Any]], None] | None = None
        self._text_streams: dict[Path, _TextStreamState] = {}
        self.channels = channels or {"harness", "tools", "chat", "model_output"}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.extra_fields: dict[str, Any] = {
            "event_schema_version": EVENT_SCHEMA_VERSION,
        }
        self.debug_subsystems: set[str] = set()
        self.log_max_mb = log_max_mb if log_max_mb is not None else 100
        self.debug_tokens = bool(debug_tokens)
        self._token_counter: dict[str, int] = {}
        for channel in self.channels:
            (self.run_dir / f"{channel}.jsonl").touch(exist_ok=True)
            (self.run_dir / f"{channel}.log").touch(exist_ok=True)
        self._write_run_header()
        if debug_subsystems:
            self.set_debug_subsystems(debug_subsystems)

    def _write_run_header(self) -> None:
        header = {
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "channels": sorted(self.channels),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        path = self.run_dir / "run_header.json"
        with self._lock:
            path.write_text(json.dumps(header, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")

    def set_debug_subsystems(self, subsystems: list[str] | None) -> None:
        if not subsystems:
            self.debug_subsystems = set()
            return
        self.debug_subsystems = {
            s.strip().lower()
            for s in subsystems
            if s.strip() and s.strip().lower() in DEBUG_SUBSYSTEMS
        }

    def set_session_id(self, session_id: str) -> Path:
        normalized = str(session_id or "").strip()
        if not normalized:
            return self.run_dir

        with self._lock:
            self.extra_fields["session_id"] = normalized
            current_name = self.run_dir.name
            parts = current_name.split("-", 1)
            suffix = parts[1] if len(parts) == 2 else current_name
            if parts and parts[0] == normalized:
                return self.run_dir

            parent = self.run_dir.parent
            candidate = parent / f"{normalized}-{suffix}"
            if candidate != self.run_dir:
                counter = 2
                while candidate.exists():
                    candidate = parent / f"{normalized}-{suffix}-{counter}"
                    counter += 1
                self.run_dir.rename(candidate)
                self.run_dir = candidate
            finalized_run_dir = str(self.run_dir)

        if hasattr(self, '_finalize_listener') and callable(self._finalize_listener):
            self._finalize_listener(finalized_run_dir)
        return self.run_dir

    def set_extra_field(self, key: str, value: Any) -> None:
        normalized = str(key or "").strip()
        if not normalized:
            return
        with self._lock:
            if value in (None, ""):
                self.extra_fields.pop(normalized, None)
            else:
                self.extra_fields[normalized] = value

    def set_trace_id(self, trace_id: str) -> None:
        self.set_extra_field("trace_id", str(trace_id or "").strip())

    def clear_trace_id(self) -> None:
        self.set_extra_field("trace_id", None)

    def set_task_id(self, task_id: str) -> None:
        self.set_extra_field("task_id", str(task_id or "").strip())

    def set_step_count(self, step_count: int) -> None:
        self.set_extra_field("step_count", int(step_count or 0))

    def set_call_count(self, call_count: int) -> None:
        self.set_extra_field("call_count", int(call_count or 0))

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

    def _should_drop_debug_event(self, channel: str, level: str, subsystem: str | None) -> bool:
        """Return True if a debug-level event should be dropped due to subsystem filters."""
        if level != "debug":
            return False
        if not self.debug_subsystems:
            return False
        resolved = (subsystem or _CHANNEL_DEFAULT_SUBSYSTEM.get(channel, "graph")).lower()
        # Always keep if the event's own subsystem is enabled.
        if resolved in self.debug_subsystems:
            return False
        # Also keep if any subsystem that owns the event's channel is enabled.
        for subsys, chans in SUBSYSTEM_CHANNELS.items():
            if subsys in self.debug_subsystems and channel in chans:
                return False
        return True

    def _maybe_sample_token(self, channel: str, event: str, data: dict[str, Any]) -> bool:
        """Return True if the token event should be logged.

        When debug_tokens is False, log every 20th token plus the first and last
        100 tokens of each call. Call count is inferred from extra_fields.
        """
        if event != "model_token":
            return True
        if self.debug_tokens:
            return True
        call_count = int(self.extra_fields.get("call_count") or 0)
        key = f"{channel}:{call_count}"
        count = self._token_counter.get(key, 0)
        self._token_counter[key] = count + 1
        # first 100 and last 100 tokens per call
        if count < 100:
            return True
        # Keep a marker for every 20th token so the stream is still observable.
        if count % 20 == 0:
            data["_sampled_token"] = True
            return True
        return False

    def _enforce_size_cap(self) -> None:
        """Rotate the largest channel file when the run directory exceeds the cap."""
        if self.log_max_mb <= 0:
            return
        try:
            total_bytes = sum(
                f.stat().st_size
                for f in self.run_dir.iterdir()
                if f.is_file() and f.suffix in {".jsonl", ".log"}
            )
        except OSError:
            return
        max_bytes = self.log_max_mb * 1024 * 1024
        if total_bytes <= max_bytes:
            return
        try:
            largest = max(
                (f for f in self.run_dir.iterdir() if f.is_file() and f.suffix in {".jsonl", ".log"}),
                key=lambda f: f.stat().st_size,
            )
        except ValueError:
            return
        rotated = largest.with_suffix(largest.suffix + ".1")
        try:
            # Move current to .1 and truncate the active file.
            if rotated.exists():
                rotated.unlink()
            largest.rename(rotated)
            largest.write_text("", encoding="utf-8")
            self._log_rotation_event(largest.name, rotated.name, total_bytes)
        except OSError:
            pass

    def _log_rotation_event(self, channel_file: str, rotated_file: str, total_bytes: int) -> None:
        self.log(
            "harness",
            "log_rotation",
            "rotated channel log due to size cap",
            level="warning",
            channel_file=channel_file,
            rotated_file=rotated_file,
            total_bytes=total_bytes,
            log_max_mb=self.log_max_mb,
        )

    def log(
        self,
        channel: str,
        event: str,
        message: str = "",
        global_path: Path | None = None,
        *,
        level: str = "info",
        subsystem: str | None = None,
        **data: Any,
    ) -> None:
        if channel not in self.channels:
            channel = list(self.channels)[0] if self.channels else "harness"
        if self._should_drop_debug_event(channel, level, subsystem):
            return
        if not self._maybe_sample_token(channel, event, data):
            return

        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        bounded_data = self._bound_payload(data)
        safe_data = redact_sensitive_data(bounded_data)
        row = {
            "timestamp": timestamp,
            "channel": channel,
            "event": event,
            "message": message,
            "level": level,
            **self.extra_fields,
            "data": safe_data,
        }
        jsonl_path = self.run_dir / f"{channel}.jsonl"
        text_path = self.run_dir / f"{channel}.log"
        with self._lock:
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
            self._write_text_log(text_path, timestamp, event, message, bounded_data, safe_data)
            if global_path:
                global_path.parent.mkdir(parents=True, exist_ok=True)
                with global_path.open("a", encoding="utf-8") as gf:
                    gf.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
            self._enforce_size_cap()
        if self._listener:
            self._listener(row)

    @staticmethod
    def _bound_payload(data: dict[str, Any]) -> dict[str, Any]:
        """Keep individual debug payloads bounded."""
        result: dict[str, Any] = {}
        for key, value in data.items():
            if key == "arguments" and isinstance(value, dict):
                result[key] = compact_tool_arguments_for_metadata(
                    str(data.get("tool_name") or ""), value
                )
            elif isinstance(value, str) and len(value) > 200:
                result[key] = value[:197] + "..."
            elif isinstance(value, list) and len(value) > 50:
                result[key] = value[:50] + [f"... ({len(value) - 50} more)"]
            else:
                result[key] = value
        return result

    def set_listener(self, listener: Callable[[dict[str, Any]], None] | None) -> None:
        self._listener = listener

    def handle_debug_signal(self, signal_path: Path) -> dict[str, Any] | None:
        """Read a debug control file and apply the command if present.

        Commands:
          - escalate:<n>  bump all subsystem loggers to DEBUG for the next N turns
          - snapshot        dump a compact recent-event summary to .debug-snapshot.jsonl
        """
        if not signal_path.exists():
            return None
        try:
            text = signal_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not text:
            return None
        try:
            signal_path.unlink()
        except OSError:
            pass

        if text.startswith("escalate:"):
            try:
                turns = int(text.split(":", 1)[1].strip())
            except ValueError:
                turns = 1
            for subsys in DEBUG_SUBSYSTEMS:
                for prefix in SUBSYSTEM_LOGGER_PREFIXES.get(subsys, ()):
                    logging.getLogger(prefix).setLevel(logging.DEBUG)
            self.log(
                "harness",
                "debug_escalation",
                "debug escalation triggered via control file",
                level="warning",
                turns=turns,
            )
            return {"command": "escalate", "turns": turns}

        if text == "snapshot":
            snapshot_path = self.run_dir / ".debug-snapshot.jsonl"
            self._write_snapshot(snapshot_path)
            self.log(
                "harness",
                "debug_snapshot",
                "wrote debug snapshot",
                level="warning",
                snapshot_path=str(snapshot_path),
            )
            return {"command": "snapshot", "snapshot_path": str(snapshot_path)}

        return None

    def _write_snapshot(self, snapshot_path: Path) -> None:
        recent: list[dict[str, Any]] = []
        for channel in self.channels:
            path = self.run_dir / f"{channel}.jsonl"
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines[-200:]:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recent.append(record)
        recent.sort(key=lambda r: r.get("timestamp", ""))
        with snapshot_path.open("w", encoding="utf-8") as fh:
            for record in recent[-500:]:
                fh.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def create_run_logger(
    base_dir: str = "logs",
    *,
    debug_subsystems: list[str] | None = None,
    log_max_mb: int | None = None,
    debug_tokens: bool = False,
) -> RunLogger:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(base_dir).resolve() / f"{run_id}-{ts}"
    return RunLogger(
        run_dir,
        debug_subsystems=debug_subsystems,
        log_max_mb=log_max_mb,
        debug_tokens=debug_tokens,
    )
