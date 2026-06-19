"""Shared utilities for Agent-Tools log/code navigation scripts.

This module is intentionally dependency-light so the tools run in any
Python 3.10+ environment. It only uses the standard library.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
SRC_DIR = REPO_ROOT / "src" / "smallctl"


WARNING_EVENTS = {
    "fama_signal_detected",
    "reflexion_created",
    "same_scope_iteration_recorded",
    "fama_mitigation_activated",
    "context_invalidated",
    "thinking_tool_protocol_sanitized",
    "inline_tool_call_recovered_from_thinking",
}

ERROR_EVENTS = {
    "error",
    "failure",
    "exception",
    "dispatch_tools_error",
    "initialize_run_error",
    "interrupt_for_human_error",
    "model_output_degenerate_loop_exhausted",
    "action_stall",
    "no_tool_recovery",
    "tool_call_aggregation_failure",
    "tool_blocked_not_exposed",
    "fama_tool_call_blocked",
    "recovery_failure_event_recorded",
}


class Colors:
    """ANSI color codes. Disabled automatically when stdout is not a TTY."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    @classmethod
    def enabled(cls) -> bool:
        return sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    if not Colors.enabled():
        return text
    return f"{color}{text}{Colors.RESET}"


def discover_runs(logs_dir: Path | None = None) -> list[Path]:
    """Return run directories sorted newest-first (by directory mtime)."""
    base = logs_dir or LOGS_DIR
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def resolve_run_dir(run_arg: str | None, logs_dir: Path | None = None) -> Path:
    """Resolve a run argument to a run directory.

    Accepts:
      - a full run directory path
      - a run id like 6cf0f870
      - "latest" or None (newest run)
      - "latest-N" (N-th most recent run, e.g. latest-1)
    """
    base = logs_dir or LOGS_DIR
    if run_arg is None or run_arg == "latest":
        runs = discover_runs(base)
        if not runs:
            raise FileNotFoundError(f"No run directories found under {base}")
        return runs[0]

    if run_arg.startswith("latest-"):
        runs = discover_runs(base)
        if not runs:
            raise FileNotFoundError(f"No run directories found under {base}")
        try:
            offset = int(run_arg.split("-", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid latest-N spec: {run_arg}") from exc
        if offset < 0 or offset >= len(runs):
            raise ValueError(f"{run_arg} out of range (only {len(runs)} runs)")
        return runs[offset]

    p = Path(run_arg).expanduser().resolve()
    if p.exists() and p.is_dir():
        return p

    # Try as a run id under logs/
    candidates = [r for r in discover_runs(base) if r.name.startswith(run_arg)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Run id prefix {run_arg!r} is ambiguous: {candidates}")

    raise FileNotFoundError(f"Could not resolve run directory: {run_arg}")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON lines, skipping malformed lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed JSONL line %s in %s: %s", lineno, path, exc)


def load_summaries(run_dir: Path) -> dict[str, Any]:
    """Load session_summary.json and task_summary.json if present."""
    summaries: dict[str, Any] = {}
    for name in ("session_summary.json", "task_summary.json"):
        path = run_dir / name
        if path.exists():
            try:
                summaries[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning("Could not load %s: %s", path, exc)
    return summaries


def load_task_summaries(run_dir: Path) -> list[dict[str, Any]]:
    """Load all tasks/task-*/task_summary.json files."""
    summaries: list[dict[str, Any]] = []
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        return summaries
    for task_dir in sorted(tasks_dir.iterdir()):
        path = task_dir / "task_summary.json"
        if path.exists():
            try:
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning("Could not load %s: %s", path, exc)
    return summaries


def event_counter(records: Iterable[dict[str, Any]]) -> Counter[str]:
    """Count records by event name."""
    return Counter(r.get("event", "") for r in records if r.get("event"))


def extract_trace_id(record: dict[str, Any]) -> str | None:
    """Return trace_id from record if present."""
    return record.get("trace_id") or record.get("data", {}).get("trace_id")


def iter_records(run_dir: Path, channel: str) -> Iterable[dict[str, Any]]:
    """Yield records from a JSONL channel."""
    jsonl_path = run_dir / f"{channel}.jsonl"
    log_path = run_dir / f"{channel}.log"
    if jsonl_path.exists():
        yield from read_jsonl(jsonl_path)
    elif log_path.exists():
        # Fallback: parse text log lines that start with a timestamp and contain JSON
        for record in _parse_text_log(log_path):
            yield record


def _parse_text_log(path: Path) -> Iterable[dict[str, Any]]:
    """Best-effort parse of .log text files into dicts with timestamp/event/message/data."""
    ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2})\s+(")?')
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = ts_re.match(line)
        if not m:
            continue
        record: dict[str, Any] = {"timestamp": m.group(1), "channel": path.stem, "event": "", "message": ""}
        rest = line[m.end():]
        # Try to find JSON payload
        json_start = rest.find("{")
        if json_start >= 0:
            message = rest[:json_start].strip().strip('"')
            payload = rest[json_start:]
            try:
                data = json.loads(payload)
                record["data"] = data
            except json.JSONDecodeError:
                record["data"] = {"raw": payload}
            # Derive event from message first word
            record["message"] = message
            record["event"] = message.split()[0] if message.split() else "unknown"
        else:
            record["message"] = rest
            record["event"] = rest.split()[0] if rest.split() else "unknown"
        yield record


def find_records_by_trace_id(run_dir: Path, trace_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return records grouped by channel for a given trace_id."""
    channels = ["harness", "tools", "chat", "model_output"]
    result: dict[str, list[dict[str, Any]]] = {ch: [] for ch in channels}
    for ch in channels:
        for rec in iter_records(run_dir, ch):
            if extract_trace_id(rec) == trace_id:
                result[ch].append(rec)
    return result


def record_level(record: dict[str, Any]) -> int:
    """Return 0=info, 1=warning, 2=error for a record.

    Some SmallCTL records carry failure_class as metadata (e.g. FAMA signals or
    reflexion entries) without being terminal errors. We surface those as
    warnings so logwatch/diagnose can count them separately.
    """
    event = str(record.get("event", "")).lower()

    if event in ERROR_EVENTS or any(evt in event for evt in ERROR_EVENTS):
        return 2

    data = record.get("data") or {}
    if isinstance(data, dict):
        if data.get("success") is False:
            return 2
        if str(data.get("status", "")).lower() in {"failed", "error", "failure"}:
            return 2
        if "error" in data or "exception" in data:
            return 2

    if event in WARNING_EVENTS:
        return 1

    if isinstance(data, dict) and (data.get("failure_class") or data.get("failure_classes")):
        return 1

    return 0


def has_error_indicator(record: dict[str, Any]) -> bool:
    """Return True if the record represents an error condition."""
    return record_level(record) == 2


def has_warning_indicator(record: dict[str, Any]) -> bool:
    """Return True if the record represents a warning/signal condition."""
    return record_level(record) == 1


def error_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter records to those with error indicators."""
    return [r for r in records if has_error_indicator(r)]


def warning_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter records to those with warning indicators."""
    return [r for r in records if has_warning_indicator(r)]


def format_record_summary(record: dict[str, Any], max_width: int = 160) -> str:
    """One-line summary of a record."""
    ts = record.get("timestamp", "")
    event = record.get("event", "")
    message = record.get("message", "")
    trace = extract_trace_id(record) or ""
    text = f"{ts}  {event}  {message}"
    if trace:
        text += f"  [{trace}]"
    if len(text) > max_width:
        text = text[: max_width - 3] + "..."
    return text


def parse_timestamp(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp with timezone offset."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def run_duration_seconds(records: Iterable[dict[str, Any]]) -> float | None:
    """Return elapsed seconds between first and last timestamp in records."""
    timestamps = [
        parse_timestamp(r.get("timestamp", ""))
        for r in records
        if r.get("timestamp")
    ]
    timestamps = [t for t in timestamps if t is not None]
    if len(timestamps) < 2:
        return None
    first = min(timestamps)
    last = max(timestamps)
    return (last - first).total_seconds()


def format_duration(seconds: float | None) -> str:
    """Format a duration in a human-readable way."""
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def ensure_logs_dir() -> Path:
    return LOGS_DIR
