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
from datetime import datetime
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


def warn_on_schema_mismatch(run_dir: Path) -> list[str]:
    """Return warnings if a run's schema version differs from the supported version."""
    warnings: list[str] = []
    header_path = run_dir / "run_header.json"
    if header_path.exists():
        try:
            header = json.loads(header_path.read_text(encoding="utf-8"))
            version = header.get("event_schema_version")
            if version is not None and version != 1:
                warnings.append(f"Run schema version {version} may not be fully supported by this tool (expected 1).")
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not parse run_header.json: {exc}")
    # Also sample the first record from each channel.
    for channel in ["harness", "tools", "chat", "model_output"]:
        for record in iter_records(run_dir, channel):
            version = record.get("event_schema_version")
            if version is not None and version != 1:
                warnings.append(f"Channel {channel} uses schema version {version} (expected 1).")
            break
    return warnings


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


ENV_BLOCKER_PATTERNS = (
    "connection refused",
    "no route to host",
    "name or service not known",
    "could not connect",
    "service not listening",
    "connection reset",
    "network is unreachable",
    "host is unreachable",
    "timed out",
    "timeout",
    "errno 111",
    "errno 113",
    "http 401",
    "unauthorized",
    "invalid token",
    "token provided",
)


# Patterns used to detect tool-call protocol mismatches in model output.
CONTROL_TOKEN_PATTERN = re.compile(r"^\s*\|>[a-z_]+<\|\s*$", re.IGNORECASE)
TOOL_CALL_WRAPPER_PATTERN = re.compile(
    r"<[/|]?\s*tool_call\b|<\|tool_call\||</tool_call\|>"
    r"|<function\s*=|<call\b",
    re.IGNORECASE,
)


def _native_tool_call_traces(tools_records: Iterable[dict[str, Any]]) -> set[str]:
    """Return trace_ids that have a real tool dispatch recorded in tools.jsonl."""
    traces: set[str] = set()
    for rec in tools_records:
        if rec.get("event") in {"dispatch_start", "dispatch_complete"}:
            tid = extract_trace_id(rec)
            if tid:
                traces.add(tid)
    return traces


def has_control_token_fragment(records: Iterable[dict[str, Any]]) -> bool:
    """Detect Gemma/Qwen-style control-token fragments in assistant text."""
    for rec in records:
        data = rec.get("data") or {}
        text = str(data.get("assistant_text") or "")
        if text and CONTROL_TOKEN_PATTERN.match(text):
            return True
    return False


def has_reasoning_tool_call_wrapper(
    records: Iterable[dict[str, Any]],
    tools_records: Iterable[dict[str, Any]],
) -> bool:
    """Detect tool-call wrappers in reasoning/assistant text without a dispatch."""
    dispatched_traces = _native_tool_call_traces(tools_records)
    for rec in records:
        data = rec.get("data") or {}
        text = str(data.get("assistant_text") or "")
        thinking = str(data.get("thinking_text") or "")
        if not TOOL_CALL_WRAPPER_PATTERN.search(f"{text}\n{thinking}"):
            continue
        tid = extract_trace_id(rec) or ""
        if tid not in dispatched_traces:
            return True
    return False


def has_tool_call_protocol_mismatch(
    records: Iterable[dict[str, Any]],
    tools_records: Iterable[dict[str, Any]],
) -> bool:
    """Detect control-token fragments or reasoning-channel tool-call wrappers."""
    return has_control_token_fragment(records) or has_reasoning_tool_call_wrapper(
        records, tools_records
    )


def has_inline_tool_call_recovery_without_dispatch(
    harness_records: Iterable[dict[str, Any]],
    tools_records: Iterable[dict[str, Any]],
) -> bool:
    """Detect harness recovery of inline tool calls from thinking that did not dispatch."""
    dispatched_traces = _native_tool_call_traces(tools_records)
    for rec in harness_records:
        if rec.get("event") != "inline_tool_call_recovered_from_thinking":
            continue
        tid = extract_trace_id(rec) or ""
        if tid not in dispatched_traces:
            return True
    return False


def has_continue_prompt_budget_loop(
    records: Iterable[dict[str, Any]],
    chat_records: Iterable[dict[str, Any]],
    threshold: int = 2,
) -> bool:
    """Detect 'continue/proceed' retries that repeatedly hit prompt-budget overflow.

    The harness records and chat conversation snapshots are scanned in timestamp
    order. A cycle counts when a user message containing 'continue' or 'proceed'
    (after a terminal task_fail) is followed by a 'PROMPT BUDGET OVERFLOW' error.
    """
    continue_re = re.compile(r"\b(continue|proceed)\b", re.IGNORECASE)
    overflow_re = re.compile(r"PROMPT\s+BUDGET\s+OVERFLOW", re.IGNORECASE)

    # Collect user message records from chat conversation snapshots.
    user_message_times: list[tuple[str, str]] = []
    for rec in chat_records:
        data = rec.get("data") or {}
        for key in ("history", "recent_messages"):
            messages = data.get(key) or []
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = str(msg.get("content") or "")
                    if continue_re.search(content):
                        ts = str(rec.get("timestamp") or "")
                        user_message_times.append((ts, content))

    # Count task_fail events.
    saw_task_fail = False
    task_fail_times: list[str] = []
    for rec in records:
        event = rec.get("event", "")
        data = rec.get("data") or {}
        if event == "dispatch_complete" and data.get("tool_name") == "task_fail" and data.get("success") is True:
            saw_task_fail = True
            ts = str(rec.get("timestamp") or "")
            task_fail_times.append(ts)

    if not saw_task_fail:
        return False

    # Count prompt budget overflow events.
    overflow_count = 0
    for rec in records:
        message = str(rec.get("message") or "")
        data = rec.get("data") or {}
        data_text = json.dumps(data, default=str, ensure_ascii=False)
        if overflow_re.search(message) or overflow_re.search(data_text):
            overflow_count += 1

    # Count continue/proceed messages that occur after a task_fail.
    continue_after_fail = 0
    # Simple heuristic: if there is at least one task_fail and the number of
    # continue messages is at least the threshold, treat it as the loop pattern.
    if task_fail_times and user_message_times:
        first_fail = min(task_fail_times)
        continue_after_fail = sum(
            1 for ts, _ in user_message_times if ts >= first_fail
        )

    return continue_after_fail >= threshold and overflow_count >= threshold


def detect_primary_blockers(
    harness_records: Iterable[dict[str, Any]],
    failed_dispatches: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return environmental/objective-level blockers such as connection refused."""
    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}

    def _scan(text: str) -> None:
        lowered = text.lower()
        for pat in ENV_BLOCKER_PATTERNS:
            if pat in lowered:
                counts[pat] += 1
                if pat not in samples:
                    samples[pat] = text.strip()[:200]

    for rec in harness_records:
        if has_error_indicator(rec):
            _scan(json.dumps(rec, default=str, ensure_ascii=False))

    for rec in failed_dispatches:
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            continue
        for key in ("error", "message", "output", "stderr", "stdout", "reason"):
            val = data.get(key)
            if val is not None:
                _scan(str(val))
        metadata = data.get("metadata") or {}
        if isinstance(metadata, dict):
            for key in ("error", "reason", "output"):
                val = metadata.get(key)
                if val is not None:
                    _scan(str(val))

    return [
        {"type": "environment", "pattern": pat, "count": count, "sample": samples.get(pat, "")}
        for pat, count in counts.most_common()
    ]


def has_strong_environment_blocker(blockers: list[dict[str, Any]], threshold: int = 2) -> bool:
    return any(b.get("count", 0) >= threshold for b in blockers)


def has_write_overwrite_guard_failures(failed_dispatches: Iterable[dict[str, Any]]) -> bool:
    """Detect staged write-session overwrite guard failures."""
    for rec in failed_dispatches:
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            continue
        text = json.dumps(data, default=str, ensure_ascii=False)
        if "chunked_write_overwrite_new_section_after_progress" in text:
            return True
        if "Refusing to overwrite the entire staged file for a new chunk section" in text:
            return True
    return False


def has_patch_first_policy_loop(
    failed_dispatches: Iterable[dict[str, Any]], threshold: int = 2
) -> bool:
    """Detect repeated file_write blocks due to the patch-first policy."""
    count = 0
    for rec in failed_dispatches:
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            continue
        tool_name = str(data.get("tool_name") or "").strip()
        if tool_name not in {"file_write", "ssh_file_write"}:
            continue
        metadata = data.get("metadata") or {}
        reason = str(metadata.get("reason") or data.get("reason") or "").strip()
        error = str(data.get("error") or "").lower()
        if reason == "patch_first_required" or "patch-first policy" in error:
            count += 1
    return count >= threshold


def file_patch_target_loop_count(failed_dispatches: Iterable[dict[str, Any]]) -> int:
    """Count failed file_patch dispatches that look like stale-target loops."""
    count = 0
    needles = (
        "patch_target_not_found",
        "target text not found",
        "target_text_not_found",
        "exact text",
        "repair_cycle_read_required",
        "fresh_file_read_required_before_patch",
        "reading the target file before patching",
        "old target text is gone",
        "patch already landed",
        "already matches",
        "no changes needed",
        "repeat_sensitive_patch_already_applied",
    )
    for rec in failed_dispatches:
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            continue
        tool_name = str(data.get("tool_name") or "").strip()
        if tool_name != "file_patch":
            continue
        text = json.dumps(data, default=str, ensure_ascii=False).lower()
        if any(needle in text for needle in needles):
            count += 1
    return count


def has_ask_human_resume_terminal_stall(
    records: Iterable[dict[str, Any]],
    after_task_id: str | None = None,
) -> bool:
    """Detect terminal-only tool exposure/stall after an interrupt resume."""
    saw_interrupt_resume = False
    saw_terminal_only_after_resume = False
    saw_stall_after_resume = False
    for rec in records:
        event = str(rec.get("event") or "")
        data = rec.get("data") or {}
        if event == "interrupt_resume":
            saw_interrupt_resume = True
            saw_terminal_only_after_resume = False
            saw_stall_after_resume = False
            continue
        if not saw_interrupt_resume:
            continue
        if after_task_id and extract_trace_id(rec):
            tid = extract_trace_id(rec) or ""
            if not (":" + after_task_id + ":" in tid or tid.endswith(":" + after_task_id)):
                continue
        if event == "chat_tool_selection" and isinstance(data, dict):
            tool_names = data.get("tool_names") or []
            reason = str(data.get("reason") or "").strip()
            if reason in {"non_lookup_chat_terminal_only", "smalltalk_terminal_only"}:
                saw_terminal_only_after_resume = True
            elif tool_names and set(tool_names).issubset({"task_complete", "task_fail"}):
                saw_terminal_only_after_resume = True
        if event in {"action_stall", "model_output_degenerate_loop_exhausted"}:
            saw_stall_after_resume = True
    return saw_interrupt_resume and saw_terminal_only_after_resume and saw_stall_after_resume


def has_chat_terminal_repetition_stall(
    records: Iterable[dict[str, Any]],
    session: dict[str, Any],
) -> bool:
    """Detect a chat-mode final task that ended with a degenerate loop and no tools."""
    last_task_id = str(session.get("latest_task_id") or session.get("current_task_id") or "").strip()
    if not last_task_id:
        return False

    saw_terminal_only = False
    saw_degenerate = False
    saw_chat_completed = False
    tool_call_count = 0
    for rec in records:
        tid = extract_trace_id(rec) or ""
        if not (":" + last_task_id + ":" in tid or tid.endswith(":" + last_task_id)):
            continue
        event = str(rec.get("event") or "")
        data = rec.get("data") or {}
        if event == "chat_tool_selection" and isinstance(data, dict):
            tool_names = data.get("tool_names") or []
            reason = str(data.get("reason") or "").strip()
            if reason in {"non_lookup_chat_terminal_only", "smalltalk_terminal_only"}:
                saw_terminal_only = True
            elif tool_names and set(tool_names).issubset({"task_complete", "task_fail"}):
                saw_terminal_only = True
        if event == "model_output_degenerate_loop_exhausted":
            saw_degenerate = True
        if event == "dispatch_tools_start":
            tool_call_count += 1
        if event == "task_finalize" and isinstance(data, dict):
            result = data.get("result", {})
            if str(result.get("status") or "").strip().lower() == "chat_completed":
                saw_chat_completed = True
    return saw_terminal_only and saw_degenerate and saw_chat_completed and tool_call_count == 0


def get_run_objective(
    session: dict[str, Any],
    task_summary: dict[str, Any] | None = None,
    task_summaries: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Return the best available task/objective text for a run."""
    for source in (session, task_summary or {}):
        for key in ("objective", "task", "current_task", "user_request", "original_task"):
            val = source.get(key)
            if val:
                return str(val).strip()
    if task_summaries:
        for t in task_summaries:
            for key in ("objective", "task", "task_text", "original_task"):
                val = t.get(key)
                if val:
                    return str(val).strip()
    return ""


def has_stderr_signature_circuit_breaker(records: Iterable[dict[str, Any]]) -> bool:
    """Detect whether the harness's stderr-signature circuit breaker tripped."""
    for rec in records:
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            continue
        if data.get("_stderr_signature_circuit_breaker") or data.get("stderr_signature_circuit_breaker"):
            return True
        message = str(rec.get("message") or "")
        if "stderr_signature_circuit_breaker" in message:
            return True
        if isinstance(data.get("recent_errors"), list):
            for err in data["recent_errors"]:
                if "stderr_signature_circuit_breaker" in str(err):
                    return True
    return False


def detect_apt_deb822_guard_misfire(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect apt_deb822 preflight blocks that occur after the validator already passed.

    A guard misfire happens when the preflight validator approved the environment
    but a later dispatch still records an apt_deb822_preflight_blocked outcome.
    These are harness/policy symptoms, not model failures, and should be surfaced
    separately so agents do not chase them as degeneration.
    """
    misfires: list[dict[str, Any]] = []
    validator_passed_traces: set[str] = set()
    validator_passed_globally = False

    for rec in records:
        event = str(rec.get("event") or "")
        data = rec.get("data") or {}
        tid = extract_trace_id(rec) or ""

        is_pass = (
            "apt_deb822_preflight_validator_passed" in event
            or "apt_deb822_validator_passed" in event
        )
        is_block = (
            "apt_deb822_preflight_blocked" in event
            or (
                isinstance(data, dict)
                and "apt_deb822_preflight_blocked" in str(data.get("reason") or "")
            )
        )

        if is_pass:
            validator_passed_globally = True
            if tid:
                validator_passed_traces.add(tid)

        if is_block:
            if tid and tid in validator_passed_traces:
                misfires.append(rec)
            elif validator_passed_globally and not tid:
                misfires.append(rec)

    return misfires
