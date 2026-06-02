#!/usr/bin/env python3
"""Enhanced log review for benchmark runs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path



@dataclass
class LogReview:
    log_dir: str = ""
    ask_human_calls: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    guard_events: list[dict] = field(default_factory=list)
    tool_failures: list[dict] = field(default_factory=list)
    step_count: int = 0
    model_finish_reason: str | None = None
    task_status: str | None = None
    ambiguity_hints: list[str] = field(default_factory=list)
    stall_classification: str | None = None
    final_class: str | None = None
    verified: bool | None = None
    backend: bool = False
    session_id: str | None = None
    run_id: str | None = None
    guard_event_summary: list[dict] = field(default_factory=list)


def correlate_run_ids(run_log_path: Path, harness_log_dirs: list[Path]) -> dict[str, str]:
    """Map challenge IDs from run log to nearest harness session IDs by timestamp."""
    mapping: dict[str, str] = {}
    if not run_log_path.exists():
        return mapping

    # Parse run log for challenge start times
    challenge_times: list[tuple[str, float]] = []
    for line in run_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("event") == "runner_status" and "challenge_id" in entry.get("data", {}):
                ts = entry.get("timestamp", "")
                if ts:
                    # Parse ISO timestamp
                    ts_clean = ts.replace("Z", "+00:00")
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts_clean)
                    challenge_times.append((
                        entry["data"]["challenge_id"],
                        dt.timestamp()
                    ))
        except Exception:
            continue

    # Map each harness log dir to its session_id and mtime
    harness_sessions: list[tuple[str, float]] = []
    for log_dir in harness_log_dirs:
        session_id = log_dir.name
        try:
            mtime = log_dir.stat().st_mtime
            harness_sessions.append((session_id, mtime))
        except Exception:
            continue

    # Match challenges to nearest harness session by time
    for challenge_id, challenge_time in challenge_times:
        best_session = None
        best_delta = float("inf")
        for session_id, session_time in harness_sessions:
            delta = abs(session_time - challenge_time)
            if delta < best_delta:
                best_delta = delta
                best_session = session_id
        if best_session:
            mapping[challenge_id] = best_session

    return mapping


def _build_guard_event_summary(guard_events: list[dict]) -> list[dict]:
    """Return top 5 guard events by frequency."""
    from collections import Counter
    event_names = [e.get("event", "unknown") for e in guard_events]
    counts = Counter(event_names)
    return [
        {"event": event, "count": count}
        for event, count in counts.most_common(5)
    ]


_GUARD_NAME_PATTERNS = re.compile(
    r"guard|blocked|loop|fama|verifier|recovery|stall|task_complete_.*_loop_status",
    re.IGNORECASE,
)

_METADATA_GUARD_KEYS = {"active_mitigation", "reason", "error_kind", "recovery_kind"}

_ASK_HUMAN_TOOL_NAME = "ask_human"


def review_logs(log_dir: Path) -> LogReview:
    """Scan smallctl JSONL logs for guards, errors, ask_human, and summary."""
    result = LogReview(log_dir=str(log_dir))

    # Derive session_id / run_id from log directory name (e.g., "session-id-20240115-120000")
    dir_name = log_dir.name
    parts = dir_name.split("-", 1)
    if len(parts) == 2 and parts[0]:
        result.run_id = parts[0]
        result.session_id = dir_name
    else:
        result.session_id = dir_name

    # Try task_summary.json first for high-level metadata
    summary_path = log_dir / "task_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result.stall_classification = summary.get("stall_classification")
            result.verified = summary.get("challenge_progress", {}).get("verified_after_last_change", False)
            err_type = summary.get("error", {}).get("type") or summary.get("error_type")
            if err_type in {"backend_stream_failure", "provider", "model_unloaded"}:
                result.backend = True
        except Exception:
            pass

    harness_path = log_dir / "harness.jsonl"
    tools_path = log_dir / "tools.jsonl"
    chat_path = log_dir / "chat.jsonl"
    model_path = log_dir / "model_output.jsonl"

    # Text fallback paths
    harness_log_path = log_dir / "harness.log"
    tools_log_path = log_dir / "tools.log"
    model_log_path = log_dir / "model_output.log"

    seen_ask_human_ids: set[str] = set()

    def _dedup_id(ev: dict) -> str:
        data = ev.get("data", {}) or {}
        return str(
            data.get("operation_id")
            or data.get("tool_call_id")
            or data.get("id")
            or ev.get("timestamp", "")
        )

    def _is_guard_event(ev: dict) -> bool:
        event = ev.get("event", "")
        if isinstance(event, str):
            if _GUARD_NAME_PATTERNS.search(event):
                return True
            if event == "context_limit":
                return True
        data = ev.get("data", {}) or {}
        if isinstance(data, dict):
            for k in _METADATA_GUARD_KEYS:
                if k in data:
                    return True
            meta = data.get("metadata", {}) or {}
            for k in _METADATA_GUARD_KEYS:
                if k in meta:
                    return True
        return False

    if harness_path.exists():
        for line in harness_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                data = ev.get("data", {}) or {}
                event = ev.get("event", "")
                if event in ("task_start", "runtime_execution"):
                    result.task_status = "started"
                if event == "task_complete":
                    result.task_status = "completed"
                if event == "task_fail":
                    result.task_status = "failed"
                if _is_guard_event(ev):
                    guard_entry = {
                        "event": event,
                        "message": ev.get("message", ""),
                        "timestamp": ev.get("timestamp", ""),
                        "data": data,
                    }
                    # Deduplicate: skip if identical to last event
                    if not result.guard_events or result.guard_events[-1] != guard_entry:
                        result.guard_events.append(guard_entry)
                if event == "tool_result_verdict" and isinstance(data, dict):
                    verdict = data.get("verdict")
                    if verdict not in (None, "continue", "ok"):
                        guard_entry = {
                            "event": "tool_result_verdict",
                            "message": ev.get("message", ""),
                            "verdict": verdict,
                            "timestamp": ev.get("timestamp", ""),
                            "data": data,
                        }
                        if not result.guard_events or result.guard_events[-1] != guard_entry:
                            result.guard_events.append(guard_entry)
            except Exception:
                continue
    elif harness_log_path.exists():
        # coarse text fallback
        text = harness_log_path.read_text(encoding="utf-8")
        for pat in ("guard", "blocked", "loop", "fama", "verifier", "recovery", "stall"):
            if pat in text.lower():
                result.guard_events.append({"event": f"text_fallback_{pat}", "message": "detected in harness.log"})

    if tools_path.exists():
        for line in tools_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                data = ev.get("data", {}) or {}
                event = ev.get("event", "")
                if event == "dispatch_start" and isinstance(data, dict):
                    if data.get("tool_name") == _ASK_HUMAN_TOOL_NAME:
                        dedup = _dedup_id(ev)
                        if dedup not in seen_ask_human_ids:
                            seen_ask_human_ids.add(dedup)
                            result.ask_human_calls.append({
                                "question": data.get("arguments", {}).get("question", "(unknown)"),
                                "timestamp": ev.get("timestamp", ""),
                                "source": "dispatch_start",
                            })
                if event == "dispatch_complete" and isinstance(data, dict):
                    if data.get("tool_name") == _ASK_HUMAN_TOOL_NAME:
                        dedup = _dedup_id(ev)
                        if dedup not in seen_ask_human_ids:
                            seen_ask_human_ids.add(dedup)
                            result.ask_human_calls.append({
                                "question": data.get("arguments", {}).get("question", "(unknown)"),
                                "timestamp": ev.get("timestamp", ""),
                                "source": "dispatch_complete",
                            })
                    if not data.get("success", True):
                        result.tool_failures.append({
                            "tool_name": data.get("tool_name"),
                            "error": data.get("error"),
                            "timestamp": ev.get("timestamp", ""),
                            "source": "dispatch_complete",
                        })
                    else:
                        # dispatch_complete success but still a failure-like result
                        result_data = data.get("result", {}) or {}
                        if isinstance(result_data, dict) and result_data.get("success") is False:
                            result.tool_failures.append({
                                "tool_name": data.get("tool_name"),
                                "error": result_data.get("error") or data.get("error"),
                                "timestamp": ev.get("timestamp", ""),
                                "source": "dispatch_complete_result_failure",
                            })
            except Exception:
                continue
    elif tools_log_path.exists():
        text = tools_log_path.read_text(encoding="utf-8")
        for m in re.finditer(r"ask_human", text, re.IGNORECASE):
            result.ask_human_calls.append({"question": "(detected in tools.log)", "source": "text_fallback"})

    if chat_path.exists():
        for line in chat_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                data = ev.get("data", {}) or {}
                event = ev.get("event", "")
                if event == "conversation_history" and isinstance(data, dict):
                    hist = data.get("history", [])
                    result.step_count = data.get("step", 0) or result.step_count
                    if hist and len(hist) > 0:
                        last = hist[-1]
                        if last.get("role") == "assistant":
                            result.model_finish_reason = "assistant_message"
                if event == "error":
                    result.errors.append({
                        "message": ev.get("message", ""),
                        "timestamp": ev.get("timestamp", ""),
                    })
            except Exception:
                continue

    if model_path.exists():
        lines = model_path.read_text(encoding="utf-8").splitlines()
        if lines:
            try:
                last = json.loads(lines[-1])
                content = last.get("data", {}).get("chunk", {}).get("choices", [{}])[0].get("delta", {}).get("content", "")
                if any(q in content.lower() for q in ("what should i", "clarify", "unclear", "ambiguous", "do you want")):
                    result.ambiguity_hints.append(content[:200])
            except Exception:
                pass
    elif model_log_path.exists():
        text = model_log_path.read_text(encoding="utf-8")
        if any(q in text.lower() for q in ("what should i", "clarify", "unclear", "ambiguous", "do you want")):
            result.ambiguity_hints.append("(detected in model_output.log)")

    result.guard_event_summary = _build_guard_event_summary(result.guard_events)
    return result
