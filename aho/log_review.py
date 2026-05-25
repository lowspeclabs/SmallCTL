#!/usr/bin/env python3
"""Enhanced log review for benchmark runs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


_GUARD_NAME_PATTERNS = re.compile(
    r"guard|blocked|loop|fama|verifier|recovery|stall|task_complete_.*_loop_status",
    re.IGNORECASE,
)

_METADATA_GUARD_KEYS = {"active_mitigation", "reason", "error_kind", "recovery_kind"}

_ASK_HUMAN_TOOL_NAME = "ask_human"


def review_logs(log_dir: Path) -> LogReview:
    """Scan smallctl JSONL logs for guards, errors, ask_human, and summary."""
    result = LogReview(log_dir=str(log_dir))

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
                    result.guard_events.append({
                        "event": event,
                        "message": ev.get("message", ""),
                        "timestamp": ev.get("timestamp", ""),
                        "data": data,
                    })
                if event == "tool_result_verdict" and isinstance(data, dict):
                    verdict = data.get("verdict")
                    if verdict not in (None, "continue", "ok"):
                        result.guard_events.append({
                            "event": "tool_result_verdict",
                            "message": ev.get("message", ""),
                            "verdict": verdict,
                            "timestamp": ev.get("timestamp", ""),
                            "data": data,
                        })
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

    return result
