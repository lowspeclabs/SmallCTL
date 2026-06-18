from __future__ import annotations

from datetime import datetime
from typing import Any

from ..state import clip_text_value


def clip_task_summary_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    clipped, truncated = clip_text_value(text, limit=limit)
    return f"{clipped} [truncated]" if truncated else clipped


def extract_task_terminal_message(result: dict[str, Any] | None, *, limit: int = 240) -> str:
    if not isinstance(result, dict) or not result:
        return ""
    message = result.get("message")
    if isinstance(message, dict):
        candidate = message.get("message") or message.get("question") or message.get("status")
        if candidate:
            return clip_task_summary_text(candidate, limit=limit)
    if isinstance(message, str) and message.strip():
        return clip_task_summary_text(message, limit=limit)
    reason = str(result.get("reason") or "").strip()
    if reason:
        return clip_task_summary_text(reason, limit=limit)
    error = result.get("error")
    if isinstance(error, dict):
        candidate = error.get("message")
        if candidate:
            return clip_task_summary_text(candidate, limit=limit)
    return ""


def task_duration_seconds(started_at: str, finished_at: str) -> float:
    try:
        started = datetime.fromisoformat(str(started_at))
        finished = datetime.fromisoformat(str(finished_at))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, round((finished - started).total_seconds(), 3))
