from __future__ import annotations

from typing import Any


def _harness_event_message_type() -> type[Any]:
    # Imported lazily to avoid a module cycle with app.py, which mixes this class in.
    from .app import HarnessEvent
    return HarnessEvent


def _extract_terminal_result_text(result: dict[str, Any]) -> str:
    if not isinstance(result, dict) or not result:
        return ""

    message = result.get("message")
    if isinstance(message, dict):
        for key in ("message", "output", "text", "question"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(message, str) and message.strip():
        return message.strip()

    assistant = str(result.get("assistant") or "").strip()
    if assistant:
        return assistant

    reason = str(result.get("reason") or "").strip()
    return reason
