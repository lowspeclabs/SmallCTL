from __future__ import annotations

from typing import Any

from ..state import json_safe_value


def cancellation_result(state: Any, *, reason: str) -> dict[str, Any]:
    verifier = getattr(state, "last_verifier_verdict", None)
    verifier = verifier if isinstance(verifier, dict) else {}
    failure_events = list(getattr(state, "failure_events", []) or [])
    recent_errors = list(getattr(state, "recent_errors", []) or [])
    verifier_failed = bool(
        verifier
        and str(verifier.get("verdict") or "").strip().lower() not in {"", "pass"}
    )
    result: dict[str, Any] = {
        "status": "cancelled_after_verifier_failure" if verifier_failed else "cancelled",
        "reason": reason,
    }
    if verifier:
        result["last_verifier_verdict"] = json_safe_value(verifier)
    if failure_events:
        result["failure_events"] = json_safe_value(failure_events[-3:])
    if recent_errors:
        result["recent_errors"] = json_safe_value(recent_errors[-3:])
    return result


def cancellation_message(state: Any) -> str:
    result = cancellation_result(state, reason="cancel_requested")
    verifier = result.get("last_verifier_verdict")
    if not isinstance(verifier, dict) or result.get("status") != "cancelled_after_verifier_failure":
        return "Run cancelled."
    bits = [
        str(verifier.get("verdict") or "fail").strip(),
        str(verifier.get("failure_mode") or "").strip(),
    ]
    exit_code = verifier.get("exit_code")
    if exit_code not in (None, ""):
        bits.append(f"exit {exit_code}")
    target = str(verifier.get("target") or verifier.get("command") or "").strip()
    if target:
        bits.append(target)
    summary = " | ".join(bit for bit in bits if bit)
    return f"Run cancelled with failing verifier: {summary}" if summary else "Run cancelled with failing verifier."
