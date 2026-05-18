from __future__ import annotations

from typing import Any


_DIAGNOSTIC_MARKERS = (
    " rca ",
    "root cause",
    "postmortem",
    "diagnose",
    "diagnostic",
    "investigate",
    "why ",
    "what failed",
    "failure analysis",
)

_NEGATIVE_VERIFICATION_MARKERS = (
    "not up",
    "site is not up",
    "not working",
    "not responding",
    "down",
    "verify failure",
    "verify failing",
    "verify it fails",
    "verify it is failing",
    "verify it is not",
    "verify that it is not",
    "verify, site is not up",
)

_FAILURE_REPORT_MARKERS = (
    "fail",
    "failed",
    "failing",
    "failure",
    "not up",
    "not working",
    "not responding",
    "could not connect",
    "cannot connect",
    "connection refused",
    "connection timed out",
    "timed out",
    "timeout",
    "missing",
    "not found",
    "blocked",
    "down",
    "rca",
    "root cause",
)


def diagnostic_failure_task(state: Any) -> bool:
    text = _state_task_text(state)
    if not text:
        return False
    padded = f" {text} "
    if any(marker in padded for marker in _DIAGNOSTIC_MARKERS):
        return True
    return any(marker in text for marker in _NEGATIVE_VERIFICATION_MARKERS)


def diagnostic_completion_reports_failure(message: str, verifier: Any) -> bool:
    if not isinstance(verifier, dict):
        return False
    if str(verifier.get("verdict") or "").strip().lower() not in {"fail", "failed", "error"}:
        return False
    text = str(message or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _FAILURE_REPORT_MARKERS):
        return True
    verifier_text = " ".join(
        str(verifier.get(key) or "").strip().lower()
        for key in ("failure_mode", "key_stdout", "key_stderr", "command", "target")
    )
    return bool(verifier_text and any(marker in verifier_text for marker in _FAILURE_REPORT_MARKERS))


def diagnostic_failure_completion_allowed(state: Any, *, message: str, verifier: Any) -> bool:
    return diagnostic_failure_task(state) and diagnostic_completion_reports_failure(message, verifier)


def _state_task_text(state: Any) -> str:
    bits: list[str] = []
    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)
    for value in (
        getattr(run_brief, "original_task", ""),
        getattr(working_memory, "current_goal", ""),
        getattr(state, "active_intent", ""),
        " ".join(str(item) for item in (getattr(state, "secondary_intents", []) or [])),
        " ".join(str(item) for item in (getattr(state, "intent_tags", []) or [])),
    ):
        value = str(value or "").strip()
        if value:
            bits.append(value)
    return " ".join(bits).casefold()
