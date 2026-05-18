from __future__ import annotations

from typing import Any

from .state import active_mitigations


def active_done_gate_fingerprints(state: Any) -> set[str]:
    fingerprints: set[str] = set()
    for mitigation in active_mitigations(state):
        if mitigation.name not in {"done_gate", "acceptance_checklist_capsule"}:
            continue
        fingerprint = fingerprint_from_reason(mitigation.reason)
        if fingerprint:
            fingerprints.add(fingerprint)
    return fingerprints


def fingerprint_from_reason(reason: str) -> str:
    text = str(reason or "").strip()
    marker = "verifier verdict "
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    if ":" not in tail:
        return ""
    return normalize_verifier_target(tail.split(":", 1)[1])


def passing_verifier_fingerprint(state: Any, *, result: Any | None = None) -> str:
    metadata = getattr(result, "metadata", None) if result is not None else None
    metadata = metadata if isinstance(metadata, dict) else {}
    verifier = metadata.get("last_verifier_verdict")
    if not isinstance(verifier, dict) or not verifier:
        current_verifier = getattr(state, "current_verifier_verdict", None)
        verifier = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    if isinstance(verifier, dict) and str(verifier.get("verdict") or "").strip().lower() == "pass":
        return normalize_verifier_target(str(verifier.get("command") or verifier.get("target") or ""))
    if str(metadata.get("verifier_verdict") or "").strip().lower() == "pass":
        return normalize_verifier_target(
            str(metadata.get("verifier_command") or metadata.get("verifier_target") or "")
        )
    return ""


def normalize_verifier_target(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.casefold()
