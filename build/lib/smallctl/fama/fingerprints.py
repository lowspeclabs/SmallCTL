from __future__ import annotations

import re
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


def _command_looks_like_install_presence_check(command: str) -> bool:
    """Return True for package-presence and service-status verifiers after an install."""
    normalized = " ".join(str(command or "").strip().lower().split())
    package_presence = bool(re.search(r"\b(dpkg\s+-l|apt\s+list|dnf\s+list\s+installed|yum\s+list\s+installed|rpm\s+-q)\s+\S+", normalized))
    service_status = bool(re.search(r"\b(systemctl\s+status|systemctl\s+is-active|service\s+\S+\s+status)\s+\S+", normalized))
    return package_presence or service_status


def _state_task_is_install_setup(state: Any) -> bool:
    task = str(
        getattr(getattr(state, "run_brief", None), "original_task", "")
        or ""
    ).strip().lower()
    return any(marker in task for marker in ("install", "setup", "deploy", "configure"))


def install_verifier_passes_objective(state: Any, *, result: Any | None = None) -> bool:
    """Return True when the latest passing verifier matches an install/setup objective."""
    if not _state_task_is_install_setup(state):
        return False
    verifier = None
    metadata = getattr(result, "metadata", None) if result is not None else None
    if isinstance(metadata, dict):
        verifier = metadata.get("last_verifier_verdict")
    if not isinstance(verifier, dict):
        current_verifier = getattr(state, "current_verifier_verdict", None)
        verifier = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    if not isinstance(verifier, dict):
        return False
    if str(verifier.get("verdict") or "").strip().lower() != "pass":
        return False
    command = str(verifier.get("command") or verifier.get("target") or "").strip()
    return _command_looks_like_install_presence_check(command)


def normalize_verifier_target(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.casefold()
