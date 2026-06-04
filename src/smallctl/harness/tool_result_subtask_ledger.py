from __future__ import annotations

from typing import Any

from ..recovery_metrics import increment_metric


def _update_subtask_ledger_from_verifier(service: Any, verifier_verdict: dict[str, Any] | None) -> None:
    if not isinstance(verifier_verdict, dict) or not verifier_verdict:
        return
    harness = getattr(service, "harness", None)
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "subtask_ledger_enabled", True)):
        return
    ledger_service = getattr(harness, "subtask_ledger", None)
    if ledger_service is None:
        return
    try:
        ledger_service.import_plan_if_needed()
        active = ledger_service.infer_or_create_active_subtask()
        command = str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip()
        verdict = str(verifier_verdict.get("verdict") or "").strip().lower()
        if command:
            ledger_service.attach_evidence(active.subtask_id, f"verifier {verdict or 'unknown'}: {command}")
        if verdict == "pass":
            if any(item in {"verifier_failed", "test_failed"} for item in getattr(active, "failure_classes", [])):
                increment_metric(harness.state, "verifier_fail_then_success_count")
            ledger_service.mark_done_if_verified(active.subtask_id, verifier_verdict)
    except Exception:
        return
