from __future__ import annotations

from typing import Any

from .config import default_ttl_steps
from .signals import ActiveMitigation, FamaFailureKind, FamaSignal, current_step


MITIGATION_RULES: dict[FamaFailureKind, list[str]] = {
    FamaFailureKind.EARLY_STOP: ["done_gate", "acceptance_checklist_capsule"],
    FamaFailureKind.LOOPING: ["micro_plan_capsule", "evidence_reuse_capsule", "tool_exposure_narrowing", "evidence_gathering_needed"],
    FamaFailureKind.REMOTE_LOCAL_CONFUSION: ["remote_scope_capsule", "remote_tool_exposure_guard"],
    FamaFailureKind.REMOTE_VERIFICATION_PENDING: ["remote_scope_capsule", "remote_verification_pending_capsule"],
    FamaFailureKind.TOOL_OUTPUT_MISREAD: ["evidence_reuse_capsule", "acceptance_checklist_capsule"],
    FamaFailureKind.BAD_TOOL_ARGS: ["micro_plan_capsule"],
    FamaFailureKind.WRITE_SESSION_STALL: ["write_session_recovery_capsule", "outline_only_recovery"],
    FamaFailureKind.BACKEND_STREAM_HALT: ["micro_plan_capsule", "outline_only_recovery"],
    FamaFailureKind.CONTEXT_DRIFT: ["micro_plan_capsule", "evidence_gathering_needed", "evidence_gathering_needed_hard_route", "remote_scope_capsule"],
    FamaFailureKind.PREFLIGHT_CONTRADICTION: ["preflight_contradiction_capsule", "micro_plan_capsule"],
    FamaFailureKind.STALE_SUCCESS_CLAIM: ["acceptance_checklist_capsule", "evidence_reuse_capsule"],
    FamaFailureKind.OBJECTIVE_MISMATCH: ["acceptance_checklist_capsule", "evidence_reuse_capsule"],
    FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE: ["repeated_remote_installer_failure_capsule", "preflight_contradiction_capsule", "evidence_reuse_capsule", "micro_plan_capsule"],
    FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS: ["preexisting_state_as_success_capsule", "acceptance_checklist_capsule", "evidence_reuse_capsule"],
}


def route_signal(signal: FamaSignal, *, state: Any, config: Any) -> list[ActiveMitigation]:
    names = list(MITIGATION_RULES.get(signal.kind, []))
    if signal.failure_class == "zero_tests_discovered" and "zero_test_recovery_capsule" not in names:
        names.append("zero_test_recovery_capsule")
    # Timeout / infinite-loop signatures should not trap the agent behind done_gate.
    # A hanging script needs a rewrite, not an acceptance checklist.
    if signal.failure_class in {"verifier_timeout", "infinite_loop_suspected"}:
        names = [n for n in names if n not in {"done_gate", "acceptance_checklist_capsule"}]
        if "rewrite_suggestion_capsule" not in names:
            names.append("rewrite_suggestion_capsule")
    if not names:
        return []
    step = current_step(state)
    expires_after_step = step + default_ttl_steps(config)
    source_signal = f"{signal.kind.value}:{signal.step}:{signal.tool_name or ''}"
    return [
        ActiveMitigation(
            name=name,
            reason=signal.evidence,
            source_signal=source_signal,
            activated_step=step,
            expires_after_step=expires_after_step,
        )
        for name in names
    ]
