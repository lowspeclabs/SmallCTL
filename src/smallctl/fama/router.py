from __future__ import annotations

import re
from typing import Any

from .config import default_ttl_steps
from .signals import ActiveMitigation, FamaFailureKind, FamaSignal, current_step


_AUTH_FAILURE_EVIDENCE_RE = re.compile(
    r"(?:permission denied|authentication failed|publickey|password required|"
    r"connection refused|host key verification|no route to host|"
    r"could not resolve hostname|name or service not known)",
    re.IGNORECASE,
)


MITIGATION_RULES: dict[FamaFailureKind, list[str]] = {
    FamaFailureKind.EARLY_STOP: ["done_gate", "acceptance_checklist_capsule"],
    FamaFailureKind.LOOPING: ["micro_plan_capsule", "evidence_reuse_capsule", "tool_exposure_narrowing", "evidence_gathering_needed"],
    FamaFailureKind.INTERACTIVE_SESSION_STALL: ["interactive_installer_stall_capsule", "tool_exposure_narrowing", "evidence_reuse_capsule"],
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
    FamaFailureKind.UPSTREAM_INSTALL_SOURCE_INVALID: ["source_invalid_install_capsule", "evidence_reuse_capsule", "tool_exposure_narrowing", "micro_plan_capsule"],
    FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS: ["preexisting_state_as_success_capsule", "acceptance_checklist_capsule", "evidence_reuse_capsule"],
    FamaFailureKind.SSH_HOST_KEY_VERIFICATION: ["ssh_host_key_recovery_capsule"],
}


def route_signal(signal: FamaSignal, *, state: Any, config: Any) -> list[ActiveMitigation]:
    names = list(MITIGATION_RULES.get(signal.kind, []))
    # SSH auth / environment failures: replace done_gate with a
    # fallback suggestion so the model tries a different approach
    # instead of retrying the same failing SSH command.
    if signal.tool_name == "ssh_exec" and signal.failure_class == "verifier_failed":
        if _AUTH_FAILURE_EVIDENCE_RE.search(signal.evidence):
            names = [n for n in names if n not in {"done_gate", "acceptance_checklist_capsule"}]
            if "remote_auth_failure_capsule" not in names:
                names.append("remote_auth_failure_capsule")
            if "micro_plan_capsule" not in names:
                names.append("micro_plan_capsule")
    if signal.failure_class == "zero_tests_discovered" and "zero_test_recovery_capsule" not in names:
        names.append("zero_test_recovery_capsule")
    if signal.failure_class == "patch_target_not_found" and "patch_target_not_found_capsule" not in names:
        names.append("patch_target_not_found_capsule")
    # Timeout / infinite-loop signatures should not trap the agent behind done_gate.
    # A hanging script needs a rewrite, not an acceptance checklist.
    if signal.failure_class in {"verifier_timeout", "infinite_loop_suspected"}:
        names = [n for n in names if n not in {"done_gate", "acceptance_checklist_capsule"}]
        if "rewrite_suggestion_capsule" not in names:
            names.append("rewrite_suggestion_capsule")
    if not names:
        return []
    # P1.2: when source is loop_guard, ensure tool_exposure_narrowing is present
    # to break the repeat cycle directly
    if signal.source == "loop_guard" and "tool_exposure_narrowing" not in names:
        names = list(names) + ["tool_exposure_narrowing"]
    # Severity-3 escalation for repeated early_stop: pivot from blocking
    # completion to admitting defeat, so the model does not cycle
    # indefinitely on an unfixable external blocker.
    if signal.kind == FamaFailureKind.EARLY_STOP and signal.severity >= 3:
        names = [n for n in names if n != "done_gate"]
        if "dead_end_pivot_capsule" not in names:
            names.append("dead_end_pivot_capsule")
        if "micro_plan_capsule" not in names:
            names.append("micro_plan_capsule")
    step = current_step(state)
    expires_after_step = step + default_ttl_steps(config)
    source_signal = f"{signal.kind.value}:{signal.step}:{signal.tool_name or ''}"
    reason = signal.evidence
    if signal.kind == FamaFailureKind.SSH_HOST_KEY_VERIFICATION and signal.next_safe_action:
        reason = signal.next_safe_action
    return [
        ActiveMitigation(
            name=name,
            reason=reason,
            source_signal=source_signal,
            activated_step=step,
            expires_after_step=expires_after_step,
        )
        for name in names
    ]
