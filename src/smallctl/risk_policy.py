from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import shell_utils as _shell_attempts
from .reasoning_policy import (
    build_claim_proof_bundle,
    classify_task as _classify_task,
    has_supported_claim,
    missing_supported_claim_message,
    task_requires_claim_support,
)
from .state import LoopState

_READ_ONLY_PHASES = {"explore", "verify"}
_SHELL_TOOLS = {"shell_exec", "ssh_exec"}


@dataclass(frozen=True)
class RiskPolicyDecision:
    allowed: bool
    requires_approval: bool = False
    reason: str = ""
    proof_bundle: dict[str, Any] | None = None
    tool_risk: str = "medium"
    task_classification: str = "implementation"
    approval_kind: str = ""


def classify_task(state: LoopState) -> str:
    return _classify_task(state)


def build_risk_proof_bundle(
    state: LoopState,
    *,
    tool_name: str,
    tool_risk: str,
    phase: str,
    action: str = "",
    expected_effect: str = "",
    rollback: str = "",
    verification: str = "",
) -> dict[str, Any]:
    task_classification = classify_task(state)
    bundle = build_claim_proof_bundle(
        state,
        tool_name=tool_name,
        action=action,
        expected_effect=expected_effect,
        rollback=rollback,
        verification=verification,
    )
    bundle.update(
        {
            "phase": phase,
            "tool_risk": tool_risk,
            "task_classification": task_classification,
            "approval_kind": "shell" if tool_name in _SHELL_TOOLS else "",
            "read_only_phase": phase in _READ_ONLY_PHASES,
        }
    )
    return bundle


def evaluate_risk_policy(
    state: LoopState,
    *,
    tool_name: str,
    tool_risk: str,
    phase: str,
    action: str = "",
    expected_effect: str = "",
    rollback: str = "",
    verification: str = "",
    approval_available: bool = False,
) -> RiskPolicyDecision:
    task_classification = classify_task(state)
    proof_bundle = build_risk_proof_bundle(
        state,
        tool_name=tool_name,
        tool_risk=tool_risk,
        phase=phase,
        action=action,
        expected_effect=expected_effect,
        rollback=rollback,
        verification=verification,
    )
    if tool_risk in {"low", "network_read"}:
        return RiskPolicyDecision(
            allowed=True,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
        )

    if (
        task_classification == "diagnosis_remediation"
        and tool_name in _SHELL_TOOLS
        and _is_read_only_evidence_action(action)
    ):
        requires_approval = approval_available and tool_name in _SHELL_TOOLS and tool_risk in {"medium", "high"}
        return RiskPolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell" if tool_name in _SHELL_TOOLS else "",
        )

    if (
        task_classification == "diagnosis_remediation"
        and tool_name == "ssh_exec"
        and not has_supported_claim(state)
        and _ssh_first_probe_allowed(state)
    ):
        requires_approval = approval_available and tool_risk in {"medium", "high"}
        return RiskPolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell",
        )

    if task_classification == "diagnosis_remediation" and not has_supported_claim(state):
        return RiskPolicyDecision(
            allowed=False,
            reason=missing_supported_claim_message(tool_name),
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell" if tool_name in _SHELL_TOOLS else "generic",
        )

    requires_approval = approval_available and tool_name in _SHELL_TOOLS and tool_risk in {"medium", "high"}
    return RiskPolicyDecision(
        allowed=True,
        requires_approval=requires_approval,
        proof_bundle=proof_bundle,
        tool_risk=tool_risk,
        task_classification=task_classification,
        approval_kind="shell" if tool_name in _SHELL_TOOLS else "generic" if tool_risk == "high" else "",
    )



_SSH_FIRST_PROBE_LIMIT = 1
_is_read_only_evidence_action = _shell_attempts.is_read_only_shell_evidence_action


def _ssh_first_probe_allowed(state: LoopState) -> bool:
    """Allow a limited number of ssh_exec probes before requiring a claim."""
    records = getattr(state, "tool_execution_records", None)
    if not isinstance(records, dict):
        return True
    ssh_attempt_count = 0
    for _op_id, record in records.items():
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "").strip() == "ssh_exec":
            ssh_attempt_count += 1
    return ssh_attempt_count < _SSH_FIRST_PROBE_LIMIT
