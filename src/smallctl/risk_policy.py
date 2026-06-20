from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from . import shell_utils as _shell_attempts
from .shell_utils import looks_like_ssh_keygen_known_hosts_removal as _is_ssh_keygen_known_hosts_removal
from .reasoning_policy import (
    build_claim_proof_bundle,
    classify_task as _classify_task,
    has_supported_claim,
    missing_supported_claim_message,
    task_requires_claim_support,
)
from .state import LoopState
from .logging_utils import log_kv, synthetic_trace_id

_LOGGER = logging.getLogger("smallctl.risk_policy")

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
        decision = RiskPolicyDecision(
            allowed=True,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            trace_id = synthetic_trace_id(state, suffix="risk")
            log_kv(
                _LOGGER,
                logging.DEBUG,
                "risk_policy_decision",
                trace_id=trace_id,
                tool_name=tool_name,
                risk_level=tool_risk,
                approval_required=decision.requires_approval,
                phase_allowed=decision.allowed,
                reason="low_risk_allowed",
                task_classification=task_classification,
                phase=phase,
            )
        return decision

    if (
        task_classification == "diagnosis_remediation"
        and tool_name in _SHELL_TOOLS
        and _is_read_only_evidence_action(action)
    ):
        requires_approval = approval_available and tool_name in _SHELL_TOOLS and tool_risk in {"medium", "high"}
        decision = RiskPolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell" if tool_name in _SHELL_TOOLS else "",
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            trace_id = synthetic_trace_id(state, suffix="risk")
            log_kv(
                _LOGGER,
                logging.DEBUG,
                "risk_policy_decision",
                trace_id=trace_id,
                tool_name=tool_name,
                risk_level=tool_risk,
                approval_required=decision.requires_approval,
                phase_allowed=decision.allowed,
                reason="diagnosis_read_only_evidence",
                task_classification=task_classification,
                phase=phase,
            )
        return decision

    if (
        task_classification == "diagnosis_remediation"
        and tool_name == "shell_exec"
        and _is_ssh_keygen_known_hosts_removal(action)
    ):
        requires_approval = approval_available and tool_risk in {"medium", "high"}
        decision = RiskPolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell",
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            trace_id = synthetic_trace_id(state, suffix="risk")
            log_kv(
                _LOGGER,
                logging.DEBUG,
                "risk_policy_decision",
                trace_id=trace_id,
                tool_name=tool_name,
                risk_level=tool_risk,
                approval_required=decision.requires_approval,
                phase_allowed=decision.allowed,
                reason="ssh_keygen_known_hosts_removal",
                task_classification=task_classification,
                phase=phase,
            )
        return decision

    if (
        task_classification == "diagnosis_remediation"
        and tool_name == "ssh_exec"
        and not has_supported_claim(state)
        and _ssh_first_probe_allowed(state)
    ):
        requires_approval = approval_available and tool_risk in {"medium", "high"}
        decision = RiskPolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell",
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            trace_id = synthetic_trace_id(state, suffix="risk")
            log_kv(
                _LOGGER,
                logging.DEBUG,
                "risk_policy_decision",
                trace_id=trace_id,
                tool_name=tool_name,
                risk_level=tool_risk,
                approval_required=decision.requires_approval,
                phase_allowed=decision.allowed,
                reason="ssh_first_probe_allowed",
                task_classification=task_classification,
                phase=phase,
            )
        return decision

    if task_classification == "diagnosis_remediation" and not has_supported_claim(state):
        decision = RiskPolicyDecision(
            allowed=False,
            reason=missing_supported_claim_message(tool_name),
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell" if tool_name in _SHELL_TOOLS else "generic",
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            trace_id = synthetic_trace_id(state, suffix="risk")
            log_kv(
                _LOGGER,
                logging.DEBUG,
                "risk_policy_decision",
                trace_id=trace_id,
                tool_name=tool_name,
                risk_level=tool_risk,
                approval_required=decision.requires_approval,
                phase_allowed=decision.allowed,
                reason=decision.reason,
                task_classification=task_classification,
                phase=phase,
            )
        return decision

    requires_approval = approval_available and tool_name in _SHELL_TOOLS and tool_risk in {"medium", "high"}
    decision = RiskPolicyDecision(
        allowed=True,
        requires_approval=requires_approval,
        proof_bundle=proof_bundle,
        tool_risk=tool_risk,
        task_classification=task_classification,
        approval_kind="shell" if tool_name in _SHELL_TOOLS else "generic" if tool_risk == "high" else "",
    )
    if _LOGGER.isEnabledFor(logging.DEBUG):
        trace_id = synthetic_trace_id(state, suffix="risk")
        log_kv(
            _LOGGER,
            logging.DEBUG,
            "risk_policy_decision",
            trace_id=trace_id,
            tool_name=tool_name,
            risk_level=tool_risk,
            approval_required=decision.requires_approval,
            phase_allowed=decision.allowed,
            reason=decision.reason or "policy_allowed",
            task_classification=task_classification,
            phase=phase,
        )
    return decision



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
