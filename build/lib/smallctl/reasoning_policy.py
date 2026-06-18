from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .phases import PHASES
from .state import ClaimRecord, LoopState

_RISKY_TOOL_NAMES = {"shell_exec", "ssh_exec", "file_write", "file_append", "file_patch", "ast_patch", "file_delete"}
_TASK_CLASSIFICATION_ALIASES = {
    "diagnosis": "diagnosis_remediation",
    "incident_response": "diagnosis_remediation",
    "remediation": "diagnosis_remediation",
    "repair": "diagnosis_remediation",
    "diagnosis_remediation": "diagnosis_remediation",
    "implementation": "implementation",
    "authoring": "implementation",
    "coding": "implementation",
}
_PHASE_NAMES = set(PHASES)


@dataclass(frozen=True)
class ClaimGateResult:
    allowed: bool
    reason: str = ""
    proof_bundle: dict[str, Any] | None = None


def classify_task(state: LoopState) -> str:
    explicit_values: list[str] = []
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict):
        explicit_values.extend(
            str(scratchpad.get(key) or "").strip()
            for key in ("_task_classification", "task_classification", "task_type")
        )
    strategy = getattr(state, "strategy", None)
    if isinstance(strategy, dict):
        explicit_values.extend(
            str(strategy.get(key) or "").strip()
            for key in ("task_classification", "task_type", "task_mode")
        )
    active_intent = str(getattr(state, "active_intent", "") or "").strip()
    if active_intent:
        explicit_values.append(active_intent)
    explicit_values.extend(
        str(tag).strip()
        for tag in getattr(state, "intent_tags", []) or []
        if str(tag).strip() and str(tag).strip().lower() not in _PHASE_NAMES
    )

    for value in explicit_values:
        normalized = value.lower().replace(" ", "_").replace("-", "_")
        if normalized in _TASK_CLASSIFICATION_ALIASES:
            return _TASK_CLASSIFICATION_ALIASES[normalized]
        if normalized in {"diagnosis_remediation", "implementation"}:
            return normalized
    return "implementation"


def task_requires_claim_support(state: LoopState) -> bool:
    return classify_task(state) == "diagnosis_remediation"


def supported_claim_records(state: LoopState) -> list[ClaimRecord]:
    reasoning_graph = getattr(state, "reasoning_graph", None)
    if reasoning_graph is None:
        return []
    claims: list[ClaimRecord] = []
    for claim in getattr(reasoning_graph, "claim_records", []) or []:
        if not isinstance(claim, ClaimRecord):
            continue
        if claim.status != "confirmed":
            continue
        if not claim.supporting_evidence_ids:
            continue
        claims.append(claim)
    return claims


def has_supported_claim(state: LoopState) -> bool:
    return bool(supported_claim_records(state))


def missing_supported_claim_message(tool_name: str) -> str:
    return (
        f"{tool_name} is blocked for diagnosis/remediation work until a supported claim exists. "
        "A supported claim must be a confirmed claim backed by actual tool evidence. "
        "`memory_update`, session notes, plans, or restating the intended command do not count. "
        "Capture real read-only evidence first."
    )


def build_claim_proof_bundle(
    state: LoopState,
    *,
    tool_name: str = "",
    action: str = "",
    expected_effect: str = "",
    rollback: str = "",
    verification: str = "",
) -> dict[str, Any]:
    claims = supported_claim_records(state)
    return {
        "tool_name": tool_name,
        "action": action,
        "expected_effect": expected_effect,
        "rollback": rollback,
        "verification": verification,
        "task_requires_claim_support": task_requires_claim_support(state),
        "supported_claim_ids": [claim.claim_id for claim in claims],
        "supporting_evidence_ids": sorted(
            {evidence_id for claim in claims for evidence_id in claim.supporting_evidence_ids}
        ),
        "claims": [
            {
                "claim_id": claim.claim_id,
                "kind": claim.kind,
                "statement": claim.statement,
                "status": claim.status,
                "confidence": claim.confidence,
            }
            for claim in claims
        ],
    }


def claim_gate_for_risky_action(
    state: LoopState,
    *,
    tool_name: str,
    action: str = "",
    expected_effect: str = "",
    rollback: str = "",
    verification: str = "",
) -> ClaimGateResult:
    if tool_name not in _RISKY_TOOL_NAMES:
        return ClaimGateResult(allowed=True)
    if not task_requires_claim_support(state):
        return ClaimGateResult(allowed=True)
    if has_supported_claim(state):
        return ClaimGateResult(allowed=True)

    proof_bundle = build_claim_proof_bundle(
        state,
        tool_name=tool_name,
        action=action,
        expected_effect=expected_effect,
        rollback=rollback,
        verification=verification,
    )
    return ClaimGateResult(
        allowed=False,
        reason=missing_supported_claim_message(tool_name),
        proof_bundle=proof_bundle,
    )
