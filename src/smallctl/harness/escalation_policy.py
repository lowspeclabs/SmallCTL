from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..recovery_metrics import increment_metric


@dataclass(frozen=True)
class EscalationPolicyDecision:
    allowed: bool
    trigger: str
    reason: str
    evidence_count: int = 0
    missing_signals: list[str] | None = None


class EscalationPolicy:
    def __init__(self, harness: Any) -> None:
        self.harness = harness

    def can_escalate(self, *, reason: str, risk_level: str = "medium") -> EscalationPolicyDecision:
        state = self.harness.state
        config = getattr(self.harness, "config", None)
        if not bool(getattr(config, "escalation_enabled", False)):
            return self._block("disabled", "Escalation is disabled.", evidence_count=0)

        scratchpad = getattr(state, "scratchpad", {})
        scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
        history = scratchpad.get("_escalation_history")
        history = history if isinstance(history, list) else []
        max_per_task = _safe_int(getattr(config, "escalation_max_per_task", 3), 3)
        if max_per_task >= 0 and len(history) >= max_per_task:
            return self._block("max_per_task", "Escalation limit reached for this task.", evidence_count=len(history))

        cooldown = _safe_int(getattr(config, "escalation_cooldown_turns", 2), 2)
        if history:
            last_step = _safe_int(history[-1].get("step_count") if isinstance(history[-1], dict) else None, -999999)
            current_step = _safe_int(getattr(state, "step_count", 0), 0)
            if current_step - last_step < cooldown:
                return self._block("cooldown", "Escalation cooldown has not elapsed.", evidence_count=len(history))

        human_block = _human_required_reason(state, scratchpad)
        if human_block:
            return self._block("approval_required", human_block, evidence_count=1)

        signals = self._stuck_signals(state, scratchpad)
        if _explicit_escalation_request(reason) and _has_prior_tool_evidence(state):
            signals.append("explicit_escalation_prior_evidence")
            signals = _dedupe(signals)
        evidence_count = len(signals)
        # In repair phase with no actionable progress, lower the evidence bar
        # so small models can escalate before they spiral into read-only loops.
        state_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        stagnation = getattr(state, "stagnation_counters", None)
        no_progress = int(stagnation.get("no_actionable_progress") or 0) if isinstance(stagnation, dict) else 0
        repair_auto_evidence = state_phase == "repair" and no_progress >= 2

        if bool(getattr(config, "escalation_require_tool_plan_evidence", True)) and not repair_auto_evidence:
            has_evidence = _has_meaningful_evidence(signals)
            if not has_evidence:
                missing = sorted(_EVIDENCE_SIGNALS - set(signals))
                return self._block(
                    "insufficient_evidence",
                    "Escalation requires verifier, tool-plan, recovery, or tool execution evidence first.",
                    evidence_count=evidence_count,
                    missing_signals=missing,
                )

        if not signals and not repair_auto_evidence:
            missing = sorted(_EVIDENCE_SIGNALS)
            return self._block(
                "not_stuck",
                "No repeated stuck signal is present yet.",
                evidence_count=0,
                missing_signals=missing,
            )

        trigger = signals[0]
        return EscalationPolicyDecision(
            allowed=True,
            trigger=trigger,
            reason=str(reason or "").strip() or f"Escalation allowed by {trigger}.",
            evidence_count=evidence_count,
        )

    def _block(
        self,
        trigger: str,
        reason: str,
        *,
        evidence_count: int,
        missing_signals: list[str] | None = None,
    ) -> EscalationPolicyDecision:
        increment_metric(self.harness.state, "escalation_policy_blocks")
        return EscalationPolicyDecision(
            False, trigger, reason, evidence_count=evidence_count, missing_signals=missing_signals
        )

    def _stuck_signals(self, state: Any, scratchpad: dict[str, Any]) -> list[str]:
        signals: list[str] = []
        threshold = max(2, _safe_int(getattr(getattr(self.harness, "config", None), "escalation_repeated_failure_threshold", 2), 2))

        failure_events = getattr(state, "failure_events", None)
        if isinstance(failure_events, list) and failure_events:
            latest_class = str(getattr(failure_events[-1], "failure_class", "") or "").strip()
            latest_kind = str(getattr(failure_events[-1], "fama_kind", "") or "").strip()
            if latest_class == "write_session_stall" or latest_kind == "write_session_stall":
                signals.append("write_session_stall")
        if isinstance(failure_events, list) and len(failure_events) >= threshold:
            if latest_class and sum(1 for event in failure_events[-threshold:] if str(getattr(event, "failure_class", "") or "") == latest_class) >= threshold:
                signals.append("repeated_failure_class")

        verifier = getattr(state, "last_verifier_verdict", None)
        if isinstance(verifier, dict) and str(verifier.get("verdict") or "").lower() in {"fail", "failed", "error"}:
            signals.append("verifier_failure")

        recent_errors = getattr(state, "recent_errors", None)
        if isinstance(recent_errors, list) and len([err for err in recent_errors if str(err).strip()]) >= threshold:
            signals.append("recent_errors")

        counters = getattr(state, "stagnation_counters", None)
        counters = counters if isinstance(counters, dict) else {}
        if _safe_int(counters.get("repeat_patch"), 0) >= threshold:
            signals.append("repeat_patch")
        if _safe_int(counters.get("no_actionable_progress"), 0) >= threshold:
            signals.append("no_actionable_progress")

        if str(scratchpad.get("_tool_plan_observations_text") or "").strip():
            signals.append("tool_plan_observations")
        if str(scratchpad.get("_tool_plan_refine_verdict") or "").strip().lower() in {"fail", "failed", "reject", "rejected"}:
            signals.append("tool_plan_refine_failed")
        if scratchpad.get("_test_time_scaling_result") or scratchpad.get("_test_time_scaling_disagreement"):
            signals.append("test_time_scaling_disagreement")
        if scratchpad.get("_hidden_tool_recovery") or scratchpad.get("_tool_loop_suppression") or scratchpad.get("_read_loop_recovery_payload"):
            signals.append("tool_loop_suppression")
        if _safe_int(scratchpad.get("_schema_validation_nudges"), 0) >= threshold or isinstance(scratchpad.get("_last_schema_validation_hint"), dict):
            signals.append("schema_validation_repair")
        if isinstance(scratchpad.get("_last_write_session_schema_failure"), dict):
            signals.append("write_session_schema_failure")
        if isinstance(scratchpad.get("_last_backend_recovery"), dict):
            signals.append("backend_stream_recovery")

        metrics = scratchpad.get("_recovery_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        if _safe_int(metrics.get("tool_plan_wrong_path_count"), 0) > 0:
            signals.append("wrong_path")
        if _safe_int(metrics.get("tool_plan_repeated_read_count"), 0) > 1:
            signals.append("tool_plan_repeated_read")
        if _safe_int(metrics.get("tool_plan_evidence_before_patch_count"), 0) > 0:
            signals.append("evidence_before_patch")

        for signal in _fama_signal_classes(scratchpad):
            if signal in {
                "wrong_path",
                "remote_local_confusion",
                "remote_verification_pending",
                "tool_schema_invalid",
                "write_session_stall",
            }:
                signals.append(signal)

        records = getattr(state, "tool_execution_records", None)
        if isinstance(records, dict) and len(records) >= threshold:
            signals.append("tool_records")

        return _dedupe(signals)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _human_required_reason(state: Any, scratchpad: dict[str, Any]) -> str:
    pending_interrupt = getattr(state, "pending_interrupt", None)
    if isinstance(pending_interrupt, dict) and pending_interrupt:
        text = jsonish_text(pending_interrupt)
        if _looks_like_human_permission_or_credential_gate(text):
            return "Human approval, permission, or credentials are currently the next required step."

    for key in (
        "_awaiting_human_approval",
        "_pending_human_approval",
        "_needs_credentials",
        "_credential_request_pending",
    ):
        if scratchpad.get(key):
            return "Human approval, permission, or credentials are currently the next required step."
    return ""


def _looks_like_human_permission_or_credential_gate(text: str) -> bool:
    normalized = str(text or "").lower()
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "approval",
            "permission",
            "credential",
            "password",
            "api key",
            "api_key",
            "token",
            "passphrase",
            "sudo",
            "confirm",
        )
    )


def jsonish_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join([str(key) for key in value.keys()] + [jsonish_text(item) for item in value.values()])
    if isinstance(value, list):
        return " ".join(jsonish_text(item) for item in value)
    return str(value or "")


_EVIDENCE_SIGNALS = {
    "tool_plan_observations",
    "verifier_failure",
    "tool_records",
    "tool_loop_suppression",
    "schema_validation_repair",
    "write_session_schema_failure",
    "wrong_path",
    "remote_local_confusion",
    "remote_verification_pending",
    "tool_schema_invalid",
    "backend_stream_recovery",
    "tool_plan_repeated_read",
    "evidence_before_patch",
    "test_time_scaling_disagreement",
    "repeat_patch",
    "no_actionable_progress",
    "write_session_stall",
    "explicit_escalation_prior_evidence",
}


def _has_meaningful_evidence(signals: list[str]) -> bool:
    return any(signal in _EVIDENCE_SIGNALS for signal in signals)


_NON_EVIDENCE_TOOLS = {
    "",
    "ask_human",
    "escalate_to_bigger_model",
    "log_note",
    "loop_status",
    "memory_update",
    "task_complete",
    "task_fail",
}


def _explicit_escalation_request(reason: str) -> bool:
    lowered = str(reason or "").lower()
    return "escalat" in lowered or "bigger model" in lowered or "larger model" in lowered


def _has_prior_tool_evidence(state: Any) -> bool:
    return bool(
        _count_evidence_tool_records(getattr(state, "tool_execution_records", None))
        or _count_evidence_artifacts(getattr(state, "artifacts", None))
        or _count_evidence_messages(getattr(state, "conversation_history", None))
        or _count_evidence_messages(getattr(state, "transcript_messages", None))
        or _count_evidence_messages(getattr(state, "recent_messages", None))
    )


def _is_evidence_tool_name(tool_name: Any) -> bool:
    return str(tool_name or "").strip() not in _NON_EVIDENCE_TOOLS


def _count_evidence_tool_records(records: Any) -> int:
    if isinstance(records, dict):
        items = records.values()
    elif isinstance(records, list):
        items = records
    else:
        return 0
    count = 0
    for record in items:
        if not isinstance(record, dict):
            continue
        if _is_evidence_tool_name(record.get("tool_name") or record.get("tool")):
            count += 1
    return count


def _count_evidence_artifacts(artifacts: Any) -> int:
    if not isinstance(artifacts, dict):
        return 0
    count = 0
    for artifact in artifacts.values():
        tool_name = getattr(artifact, "tool_name", None)
        if isinstance(artifact, dict):
            tool_name = artifact.get("tool_name")
        if _is_evidence_tool_name(tool_name):
            count += 1
    return count


def _count_evidence_messages(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
            name = message.get("name")
        else:
            role = getattr(message, "role", None)
            name = getattr(message, "name", None)
        if str(role or "").strip() == "tool" and _is_evidence_tool_name(name):
            count += 1
    return count


def _fama_signal_classes(scratchpad: dict[str, Any]) -> list[str]:
    fama = scratchpad.get("_fama")
    if not isinstance(fama, dict):
        return []
    signals = fama.get("signals")
    if not isinstance(signals, list):
        return []
    classes: list[str] = []
    for item in signals[-8:]:
        if not isinstance(item, dict):
            continue
        for key in ("failure_class", "kind"):
            value = str(item.get(key) or "").strip()
            if value:
                classes.append(value)
    return classes


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
