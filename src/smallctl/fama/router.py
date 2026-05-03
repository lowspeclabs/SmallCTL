from __future__ import annotations

from typing import Any

from .config import default_ttl_steps
from .signals import ActiveMitigation, FamaFailureKind, FamaSignal, current_step


MITIGATION_RULES: dict[FamaFailureKind, list[str]] = {
    FamaFailureKind.EARLY_STOP: ["done_gate", "acceptance_checklist_capsule"],
    FamaFailureKind.LOOPING: ["micro_plan_capsule", "evidence_reuse_capsule", "tool_exposure_narrowing"],
    FamaFailureKind.REMOTE_LOCAL_CONFUSION: ["remote_scope_capsule", "remote_tool_exposure_guard"],
    FamaFailureKind.TOOL_OUTPUT_MISREAD: ["evidence_reuse_capsule", "acceptance_checklist_capsule"],
    FamaFailureKind.BAD_TOOL_ARGS: ["micro_plan_capsule"],
    FamaFailureKind.WRITE_SESSION_STALL: ["write_session_recovery_capsule", "outline_only_recovery"],
    FamaFailureKind.BACKEND_STREAM_HALT: ["micro_plan_capsule", "outline_only_recovery"],
    FamaFailureKind.CONTEXT_DRIFT: ["micro_plan_capsule", "remote_scope_capsule"],
}


def route_signal(signal: FamaSignal, *, state: Any, config: Any) -> list[ActiveMitigation]:
    names = MITIGATION_RULES.get(signal.kind, [])
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
