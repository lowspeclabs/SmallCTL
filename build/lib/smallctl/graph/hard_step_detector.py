from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..state import LoopState, PlanStep


HIGH_RISK_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "shell_exec",
    "ssh_exec",
    "ssh_file_write",
    "ssh_file_patch",
}


@dataclass(frozen=True)
class HardStepDecision:
    should_scale: bool
    reason: str = ""


class HardStepDetector:
    def should_scale(self, *, step: PlanStep, state: LoopState, config: Any) -> bool:
        return self.decide(step=step, state=state, config=config).should_scale

    def decide(self, *, step: PlanStep, state: LoopState, config: Any) -> HardStepDecision:
        if not bool(getattr(config, "test_time_scaling_enabled", False)):
            return HardStepDecision(False, "disabled")
        runtimes = getattr(config, "test_time_scaling_runtimes", ["staged_execution"]) or []
        runtime_names = {str(item).strip() for item in runtimes if str(item).strip()}
        if runtime_names and "staged_execution" not in runtime_names:
            return HardStepDecision(False, "runtime_not_enabled")

        trigger = str(getattr(config, "test_time_scaling_trigger", "retry_or_explicit") or "").strip().lower()
        if not trigger:
            trigger = "retry_or_explicit"
        if trigger == "any":
            return HardStepDecision(True, "any")

        explicit_hard = str(getattr(step, "difficulty", "") or "").strip().lower() == "hard"
        retrying = int(getattr(step, "retry_count", 0) or 0) > 0
        failure_text = " ".join(str(item) for item in getattr(step, "failure_reasons", []) or []).lower()
        verifier_failure = bool(failure_text and any(token in failure_text for token in ("verifier", "verification", "tool")))
        risky_tools = bool(set(getattr(step, "tool_allowlist", []) or []) & HIGH_RISK_TOOLS)

        if trigger == "explicit":
            return HardStepDecision(explicit_hard, "explicit_hard" if explicit_hard else "not_explicit")
        if trigger == "retry_or_explicit":
            if explicit_hard:
                return HardStepDecision(True, "explicit_hard")
            if retrying:
                return HardStepDecision(True, "retry")
            return HardStepDecision(False, "not_retry_or_explicit")
        if trigger == "heuristic":
            if explicit_hard:
                return HardStepDecision(True, "explicit_hard")
            if retrying:
                return HardStepDecision(True, "retry")
            if verifier_failure:
                return HardStepDecision(True, "prior_verifier_failure")
            if risky_tools:
                return HardStepDecision(True, "risk_tool_allowlist")
            return HardStepDecision(False, "heuristic_no_signal")
        return HardStepDecision(False, "unknown_trigger")
