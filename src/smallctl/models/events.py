from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class UIEventType(str, Enum):
    USER = "user"
    STATUS = "status"
    THINKING = "thinking"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    SYSTEM = "system"
    METRICS = "metrics"
    SHELL_STREAM = "shell_stream"
    ALERT = "alert"


@dataclass
class UIEvent:
    event_type: UIEventType
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp,
        }


def format_verifier_verdict(verdict: Any) -> str:
    if not isinstance(verdict, dict) or not verdict:
        return ""
    verdict_name = str(verdict.get("verdict") or "").strip()
    target = str(verdict.get("target") or verdict.get("command") or "").strip()
    exit_code = verdict.get("exit_code")
    status = verdict_name or "unknown"
    if verdict_name == "needs_human":
        status = "needs human"
    bits = [status]
    if target:
        bits.append(target)
    if exit_code not in (None, ""):
        bits.append(f"exit {exit_code}")
    return " | ".join(bits)


def format_acceptance_progress(checklist: list[dict[str, Any]] | None) -> str:
    if not checklist:
        return ""
    total = len(checklist)
    satisfied = sum(1 for item in checklist if bool(item.get("satisfied")))
    return f"{satisfied}/{total}"


def compute_activity_for_event(
    event: UIEvent,
    *,
    active_task_done: bool | None = None,
) -> str | None:
    if "status_activity" in event.data:
        return str(event.data.get("status_activity") or "").strip()

    if event.event_type == UIEventType.TOOL_CALL:
        tool_name = str(event.content or event.data.get("tool_name") or "").strip()
        return f"running {tool_name}..." if tool_name else "running tool..."

    if event.event_type == UIEventType.TOOL_RESULT:
        if active_task_done is False:
            return "thinking..."
        return ""

    if event.event_type in {UIEventType.ASSISTANT, UIEventType.THINKING}:
        return "responding..."

    if event.event_type == UIEventType.ERROR:
        return ""

    return None


@dataclass(frozen=True)
class UIStatusSnapshot:
    model: str = "n/a"
    phase: str = "explore"
    step: int | str = 0
    mode: str = "execution"
    plan: str = ""
    active_step: str = ""
    activity: str = ""
    contract_flow_ui: bool = False
    contract_phase: str = ""
    acceptance_progress: str = ""
    latest_verdict: str = ""
    token_usage: int = 0
    token_total: int = 0
    token_limit: int = 0
    api_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "phase": self.phase,
            "step": self.step,
            "mode": self.mode,
            "plan": self.plan,
            "active_step": self.active_step,
            "activity": self.activity,
            "contract_flow_ui": self.contract_flow_ui,
            "contract_phase": self.contract_phase,
            "acceptance_progress": self.acceptance_progress,
            "latest_verdict": self.latest_verdict,
            "token_usage": self.token_usage,
            "token_total": self.token_total,
            "token_limit": self.token_limit,
            "api_errors": self.api_errors,
        }

    @classmethod
    def from_harness(
        cls,
        harness: Any,
        harness_kwargs: dict[str, Any],
        *,
        activity: str = "",
        api_errors: int | None = None,
    ) -> "UIStatusSnapshot":
        model = str(harness_kwargs.get("model", "n/a"))
        phase = str(harness_kwargs.get("phase", "explore"))
        step: int | str = 0
        mode = "execution"
        plan_label = ""
        active_step_label = ""
        contract_phase = ""
        acceptance_progress = ""
        latest_verdict = ""
        contract_flow_ui = bool(harness_kwargs.get("contract_flow_ui", False))
        token_usage = 0
        token_total = 0
        token_limit = 0

        if api_errors is None and harness is not None:
            api_errors = int(getattr(harness.state, "scratchpad", {}).get("_ui_api_error_count", 0) or 0)
        if api_errors is None:
            api_errors = 0

        if harness is not None:
            phase = str(getattr(harness.state, "current_phase", phase) or phase)
            step = getattr(harness.state, "step_count", 0)
            mode = "planning" if bool(getattr(harness.state, "planning_mode_enabled", False)) else "execution"
            plan = getattr(harness.state, "active_plan", None) or getattr(harness.state, "draft_plan", None)
            if plan is not None:
                total_steps = max(1, len(plan.iter_steps()))
                completed_steps = sum(1 for item in plan.iter_steps() if item.status == "completed")
                plan_label = f"{completed_steps}/{total_steps}"
                active_step = plan.active_step()
                if active_step is not None:
                    active_step_label = active_step.step_id
            contract_phase = str(harness.state.contract_phase() or "")
            acceptance_progress = format_acceptance_progress(harness.state.acceptance_checklist())
            latest_verdict = format_verifier_verdict(harness.state.current_verifier_verdict())
            token_usage = int(
                harness.state.scratchpad.get("context_used_tokens", getattr(harness.state, "token_usage", 0))
            )
            token_total = int(getattr(harness.state, "token_usage", 0) or 0)
            context_policy = getattr(harness, "context_policy", None)
            server_context_limit = getattr(harness, "server_context_limit", None)
            guards = getattr(harness, "guards", None)
            token_limit = int(
                getattr(context_policy, "max_prompt_tokens", None)
                or server_context_limit
                or getattr(guards, "max_tokens", 0)
                or 0
            )

        return cls(
            model=model,
            phase=phase,
            step=step,
            mode=mode,
            plan=plan_label,
            active_step=active_step_label,
            activity=activity,
            contract_flow_ui=contract_flow_ui,
            contract_phase=contract_phase,
            acceptance_progress=acceptance_progress,
            latest_verdict=latest_verdict,
            token_usage=max(0, token_usage),
            token_total=max(0, token_total),
            token_limit=max(0, token_limit),
            api_errors=max(0, int(api_errors)),
        )
