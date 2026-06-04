from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope

_STAGED_CONTROL_TOOLS = {"loop_status", "step_complete", "step_fail", "ask_human"}


def _fama_dispatch_block(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Any | None,
    phase: str,
) -> ToolEnvelope | None:
    if state is None:
        return None
    try:
        from ..fama.tool_policy import enforce_fama_tool_call
    except Exception:
        return None
    return enforce_fama_tool_call(
        tool_name,
        arguments if isinstance(arguments, dict) else {},
        state=state,
        mode=phase,
        config=None,
    )


def _staged_tool_allowlist_error(state: Any | None, tool_name: str) -> ToolEnvelope | None:
    if state is None:
        return None
    if not bool(getattr(state, "plan_execution_mode", False)):
        return None
    active_step_id = str(getattr(state, "active_step_id", "") or "").strip()
    if not active_step_id:
        return None

    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    step = plan.find_step(active_step_id) if plan is not None and hasattr(plan, "find_step") else None
    plan_id = str(getattr(plan, "plan_id", "") or "")
    active_step_run_id = str(getattr(state, "active_step_run_id", "") or "")
    attempt = 1
    if step is not None:
        attempt = int(getattr(step, "retry_count", 0) or 0) + 1

    if tool_name in _STAGED_CONTROL_TOOLS:
        return None
    if tool_name == "task_complete":
        return ToolEnvelope(
            success=False,
            error="`task_complete` is not allowed during staged execution. Use `step_complete` for the active step.",
            metadata={
                "reason": "task_complete_blocked_in_staged_execution",
                "tool_name": tool_name,
                "plan_id": plan_id,
                "step_id": active_step_id,
                "step_run_id": active_step_run_id,
                "attempt": attempt,
            },
        )

    allowed = set(getattr(step, "tool_allowlist", []) or []) if step is not None else set()
    if tool_name in allowed:
        return None
    return ToolEnvelope(
        success=False,
        error=f"Tool `{tool_name}` is not allowed for active staged step `{active_step_id}`.",
        metadata={
            "reason": "tool_not_allowed_for_step",
            "tool_name": tool_name,
            "allowed_tools": sorted(allowed),
            "plan_id": plan_id,
            "step_id": active_step_id,
            "step_run_id": active_step_run_id,
            "attempt": attempt,
        },
    )
