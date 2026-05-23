from __future__ import annotations

from typing import Any

from .common import fail


def _format_advisory_text(advisory: dict[str, Any]) -> str:
    """Convert a raw escalation advisory dict into structured, model-readable text."""
    lines: list[str] = []
    lines.append("ESCALATION ADVISORY")
    lines.append(f"Verdict: {advisory.get('verdict', 'unknown')}")
    lines.append(f"Confidence: {advisory.get('confidence', 'N/A')}")
    lines.append(f"Failure Diagnosis: {advisory.get('failure_diagnosis', 'N/A')}")

    action = advisory.get("recommended_next_action")
    if isinstance(action, dict):
        action_type = str(action.get("type") or "none").strip()
        if action_type == "tool_call":
            tool = str(action.get("tool") or "").strip()
            reason = str(action.get("reason") or "").strip()
            args = action.get("args")
            if tool:
                lines.append(f"Recommended Next Action: tool_call({tool})")
                if reason:
                    lines.append(f"  Reason: {reason}")
                if isinstance(args, dict) and args:
                    lines.append(f"  Suggested args: {args}")
        elif action_type and action_type != "none":
            reason = str(action.get("reason") or "").strip()
            lines.append(f"Recommended Next Action: {action_type}")
            if reason:
                lines.append(f"  Reason: {reason}")
        else:
            lines.append("Recommended Next Action: none")
    else:
        lines.append("Recommended Next Action: none")

    repair = advisory.get("repair_plan")
    if repair:
        lines.append(f"Repair Plan: {repair}")

    risks = advisory.get("risk_notes") or []
    if isinstance(risks, list) and risks:
        lines.append(f"Risk Notes: {', '.join(str(r) for r in risks)}")

    if advisory.get("requires_human_approval"):
        lines.append("REQUIRES HUMAN APPROVAL")

    return "\n".join(lines)


async def escalate_to_bigger_model(
    *,
    reason: str,
    question: str,
    requested_output: str,
    risk_level: str = "medium",
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    del state
    if harness is None:
        return {"success": False, "status": "error", "error": "Harness is required for escalation."}
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=reason,
        question=question,
        requested_output=requested_output,
        risk_level=risk_level,
    )
    if isinstance(result, dict) and result.get("success") is False:
        message = str(result.get("error") or result.get("reason") or result.get("status") or "Escalation failed.")
        missing = result.get("missing_signals")
        if isinstance(missing, list) and missing:
            message += f" Missing evidence signals: {', '.join(missing)}."
        metadata = {key: value for key, value in result.items() if key not in {"error", "reason"}}
        metadata["escalation_result"] = result
        return fail(message, metadata=metadata)

    # Reformat the raw advisory JSON into structured text the acting model can easily consume.
    if isinstance(result, dict):
        result["advisory_text"] = _format_advisory_text(result)
        # Inject the recommended next action into working memory so the model
        # treats it as a concrete goal rather than drifting back to task_complete.
        action = result.get("recommended_next_action")
        if isinstance(action, dict):
            action_type = str(action.get("type") or "").strip()
            if action_type == "tool_call":
                tool = str(action.get("tool") or "").strip()
                reason = str(action.get("reason") or "").strip()
                args = action.get("args")
                if tool:
                    next_step = f"Next step (from bigger model): call {tool}"
                    if reason:
                        next_step += f" — {reason}"
                    if isinstance(args, dict) and args:
                        next_step += f" with args {args}"
                    _inject_next_action(state, next_step)
            elif action_type and action_type != "none":
                reason = str(action.get("reason") or "").strip()
                next_step = f"Next step (from bigger model): {action_type}"
                if reason:
                    next_step += f" — {reason}"
                _inject_next_action(state, next_step)
        repair_plan = result.get("repair_plan")
        if isinstance(repair_plan, str) and repair_plan.strip():
            _inject_next_action(state, f"Repair plan (from bigger model): {repair_plan.strip()}")
    return result


def _inject_next_action(state: Any, text: str) -> None:
    if state is None:
        return
    working_memory = getattr(state, "working_memory", None)
    if working_memory is not None:
        existing = list(getattr(working_memory, "next_actions", []) or [])
        existing.append(text)
        working_memory.next_actions = existing[-4:]
    # Also update current_goal if it exists, to resteer the model
    goal = getattr(working_memory, "current_goal", "") or ""
    if goal and "escalat" not in goal.lower():
        setattr(working_memory, "current_goal", f"{goal}\n\n[ESCATION GUIDANCE] {text}")
