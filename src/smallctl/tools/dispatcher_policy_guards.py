from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope

_STAGED_CONTROL_TOOLS = {"loop_status", "step_complete", "step_fail", "ask_human"}

_RESEARCH_TOOLS = {
    "web_search",
    "web_fetch",
    "file_read",
    "dir_list",
    "artifact_read",
    "artifact_print",
    "artifact_grep",
    "ask_human",
    "search",
    "grep",
    "git_status",
    "git_log",
    "git_diff",
}
_MUTATION_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
}
_ASK_USER_TOOLS = {"ask_human", "escalate_to_bigger_model"}
_STOP_BLOCKED_TOOLS = {"task_fail"}

_TOOL_TO_ACTION_CLASS: dict[str, str] = {}
for _tool in _RESEARCH_TOOLS:
    _TOOL_TO_ACTION_CLASS[_tool] = "research"
for _tool in _MUTATION_TOOLS:
    _TOOL_TO_ACTION_CLASS[_tool] = "mutation"
for _tool in _ASK_USER_TOOLS:
    _TOOL_TO_ACTION_CLASS[_tool] = "ask_user"
for _tool in _STOP_BLOCKED_TOOLS:
    _TOOL_TO_ACTION_CLASS[_tool] = "stop_blocked"


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


def _verifier_loop_dispatch_block(state: Any | None, tool_name: str) -> ToolEnvelope | None:
    """Enforce the verifier-loop hard-stop at dispatch time.

    When a verifier has rejected task_complete three or more times, the harness
    sets `_verifier_loop_required_action_classes` to limit the next tool to
    research, mutation, ask_user, or stop_blocked.  This function blocks tools
    that fall outside those allowed classes so a stuck model cannot keep issuing
    verifiers or task_complete calls.
    """
    if state is None:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    required_classes = scratchpad.get("_verifier_loop_required_action_classes")
    if not isinstance(required_classes, list) or not required_classes:
        return None
    rejection_count = int(scratchpad.get("_verifier_loop_rejection_count", 0) or 0)
    if rejection_count < 3:
        return None

    allowed_classes = set(required_classes)
    action_class = _TOOL_TO_ACTION_CLASS.get(tool_name)

    # Always block the verifier tools and task_complete during a hard stop.
    if tool_name in {"shell_exec", "ssh_exec", "task_complete"}:
        return ToolEnvelope(
            success=False,
            status="blocked",
            error=(
                f"VERIFIER LOOP HARD STOP: {tool_name} is blocked after {rejection_count} verifier rejections. "
                f"Allowed action classes are: {', '.join(sorted(allowed_classes))}. "
                "Research the failure, make a targeted mutation, ask the user, or call task_fail."
            ),
            metadata={
                "tool_name": tool_name,
                "reason": "verifier_loop_hard_stop",
                "rejection_count": rejection_count,
                "allowed_action_classes": sorted(allowed_classes),
                "active_mitigation": "verifier_loop_dispatch_block",
            },
        )

    if action_class is None or action_class not in allowed_classes:
        return ToolEnvelope(
            success=False,
            status="blocked",
            error=(
                f"VERIFIER LOOP HARD STOP: {tool_name} is not in an allowed action class after {rejection_count} verifier rejections. "
                f"Allowed classes are: {', '.join(sorted(allowed_classes))}."
            ),
            metadata={
                "tool_name": tool_name,
                "reason": "verifier_loop_hard_stop",
                "rejection_count": rejection_count,
                "allowed_action_classes": sorted(allowed_classes),
                "action_class": action_class,
                "active_mitigation": "verifier_loop_dispatch_block",
            },
        )

    return None
