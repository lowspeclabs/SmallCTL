from __future__ import annotations

from typing import Any


def _task_is_remote_execute(task_context: Any | None) -> bool:
    if task_context is None:
        return False
    return getattr(task_context, "remote_session_active", False)


def _remote_scope_is_active(state: Any | None) -> bool:
    if state is None:
        return False
    task_context = getattr(state, "task_context", None)
    if task_context is None and hasattr(state, "task_context"):
        task_context = state.task_context
    if _task_is_remote_execute(task_context):
        return True
    if str(getattr(state, "task_mode", "") or "").strip() == "remote_execute":
        return True
    # Fallback: task handoff context may carry remote_execute mode
    handoff = getattr(state, "scratchpad", {}).get("_last_task_handoff") if hasattr(state, "scratchpad") else None
    if isinstance(handoff, dict):
        return str(handoff.get("task_mode") or "").strip() == "remote_execute"
    return False


def _has_single_confirmed_ssh_target(state: Any | None) -> bool:
    if state is None:
        return False
    if not hasattr(state, "active_plan") or state.active_plan is None:
        return False
    p = state.active_plan
    if not hasattr(p, "find_step") or not hasattr(p, "steps"):
        return False
    ssh_hosts: set[str] = set()
    for step in getattr(p, "steps", []):
        if step is None:
            continue
        tool_chain = getattr(step, "tool_chain", None)
        if not isinstance(tool_chain, list):
            continue
        for entry in tool_chain:
            if isinstance(entry, dict) and entry.get("tool_name") == "ssh_exec":
                args = entry.get("arguments", {})
                if isinstance(args, dict):
                    h = str(args.get("host") or "").strip()
                    if h:
                        ssh_hosts.add(h)
    if len(ssh_hosts) != 1:
        return False
    task_context = getattr(state, "task_context", None)
    if task_context is None:
        return False
    if not _task_is_remote_execute(task_context):
        return False
    return True
