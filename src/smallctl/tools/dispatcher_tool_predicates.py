from __future__ import annotations

from typing import Any


def _ssh_exec_available(registry: Any, *, phase: str | None, state: Any | None) -> bool:
    get_tool = getattr(registry, "get", None)
    if not callable(get_tool):
        return False
    ssh_spec = get_tool("ssh_exec")
    if ssh_spec is None:
        return False
    phase_allowed = getattr(ssh_spec, "phase_allowed", None)
    if callable(phase_allowed) and phase and not phase_allowed(phase):
        return False
    profile_allowed = getattr(ssh_spec, "profile_allowed", None)
    profiles = set(getattr(state, "active_tool_profiles", []) or [])
    if callable(profile_allowed) and not profile_allowed(profiles):
        return False
    return True


def _recent_ssh_auth_failure(state: Any | None) -> bool:
    """Return True if the most recent ssh_exec attempt failed with auth denied."""
    if state is None:
        return False
    records = getattr(state, "tool_execution_records", None)
    if not isinstance(records, dict) or not records:
        return False
    for record in reversed(list(records.values())):
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "").strip() != "ssh_exec":
            continue
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        if bool(result.get("success")):
            return False
        error = str(result.get("error") or "").lower()
        stderr = ""
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            output = metadata.get("output")
            if isinstance(output, dict):
                stderr = str(output.get("stderr") or "").lower()
            else:
                stderr = str(metadata.get("stderr") or "").lower()
        combined = f"{error}\n{stderr}"
        if "permission denied" in combined and ("publickey" in combined or "password" in combined):
            return True
        return False
    return False


def _escalation_recommends_local_shell(state: Any | None) -> bool:
    """Check if recent escalation guidance recommends using local shell_exec."""
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    # Check recent escalation entries
    escalation_history = scratchpad.get("escalation_history", [])
    for entry in reversed(escalation_history[-3:]):
        if not isinstance(entry, dict):
            continue
        action = entry.get("recommended_next_action") or {}
        if isinstance(action, dict):
            action_type = str(action.get("type", "")).lower()
            reason = str(action.get("reason", "")).lower()
            if action_type == "shell_exec" or "shell_exec" in reason:
                return True
        repair_plan = str(entry.get("repair_plan", "")).lower()
        if "shell_exec" in repair_plan and "local" in repair_plan:
            return True
    # Also check working memory for escalation guidance
    working_memory = getattr(state, "working_memory", None)
    if working_memory is not None:
        notes = str(getattr(working_memory, "notes", "") or "").lower()
        if "shell_exec" in notes and "escalation" in notes:
            return True
    return False
