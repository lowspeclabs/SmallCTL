from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from .dispatcher_remote_paths import (
    command_mentions_remote_absolute_path,
    looks_like_remote_infrastructure_probe_command,
)
from .dispatcher_scope_predicates import (
    _has_single_confirmed_ssh_target,
    _remote_scope_is_active,
)
from .dispatcher_tool_predicates import (
    _recent_ssh_auth_failure,
)


def _guard_ssh_auth_recovery(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[ToolEnvelope | None, dict[str, Any]]:
    if not isinstance(arguments, dict) or state is None:
        return None, {}
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None, {}
    from .dispatcher_ssh_auth import ssh_auth_recovery_entry_key
    recovery_state = scratchpad.get("_ssh_auth_recovery_state")
    if not isinstance(recovery_state, dict):
        return None, {}
    host = str(arguments.get("host") or "").strip().lower()
    user = str(arguments.get("user") or "").strip()
    if not host:
        return None, {}
    record = recovery_state.get(ssh_auth_recovery_entry_key(host, user))
    if not isinstance(record, dict):
        return None, {}

    password = str(arguments.get("password") or "").strip()
    prior_fingerprint = str(record.get("password_fingerprint") or "").strip()
    from .dispatcher_ssh_auth import password_fingerprint
    current_fingerprint = password_fingerprint(password)
    password_retry_allowed = bool(password) and (
        not prior_fingerprint or current_fingerprint != prior_fingerprint or not bool(record.get("password_provided"))
    )
    metadata = {
        "ssh_auth_recovery_required": True,
        "ssh_auth_recovery_failure_count": int(record.get("failure_count") or 0),
    }
    if password_retry_allowed:
        metadata["ssh_auth_recovery_branch"] = "retry_with_password"
        return None, metadata

    required_arguments = {
        "host": host,
        "command": str(arguments.get("command") or "").strip(),
    }
    if user:
        required_arguments["user"] = user
    error = (
        "SSH authentication previously failed for this target. Next step must be exactly one of: "
        "retry `ssh_exec` with a corrected `password` using a safe command like `whoami`, "
        "call `ask_human` for corrected credentials, or stop with `task_fail`. "
        "Do not retry key-only auth, do not use raw `shell_exec` SSH, and do not use SSH file mutation tools as an auth probe."
    )
    return ToolEnvelope(
        success=False,
        error=error,
        metadata={
            "tool_name": "ssh_exec",
            "reason": "ssh_auth_recovery_required",
            "last_error": str(record.get("last_error") or "").strip(),
            "last_command": str(record.get("last_command") or "").strip(),
            "next_required_action": {
                "tool_names": ["ssh_exec", "ask_human", "task_fail"],
                "required_arguments": required_arguments,
                "notes": [
                    "If you retry ssh_exec, include a corrected password.",
                    "If you do not have corrected credentials, ask the user instead of improvising with shell_exec.",
                ],
            },
        },
    ), metadata


def _guard_remote_file_tool_request(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
    ssh_available: bool = True,
) -> ToolEnvelope | None:
    _REMOTE_GUARDED_FILE_TOOLS = {"dir_list", "file_read", "file_write", "file_patch", "ast_patch"}
    if tool_name not in _REMOTE_GUARDED_FILE_TOOLS or not isinstance(arguments, dict):
        return None

    # Fix: if task was explicitly reclassified to local_execute, allow local filesystem tools
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    if task_mode in {"local_execute", "chat", "plan_only"}:
        return None

    from .dispatcher_remote_paths import looks_like_remote_absolute_path
    path = str(arguments.get("path") or "").strip()
    if not path or not looks_like_remote_absolute_path(path, state=state):
        return None

    if not (_remote_scope_is_active(state) or _has_single_confirmed_ssh_target(state)):
        return None

    suggested_tool = _suggested_remote_file_tool(tool_name, state=state)
    metadata = {
        "tool_name": tool_name,
        "reason": "remote_path_requires_ssh_exec"
        if suggested_tool == "ssh_exec"
        else "remote_path_requires_typed_ssh_file_tool",
        "path": path,
    }
    if ssh_available:
        metadata["suggested_tool"] = suggested_tool
        error = f"This path appears to be on the remote host. Use `{suggested_tool}`, not local `{tool_name}`."
    else:
        error = (
            f"This path appears to be on the remote host, but SSH tools are not currently available. "
            "Resume with the network/SSH tool profile or ask for help instead of using a local file tool."
        )
    return ToolEnvelope(success=False, error=error, metadata=metadata)


def _suggested_remote_file_tool(tool_name: str, *, state: Any | None = None) -> str:
    if tool_name == "file_read":
        return "ssh_file_read"
    if tool_name == "file_write":
        return "ssh_file_write"
    if tool_name in {"file_patch", "ast_patch"}:
        from .dispatcher_ssh_context import ssh_task_context_texts
        task_text = " ".join(ssh_task_context_texts(state)).lower() if state is not None else ""
        if any(marker in task_text for marker in ("style block", "<style>", "between ", "bounded block", "inline style")):
            return "ssh_file_replace_between"
        return "ssh_file_patch"
    return "ssh_exec"


def _guard_remote_shell_tool_request(
    command: str,
    *,
    state: Any | None = None,
    ssh_available: bool = True,
) -> ToolEnvelope | None:
    if not command:
        return None
    if _recent_ssh_auth_failure(state):
        return None

    # Fix: if task was explicitly reclassified to local_execute, allow local shell tools
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    if task_mode in {"local_execute", "chat", "plan_only"}:
        return None

    if not (_remote_scope_is_active(state) or _has_single_confirmed_ssh_target(state)):
        return None

    if command_mentions_remote_absolute_path(command, state=state):
        metadata = {
            "tool_name": "shell_exec",
            "reason": "remote_path_requires_ssh_exec",
            "command": command,
        }
        if ssh_available:
            metadata["suggested_tool"] = "ssh_exec"
            error = "This command references remote-looking host paths. Use `ssh_exec`, not local `shell_exec`."
        else:
            error = (
                "This command references remote-looking host paths, but `ssh_exec` is not currently available. "
                "Resume with the network/SSH tool profile or ask for help instead of running it locally."
            )
        return ToolEnvelope(
            success=False,
            error=error,
            metadata=metadata,
        )

    if looks_like_remote_infrastructure_probe_command(command):
        metadata = {
            "tool_name": "shell_exec",
            "reason": "remote_task_requires_ssh_exec",
            "command": command,
        }
        if ssh_available:
            metadata["suggested_tool"] = "ssh_exec"
            error = "This infrastructure check must run on the remote host. Use `ssh_exec`, not local `shell_exec`."
        else:
            error = (
                "This infrastructure check must run on the remote host, but `ssh_exec` is not currently available. "
                "Resume with the network/SSH tool profile or ask for help instead of running it locally."
            )
        return ToolEnvelope(
            success=False,
            error=error,
            metadata=metadata,
        )
    return None
