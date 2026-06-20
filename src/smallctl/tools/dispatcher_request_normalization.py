from __future__ import annotations

from typing import Any

_TOOL_ALIAS_REPAIRS = {
    "use_shell_exec": "shell_exec",
    "use_ssh_exec": "ssh_exec",
    "artifact_write": "file_write",
}
_WRITE_SESSION_PATH_REPAIR_TOOLS = {"file_write", "file_append", "file_patch", "ast_patch"}


def normalize_initial_tool_request(
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    original_tool_name = str(tool_name or "").strip()
    repaired_tool_name = _TOOL_ALIAS_REPAIRS.get(original_tool_name, original_tool_name)
    if repaired_tool_name != original_tool_name:
        metadata.update(
            {
                "repaired_tool_alias_from": original_tool_name,
                "repaired_tool_alias_to": repaired_tool_name,
                "routing_reason": "tool_alias_repair",
            }
        )

    return repaired_tool_name, arguments, metadata


def repair_ssh_exec_malformed_args(
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unwrap common small-model hallucinations in ssh_exec argument JSON."""
    if not isinstance(arguments, dict):
        return arguments, {}
    repaired = dict(arguments)
    metadata: dict[str, Any] = {}

    nested = repaired.pop("arguments", None)
    if isinstance(nested, dict):
        nested_cmd = nested.get("arg") or nested.get("command")
        if nested_cmd:
            if not repaired.get("command"):
                repaired["command"] = str(nested_cmd).strip()
            metadata["repaired_ssh_exec_nested_args"] = True

    inner_name = repaired.get("name")
    if isinstance(inner_name, str) and inner_name.strip() and inner_name.strip() not in {
        "ssh_exec",
        "shell_exec",
        "ssh_file_read",
        "ssh_file_write",
        "ssh_file_patch",
        "ssh_file_replace_between",
    }:
        repaired.pop("name", None)
        metadata["repaired_ssh_exec_hallucinated_name"] = True

    if metadata:
        metadata["routing_reason"] = "ssh_exec_malformed_args_repair"
    return repaired, metadata


def repair_write_session_path_from_state(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if tool_name not in _WRITE_SESSION_PATH_REPAIR_TOOLS or not isinstance(arguments, dict):
        return arguments, {}
    if value_present(arguments.get("path")):
        return arguments, {}

    session = getattr(state, "write_session", None)
    if session is None:
        return arguments, {}
    if str(getattr(session, "status", "") or "").strip().lower() == "complete":
        return arguments, {}

    requested_session_id = str(arguments.get("write_session_id") or "").strip()
    active_session_id = str(getattr(session, "write_session_id", "") or "").strip()
    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    if not requested_session_id or not active_session_id or requested_session_id != active_session_id:
        return arguments, {}
    if not target_path:
        return arguments, {}

    repaired = dict(arguments)
    repaired["path"] = target_path
    return repaired, {
        "argument_repair": "active_write_session_path",
        "repaired_write_session_path": True,
        "write_session_id": active_session_id,
        "target_path": target_path,
    }


def value_present(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))
