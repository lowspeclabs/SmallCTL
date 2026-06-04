from __future__ import annotations

import re
from typing import Any

from ..models.tool_result import ToolEnvelope
from . import network
from .dispatcher_artifact_normalization import (
    normalize_artifact_read_request as _normalize_artifact_read_request,
    normalize_web_fetch_request as _normalize_web_fetch_request,
)
from .dispatcher_request_normalization import (
    normalize_initial_tool_request as _normalize_initial_tool_request,
    repair_ssh_exec_malformed_args as _repair_ssh_exec_malformed_args,
    repair_write_session_path_from_state as _repair_write_session_path_from_state,
)
from .dispatcher_shell_guards import (
    guard_harness_tool_as_ssh_shell_command as _guard_harness_tool_as_ssh_shell_command,
    guard_nested_raw_ssh_in_ssh_exec as _guard_nested_raw_ssh_in_ssh_exec,
    looks_like_raw_ssh_shell_command as _looks_like_raw_ssh_shell_command,
    raw_ssh_shell_block_envelope as _raw_ssh_shell_block_envelope,
)
from .dispatcher_tool_guards import (
    _guard_remote_file_tool_request,
    _guard_remote_shell_tool_request,
    _guard_ssh_auth_recovery,
)
from .dispatcher_tool_predicates import (
    _escalation_recommends_local_shell,
    _recent_ssh_auth_failure,
    _ssh_exec_available,
)
from .dispatcher_ssh_recovery import (
    _pin_and_guard_ssh_credentials,
    _recover_ssh_arguments_from_task_context,
)
from .dispatcher_remote_detection import task_clearly_targets_remote_ssh_host as _task_clearly_targets_remote_ssh_host

_SSH_FILE_TOOLS = {"ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}


def normalize_tool_request(
    registry: Any,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    phase: str | None = None,
    state: Any | None = None,
) -> tuple[str, dict[str, Any], ToolEnvelope | None, dict[str, Any]]:
    tool_name, arguments, normalization_metadata = _normalize_initial_tool_request(tool_name, arguments)

    if tool_name == "artifact_read":
        tool_name, arguments, artifact_metadata = _normalize_artifact_read_request(arguments, state=state)
        normalization_metadata.update(artifact_metadata)
    elif tool_name == "web_fetch":
        arguments, web_fetch_metadata = _normalize_web_fetch_request(arguments, state=state)
        normalization_metadata.update(web_fetch_metadata)

    arguments, write_session_metadata = _repair_write_session_path_from_state(
        tool_name,
        arguments,
        state=state,
    )
    normalization_metadata.update(write_session_metadata)
    ssh_available = _ssh_exec_available(registry, phase=phase, state=state)

    remote_file_guard = _guard_remote_file_tool_request(
        tool_name,
        arguments,
        state=state,
        ssh_available=ssh_available,
    )
    if remote_file_guard is not None:
        return tool_name, arguments, remote_file_guard, normalization_metadata

    if tool_name in _SSH_FILE_TOOLS:
        pre_recovered, pre_metadata = _recover_ssh_arguments_from_task_context(
            arguments,
            state=state,
        )
        normalization_metadata.update(pre_metadata)
        try:
            normalized_arguments = network.normalize_ssh_arguments(pre_recovered)
            normalized_arguments, ssh_metadata = _recover_ssh_arguments_from_task_context(
                normalized_arguments,
                state=state,
            )
            for _preserve_key in ("recovered_ssh_host", "recovered_ssh_user", "recovered_ssh_password_source", "routing_reason"):
                if _preserve_key in normalization_metadata:
                    ssh_metadata[_preserve_key] = normalization_metadata[_preserve_key]
            if "recovered_ssh_password_source" in normalization_metadata:
                ssh_metadata["ssh_password_origin"] = normalization_metadata["recovered_ssh_password_source"]
                ssh_metadata["ssh_password_recovered"] = True
            normalization_metadata.update(ssh_metadata)
            normalized_arguments, pin_block, pin_metadata = _pin_and_guard_ssh_credentials(
                normalized_arguments,
                state=state,
                normalization_metadata=normalization_metadata,
            )
            normalization_metadata.update(pin_metadata)
            if pin_block is not None:
                return tool_name, normalized_arguments, pin_block, normalization_metadata
            return tool_name, normalized_arguments, None, normalization_metadata
        except ValueError as exc:
            return (
                tool_name,
                arguments,
                ToolEnvelope(
                    success=False,
                    error=str(exc),
                    metadata={
                        "tool_name": tool_name,
                        "reason": "invalid_ssh_target",
                    },
                ),
                normalization_metadata,
            )

    if tool_name == "ssh_exec":
        arguments, repair_metadata = _repair_ssh_exec_malformed_args(arguments)
        normalization_metadata.update(repair_metadata)
        command = str(arguments.get("command") or "").strip() if isinstance(arguments, dict) else ""
        harness_tool_shell_error = _guard_harness_tool_as_ssh_shell_command(command)
        if harness_tool_shell_error is not None:
            return tool_name, arguments, harness_tool_shell_error, normalization_metadata
        nested_raw_ssh_error = _guard_nested_raw_ssh_in_ssh_exec(command)
        if nested_raw_ssh_error is not None:
            return tool_name, arguments, nested_raw_ssh_error, normalization_metadata
        pre_recovered, pre_metadata = _recover_ssh_arguments_from_task_context(
            arguments,
            state=state,
        )
        normalization_metadata.update(pre_metadata)
        try:
            normalized_arguments = network.normalize_ssh_arguments(pre_recovered)
            normalized_arguments, ssh_metadata = _recover_ssh_arguments_from_task_context(
                normalized_arguments,
                state=state,
            )
            for _preserve_key in ("recovered_ssh_host", "recovered_ssh_user", "recovered_ssh_password_source", "routing_reason"):
                if _preserve_key in normalization_metadata:
                    ssh_metadata[_preserve_key] = normalization_metadata[_preserve_key]
            if "recovered_ssh_password_source" in normalization_metadata:
                ssh_metadata["ssh_password_origin"] = normalization_metadata["recovered_ssh_password_source"]
                ssh_metadata["ssh_password_recovered"] = True
            normalization_metadata.update(ssh_metadata)
            normalized_arguments, pin_block, pin_metadata = _pin_and_guard_ssh_credentials(
                normalized_arguments,
                state=state,
                normalization_metadata=normalization_metadata,
            )
            normalization_metadata.update(pin_metadata)
            if pin_block is not None:
                return tool_name, normalized_arguments, pin_block, normalization_metadata
            auth_recovery_error, auth_recovery_metadata = _guard_ssh_auth_recovery(
                normalized_arguments,
                state=state,
            )
            normalization_metadata.update(auth_recovery_metadata)
            if auth_recovery_error is not None:
                return tool_name, normalized_arguments, auth_recovery_error, normalization_metadata
            return tool_name, normalized_arguments, None, normalization_metadata
        except ValueError as exc:
            return (
                tool_name,
                arguments,
                ToolEnvelope(
                    success=False,
                    error=str(exc),
                    metadata={
                        "tool_name": tool_name,
                        "reason": "invalid_ssh_target",
                    },
                ),
                normalization_metadata,
            )

    if tool_name != "shell_exec":
        return tool_name, arguments, None, normalization_metadata

    ssh_spec = registry.get("ssh_exec") if hasattr(registry, "get") else None
    phase_allowed = getattr(ssh_spec, "phase_allowed", None)
    ssh_spec_phase_available = not (
        ssh_spec is None or (phase and callable(phase_allowed) and not phase_allowed(phase))
    )

    command = str(arguments.get("command", "") or "").strip() if isinstance(arguments, dict) else ""
    if not command:
        return tool_name, arguments, None, normalization_metadata

    rewritten_args = None
    if ssh_spec_phase_available:
        try:
            rewritten_args = network.parse_ssh_exec_args_from_shell_command(command)
        except ValueError as exc:
            return (
                tool_name,
                arguments,
                ToolEnvelope(
                    success=False,
                    error=str(exc),
                    metadata={
                        "tool_name": "ssh_exec",
                        "reason": "invalid_ssh_target",
                        "rewritten_from_tool": "shell_exec",
                    },
                ),
                normalization_metadata,
            )
    raw_ssh_shell_attempt = _looks_like_raw_ssh_shell_command(command)
    if rewritten_args is None:
        if raw_ssh_shell_attempt:
            return (
                tool_name,
                arguments,
                _raw_ssh_shell_block_envelope(command, ssh_available=ssh_available),
                normalization_metadata,
            )
        remote_shell_guard = _guard_remote_shell_tool_request(
            command,
            state=state,
            ssh_available=ssh_available,
        )
        if remote_shell_guard is not None:
            return tool_name, arguments, remote_shell_guard, normalization_metadata
        if _escalation_recommends_local_shell(state):
            return tool_name, arguments, None, normalization_metadata
        if not _recent_ssh_auth_failure(state) and _task_clearly_targets_remote_ssh_host(state) and not re.search(r"\b(?:ssh|scp|sftp)\b", command):
            metadata = {
                "tool_name": tool_name,
                "reason": "remote_task_requires_ssh_exec",
            }
            if ssh_available:
                metadata["suggested_tool"] = "ssh_exec"
                error = "This is a remote task. Use `ssh_exec`, not local `shell_exec`."
            else:
                error = (
                    "This is a remote task, but `ssh_exec` is not currently available. "
                    "Resume with the network/SSH tool profile or ask for help instead of using local `shell_exec`."
                )
            return (
                tool_name,
                arguments,
                ToolEnvelope(
                    success=False,
                    error=error,
                    metadata=metadata,
                ),
                normalization_metadata,
            )
        return tool_name, arguments, None, normalization_metadata

    rewritten_args, ssh_metadata = _recover_ssh_arguments_from_task_context(
        rewritten_args,
        state=state,
    )
    normalization_metadata.update(ssh_metadata)
    auth_recovery_error, auth_recovery_metadata = _guard_ssh_auth_recovery(
        rewritten_args,
        state=state,
    )
    normalization_metadata.update(auth_recovery_metadata)
    if auth_recovery_error is not None:
        return "ssh_exec", rewritten_args, auth_recovery_error, normalization_metadata
    if "timeout_sec" in arguments and "timeout_sec" not in rewritten_args:
        rewritten_args["timeout_sec"] = arguments["timeout_sec"]
    normalization_metadata.update(
        {
            "rewritten_from_tool": "shell_exec",
            "routing_reason": "ssh_shell_command",
        }
    )
    return "ssh_exec", rewritten_args, None, normalization_metadata
