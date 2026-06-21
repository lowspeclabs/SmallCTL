from __future__ import annotations

import re
from typing import Any

from ..models.tool_result import ToolEnvelope
from .dispatcher_remote_paths import (
    _IPV4_RE,
    _AT_HOST_TARGET_RE,
    _REMOTE_TASK_HINT_RE,
    _SSH_TASK_TARGET_RE,
)
from .dispatcher_schema_helpers import coerce_value as _coerce_value
from .dispatcher_shell_guards import (
    guard_harness_tool_as_ssh_shell_command as _guard_harness_tool_as_ssh_shell_command,
    guard_nested_raw_ssh_in_ssh_exec as _guard_nested_raw_ssh_in_ssh_exec,
    looks_like_raw_ssh_shell_command as _looks_like_raw_ssh_shell_command,
    raw_ssh_shell_block_envelope as _raw_ssh_shell_block_envelope,
)
from .dispatcher_ssh_auth import (
    password_fingerprint as _password_fingerprint,
    ssh_auth_debug_metadata as _ssh_auth_debug_metadata,
    ssh_auth_recovery_entry_key as _ssh_auth_recovery_entry_key,
)
from .dispatcher_ssh_context import (
    infer_ssh_user_from_state_context as _infer_ssh_user_from_state_context,
    ssh_task_context_texts as _ssh_task_context_texts,
)
from .dispatcher_ssh_memory import (
    explicit_ssh_password_matches_current_user_context as _explicit_ssh_password_matches_current_user_context,
    infer_ssh_password as _infer_ssh_password,
    infer_ssh_user_from_execution_records as _infer_ssh_user_from_execution_records,
    infer_ssh_user_from_session_memory as _infer_ssh_user_from_session_memory,
    session_ssh_target_record as _session_ssh_target_record,
)


def _infer_ssh_host_from_context(state: Any | None) -> str:
    """Infer an SSH host from session state when the model omits target/host."""
    if state is None:
        return ""
    # Prefer single confirmed session target
    from ..remote_scope import _confirmed_session_targets
    confirmed = _confirmed_session_targets(state)
    if len(confirmed) == 1:
        return confirmed[0]["host"]
    # Fall back to execution records: most recent ssh_exec host
    records = getattr(state, "tool_execution_records", None)
    if isinstance(records, dict) and records:
        for record in reversed(list(records.values())):
            if not isinstance(record, dict):
                continue
            tool_name = str(record.get("tool_name") or "").strip()
            if tool_name not in {"ssh_exec", "ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}:
                continue
            args = record.get("args")
            if not isinstance(args, dict):
                continue
            record_host = str(args.get("host") or "").strip()
            if record_host:
                return record_host
    # Fall back to task context text
    for text in _ssh_task_context_texts(state):
        if not text:
            continue
        match = _SSH_TASK_TARGET_RE.search(text)
        if match:
            host = str(match.group("host") or "").strip()
            if host:
                return host
        match = _AT_HOST_TARGET_RE.search(text)
        if match:
            host = str(match.group("host") or "").strip()
            if host:
                return host
        match = _IPV4_RE.search(text)
        if match and _REMOTE_TASK_HINT_RE.search(text):
            host = str(match.group(0) or "").strip()
            if host:
                return host
    return ""


def _recover_ssh_arguments_from_task_context(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
    credential_store: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(arguments, dict):
        return arguments, {}

    repaired = dict(arguments)
    metadata: dict[str, Any] = {}
    host = str(repaired.get("host") or "").strip()
    user = str(repaired.get("user") or "").strip()
    password = str(repaired.get("password") or "").strip()
    command = str(repaired.get("command") or "").strip()

    # If the model echoed a redacted placeholder back as the password value,
    # treat it as missing so we can recover the real credential from context.
    if password.startswith("[REDACTED]"):
        repaired.pop("password", None)
        password = ""

    password_source = "explicit" if password else "none"

    target = str(repaired.get("target") or "").strip()
    if not host and not target:
        inferred_host = _infer_ssh_host_from_context(state)
        if inferred_host:
            repaired["host"] = inferred_host
            host = inferred_host
            metadata["recovered_ssh_host"] = inferred_host
            metadata["routing_reason"] = "ssh_host_recovery"

    if host and not user:
        inferred_user = _infer_ssh_user_from_state_context(host, state=state)
        user_source = "task_context"
        if not inferred_user:
            inferred_user = _infer_ssh_user_from_execution_records(host, state=state)
            user_source = "prior_ssh_exec"
        if not inferred_user:
            inferred_user = _infer_ssh_user_from_session_memory(host, state=state)
            user_source = "session_memory"
        if inferred_user:
            repaired["user"] = inferred_user
            user = inferred_user
            metadata.update(
                {
                    "recovered_ssh_user": inferred_user,
                    "recovered_ssh_user_source": user_source,
                    "routing_reason": "ssh_task_context_user_recovery",
                }
            )

    if host and not password:
        inferred_password, password_source = _infer_ssh_password(
            host,
            user=user,
            state=state,
            credential_store=credential_store,
        )
        if inferred_password:
            repaired["password"] = inferred_password
            password = inferred_password
            metadata.update(
                {
                    "recovered_ssh_password": True,
                    "recovered_ssh_password_source": password_source,
                }
            )
            metadata["routing_reason"] = metadata.get("routing_reason") or f"ssh_password_recovery_{password_source}"

    if not command and _task_requests_ssh_connection_probe(state):
        repaired["command"] = "whoami"
        metadata["recovered_ssh_command"] = "whoami"
        metadata["routing_reason"] = metadata.get("routing_reason") or "ssh_connection_probe_recovery"

    metadata.update(_ssh_auth_debug_metadata(repaired, password_source=password_source))
    return repaired, metadata


# Import at bottom to avoid circular import issues at module load time
from .dispatcher_remote_detection import task_requests_ssh_connection_probe as _task_requests_ssh_connection_probe


def _pin_and_guard_ssh_credentials(
    arguments: dict[str, Any],
    *,
    state: Any | None,
    normalization_metadata: dict[str, Any],
    credential_store: Any | None = None,
) -> tuple[dict[str, Any], ToolEnvelope | None, dict[str, Any]]:
    """Pin confirmed SSH credentials and block dispatch when the model contradicts them."""
    if not isinstance(arguments, dict) or state is None:
        return arguments, None, {}

    host = str(arguments.get("host") or "").strip().lower()
    if not host:
        return arguments, None, {}

    target = _session_ssh_target_record(host, state=state)
    if not isinstance(target, dict) or not bool(target.get("confirmed")):
        return arguments, None, {}

    confirmed_user = str(target.get("user") or "").strip()
    confirmed_password = ""
    if credential_store is not None:
        confirmed_password = credential_store.get_ssh_password(host, confirmed_user) or ""
    if not confirmed_password:
        confirmed_password = str(target.get("password") or "").strip()
    confirmed_port = target.get("port")
    confirmed_identity = str(target.get("identity_file") or "").strip()

    user = str(arguments.get("user") or "").strip()
    password = str(arguments.get("password") or "").strip()

    password_origin = str(normalization_metadata.get("ssh_password_origin") or "").strip()
    user_was_recovered = str(normalization_metadata.get("recovered_ssh_user_source") or "").strip() != ""

    metadata: dict[str, Any] = {}
    repaired = dict(arguments)

    if not user and confirmed_user:
        repaired["user"] = confirmed_user
        metadata["pinned_ssh_user"] = confirmed_user
        user = confirmed_user

    if not password and confirmed_password:
        repaired["password"] = confirmed_password
        metadata["pinned_ssh_password"] = True
        password = confirmed_password

    if "port" not in repaired and isinstance(confirmed_port, int):
        repaired["port"] = confirmed_port
        metadata["pinned_ssh_port"] = confirmed_port

    if not str(repaired.get("identity_file") or "").strip() and confirmed_identity:
        repaired["identity_file"] = confirmed_identity
        metadata["pinned_ssh_identity_file"] = confirmed_identity

    if user and confirmed_user and user != confirmed_user and not user_was_recovered:
        metadata["ssh_credential_block_reason"] = "user_mismatch"
        error = (
            f"SSH user mismatch for {host}: the confirmed session user is `{confirmed_user}`, "
            f"but the request uses `{user}`. Use the confirmed credentials or ask the user for new ones."
        )
        return repaired, ToolEnvelope(
            success=False,
            error=error,
            metadata={
                "reason": "ssh_credential_pinning_blocked",
                "block_reason": "user_mismatch",
                "expected_user": confirmed_user,
                "provided_user": user,
            },
        ), metadata

    if password and confirmed_password:
        current_fp = _password_fingerprint(password)
        confirmed_fp = _password_fingerprint(confirmed_password)
        if current_fp != confirmed_fp:
            if (
                password_origin == "explicit"
                and _explicit_ssh_password_matches_current_user_context(
                    host,
                    password,
                    user=user,
                    state=state,
                )
            ):
                metadata["ssh_credential_rotation_candidate"] = True
                metadata["ssh_credential_rotation_source"] = "current_user_context"
                return repaired, None, metadata

            repaired["password"] = confirmed_password
            metadata["pinned_ssh_password"] = True
            metadata["pinned_ssh_password_overrode_mismatch"] = True
            metadata["pinned_ssh_password_overrode_origin"] = password_origin or "unknown"
            metadata["provided_ssh_password_fingerprint"] = current_fp
            metadata["confirmed_ssh_password_fingerprint"] = confirmed_fp

    return repaired, None, metadata
