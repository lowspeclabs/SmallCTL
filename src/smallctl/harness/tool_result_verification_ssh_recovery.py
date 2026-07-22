from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from ..tools.dispatcher_ssh_auth import password_fingerprint, ssh_auth_recovery_entry_key
from .tool_result_verification_constants import _RAW_SSH_COMMAND_RE, _SSH_AUTH_RECOVERY_KEY
from .tool_result_verification_helpers import snip_text as _snip_text


def _update_ssh_auth_recovery_state(
    state: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
) -> None:
    if tool_name != "ssh_exec":
        return
    args = arguments if isinstance(arguments, dict) else {}
    command = str(args.get("command") or "").strip()
    if _RAW_SSH_COMMAND_RE.match(command):
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    recovery_state = scratchpad.setdefault(_SSH_AUTH_RECOVERY_KEY, {})
    if not isinstance(recovery_state, dict):
        recovery_state = {}
        scratchpad[_SSH_AUTH_RECOVERY_KEY] = recovery_state

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("reason") == "tool_not_exposed_this_turn":
        return None
    output = result.output if isinstance(result.output, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    host = str(args.get("host") or metadata.get("host") or "").strip().lower()
    user = str(args.get("user") or "").strip()
    if not host:
        return
    entry_key = ssh_auth_recovery_entry_key(host, user)
    reached_remote_host = (
        result.success
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    )
    if reached_remote_host:
        recovery_state.pop(entry_key, None)
        return

    error_text = " ".join(
        part
        for part in [
            str(result.error or "").strip(),
            str(output.get("stderr") or "").strip(),
            str(metadata.get("error") or "").strip(),
        ]
        if part
    ).strip()
    if "permission denied" not in error_text.lower():
        return

    prior = recovery_state.get(entry_key)
    prior_failures = 0
    if isinstance(prior, dict):
        try:
            prior_failures = max(0, int(prior.get("failure_count") or 0))
        except (TypeError, ValueError):
            prior_failures = 0
    password = str(args.get("password") or "").strip()
    recovery_state[entry_key] = {
        "host": host,
        "user": user,
        "failure_count": prior_failures + 1,
        "password_provided": bool(password),
        "password_fingerprint": password_fingerprint(password),
        "ssh_auth_mode": str(metadata.get("ssh_auth_mode") or "").strip(),
        "ssh_auth_transport": str(metadata.get("ssh_auth_transport") or "").strip(),
        "last_command": str(args.get("command") or "").strip(),
        "last_error": _snip_text(error_text, limit=240),
    }
