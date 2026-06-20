from __future__ import annotations

import re
from typing import Any

from .shell_support_constants import _DEB822_FIELDS, _DEB822_PREFLIGHT_KEY


def _guard_fail(
    message: str,
    *,
    reason: str,
    command: str,
    error_kind: str | None = None,
    next_required_tool: dict[str, Any] | None = None,
    next_required_action: dict[str, Any] | str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent guard failure result."""
    from .common import fail
    metadata: dict[str, Any] = {
        "reason": reason,
        "command": command,
    }
    if error_kind is not None:
        metadata["error_kind"] = error_kind
    if next_required_tool is not None:
        metadata["next_required_tool"] = next_required_tool
    if next_required_action is not None:
        metadata["next_required_action"] = next_required_action
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return fail(message, metadata=metadata)


def validate_sources_file(content: str) -> dict[str, Any]:
    """Validate a deb822 .sources file content.

    Uses ``debian.deb822.Sources`` when available for structured parsing,
    otherwise falls back to a naive substring check.

    Returns:
        {"valid": True} on success, or
        {"valid": False, "error": "...", "missing_fields": [...]} on failure.
    """
    text = str(content or "")
    try:
        from debian.deb822 import Sources
    except ImportError:
        missing = [f for f in _DEB822_FIELDS if f not in text]
        if missing:
            return {
                "valid": False,
                "error": f"missing deb822 fields: {', '.join(missing)}",
                "missing_fields": missing,
            }
        return {"valid": True}

    try:
        s = Sources(text)
        missing = []
        for field in _DEB822_FIELDS:
            key = field.rstrip(":")
            val = s.get(key)
            if not val or not str(val).strip():
                missing.append(field)
        if missing:
            return {
                "valid": False,
                "error": f"missing required deb822 fields: {', '.join(missing)}",
                "missing_fields": missing,
            }
        return {"valid": True}
    except Exception as exc:
        return {"valid": False, "error": str(exc), "missing_fields": []}


def _is_deb822_preflight_clean(state: Any, host: str, user: str) -> bool:
    """Return True if deb822 has been validated for this host/user recently."""
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    preflights = scratchpad.get(_DEB822_PREFLIGHT_KEY)
    if not isinstance(preflights, dict):
        return False
    key = "|".join([str(host or "").strip().lower(), str(user or "").strip().lower()])
    entry = preflights.get(key)
    if isinstance(entry, dict):
        status = str(entry.get("status") or "").strip()
        created = int(entry.get("created_at_step", 0) or 0)
        if status == "clean" and int(getattr(state, "step_count", 0) or 0) - created <= 20:
            return True
    return False


def _mark_deb822_preflight_clean(state: Any, host: str, user: str) -> None:
    """Record that deb822 has been validated for this host/user."""
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    preflights = scratchpad.get(_DEB822_PREFLIGHT_KEY)
    if not isinstance(preflights, dict):
        preflights = {}
        scratchpad[_DEB822_PREFLIGHT_KEY] = preflights
    key = "|".join([str(host or "").strip().lower(), str(user or "").strip().lower()])
    preflights[key] = {
        "status": "clean",
        "created_at_step": int(getattr(state, "step_count", 0) or 0),
    }
    if hasattr(state, "touch"):
        state.touch()


def _looks_like_deb822_validator(command: str) -> bool:
    """Return True if *command* matches the deb822 validator we generate."""
    raw = str(command or "").strip()
    return (
        "python3" in raw
        and "Path('/etc/apt/sources.list.d/debian.sources')" in raw
        and "deb822 OK" in raw
    )


def _apt_deb822_preflight_guard(
    command: str,
    *,
    tool_name: str,
    state: Any = None,
    host: str = "",
    user: str = "",
) -> dict[str, Any] | None:
    raw = str(command or "").strip()
    lowered = raw.lower()
    if not re.search(r"\b(?:apt|apt-get)\b", lowered):
        return None
    if not re.search(r"\b(?:install|update|upgrade|dist-upgrade|full-upgrade|autoremove|purge|remove)\b", lowered):
        return None
    if all(field.lower() in lowered for field in _DEB822_FIELDS) and "debian.sources" in lowered:
        return None

    # Allow through if this session already validated deb822 for this host/user.
    if _is_deb822_preflight_clean(state, host, user):
        return None

    # Only gate on deb822 if a prior apt update actually failed with a
    # source-format error.  Do not proactively block all apt operations —
    # most systems have healthy sources.  Critically, GPG/signature errors
    # (SHA1 rejections, key import failures, sqv errors) are *not*
    # deb822-format problems and must not trigger the deb822 gate.
    if state is not None:
        scratchpad = getattr(state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            guard_state = scratchpad.get("_apt_sources_list_d_guard")
            if isinstance(guard_state, dict):
                update_succeeded = guard_state.get("apt_update_succeeded")
                if update_succeeded is not False:
                    return None
                # If the failure was a GPG / signature issue rather than
                # a source-file format problem, the deb822 gate is
                # irrelevant — let the operation through.
                last_err = str(guard_state.get("last_update_error") or "").lower()
                _gpg_markers = (
                    "gpg",
                    "signature",
                    "no_pubkey",
                    "no pubkey",
                    "keyring",
                    "jcameron",
                    "sqv",
                    "sha1",
                    "security policy",
                    "openpgp",
                )
                if any(m in last_err for m in _gpg_markers):
                    return None
            else:
                return None
        else:
            return None
    else:
        return None

    # One-liner validator (chainable with &&) — no heredoc delimiter issues.
    validator = (
        "python3 -c \"from pathlib import Path; "
        "p = Path('/etc/apt/sources.list.d/debian.sources'); "
        "s = p.read_text(); "
        "missing = [k for k in ('Types:', 'URIs:', 'Suites:', 'Components:') if k not in s]; "
        "assert not missing, 'debian.sources missing deb822 fields: ' + ', '.join(missing); "
        "print('deb822 OK')\""
    )
    return _guard_fail(
        f"`{tool_name}` blocked apt package operation until `/etc/apt/sources.list.d/debian.sources` is validated as deb822.",
        reason="apt_deb822_preflight_required",
        command=raw,
        next_required_action={
            "tool_name": tool_name,
            "required_arguments": {"command": validator},
            "notes": [
                "Run the validator first as a standalone ssh_exec/shell_exec call.",
                "Once validation passes, apt commands will be allowed for the remainder of the session.",
                "For small sources files, prefer full-file ssh_file_write over boundary-anchored replacement.",
            ],
        },
        extra_metadata={
            "required_fields": list(_DEB822_FIELDS),
        },
    )


def _apt_sources_list_d_guard(
    command: str,
    *,
    tool_name: str,
    state: Any = None,
    host: str = "",
    user: str = "",
) -> dict[str, Any] | None:
    """Block apt operations if a repo file under /etc/apt/sources.list.d was created
    but apt-get update has not yet succeeded, or if the last apt-get update failed
    with a malformed-entry error.
    """
    raw = str(command or "").strip()
    lowered = raw.lower()
    if not re.search(r"\b(?:apt|apt-get)\b", lowered):
        return None
    # Always allow apt-get update / apt update as the remediation step
    if re.search(r"\bupdate\b", lowered) and not re.search(r"\b(?:install|upgrade|dist-upgrade|full-upgrade|autoremove|purge|remove)\b", lowered):
        return None
    if not re.search(r"\b(?:install|upgrade|dist-upgrade|full-upgrade|autoremove|purge|remove)\b", lowered):
        return None

    if state is None:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None

    key = "_apt_sources_list_d_guard"
    guard_state = scratchpad.get(key)
    if not isinstance(guard_state, dict):
        return None

    # Check for malformed-entry failure
    last_update_error = str(guard_state.get("last_update_error") or "").strip()
    if "malformed entry" in last_update_error.lower() or "malformed" in last_update_error.lower():
        malformed_file = str(guard_state.get("malformed_file") or "").strip()
        extra = {}
        if malformed_file:
            extra["malformed_file"] = malformed_file
        return _guard_fail(
            f"`{tool_name}` blocked apt operation because the last `apt-get update` reported a malformed source list. "
            "Repair or remove the malformed file before proceeding.",
            reason="apt_malformed_sources_list",
            command=raw,
            next_required_action={
                "strategy": "repair_malformed_apt_source",
                "steps": [
                    f"Inspect and fix or remove the malformed file ({malformed_file or 'under /etc/apt/sources.list.d/'}).",
                    "Run `apt-get update` again to confirm it succeeds.",
                    "Then retry the original apt command.",
                ],
            },
            extra_metadata={
                "host": host,
                "user": user,
                **extra,
            },
        )

    # Check if sources.list.d was modified but update has not succeeded
    if guard_state.get("sources_list_d_modified") and not guard_state.get("apt_update_succeeded"):
        return _guard_fail(
            f"`{tool_name}` blocked apt operation because a repo file under `/etc/apt/sources.list.d/` was created "
            "but `apt-get update` has not yet succeeded. Run `apt-get update` first.",
            reason="apt_update_required_after_sources_change",
            command=raw,
            next_required_action="apt-get update",
            extra_metadata={
                "host": host,
                "user": user,
            },
        )

    return None


def record_apt_update_result(
    state: Any,
    *,
    command: str,
    success: bool,
    stderr: str = "",
    host: str = "",
    user: str = "",
) -> None:
    """Observe apt-get update results to track whether apt sources are healthy."""
    if state is None:
        return
    raw = str(command or "").strip().lower()
    if not re.search(r"\bapt-get\b.*\bupdate\b|\bapt\b.*\bupdate\b", raw):
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    key = "_apt_sources_list_d_guard"
    guard_state = scratchpad.setdefault(key, {})
    if not isinstance(guard_state, dict):
        guard_state = {}
        scratchpad[key] = guard_state

    if success:
        guard_state["apt_update_succeeded"] = True
        guard_state["last_update_error"] = ""
        guard_state["malformed_file"] = ""
    else:
        stderr_text = str(stderr or "").strip()
        guard_state["apt_update_succeeded"] = False
        # Try to extract malformed file path from error
        malformed_match = re.search(r"malformed entry.*?in list file\s+(.+?)(?:\s|$)", stderr_text, re.IGNORECASE)
        if malformed_match:
            guard_state["last_update_error"] = stderr_text
            guard_state["malformed_file"] = malformed_match.group(1).strip()
        else:
            guard_state["last_update_error"] = stderr_text


def record_sources_list_d_modification(
    state: Any,
    *,
    path: str,
    host: str = "",
    user: str = "",
) -> None:
    """Record that a file under /etc/apt/sources.list.d/ was modified."""
    if state is None:
        return
    normalized = str(path or "").strip()
    if not normalized.startswith("/etc/apt/sources.list.d/"):
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    key = "_apt_sources_list_d_guard"
    guard_state = scratchpad.setdefault(key, {})
    if not isinstance(guard_state, dict):
        guard_state = {}
        scratchpad[key] = guard_state
    guard_state["sources_list_d_modified"] = True
    guard_state["apt_update_succeeded"] = False
    guard_state["modified_file"] = normalized
    guard_state["host"] = str(host or "").strip().lower()
    guard_state["user"] = str(user or "").strip().lower()


def classify_shell_outcome(command: str, returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    """Classify expected miss commands as empty_result when stdout/stderr semantics are clear."""
    cmd = str(command or "").strip()
    if returncode == 0:
        return {"status": "success", "kind": "ok"}
    combined = f"{stdout}\n{stderr}".lower()
    if (
        ("curl" in cmd.lower() or "wget" in cmd.lower())
        and (" sh " in f" {cmd.lower()} " or " bash " in f" {cmd.lower()} ")
        and ("404" in combined or "not found" in combined)
    ):
        return {
            "status": "failure",
            "kind": "error",
            "exit_code": returncode,
            "failure_mode": "remote_installer_download_error",
            "next_required_action": "Stop executing the downloaded file; fetch and inspect the installer source or verify the URL first.",
        }
    absence_patterns = [
        (r'\bfind\s+.*-name\s+', "absence_probe"),
        (r'\bwhich\s+', "absence_probe"),
        (r'\bgrep\s+-[qLl]', "absence_probe"),
        (r'\btest\s+-[efdx]', "absence_probe"),
    ]
    for pat, kind in absence_patterns:
        if re.search(pat, cmd) and not stdout.strip():
            return {"status": "success", "kind": "empty_result", "exit_code": returncode}
    return {"status": "failure", "kind": "error", "exit_code": returncode}
