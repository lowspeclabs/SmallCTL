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
        if status == "clean" and int(getattr(state, "step_count", 0) or 0) - created <= 8:
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


def classify_shell_outcome(command: str, returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    """Classify expected miss commands as empty_result when stdout/stderr semantics are clear."""
    cmd = str(command or "").strip()
    if returncode == 0:
        return {"status": "success", "kind": "ok"}
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
