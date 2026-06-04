from __future__ import annotations

import re
from typing import Any

from .shell_support_constants import _DEB822_FIELDS


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


def _apt_deb822_preflight_guard(command: str, *, tool_name: str) -> dict[str, Any] | None:
    raw = str(command or "").strip()
    lowered = raw.lower()
    if not re.search(r"\b(?:apt|apt-get)\b", lowered):
        return None
    if not re.search(r"\b(?:install|update|upgrade|dist-upgrade|full-upgrade|autoremove|purge|remove)\b", lowered):
        return None
    if all(field.lower() in lowered for field in _DEB822_FIELDS) and "debian.sources" in lowered:
        return None
    validator = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path('/etc/apt/sources.list.d/debian.sources')\n"
        "s = p.read_text()\n"
        "missing = [k for k in ('Types:', 'URIs:', 'Suites:', 'Components:') if k not in s]\n"
        "assert not missing, 'debian.sources missing deb822 fields: ' + ', '.join(missing)\n"
        "print('deb822 OK')\n"
        "PY"
    )
    return _guard_fail(
        f"`{tool_name}` blocked apt package operation until `/etc/apt/sources.list.d/debian.sources` is validated as deb822.",
        reason="apt_deb822_preflight_required",
        command=raw,
        next_required_action={
            "tool_name": tool_name,
            "required_arguments": {"command": validator},
            "notes": [
                "Run the validator first, or combine this exact validation before apt with &&.",
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
