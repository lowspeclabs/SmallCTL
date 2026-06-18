from __future__ import annotations

from typing import Any

from .tool_result_verification_constants import (
    _LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE,
    _LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE,
)


def _is_long_running_remote_command_timeout(
    *,
    tool_name: str,
    command: str,
    result: Any,
    stdout: str,
    stderr: str,
) -> bool:
    if tool_name != "ssh_exec":
        return False
    cmd = str(command or "").strip()
    if not cmd:
        return False
    lowered = cmd.lower()
    if _LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE.search(cmd):
        return True
    out = str(stdout or "").strip()
    err = str(stderr or "").strip()
    combined = "\n".join(part for part in (out, err) if part).strip()
    if _LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE.search(combined):
        return True
    if "timeout" in lowered or "timed out" in combined.lower():
        return True
    if result and getattr(result, "error", None):
        error = str(result.error or "").lower()
        if "timeout" in error or "timed out" in error:
            return True
    return False
