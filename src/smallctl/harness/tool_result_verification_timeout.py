from __future__ import annotations

import re
from typing import Any

from .tool_result_verification_constants import (
    _LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE,
    _LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE,
)

_DOCKER_DETACHED_RUN_RE = re.compile(r"\bdocker\s+run\b.*\s-d(?:\s|$)", re.IGNORECASE | re.DOTALL)
_DOCKER_DETACHED_SUCCESS_RE = re.compile(
    r"\bStatus:\s+Downloaded\s+newer\s+image\b"
    r"|"
    r"^[0-9a-f]{12,64}$",
    re.IGNORECASE | re.MULTILINE,
)


def _docker_detached_run_timed_out_after_success(command: str, stdout: str, stderr: str) -> bool:
    if not _DOCKER_DETACHED_RUN_RE.search(str(command or "")):
        return False
    combined = "\n".join(part for part in (str(stdout or ""), str(stderr or "")) if part).strip()
    return bool(combined and _DOCKER_DETACHED_SUCCESS_RE.search(combined))


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
    out = str(stdout or "").strip()
    err = str(stderr or "").strip()
    combined = "\n".join(part for part in (out, err) if part).strip()
    if _docker_detached_run_timed_out_after_success(cmd, out, err):
        return False
    if _LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE.search(cmd):
        return True
    if _LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE.search(combined):
        return True
    if "timeout" in lowered or "timed out" in combined.lower():
        return True
    if result and getattr(result, "error", None):
        error = str(result.error or "").lower()
        if "timeout" in error or "timed out" in error:
            return True
    return False
