from __future__ import annotations

import re
from typing import Any

_SSH_TASK_TARGET_RE = re.compile(
    r"\b(?:ssh|scp|sftp)\s+(?:[A-Za-z0-9._-]+@)?(?P<host>[A-Za-z0-9._-]+)\b",
    re.IGNORECASE,
)
_AT_HOST_TARGET_RE = re.compile(r"\b[A-Za-z0-9._-]+@(?P<host>[A-Za-z0-9._-]+)\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_REMOTE_TASK_HINT_RE = re.compile(r"\b(?:remote|ssh|username|password|server|host)\b", re.IGNORECASE)

_REMOTE_COMMAND_PATH_RE = re.compile(
    r"(?<![\w/])/(?:(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:\.[A-Za-z0-9._-]+)?)"
)
_REMOTE_INFRA_PROBE_RE = re.compile(
    r"\b(?:which|command\s+-v|type|whereis)\s+nginx\b"
    r"|"
    r"\bnginx\s+-[A-Za-z]"
    r"|"
    r"\b(?:systemctl|service)\s+(?:status|is-active|is-enabled|reload|restart|start|stop)\s+nginx\b",
    re.IGNORECASE,
)
_REMOTE_ABSOLUTE_PATH_PREFIXES = (
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/media",
    "/mnt",
    "/opt",
    "/proc",
    "/root",
    "/run",
    "/srv",
    "/sys",
    "/usr",
    "/var",
)


def looks_like_remote_absolute_path(path: str, *, state: Any | None = None) -> bool:
    candidate = str(path or "").strip()
    if not candidate.startswith("/"):
        return False

    cwd = str(getattr(state, "cwd", "") or "").rstrip("/")
    if cwd and (candidate == cwd or candidate.startswith(cwd + "/")):
        return False
    if candidate == "/tmp" or candidate.startswith("/tmp/"):
        return False

    if candidate in _REMOTE_ABSOLUTE_PATH_PREFIXES:
        return True
    return candidate.startswith(tuple(prefix + "/" for prefix in _REMOTE_ABSOLUTE_PATH_PREFIXES))


def command_mentions_remote_absolute_path(command: str, *, state: Any | None = None) -> bool:
    for match in _REMOTE_COMMAND_PATH_RE.finditer(str(command or "")):
        if looks_like_remote_absolute_path(match.group(0), state=state):
            return True
    return False


def looks_like_remote_infrastructure_probe_command(command: str) -> bool:
    return _REMOTE_INFRA_PROBE_RE.search(str(command or "").strip()) is not None
