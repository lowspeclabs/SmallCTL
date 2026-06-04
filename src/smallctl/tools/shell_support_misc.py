from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _shell_workspace_relative_hint(command: str, cwd: str | None = None) -> str | None:
    raw_command = str(command or "")
    match = re.search(r"(?<![\w/])(/temp(?:/[^\s\"'`]+)*)", raw_command)
    if match is None:
        return None

    suspicious_path = match.group(1)
    trimmed = suspicious_path.lstrip("/")
    if not trimmed:
        return None

    base = Path(cwd) if cwd else Path.cwd()
    workspace_candidate = (base / Path(trimmed)).resolve()
    if not (workspace_candidate.exists() or workspace_candidate.parent.exists()):
        return None

    return (
        f"That command used the root-level `{suspicious_path}` path. "
        f"If you meant the workspace copy, retry with `{( './' + trimmed)}` instead."
    )


def _shell_status_update_interval(timeout_sec: int) -> float:
    return max(1.0, min(max(1, timeout_sec) / 3.0, 10.0))


def _build_shell_status_update(command: str, *, elapsed_sec: float, timeout_sec: int) -> str:
    elapsed_text = f"{elapsed_sec:.0f}s"
    timeout_text = f"{max(1, timeout_sec)}s"
    return f"[still running after {elapsed_text} of {timeout_text}] {command}"
