from __future__ import annotations

import shlex
from typing import Any

from .tool_result_verification_constants import _REMOTE_READBACK_COMMANDS


def _simple_remote_readback_path(command: str) -> str:
    try:
        tokens = shlex.split(str(command or "").strip())
    except ValueError:
        return ""
    if not tokens:
        return ""
    if tokens[0] == "sudo":
        tokens = tokens[1:]
    if not tokens:
        return ""

    command_name = tokens[0]
    if command_name in _REMOTE_READBACK_COMMANDS:
        path_operands = [token for token in tokens[1:] if token.startswith("/")]
        return path_operands[0] if len(path_operands) == 1 else ""

    if command_name == "sed" and "-n" in tokens:
        path_operands = [token for token in tokens[1:] if token.startswith("/")]
        return path_operands[0] if len(path_operands) == 1 else ""

    return ""
