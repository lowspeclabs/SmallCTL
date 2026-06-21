from __future__ import annotations

from pathlib import Path
from typing import Any

from .shell_parsing import (
    _split_shell_command_segments,
    _split_shell_words,
    _strip_environment_and_wrappers,
)
from .shell_support_common import _guard_fail
from .shell_support_constants import (
    _DETACHED_COMMAND_MARKERS,
    _FOLLOW_FLAGS,
    _FOREGROUND_BINARIES,
    _FOREGROUND_SUBCOMMANDS,
    _INSPECTION_FLAGS,
    _PACKAGE_RUNNERS,
    _SERVICE_MANAGER_COMMANDS,
)


def _foreground_command_guard(
    command: str,
    *,
    tool_name: str,
    allow_background_parameter: bool = False,
) -> dict[str, Any] | None:
    reason = _likely_long_running_foreground_reason(command)
    if reason is None:
        return None

    background_text = " or `background=True`" if allow_background_parameter else ""
    return _guard_fail(
        f"`{tool_name}` blocked a likely long-running foreground command: `{command}`. "
        "Start services with a service manager, a detached/background command, or a bounded command "
        f"such as `timeout 20s ...`{background_text}, then verify with a separate health check.",
        reason="long_running_foreground_command",
        command=command,
        next_required_action={
            "strategy": "detach_or_bound_then_verify",
            "notes": [
                "Use a service manager for daemons when available.",
                "Use a detached/background launch when the command is expected to keep running.",
                "Use a bounded `timeout` wrapper only when sampling foreground output is intentional.",
                "Run a separate verification command after launch.",
            ],
        },
        extra_metadata={
            "foreground_detection": reason,
        },
    )


def _likely_long_running_foreground_reason(command: str) -> str | None:
    raw = str(command or "").strip()
    if not raw:
        return None
    if _has_detached_or_bounded_marker(raw):
        return None

    commands = [raw]
    words = _split_shell_words(raw)
    if len(words) >= 3 and words[0] in {"bash", "sh", "/bin/bash", "/bin/sh"} and words[1] in {"-c", "-lc"}:
        commands.append(words[2])

    for candidate in commands:
        for segment in _split_shell_command_segments(candidate):
            reason = _likely_long_running_simple_command_reason(segment)
            if reason is not None:
                return reason
    return None


def _has_detached_or_bounded_marker(command: str) -> bool:
    raw = str(command or "").strip()
    if not raw:
        return False
    if raw.endswith("&") and not raw.endswith("&&"):
        return True
    lowered = raw.lower()
    if lowered.startswith("timeout ") or lowered.startswith("/usr/bin/timeout "):
        return True
    return any(marker in lowered for marker in _DETACHED_COMMAND_MARKERS)


def _likely_long_running_simple_command_reason(command: str) -> str | None:
    words = _split_shell_words(command)
    if not words:
        return None
    words = _strip_environment_and_wrappers(words)
    if not words:
        return None

    executable = Path(words[0]).name.lower()
    args = [word.lower() for word in words[1:]]
    if any(arg in _INSPECTION_FLAGS for arg in args):
        return None
    if executable in _SERVICE_MANAGER_COMMANDS:
        return None
    if executable == "docker":
        if len(args) >= 2 and args[0] == "logs" and any(arg in _FOLLOW_FLAGS for arg in args[1:]):
            return "follow_output"
        if "run" in args and "-d" in args:
            return None
        if len(args) >= 3 and args[0] == "compose" and args[1] == "up" and "-d" in args[2:]:
            return None
        if len(args) >= 2 and args[0] == "compose" and args[1] == "logs" and any(arg in _FOLLOW_FLAGS for arg in args[2:]):
            return "follow_output"
    if executable == "kubectl" and len(args) >= 1 and args[0] == "logs" and any(arg in _FOLLOW_FLAGS for arg in args[1:]):
        return "follow_output"
    if executable in {"tail", "journalctl"} and any(arg in _FOLLOW_FLAGS for arg in args):
        return "follow_output"
    if executable in _PACKAGE_RUNNERS:
        if args[:2] == ["run", "dev"] or (args and args[0] in _FOREGROUND_SUBCOMMANDS):
            return "package_runner_foreground"
        if executable in {"yarn", "pnpm", "bun"} and args and args[0] == "run" and len(args) > 1 and args[1] in _FOREGROUND_SUBCOMMANDS:
            return "package_runner_foreground"
    if executable in {"python", "python3"} and len(args) >= 2 and args[0] == "-m" and args[1] in {"http.server", "uvicorn"}:
        return "python_module_server"
    if executable in _FOREGROUND_BINARIES:
        if executable == "caddy" and args and args[0] == "start":
            return None
        if executable == "caddy" and args and args[0] == "reload":
            return None
        if args and args[0] in {"run", *list(_FOREGROUND_SUBCOMMANDS)}:
            return "service_foreground_subcommand"
        if executable in {"uvicorn", "gunicorn", "redis-server", "http-server", "vite", "webpack-dev-server", "nodemon"}:
            return "service_foreground_binary"
    return None
