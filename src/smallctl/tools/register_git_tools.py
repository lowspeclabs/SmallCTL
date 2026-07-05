from __future__ import annotations

from typing import Any, Awaitable, Callable

from . import git_tools
from .base import path_field


def register_git_tools(
    *,
    register: Callable[[list[Any]], None],
    make_registration: Callable[..., Any],
    inject_cwd: Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]],
    core_profile: str,
) -> None:
    register(
        [
            make_registration(
                name="git_status",
                description="Show git status for a workspace directory. Returns clean/dirty flag and short or full status output.",
                schema={
                    "type": "object",
                    "properties": {
                        "path": path_field("Workspace-relative directory path. Defaults to cwd."),
                        "short": {"type": "boolean", "description": "Use --short format. Default true."},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                handler=inject_cwd(git_tools.git_status),
                category="git",
                risk="low",
                allowed_modes={"chat", "loop", "planning"},
                profiles={core_profile},
            ),
            make_registration(
                name="git_diff",
                description="Show git diff for a workspace directory. Optionally show cached/staged changes and limit to a specific file.",
                schema={
                    "type": "object",
                    "properties": {
                        "path": path_field("Workspace-relative directory path. Defaults to cwd."),
                        "cached": {"type": "boolean", "description": "Show staged changes. Default false."},
                        "target": {"type": "string", "description": "Optional workspace-relative file path to limit diff to."},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                handler=inject_cwd(git_tools.git_diff),
                category="git",
                risk="low",
                allowed_modes={"chat", "loop", "planning"},
                profiles={core_profile},
            ),
            make_registration(
                name="read_log",
                description="Read the tail of a workspace log file. Bounded line-oriented read to avoid context overflow.",
                schema={
                    "type": "object",
                    "properties": {
                        "path": path_field("Workspace-relative log file path."),
                        "lines": {"type": "integer", "description": "Number of lines to read. Default 100."},
                        "offset": {"type": "integer", "description": "Optional 0-based line offset to start reading from."},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=inject_cwd(git_tools.read_log),
                category="filesystem",
                risk="low",
                allowed_modes={"chat", "loop", "planning"},
                profiles={core_profile},
            ),
        ]
    )
