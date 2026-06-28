from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ToolPlanMode = Literal["tool_plan"]

PARALLELIZABLE_TOOL_PLAN_TOOLS = frozenset({
    "file_read",
    "dir_list",
    "grep",
    "find_files",
    "artifact_read",
    "artifact_grep",
    "web_search",
    "web_fetch",
    "ssh_file_read",
    "ssh_dir_list",
    "git_status",
    "git_diff",
    "read_log",
})

READONLY_TOOL_PLAN_TOOLS = set(PARALLELIZABLE_TOOL_PLAN_TOOLS)

MUTATING_TOOL_PLAN_BLOCKLIST = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "shell_exec",
    "ssh_exec",
    "ssh_file_write",
    "ssh_file_patch",
    "ansible",
    "task_complete",
    "task_fail",
    "ask_human",
    "memory_update",
    "log_note",
}

TOOL_PLAN_ALIASES = {
    "search": "grep",
    "repo_search": "grep",
    "grep_search": "grep",
    "read_file": "file_read",
    "list_dir": "dir_list",
    "fetch_url": "web_fetch",
}


@dataclass(slots=True)
class ToolPlanStep:
    id: str
    tool: str
    args: dict[str, Any]
    reason: str = ""
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False


@dataclass(slots=True)
class ToolPlan:
    mode: ToolPlanMode
    objective: str
    steps: list[ToolPlanStep]
    max_steps: int = 6
