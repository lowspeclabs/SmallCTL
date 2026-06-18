from __future__ import annotations

from .tool_plan_schema import READONLY_TOOL_PLAN_TOOLS

PROMPT_VARIANTS = (
    "Use the standard staged-step strategy. Prefer concrete tool progress and call step_complete only when verified.",
    "Be conservative: inspect or verify before mutation, prefer the smallest change, and avoid unrelated edits.",
    "Try an alternate route from the obvious one while still obeying the active step contract and tool allowlist.",
    "Debug first: identify the likely failure mode, gather targeted evidence, then act.",
    "Minimize risk: avoid shell or remote tools unless they are necessary for this exact step.",
)

HIGH_RISK_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "shell_exec",
    "ssh_exec",
    "ssh_file_write",
    "ssh_file_patch",
}

READ_ONLY_CONTROL_TOOLS = {"loop_status"}

READ_ONLY_STAGED_TOOLS = frozenset(set(READONLY_TOOL_PLAN_TOOLS) | READ_ONLY_CONTROL_TOOLS)

LOCAL_FILE_MUTATION_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
}

SHELL_EXECUTION_TOOLS = {"shell_exec", "ssh_exec"}
REMOTE_MUTATION_TOOLS = {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}
