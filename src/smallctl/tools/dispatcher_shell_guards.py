from __future__ import annotations

import re

from ..models.tool_result import ToolEnvelope

_HARNESS_TOOL_SHELL_NAMES = {
    "artifact_grep",
    "artifact_print",
    "artifact_read",
    "ask_human",
    "ast_patch",
    "dir_list",
    "file_delete",
    "file_download",
    "file_patch",
    "file_read",
    "file_write",
    "find_files",
    "http_get",
    "http_post",
    "index_query_symbol",
    "json_query",
    "log_note",
    "loop_status",
    "memory_update",
    "plan_set",
    "process_kill",
    "shell_exec",
    "ssh_exec",
    "ssh_file_patch",
    "ssh_file_read",
    "ssh_file_replace_between",
    "ssh_file_write",
    "task_complete",
    "task_fail",
    "web_fetch",
    "web_search",
    "yaml_read",
}
_HARNESS_TOOL_AS_SHELL_RE = re.compile(
    r"^\s*(?P<tool>[A-Za-z_][A-Za-z0-9_]*)\s+(?:[A-Za-z_][A-Za-z0-9_]*=|[{'\"])",
    re.DOTALL,
)
_RAW_SSH_SHELL_RE = re.compile(r"^\s*(?:ssh\b|scp\b|sftp\b|sshpass\b)", re.IGNORECASE)


def guard_harness_tool_as_ssh_shell_command(command: str) -> ToolEnvelope | None:
    match = _HARNESS_TOOL_AS_SHELL_RE.match(str(command or ""))
    if not match:
        return None
    attempted = str(match.group("tool") or "").strip()
    if attempted not in _HARNESS_TOOL_SHELL_NAMES:
        return None
    return ToolEnvelope(
        success=False,
        error=(
            f"You tried to run harness tool `{attempted}` as a remote shell command. "
            f"Call `{attempted}` directly with JSON arguments instead of passing it to `ssh_exec`."
        ),
        metadata={
            "tool_name": "ssh_exec",
            "reason": "harness_tool_as_remote_shell_command",
            "suggested_tool": attempted,
        },
    )


def looks_like_raw_ssh_shell_command(command: str) -> bool:
    return bool(_RAW_SSH_SHELL_RE.match(str(command or "").strip()))


def guard_nested_raw_ssh_in_ssh_exec(command: str) -> ToolEnvelope | None:
    if not looks_like_raw_ssh_shell_command(command):
        return None
    return ToolEnvelope(
        success=False,
        error=(
            "Do not put raw `ssh`/`sshpass`/`scp`/`sftp` inside `ssh_exec.command`; "
            "`ssh_exec` already opens the SSH connection. Use canonical `ssh_exec` arguments, "
            "for example `ssh_exec(host='192.168.1.89', user='root', password='...', command='whoami')`."
        ),
        metadata={
            "tool_name": "ssh_exec",
            "reason": "nested_raw_ssh_in_ssh_exec",
            "command": command,
            "suggested_tool": "ssh_exec",
            "suggested_command": "whoami",
        },
    )


def raw_ssh_shell_block_envelope(command: str, *, ssh_available: bool) -> ToolEnvelope:
    if ssh_available:
        error = (
            "Raw `ssh`/`scp`/`sftp` shell commands are not allowed here. "
            "Use canonical `ssh_exec` for remote commands or `ssh_file_read` / `ssh_file_write` / "
            "`ssh_file_patch` / `ssh_file_replace_between` for remote file operations."
        )
    else:
        error = (
            "Raw `ssh`/`scp`/`sftp` shell commands are blocked, and canonical SSH tools are not currently available. "
            "Resume with the network/SSH tool profile or ask for help instead of using local `shell_exec`."
        )
    return ToolEnvelope(
        success=False,
        error=error,
        metadata={
            "tool_name": "shell_exec",
            "reason": "raw_ssh_shell_blocked",
            "suggested_tools": [
                "ssh_exec",
                "ssh_file_read",
                "ssh_file_write",
                "ssh_file_patch",
                "ssh_file_replace_between",
            ],
            "command": command,
        },
    )
