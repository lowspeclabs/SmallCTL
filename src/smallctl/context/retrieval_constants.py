from __future__ import annotations

CHAT_SUPPRESSED_TOOL_NAMES = {
    "shell_exec",
    "ssh_exec",
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "process_kill",
    "http_post",
    "file_download",
}

REMOTE_FILE_TOOLS = {
    "ssh_exec",
    "ssh_file_read",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
}

DIAGNOSTIC_FAILURE_TOOL_PENALTIES = {
    "artifact_grep": 24.0,
    "artifact_read": 18.0,
    "web_search": 12.0,
}
