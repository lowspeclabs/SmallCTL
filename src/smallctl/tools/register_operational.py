from __future__ import annotations

from typing import Any, Awaitable, Callable

from . import http, network, shell, ssh_files
from . import network_interactive_sessions


def register_operational_tools(
    *,
    register: Callable[[list[Any]], None],
    make_registration: Callable[..., Any],
    inject_state: Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]],
    inject_state_and_harness: Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]],
    core_profile: str,
    support_profile: str,
    network_profile: str,
    network_raw_profile: str,
) -> None:
    register(
        [
            make_registration(
                name="ssh_exec",
                description=(
                    "Execute a REMOTE command on a remote host via SSH with live streaming support. "
                    "Prefer `target='user@host'` when a username is known, for example "
                    "`target='root@192.168.1.63'`, rather than splitting identity across separate fields. "
                    "For interactive installers or scripts that prompt for input, use `ssh_session_start` instead. "
                    "Exit code 1 from diagnostic probes (systemctl status, dpkg -l, apt list, which, etc.) that report 'not found' is valid negative intelligence, not an error."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                        "host": {"type": "string", "description": "Target hostname or IP."},
                        "command": {"type": "string", "description": "Command to run remotely."},
                        "user": {"type": "string", "description": "SSH username."},
                        "username": {"type": "string", "description": "Alias for `user`; useful when the task says 'username root'."},
                        "port": {"type": "integer", "default": 22},
                        "identity_file": {"type": "string", "description": "Path to SSH private key."},
                        "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                        "timeout_sec": {"type": "integer", "default": 120},
                        "stdin_data": {
                            "type": "string",
                            "description": "Optional data to write to the remote command's stdin, for commands that read a finite answer stream.",
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(network.ssh_exec),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="interactive_run",
                description=(
                    "Run a command in a fresh interactive SSH session and feed it a sequence of answers. "
                    "This is the preferred high-level tool for straightforward interactive remote installers "
                    "(e.g., Pi-hole, Webmin) when the model would otherwise manage session IDs manually. "
                    "For complex multi-prompt installers with unknown prompts, use `ssh_session_start` followed by "
                    "`ssh_session_read`/`ssh_session_send` instead."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                        "host": {"type": "string", "description": "Target hostname or IP."},
                        "command": {"type": "string", "description": "Interactive command to run remotely."},
                        "answers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ordered list of answers to send when the remote command prompts for input.",
                        },
                        "user": {"type": "string", "description": "SSH username."},
                        "username": {"type": "string", "description": "Alias for `user`."},
                        "port": {"type": "integer", "default": 22},
                        "identity_file": {"type": "string", "description": "Path to SSH private key."},
                        "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                        "timeout_sec": {"type": "integer", "default": 900},
                        "wait_sec": {"type": "number", "default": 1.0, "description": "Seconds to wait after each read/send."},
                        "max_chars": {"type": "integer", "default": 6000},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(network_interactive_sessions.interactive_run),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_session_start",
                description=(
                    "Start an interactive REMOTE SSH command with a pseudo-terminal. This is the preferred tool for "
                    "running interactive installers or scripts that prompt for input (e.g., FOG install, package config). "
                    "After starting, poll output with `ssh_session_read`, send answers with `ssh_session_send`, and "
                    "close with `ssh_session_close`. If the installer supports a non-interactive flag (like `--autoaccept`), "
                    "consider using `ssh_exec` with that flag instead."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                        "host": {"type": "string", "description": "Target hostname or IP."},
                        "command": {"type": "string", "description": "Interactive command to run remotely."},
                        "user": {"type": "string", "description": "SSH username."},
                        "username": {"type": "string", "description": "Alias for `user`."},
                        "port": {"type": "integer", "default": 22},
                        "identity_file": {"type": "string", "description": "Path to SSH private key."},
                        "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                        "timeout_sec": {"type": "integer", "default": 900},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(network.ssh_session_start),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_session_read",
                description="Read current output and detected prompt state from an active interactive SSH session.",
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "wait_sec": {"type": "number", "default": 1.0},
                        "max_chars": {"type": "integer", "default": 6000},
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(network.ssh_session_read),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_session_send",
                description="Send input to an active interactive SSH session. By default this submits the input with a trailing newline.",
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "input": {"type": "string", "description": "Text to write to remote stdin. Omit the trailing newline for normal shell commands; the tool appends it by default."},
                        "send_newline": {"type": "boolean", "default": True, "description": "Append a trailing newline before writing. Set false for raw password or single-keystroke input."},
                        "wait_sec": {"type": "number", "default": 0.5},
                        "max_chars": {"type": "integer", "default": 6000},
                    },
                    "required": ["session_id", "input"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(network.ssh_session_send),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_session_close",
                description="Close an active interactive SSH session, optionally terminating the remote command.",
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "terminate": {"type": "boolean", "default": True},
                        "max_chars": {"type": "integer", "default": 6000},
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(network.ssh_session_close),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_file_read",
                description=(
                    "Read a REMOTE file over SSH with structured output. Prefer this over `ssh_exec` with cat/head/sed "
                    "for remote file inspection. "
                    "This tool operates on the REMOTE host filesystem ONLY. After writing a file remotely with `ssh_file_write`, verify it with `ssh_file_read`, never with `file_read`."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                        "host": {"type": "string", "description": "Target hostname or IP."},
                        "user": {"type": "string", "description": "SSH username."},
                        "username": {"type": "string", "description": "Alias for `user`."},
                        "port": {"type": "integer", "default": 22},
                        "identity_file": {"type": "string", "description": "Path to SSH private key."},
                        "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                        "path": {"type": "string", "description": "Remote file path."},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "max_bytes": {"type": "integer", "default": 262144},
                        "truncate": {"type": "boolean", "default": True, "description": "When false, fail instead of truncating oversized files."},
                        "timeout_sec": {"type": "integer", "default": 120},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(ssh_files.ssh_file_read),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_file_write",
                description=(
                    "Write or overwrite a REMOTE file over SSH with atomic replace and readback hash verification. "
                    "Prefer this over `ssh_exec` with here-docs, tee, or shell redirection for remote file writes. "
                    "This tool operates on the REMOTE host filesystem ONLY. The `path` argument is resolved on the REMOTE host, "
                    "not the local orchestrator. Do NOT pass local absolute paths (e.g. /home/stephen/...) unless the remote host "
                    "has an identical directory structure. After writing, verify the file with `ssh_file_read`, never with `file_read`."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                        "host": {"type": "string", "description": "Target hostname or IP."},
                        "user": {"type": "string", "description": "SSH username."},
                        "username": {"type": "string", "description": "Alias for `user`."},
                        "port": {"type": "integer", "default": 22},
                        "identity_file": {"type": "string", "description": "Path to SSH private key."},
                        "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                        "path": {"type": "string", "description": "Remote file path."},
                        "content": {"type": "string", "description": "Content to write."},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "mode": {"type": "string", "description": "overwrite, create, or append."},
                        "create_parent_dirs": {"type": "boolean", "default": False},
                        "backup": {"type": "boolean", "default": True},
                        "expected_sha256": {"type": "string", "description": "Optional hash of the current remote file."},
                        "source_artifact_id": {"type": "string", "description": "Artifact id whose sha256 should match the current remote file before writing."},
                        "timeout_sec": {"type": "integer", "default": 120},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(ssh_files.ssh_file_write),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_file_patch",
                description=(
                    "Patch a REMOTE file over SSH by replacing exact target text with replacement text. "
                    "Prefer this over `ssh_exec` with sed/perl for precise remote edits. "
                    "This tool operates on the REMOTE host filesystem ONLY."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                        "host": {"type": "string", "description": "Target hostname or IP."},
                        "user": {"type": "string", "description": "SSH username."},
                        "username": {"type": "string", "description": "Alias for `user`."},
                        "port": {"type": "integer", "default": 22},
                        "identity_file": {"type": "string", "description": "Path to SSH private key."},
                        "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                        "path": {"type": "string", "description": "Remote file path."},
                        "target_text": {"type": "string", "description": "Exact text to replace."},
                        "replacement_text": {"type": "string", "description": "Replacement text. Use an empty string to delete the target text."},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "expected_occurrences": {"type": "integer", "default": 1},
                        "backup": {"type": "boolean", "default": True},
                        "expected_sha256": {"type": "string", "description": "Optional hash of the current remote file."},
                        "source_artifact_id": {"type": "string", "description": "Artifact id whose sha256 should match the current remote file before patching."},
                        "whitespace_normalized": {"type": "boolean", "default": False, "description": "Opt-in relaxed matching that ignores whitespace differences. Prefer dry_run first."},
                        "dry_run": {"type": "boolean", "default": False, "description": "Preview matches and planned hash without writing the remote file."},
                        "timeout_sec": {"type": "integer", "default": 120},
                    },
                    "required": ["path", "target_text", "replacement_text"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(ssh_files.ssh_file_patch),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="ssh_file_replace_between",
                description=(
                    "Replace a bounded REMOTE file block over SSH using exact start_text and end_text. "
                    "Use this for multiline blocks such as `<style>...</style>` instead of shell regex. "
                    "This tool operates on the REMOTE host filesystem ONLY."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Preferred SSH target in `user@host` or `host` form."},
                        "host": {"type": "string", "description": "Target hostname or IP."},
                        "user": {"type": "string", "description": "SSH username."},
                        "username": {"type": "string", "description": "Alias for `user`."},
                        "port": {"type": "integer", "default": 22},
                        "identity_file": {"type": "string", "description": "Path to SSH private key."},
                        "password": {"type": "string", "description": "Optional SSH password. Uses `sshpass` when provided."},
                        "path": {"type": "string", "description": "Remote file path."},
                        "start_text": {"type": "string", "description": "Exact start bound."},
                        "end_text": {"type": "string", "description": "Exact end bound."},
                        "replacement_text": {"type": "string", "description": "Replacement text. Use an empty string to delete the bounded block."},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "include_bounds": {"type": "boolean", "default": True},
                        "expected_occurrences": {"type": "integer", "default": 1},
                        "backup": {"type": "boolean", "default": True},
                        "expected_sha256": {"type": "string", "description": "Optional hash of the current remote file."},
                        "source_artifact_id": {"type": "string", "description": "Artifact id whose sha256 should match the current remote file before replacement."},
                        "whitespace_normalized": {"type": "boolean", "default": False, "description": "Opt-in relaxed matching that ignores whitespace differences. Prefer dry_run first."},
                        "dry_run": {"type": "boolean", "default": False, "description": "Preview matches and planned hash without writing the remote file."},
                        "timeout_sec": {"type": "integer", "default": 120},
                    },
                    "required": ["path", "start_text", "end_text", "replacement_text"],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(ssh_files.ssh_file_replace_between),
                category="network",
                risk="high",
                allowed_modes={"chat", "loop", "planning"},
                profiles={network_profile},
            ),
            make_registration(
                name="shell_exec",
                description=(
                    "Execute a shell command after user approval, launch it in background, or poll a background job with job_id. "
                    "On Unix, commands run through /bin/sh by default. Use 'bash -c' explicitly if you need Bash-only features "
                    "(e.g., here-strings <<<, arrays, or source .venv/bin/activate)."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "job_id": {"type": "string"},
                        "background": {"type": "boolean"},
                        "timeout_sec": {"type": "integer"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                handler=inject_state_and_harness(shell.shell_exec),
                category="shell",
                risk="high",
                allowed_modes={"loop"},
                profiles={core_profile},
            ),
            make_registration(
                name="process_kill",
                description="Kill a tracked background process.",
                schema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
                handler=inject_state(shell.process_kill),
                category="shell",
                risk="high",
                allowed_modes={"loop"},
                profiles={support_profile},
            ),
            make_registration(
                name="http_get",
                description="Run an HTTP GET request.",
                schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "headers": {"type": "object"},
                        "timeout_sec": {"type": "integer"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                handler=http.http_get,
                category="http",
                risk="medium",
                allowed_modes={"loop"},
                profiles={network_profile, network_raw_profile},
            ),
            make_registration(
                name="http_post",
                description="Run an HTTP POST request.",
                schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "json_body": {"type": "object"},
                        "headers": {"type": "object"},
                        "timeout_sec": {"type": "integer"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                handler=http.http_post,
                category="http",
                risk="high",
                allowed_modes={"loop"},
                profiles={network_profile, network_raw_profile},
            ),
            make_registration(
                name="file_download",
                description="Download a file from URL.",
                schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "output_path": {"type": "string"},
                        "headers": {"type": "object"},
                        "timeout_sec": {"type": "integer"},
                    },
                    "required": ["url", "output_path"],
                    "additionalProperties": False,
                },
                handler=inject_state(http.file_download),
                category="http",
                risk="high",
                allowed_modes={"loop"},
                profiles={network_profile, network_raw_profile},
            ),
        ]
    )
