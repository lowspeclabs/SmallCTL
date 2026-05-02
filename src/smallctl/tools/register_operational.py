from __future__ import annotations

from typing import Any, Awaitable, Callable

from . import http, network, shell, ssh_files


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
                    "Execute a command on a remote host via SSH with live streaming support. "
                    "Prefer `target='user@host'` when a username is known, for example "
                    "`target='root@192.168.1.63'`, rather than splitting identity across separate fields."
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
                        "timeout_sec": {"type": "integer", "default": 60},
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
                name="ssh_file_read",
                description=(
                    "Read a remote file over SSH with structured output. Prefer this over `ssh_exec` with cat/head/sed "
                    "for remote file inspection."
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
                        "timeout_sec": {"type": "integer", "default": 60},
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
                    "Write or overwrite a remote file over SSH with atomic replace and readback hash verification. "
                    "Prefer this over `ssh_exec` with here-docs, tee, or shell redirection for remote file writes."
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
                        "timeout_sec": {"type": "integer", "default": 60},
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
                    "Patch a remote file over SSH by replacing exact target text with replacement text. "
                    "Prefer this over `ssh_exec` with sed/perl for precise remote edits."
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
                        "replacement_text": {"type": "string", "description": "Replacement text."},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "expected_occurrences": {"type": "integer", "default": 1},
                        "backup": {"type": "boolean", "default": True},
                        "expected_sha256": {"type": "string", "description": "Optional hash of the current remote file."},
                        "source_artifact_id": {"type": "string", "description": "Artifact id whose sha256 should match the current remote file before patching."},
                        "whitespace_normalized": {"type": "boolean", "default": False, "description": "Opt-in relaxed matching that ignores whitespace differences. Prefer dry_run first."},
                        "dry_run": {"type": "boolean", "default": False, "description": "Preview matches and planned hash without writing the remote file."},
                        "timeout_sec": {"type": "integer", "default": 60},
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
                    "Replace a bounded remote file block over SSH using exact start_text and end_text. "
                    "Use this for multiline blocks such as `<style>...</style>` instead of shell regex."
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
                        "replacement_text": {"type": "string", "description": "Replacement text."},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "include_bounds": {"type": "boolean", "default": True},
                        "expected_occurrences": {"type": "integer", "default": 1},
                        "backup": {"type": "boolean", "default": True},
                        "expected_sha256": {"type": "string", "description": "Optional hash of the current remote file."},
                        "source_artifact_id": {"type": "string", "description": "Artifact id whose sha256 should match the current remote file before replacement."},
                        "whitespace_normalized": {"type": "boolean", "default": False, "description": "Opt-in relaxed matching that ignores whitespace differences. Prefer dry_run first."},
                        "dry_run": {"type": "boolean", "default": False, "description": "Preview matches and planned hash without writing the remote file."},
                        "timeout_sec": {"type": "integer", "default": 60},
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
                description="Execute a shell command after user approval, launch it in background, or poll a background job with job_id.",
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
                handler=http.file_download,
                category="http",
                risk="high",
                allowed_modes={"loop"},
                profiles={network_profile, network_raw_profile},
            ),
        ]
    )
