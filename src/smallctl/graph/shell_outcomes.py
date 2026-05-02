from __future__ import annotations

import re
from typing import Any

from .tool_call_parser import _tool_call_fingerprint
from .state import ToolExecutionRecord


def _clear_shell_human_retry_state(harness: Any) -> None:
    for key in (
        "_shell_human_blocked_command_fingerprint",
        "_shell_human_blocked_command",
        "_shell_human_blocked_reason",
        "_shell_human_blocked_question",
        "_shell_human_blocked_tool_call_id",
    ):
        harness.state.scratchpad.pop(key, None)


def _remember_shell_human_retry_state(harness: Any, record: ToolExecutionRecord) -> None:
    command = str(record.args.get("command", "") or record.result.metadata.get("command", "") or "").strip()
    if not command:
        return

    harness.state.scratchpad["_shell_human_blocked_command_fingerprint"] = _tool_call_fingerprint(
        "shell_exec",
        {"command": command},
    )
    harness.state.scratchpad["_shell_human_blocked_command"] = command
    harness.state.scratchpad["_shell_human_blocked_reason"] = str(record.result.metadata.get("reason", "") or "").strip()
    harness.state.scratchpad["_shell_human_blocked_question"] = str(record.result.metadata.get("question", "") or "").strip()
    harness.state.scratchpad["_shell_human_blocked_tool_call_id"] = str(record.tool_call_id or "")


def _shell_human_retry_hint(harness: Any, pending: Any) -> str | None:
    if getattr(pending, "tool_name", "") != "shell_exec":
        return None
    blocked_fingerprint = harness.state.scratchpad.get("_shell_human_blocked_command_fingerprint")
    if not isinstance(blocked_fingerprint, str) or not blocked_fingerprint:
        return None

    current_fingerprint = _tool_call_fingerprint("shell_exec", dict(getattr(pending, "args", {}) or {}))
    if current_fingerprint != blocked_fingerprint:
        return None

    command = str(harness.state.scratchpad.get("_shell_human_blocked_command", "") or "").strip()
    if not command:
        return None
    reason = str(harness.state.scratchpad.get("_shell_human_blocked_reason", "") or "").strip()
    question = str(harness.state.scratchpad.get("_shell_human_blocked_question", "") or "").strip()

    if reason == "unsupported_shell_syntax":
        return (
            f"You already got a shell-syntax prompt for `{command}`. "
            "Do not retry the same command; rewrite it using POSIX syntax or wrap it in `bash -lc`."
        )
    if reason in {"password_prompt_detected", "password_prompt_timeout"}:
        return (
            f"You already hit a password prompt for `{command}`. "
            "Do not retry the same command; ask the user for credentials or switch to a passwordless path."
        )
    if question:
        return (
            f"You already hit a human-input prompt for `{command}`: {question}. "
            "Do not retry the same command; ask the user or change strategy."
        )
    return (
        f"You already hit a human-input prompt for `{command}`. "
        "Do not retry the same command; ask the user or change strategy."
    )


def _shell_ssh_retry_hint(harness: Any, pending: Any) -> str | None:
    if getattr(pending, "tool_name", "") != "shell_exec":
        return None

    command = str(getattr(pending, "args", {}).get("command", "") or "").strip()
    if not command:
        return None
    if not re.search(r"\b(?:ssh|scp|sftp)\b", command):
        return None

    nudge_key = f"ssh_exec::{command}"
    if harness.state.scratchpad.get("_shell_ssh_routing_nudged") == nudge_key:
        return None
    harness.state.scratchpad["_shell_ssh_routing_nudged"] = nudge_key

    return (
        f"You are trying to run an SSH command through `shell_exec`: `{command}`. "
        "Use `ssh_exec` for remote SSH work and reserve `shell_exec` for local shell commands."
    )


def _shell_workspace_relative_retry_hint(harness: Any, pending: Any) -> str | None:
    if getattr(pending, "tool_name", "") != "shell_exec":
        return None

    command = str(getattr(pending, "args", {}).get("command", "") or "").strip()
    if not command:
        return None

    match = re.search(r"(?<![\w/])(/temp(?:/[^\s\"'`]+)*)", command)
    if match is None:
        return None

    nudge_key = f"shell_exec::{command}::workspace_relative"
    if harness.state.scratchpad.get("_shell_workspace_relative_retry_nudged") == nudge_key:
        return None
    harness.state.scratchpad["_shell_workspace_relative_retry_nudged"] = nudge_key

    suspicious_path = match.group(1)
    trimmed = suspicious_path.lstrip("/")
    return (
        f"You used the root-level `{suspicious_path}` path in `shell_exec`: `{command}`. "
        f"Use the workspace copy at `{('./' + trimmed)}` instead of retrying the same absolute path."
    )
