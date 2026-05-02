from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .models.tool_result import ToolEnvelope

_SHELL_WRAPPER_TOKENS = {
    "bash",
    "sh",
    "zsh",
    "dash",
    "ksh",
    "pwsh",
    "powershell",
    "cmd",
    "cmd.exe",
}
_SHELL_WRAPPER_COMMAND_FLAGS = {"-c", "-lc", "/c", "-Command", "-command"}
_BENIGN_SHELL_REDIRECTION_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))(?:\d?>|\d?>>|>>|>)\s*/dev/(?:null|stdout|stderr|fd/\d+)\b"
    r"|"
    r"(?:(?<=^)|(?<=\s))\d?>&(?:1|2)\b"
)
_READ_ONLY_ROOT_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "grep",
    "rg",
    "cat",
    "head",
    "tail",
    "awk",
    "wc",
    "stat",
    "which",
    "pytest",
    "apt-cache",
    "journalctl",
    "uname",
    "id",
    "whoami",
    "hostname",
    "df",
    "free",
    "uptime",
    "env",
    "printenv",
    "date",
    "lsblk",
    "ip",
    "ss",
    "netstat",
    "lsof",
    "ps",
    "top",
    "file",
}


def file_read_cache_key(cwd: str, payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    try:
        resolved = str(candidate.resolve())
    except Exception:
        resolved = str(candidate)
    start_line = payload.get("requested_start_line", payload.get("start_line"))
    end_line = payload.get("requested_end_line", payload.get("end_line"))
    max_bytes = payload.get("max_bytes", 100_000)
    return f"{resolved}|{start_line}|{end_line}|{max_bytes}"


def shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def looks_like_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    key, _value = token.split("=", 1)
    return key.isidentifier()


def shell_unwrap_command(tokens: list[str]) -> str:
    if not tokens:
        return ""
    first = tokens[0].lower()
    if first == "sudo":
        inner_tokens = tokens[1:]
        while inner_tokens and inner_tokens[0].startswith("-"):
            inner_tokens = inner_tokens[1:]
        return " ".join(inner_tokens)
    if len(tokens) < 2:
        return tokens[0]
    if tokens[1] in {"-c", "-lc", "/c", "-Command", "-command"}:
        return " ".join(tokens[2:])
    return " ".join(tokens[1:])


def shell_command_root(command: str) -> str | None:
    tokens = shell_tokens(command)
    if not tokens:
        return None
    first = tokens[0].lower()
    wrapper_tokens = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell", "cmd", "cmd.exe", "sudo"}
    if first == "env":
        inner_tokens = leading_command_tokens(command)
        if not inner_tokens or inner_tokens == tokens:
            return first
        return shell_command_root(" ".join(inner_tokens))
    if first in wrapper_tokens:
        inner = shell_unwrap_command(tokens)
        if inner == command:
            return first
        return shell_command_root(inner)
    for token in tokens:
        if token.startswith("-"):
            continue
        if "=" in token and token.split("=", 1)[0].isidentifier():
            continue
        return token.lower()
    return first


def shell_attempt_family_key(command: str) -> str | None:
    root = shell_command_root(command)
    if not root:
        return None
    return f"shell_exec:{root}"


def shell_attempt_is_diagnostic(command: str) -> bool:
    tokens = shell_tokens(command)
    if not tokens:
        return False
    first = tokens[0].lower()
    wrapper_tokens = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell", "cmd", "cmd.exe", "sudo"}
    if first in wrapper_tokens:
        inner = shell_unwrap_command(tokens)
        return shell_attempt_is_diagnostic(inner)
    lowered = [token.lower() for token in tokens]
    return any(token in {"-h", "--help", "/?"} for token in lowered) or "help" in lowered[1:]


def leading_command_tokens(command: str, *, max_depth: int = 4) -> list[str]:
    current = command
    for _ in range(max_depth):
        tokens = shell_tokens(current)
        if not tokens:
            return []
        if len(tokens) >= 3 and tokens[0].lower() in _SHELL_WRAPPER_TOKENS and tokens[1] in _SHELL_WRAPPER_COMMAND_FLAGS:
            current = shell_unwrap_command(tokens)
            continue
        index = 0
        if tokens[0].lower() == "env":
            index = 1
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
        while index < len(tokens) and looks_like_env_assignment(tokens[index]):
            index += 1
        if index:
            current = " ".join(tokens[index:])
            continue
        return tokens
    return shell_tokens(current)


def split_shell_segments(command: str) -> list[str]:
    command = strip_benign_shell_redirections(command)
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    escape = False
    index = 0

    while index < len(command):
        char = command[index]

        if escape:
            current.append(char)
            escape = False
            index += 1
            continue

        if char == "\\" and quote != "'":
            current.append(char)
            escape = True
            index += 1
            continue

        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue

        if char == "`" or command.startswith("$(", index):
            return []

        if char in {"<", ">", "(", ")"}:
            return []

        if command.startswith("&&", index) or command.startswith("||", index):
            segment = "".join(current).strip()
            if not segment:
                return []
            segments.append(segment)
            current = []
            index += 2
            continue

        if char in {";", "|"}:
            segment = "".join(current).strip()
            if not segment:
                return []
            segments.append(segment)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    if quote or escape:
        return []

    segment = "".join(current).strip()
    if not segment:
        return []
    segments.append(segment)
    return segments


def strip_benign_shell_redirections(command: str) -> str:
    cleaned = _BENIGN_SHELL_REDIRECTION_RE.sub(" ", str(command or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


def is_read_only_shell_segment(segment: str) -> bool:
    command = str(segment or "").strip()
    if not command:
        return False

    tokens = shell_tokens(command)
    if not tokens:
        return False

    first = tokens[0].lower()
    if first in _SHELL_WRAPPER_TOKENS or first == "sudo":
        unwrapped = shell_unwrap_command(tokens)
        if unwrapped and unwrapped != command:
            return is_read_only_shell_evidence_action(unwrapped)

    tokens = leading_command_tokens(command)
    if not tokens:
        return False

    root = tokens[0].lower()
    if root in _READ_ONLY_ROOT_COMMANDS:
        return True
    if root == "apt":
        return len(tokens) >= 2 and tokens[1] in {"list", "show", "search", "policy"}
    if root == "sed":
        return len(tokens) >= 2 and tokens[1] == "-n"
    if root == "command":
        return len(tokens) >= 2 and tokens[1] == "-v"
    if root == "git":
        return len(tokens) >= 2 and tokens[1] in {"status", "diff", "show", "log"}
    if root in {"python", "python3"}:
        return len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest"
    if root == "dpkg":
        return len(tokens) >= 2 and tokens[1] in {"-l", "--list"}
    if root == "rpm":
        return len(tokens) >= 2 and tokens[1] in {"-q", "-qa", "--query", "--queryformat", "--querytags"}
    if root == "systemctl":
        return len(tokens) >= 2 and tokens[1] in {"status", "show", "is-active", "is-enabled", "list-units", "list-unit-files"}
    if root in {"docker", "podman", "docker-compose"}:
        if len(tokens) < 2:
            return False
        return tokens[1] in {
            "--version",
            "version",
            "ps",
            "images",
            "inspect",
            "info",
            "logs",
            "stats",
            "top",
            "port",
            "network",
            "volume",
            "container",
            "image",
            "system",
        }
    return False


def is_read_only_shell_evidence_action(action: str) -> bool:
    command = str(action or "").strip().lower()
    if not command:
        return False
    segments = split_shell_segments(str(action or "").strip())
    return bool(segments) and all(is_read_only_shell_segment(segment) for segment in segments)


def _extract_command_from_result(result: ToolEnvelope, *, tool_name: str) -> str:
    arguments = result.metadata.get("arguments") if isinstance(result.metadata, dict) else None
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or "").strip()
    if not command and isinstance(result.metadata, dict):
        command = str(result.metadata.get("command") or "").strip()
    if not command and tool_name == "ssh_exec" and isinstance(arguments, dict):
        command = str(arguments.get("host") or "").strip()
    if not command and tool_name == "ssh_exec" and isinstance(result.metadata, dict):
        command = str(result.metadata.get("host") or "").strip()
    return command


def mark_artifact_superseded(
    *,
    state: Any,
    artifact_id: str,
    superseded_by: str,
    family_key: str,
    reason: str,
) -> None:
    artifact = state.artifacts.get(artifact_id)
    if artifact is None:
        return
    artifact.metadata["attempt_family"] = family_key
    artifact.metadata["superseded_by"] = superseded_by
    artifact.metadata["attempt_status"] = "superseded"
    artifact.metadata["superseded_reason"] = reason


def consolidate_shell_attempt_family(
    *,
    state: Any,
    artifact_id: str,
    result: ToolEnvelope,
    tool_name: str = "shell_exec",
) -> None:
    command = _extract_command_from_result(result, tool_name=tool_name)
    if not command:
        return

    family_key = shell_attempt_family_key(command)
    if not family_key:
        return

    family_state = state.scratchpad.setdefault("_shell_attempt_families", {})
    if not isinstance(family_state, dict):
        family_state = {}
        state.scratchpad["_shell_attempt_families"] = family_state

    record = family_state.get(family_key)
    if not isinstance(record, dict):
        record = {
            "tool_name": tool_name,
            "members": [],
            "canonical_artifact_id": None,
            "resolved": False,
        }
        family_state[family_key] = record

    members = record.get("members")
    if not isinstance(members, list):
        members = []
        record["members"] = members

    is_diagnostic = shell_attempt_is_diagnostic(command)
    root = shell_command_root(command)
    canonical_artifact_id = record.get("canonical_artifact_id")
    canonical_artifact_id = canonical_artifact_id if isinstance(canonical_artifact_id, str) and canonical_artifact_id else None

    artifact = state.artifacts.get(artifact_id)
    if artifact is None:
        return

    artifact.metadata["attempt_family"] = family_key
    if root:
        artifact.metadata["attempt_family_root"] = root

    if canonical_artifact_id:
        artifact.metadata["attempt_status"] = "redundant"
        artifact.metadata["superseded_by"] = canonical_artifact_id
        artifact.metadata["canonical_attempt_artifact_id"] = canonical_artifact_id
        members.append(artifact_id)
        return

    previous_members = [member_id for member_id in members if member_id != artifact_id]
    members.append(artifact_id)

    if result.success and not is_diagnostic:
        record["resolved"] = True
        record["canonical_artifact_id"] = artifact_id
        artifact.metadata["attempt_status"] = "canonical"
        artifact.metadata["canonical_attempt_artifact_id"] = artifact_id
        for member_id in previous_members:
            mark_artifact_superseded(
                state=state,
                artifact_id=member_id,
                superseded_by=artifact_id,
                family_key=family_key,
                reason="resolved_by_success",
            )
        return

    artifact.metadata["attempt_status"] = "diagnostic" if is_diagnostic and result.success else "failed"
    for member_id in previous_members:
        mark_artifact_superseded(
            state=state,
            artifact_id=member_id,
            superseded_by=artifact_id,
            family_key=family_key,
            reason="replaced_by_new_attempt",
        )
