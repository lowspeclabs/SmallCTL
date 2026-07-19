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

# Exact read-only subcommand allowlist for container CLIs. Anything not listed
# here (rm, rmi, prune, kill, stop, start, restart, pause, unpause, create,
# run, exec, build, pull, push, tag, commit, cp, rename, update, save, load,
# import, export, wait, attach, ...) fails closed toward NOT read-only.
_CONTAINER_READ_ONLY_ACTIONS = {
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
}
_CONTAINER_READ_ONLY_GROUP_ACTIONS = {
    "container": {"ls", "inspect", "logs", "stats", "top", "port"},
    "image": {"ls", "inspect"},
    "network": {"ls", "inspect"},
    "volume": {"ls", "inspect"},
    "system": {"info", "version"},
}

_SED_MUTATING_SCRIPT_COMMANDS = frozenset("erwRW")
_SED_TEXT_COMMANDS = frozenset("aic")
_SED_LABEL_COMMANDS = frozenset(":btT")
_SED_SAFE_SHORT_OPTIONS = frozenset("nrEsuz")
_SED_SAFE_LONG_OPTIONS = {
    "--quiet",
    "--silent",
    "--regexp-extended",
    "--extended-regexp",
    "--posix",
    "--debug",
    "--follow-symlinks",
    "--separate",
    "--sandbox",
    "--unbuffered",
    "--null-data",
    "--version",
    "--help",
}

_SSH_KEYGEN_SHELL_METACHAR_TOKENS = {
    "&&",
    "||",
    ";",
    "|",
    "&",
    "$(",
    "`",
    ">",
    ">>",
    "<",
    "<<",
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


def ssh_file_read_cache_key(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    host = str(payload.get("host") or "").strip().lower()
    start_line = payload.get("requested_start_line", payload.get("start_line"))
    end_line = payload.get("requested_end_line", payload.get("end_line"))
    max_bytes = payload.get("max_bytes", 100_000)
    return f"ssh://{host}{path}|{start_line}|{end_line}|{max_bytes}"


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


def strip_benign_shell_redirections(command: str, *, preserve_newlines: bool = False) -> str:
    cleaned = _BENIGN_SHELL_REDIRECTION_RE.sub(" ", str(command or ""))
    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
        return "\n".join(line for line in lines if line).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def _has_file_writing_redirection(command: str) -> bool:
    """Return True if the command redirects output to a real file path."""
    cleaned = strip_benign_shell_redirections(command)
    if re.search(r"[\d\s]?>", cleaned):
        return True
    return False


def _is_read_only_container_command(tokens: list[str]) -> bool:
    """Classify docker/podman/docker-compose invocations via an exact subcommand allowlist."""
    if len(tokens) < 2:
        return False
    action = tokens[1].lower()
    if action in _CONTAINER_READ_ONLY_ACTIONS:
        return True
    group = _CONTAINER_READ_ONLY_GROUP_ACTIONS.get(action)
    if group is None or len(tokens) < 3:
        return False
    return tokens[2].lower() in group


def _skip_sed_delimited(script: str, index: int, delimiter: str) -> int:
    """Skip a delimited sed section (regex, replacement), honoring backslash escapes."""
    length = len(script)
    while index < length:
        char = script[index]
        if char == "\\":
            index += 2
            continue
        if char == delimiter:
            return index + 1
        index += 1
    return length


def _sed_script_is_read_only(script: str) -> bool:
    """Return False when a sed script can execute commands or read/write files.

    Rejects the GNU `e` execute command, the `w`/`W` write-file commands, the
    `r`/`R` read-file commands, and the `s///w` and `s///e` substitution flags.
    """
    index = 0
    length = len(script)
    while index < length:
        char = script[index]
        if char in " \t\n;{}":
            index += 1
            continue
        if char.isdigit() or char in "$,~+!":
            index += 1
            continue
        if char == "#":
            newline = script.find("\n", index)
            index = length if newline < 0 else newline + 1
            continue
        if char == "/":
            index = _skip_sed_delimited(script, index + 1, "/")
            continue
        if char == "\\":
            # Address form \%re% uses the next character as the delimiter.
            if index + 1 >= length:
                return False
            index = _skip_sed_delimited(script, index + 2, script[index + 1])
            continue
        if char in _SED_MUTATING_SCRIPT_COMMANDS:
            return False
        if char in {"s", "y"}:
            delimiter_index = index + 1
            if delimiter_index >= length or script[delimiter_index] in " \t\n":
                return False
            delimiter = script[delimiter_index]
            index = _skip_sed_delimited(script, delimiter_index + 1, delimiter)
            if index >= length:
                return False
            index = _skip_sed_delimited(script, index + 1, delimiter)
            if char == "s":
                flags_start = index
                while index < length and script[index].isalnum():
                    index += 1
                flags = script[flags_start:index]
                if "w" in flags or "e" in flags:
                    return False
            continue
        if char in _SED_LABEL_COMMANDS:
            index += 1
            while index < length and script[index] not in ";\n":
                index += 1
            continue
        if char in _SED_TEXT_COMMANDS:
            # GNU one-line a/i/c text runs to an unescaped newline.
            index += 1
            while index < length and script[index] in " \t":
                index += 1
            while index < length:
                if script[index] == "\\" and index + 1 < length:
                    index += 2
                    continue
                if script[index] == "\n":
                    break
                index += 1
            continue
        index += 1
    return True


def _sed_scripts_from_tokens(tokens: list[str]) -> list[str] | None:
    """Extract sed script arguments, or None when invocation safety is undecidable.

    Returns None for in-place editing (-i/--in-place), scripts loaded from a
    file (-f/--file), and unrecognized options (fail closed).
    """
    scripts: list[str] = []
    saw_script = False
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            if not saw_script and index < len(tokens):
                scripts.append(tokens[index])
            break
        if token.startswith("--"):
            name, _, attached = token.partition("=")
            if name == "--in-place":
                return None
            if name == "--expression":
                if attached:
                    scripts.append(attached)
                else:
                    index += 1
                    if index >= len(tokens):
                        return None
                    scripts.append(tokens[index])
                saw_script = True
            elif name == "--file":
                return None
            elif name in _SED_SAFE_LONG_OPTIONS:
                pass
            else:
                return None
            index += 1
            continue
        if token.startswith("-") and token != "-":
            cluster = token[1:]
            position = 0
            while position < len(cluster):
                letter = cluster[position]
                if letter == "i":
                    return None
                if letter == "e":
                    attached_script = cluster[position + 1 :]
                    if attached_script:
                        scripts.append(attached_script)
                    else:
                        index += 1
                        if index >= len(tokens):
                            return None
                        scripts.append(tokens[index])
                    saw_script = True
                    break
                if letter == "f":
                    return None
                if letter == "l":
                    # -l consumes a line-length argument when not attached.
                    if position + 1 >= len(cluster):
                        index += 1
                        if index >= len(tokens):
                            return None
                    break
                if letter not in _SED_SAFE_SHORT_OPTIONS:
                    return None
                position += 1
            index += 1
            continue
        if not saw_script:
            scripts.append(token)
            saw_script = True
        index += 1
    return scripts


def _is_read_only_sed_invocation(command: str, tokens: list[str]) -> bool:
    scripts = _sed_scripts_from_tokens(tokens)
    if scripts is None:
        return False
    if not all(_sed_script_is_read_only(script) for script in scripts):
        return False
    return not _has_file_writing_redirection(command)


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
        if root == "find":
            mutating_actions = {
                "-delete", "-exec", "-execdir", "-ok", "-okdir",
            }
            if any(tok.lower() in mutating_actions for tok in tokens):
                return False
            return not _has_file_writing_redirection(command)
        return True
    if root == "awk":
        if "system(" in command or "system (" in command:
            return False
        return not _has_file_writing_redirection(command)
    if root == "apt":
        return len(tokens) >= 2 and tokens[1] in {"list", "show", "search", "policy"}
    if root == "sed":
        return _is_read_only_sed_invocation(command, tokens)
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
        return _is_read_only_container_command(tokens)
    return False


def is_read_only_shell_evidence_action(action: str) -> bool:
    command = str(action or "").strip().lower()
    if not command:
        return False
    segments = split_shell_segments(str(action or "").strip())
    return bool(segments) and all(is_read_only_shell_segment(segment) for segment in segments)


def looks_like_ssh_keygen_known_hosts_removal(command: str) -> dict[str, str] | None:
    """Return parsed host/file if command is a safe local ssh-keygen -R removal."""
    text = str(command or "").strip()
    if not text:
        return None
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if not tokens:
        return None
    if tokens[0] != "ssh-keygen":
        return None

    for token in tokens[1:]:
        if token in _SSH_KEYGEN_SHELL_METACHAR_TOKENS:
            return None
        if "`" in token or "$(" in token:
            return None
        if re.match(r"^\d*[<>]", token):
            return None

    host: str | None = None
    known_hosts_file: str | None = None
    seen_options: set[str] = set()
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-R" or token.startswith("-R"):
            if "R" in seen_options:
                return None
            seen_options.add("R")
            if token == "-R":
                index += 1
                if index >= len(tokens):
                    return None
                host = tokens[index]
            else:
                host = token[2:]
            if not host or host.startswith("-"):
                return None
            index += 1
            continue
        if token == "-f" or token.startswith("-f"):
            if "f" in seen_options:
                return None
            seen_options.add("f")
            if token == "-f":
                index += 1
                if index >= len(tokens):
                    return None
                known_hosts_file = tokens[index]
            else:
                known_hosts_file = token[2:]
            if not known_hosts_file or known_hosts_file.startswith("-"):
                return None
            index += 1
            continue
        if token.startswith("-"):
            return None
        return None

    if host is None:
        return None
    if known_hosts_file is None:
        known_hosts_file = "~/.ssh/known_hosts"
    return {"host": host, "known_hosts_file": known_hosts_file}


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
