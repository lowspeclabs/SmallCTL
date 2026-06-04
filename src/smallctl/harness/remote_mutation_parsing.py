from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

_REMOTE_PATH_RE = re.compile(r"(?<![\w/])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:\.[A-Za-z0-9._-]+)?")
_REMOTE_DELETION_RE = re.compile(r"\b(?:rm|truncate)\b", re.IGNORECASE)
_BENIGN_REMOTE_MUTATION_PATHS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
_BENIGN_REMOTE_MUTATION_PATH_RE = re.compile(r"^/dev/fd/\d+$")
_REMOTE_SHELL_INTERPRETER_PATHS = frozenset(
    {"/bin/bash", "/bin/sh", "/usr/bin/bash", "/usr/bin/sh", "/usr/bin/env"}
)
_REMOTE_PATH_GLOB_CHARS = frozenset("*?[")
_REMOTE_CONTROL_TOKENS = frozenset({"&&", "||", "|", ";", "\n", "&"})
_REMOTE_REDIRECT_TOKENS = frozenset({">", ">>", "<", "<<", "<<<", "<>", ">|"})
_REMOTE_OUTPUT_REDIRECT_TOKENS = frozenset({">", ">>", "<>", ">|"})
_REMOTE_DELETION_COMMANDS = frozenset({"rm", "truncate"})
_REMOTE_PATH_MUTATOR_COMMANDS = frozenset({"sed", "perl", "tee", "cp", "mv", "install"})


def guess_remote_mutation_paths(command: str, *, deletion: bool = False) -> list[str]:
    if deletion:
        return guess_remote_deletion_paths(command)
    paths: list[str] = []
    collect_remote_redirection_targets(command, paths)
    collect_remote_mutator_operands(command, paths)
    if python_open_write_mutation(command):
        for match in _REMOTE_PATH_RE.finditer(str(command or "")):
            append_remote_mutation_path(paths, match.group(0))
    return paths[:12]


def collect_remote_redirection_targets(command: str, paths: list[str]) -> None:
    for tokens in remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            target_index = redirection_target_index(tokens, index, output_only=True)
            if target_index is None:
                index += 1
                continue
            if target_index < len(tokens):
                target = redirection_target_token(tokens[target_index])
                append_remote_mutation_path(paths, target)
            index = target_index + 1


def collect_remote_mutator_operands(command: str, paths: list[str]) -> None:
    for tokens in remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            token = shell_command_name(tokens[index])
            if token not in _REMOTE_PATH_MUTATOR_COMMANDS:
                index += 1
                continue
            index = collect_mutator_path_operands(tokens, index, paths)


def collect_mutator_path_operands(tokens: list[str], command_index: int, paths: list[str]) -> int:
    command = shell_command_name(tokens[command_index])
    index = command_index + 1
    operands: list[str] = []
    while index < len(tokens):
        token = tokens[index]
        if token in _REMOTE_CONTROL_TOKENS or token == "+":
            break
        target_index = redirection_target_index(tokens, index, output_only=False)
        if target_index is not None:
            index = target_index + 1
            continue
        if token == "--" or token.startswith("-") or token in {"{}", r"\;"}:
            index += 1
            continue
        normalized = normalize_remote_mutation_operand(token)
        if normalized:
            operands.append(normalized)
        index += 1

    if command in {"cp", "install"} and operands:
        append_remote_mutation_path(paths, operands[-1])
    else:
        for operand in operands:
            append_remote_mutation_path(paths, operand)
    return max(index, command_index + 1)


def python_open_write_mutation(command: str) -> bool:
    return bool(
        re.search(
            r"\bpython3?\s+-c\b.*\bopen\s*\([^)]*['\"]w",
            str(command or ""),
            re.IGNORECASE | re.DOTALL,
        )
    )


def guess_remote_deletion_paths(command: str) -> list[str]:
    paths: list[str] = []
    for tokens in remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            token = shell_command_name(tokens[index])
            if token in _REMOTE_DELETION_COMMANDS:
                index = collect_deletion_operands(tokens, index + 1, paths)
                continue
            index += 1
    return paths[:12]


def guess_remote_deletion_directory_empty_checks(command: str) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    for tokens in remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            token = shell_command_name(tokens[index])
            if token in _REMOTE_DELETION_COMMANDS:
                index = collect_deletion_glob_checks(tokens, index + 1, checks, seen)
                continue
            index += 1
    return checks[:12]


def remote_shell_command_lines(command: str) -> list[list[str]]:
    lines: list[list[str]] = []
    for raw_line in str(command or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            tokens = shlex.split(line, comments=True, posix=True)
        if tokens:
            lines.append(tokens)
    return lines


def collect_deletion_operands(tokens: list[str], start_index: int, paths: list[str]) -> int:
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if token in _REMOTE_CONTROL_TOKENS or token == "+":
            return index + 1
        if token_starts_redirection(tokens, index):
            index = skip_redirection(tokens, index)
            continue
        if token == "--" or token.startswith("-") or token in {"{}", r"\;"}:
            index += 1
            continue
        path = normalize_remote_deletion_operand(token)
        if path and path not in paths:
            paths.append(path)
        index += 1
    return index


def collect_deletion_glob_checks(
    tokens: list[str],
    start_index: int,
    checks: list[dict[str, str]],
    seen: set[str],
) -> int:
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if token in _REMOTE_CONTROL_TOKENS or token == "+":
            return index + 1
        if token_starts_redirection(tokens, index):
            index = skip_redirection(tokens, index)
            continue
        if token == "--" or token.startswith("-") or token in {"{}", r"\;"}:
            index += 1
            continue
        check = remote_deletion_glob_empty_check(token)
        path = check.get("path", "") if check else ""
        if check and path not in seen:
            seen.add(path)
            checks.append(check)
        index += 1
    return index


def shell_command_name(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    return Path(value).name.lower()


def token_starts_redirection(tokens: list[str], index: int) -> bool:
    token = str(tokens[index] or "").strip()
    if token in _REMOTE_REDIRECT_TOKENS:
        return True
    if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _REMOTE_REDIRECT_TOKENS:
        return True
    return bool(re.match(r"^\d*(?:>>?|<<?|<>)", token))


def skip_redirection(tokens: list[str], index: int) -> int:
    token = str(tokens[index] or "").strip()
    if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _REMOTE_REDIRECT_TOKENS:
        index += 2
    elif token in _REMOTE_REDIRECT_TOKENS:
        index += 1
    else:
        index += 1
        return index
    if index < len(tokens) and tokens[index] not in _REMOTE_CONTROL_TOKENS:
        index += 1
    return index


def redirection_target_index(tokens: list[str], index: int, *, output_only: bool) -> int | None:
    token = str(tokens[index] or "").strip()
    redirect_token = ""
    target_index = index + 1
    if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _REMOTE_REDIRECT_TOKENS:
        redirect_token = str(tokens[index + 1])
        target_index = index + 2
    elif token in _REMOTE_REDIRECT_TOKENS:
        redirect_token = token
    else:
        compact = re.match(r"^\d*(>>?|<>|>\|)(/\S+)$", token)
        if not compact:
            return None
        redirect_token = compact.group(1)
        if output_only and redirect_token not in _REMOTE_OUTPUT_REDIRECT_TOKENS:
            return None
        return index

    if output_only and redirect_token not in _REMOTE_OUTPUT_REDIRECT_TOKENS:
        return None
    if redirect_token in {"<<", "<<<"}:
        return None
    if target_index < len(tokens) and tokens[target_index] not in _REMOTE_CONTROL_TOKENS:
        return target_index
    return None


def redirection_target_token(token: str) -> str:
    value = str(token or "").strip()
    compact = re.match(r"^\d*(?:>>?|<>|>\|)(/\S+)$", value)
    if compact:
        return compact.group(1)
    return value


def normalize_remote_deletion_operand(token: str) -> str:
    candidate = str(token or "").strip().strip("`'")
    candidate = candidate.rstrip(";,")
    if not candidate.startswith("/"):
        return ""
    if any(char in candidate for char in _REMOTE_PATH_GLOB_CHARS):
        return ""
    if candidate.endswith("/"):
        candidate = candidate.rstrip("/")
    if not _REMOTE_PATH_RE.fullmatch(candidate):
        return ""
    if remote_path_should_be_ignored(candidate):
        return ""
    return candidate


def normalize_remote_mutation_operand(token: str) -> str:
    candidate = str(token or "").strip().strip("`'")
    candidate = candidate.rstrip(";,")
    if not candidate.startswith("/"):
        return ""
    if any(char in candidate for char in _REMOTE_PATH_GLOB_CHARS):
        return ""
    if candidate.endswith("/"):
        candidate = candidate.rstrip("/")
    if not _REMOTE_PATH_RE.fullmatch(candidate):
        return ""
    if remote_path_should_be_ignored(candidate) or remote_path_is_known_directory(candidate):
        return ""
    return candidate


def append_remote_mutation_path(paths: list[str], path: str) -> None:
    normalized = normalize_remote_mutation_operand(path)
    if normalized and normalized not in paths:
        paths.append(normalized)


def remote_path_is_known_directory(path: str) -> bool:
    return str(path or "").strip().rstrip("/") in {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
    }


def remote_deletion_glob_empty_check(token: str) -> dict[str, str] | None:
    candidate = str(token or "").strip().strip("`'\"").rstrip(";,")
    if not candidate.startswith("/") or not any(char in candidate for char in _REMOTE_PATH_GLOB_CHARS):
        return None
    if not candidate.endswith("/*"):
        return None
    parent = candidate[:-2].rstrip("/")
    if not parent or not _REMOTE_PATH_RE.fullmatch(parent):
        return None
    if remote_path_should_be_ignored(parent):
        return None
    return {"path": parent, "glob": candidate}


def remote_path_should_be_ignored(path: str) -> bool:
    normalized = str(path or "").strip()
    if (
        normalized in _BENIGN_REMOTE_MUTATION_PATHS
        or normalized in _REMOTE_SHELL_INTERPRETER_PATHS
        or bool(_BENIGN_REMOTE_MUTATION_PATH_RE.match(normalized))
    ):
        return True
    # Ignore ephemeral temp files that are not task deliverables
    return any(
        normalized.startswith(prefix)
        for prefix in ("/tmp/", "/var/tmp/", "/dev/shm/")
    )
