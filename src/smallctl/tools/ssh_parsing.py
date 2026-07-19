from __future__ import annotations

import re
import shlex
from typing import Any

_IGNORABLE_SSH_FLAGS = {
    "-4",
    "-6",
    "-A",
    "-a",
    "-C",
    "-f",
    "-G",
    "-g",
    "-K",
    "-k",
    "-M",
    "-N",
    "-n",
    "-q",
    "-s",
    "-T",
    "-t",
    "-tt",
    "-v",
    "-vv",
    "-vvv",
    "-X",
    "-x",
    "-Y",
    "-y",
}
_SAFE_SSH_OPTION_KEYS = {
    "BatchMode",
    "ConnectTimeout",
    "IdentityFile",
    "NumberOfPasswordPrompts",
    "PasswordAuthentication",
    "Port",
    "PreferredAuthentications",
    "PubkeyAuthentication",
    "StrictHostKeyChecking",
    "User",
}
_LOCAL_SHELL_CONTROL_TOKENS = {"|", "||", "&&", ";", ";&", ";;&"}
_ROOT_SUDO_PREFIX_RE = re.compile(
    r"^\s*sudo(?:\s+-(?:n|E|H|S))*\s+(?:(?:-i|-s)\s+)?(?:--\s+)?(?P<command>.+?)\s*$",
    re.DOTALL,
)
_ROOT_SUDO_SEGMENT_RE = re.compile(
    r"(?P<prefix>^|(?:&&|\|\||;|\|)\s*)sudo(?:\s+-(?:n|E|H|S))*\s+(?:(?:-i|-s)\s+)?(?:--\s+)?",
)


def strip_redundant_root_sudo(command: str, user: str | None) -> tuple[str, bool]:
    if str(user or "").strip().lower() != "root":
        return command, False
    text = str(command or "").strip()
    if not text.startswith("sudo"):
        return command, False
    match = _ROOT_SUDO_PREFIX_RE.match(text)
    if not match:
        return command, False
    stripped = str(match.group("command") or "").strip()
    stripped = _ROOT_SUDO_SEGMENT_RE.sub(lambda match: str(match.group("prefix") or ""), stripped)
    return stripped, bool(stripped and stripped != text)


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def normalize_optional_ssh_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _reject_dash_prefixed_ssh_value(name: str, value: str | None) -> None:
    if value is not None and value.startswith("-"):
        raise ValueError(
            f"SSH {name} must not start with `-`; values beginning with a dash "
            "are parsed by ssh as options and can inject local commands."
        )


def normalize_ssh_target(*, host: str, user: str | None = None) -> tuple[str, str | None]:
    host_text = str(host or "").strip()
    user_text = normalize_optional_ssh_string(user)
    if not host_text:
        return "", user_text
    if host_text.count("@") > 1:
        raise ValueError("SSH target must contain at most one `@` separator.")
    if "@" not in host_text:
        _reject_dash_prefixed_ssh_value("host", host_text)
        _reject_dash_prefixed_ssh_value("user", user_text)
        return host_text, user_text

    embedded_user, bare_host = host_text.rsplit("@", 1)
    embedded_user = embedded_user.strip()
    bare_host = bare_host.strip()
    if not embedded_user or not bare_host:
        raise ValueError("SSH target must be either `host` plus `user` or `user@host`.")
    if user_text is not None and user_text != embedded_user:
        raise ValueError("SSH target must be either `host` plus `user` or `user@host`.")
    _reject_dash_prefixed_ssh_value("host", bare_host)
    _reject_dash_prefixed_ssh_value("user", embedded_user)
    return bare_host, embedded_user


def normalize_ssh_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}

    normalized = dict(arguments)
    target_text = normalize_optional_ssh_string(normalized.pop("target", None))
    explicit_host = normalize_optional_ssh_string(normalized.get("host"))
    alias_user = normalize_optional_ssh_string(normalized.pop("username", None))
    explicit_user = normalize_optional_ssh_string(normalized.get("user"))
    if alias_user:
        if explicit_user and explicit_user != alias_user:
            raise ValueError("Conflicting SSH usernames provided via `user` and `username`.")
        normalized["user"] = alias_user
        explicit_user = alias_user
    elif explicit_user is None:
        normalized.pop("user", None)
    else:
        normalized["user"] = explicit_user

    if target_text:
        target_host, target_user = normalize_ssh_target(host=target_text, user=explicit_user)
        if explicit_host:
            explicit_host_text, explicit_host_user = normalize_ssh_target(host=explicit_host, user=explicit_user)
            if explicit_host_text != target_host:
                raise ValueError("Conflicting SSH targets provided via `target` and `host`.")
            if explicit_host_user and target_user and explicit_host_user != target_user:
                raise ValueError("Conflicting SSH usernames provided via `target` and `host`.")
        normalized["host"] = target_host
        if target_user:
            normalized["user"] = target_user
            explicit_user = target_user

    host_text = normalize_optional_ssh_string(normalized.get("host")) or ""
    host_text, user_text = normalize_ssh_target(host=host_text, user=explicit_user)
    if not host_text:
        raise ValueError("SSH target requires either `target` or `host`.")
    normalized["host"] = host_text
    if user_text:
        normalized["user"] = user_text
    else:
        normalized.pop("user", None)
    identity_file_text = normalize_optional_ssh_string(normalized.get("identity_file"))
    _reject_dash_prefixed_ssh_value("identity_file", identity_file_text)
    return normalized


def split_ssh_option_value(option: str) -> tuple[str, str | None]:
    cleaned = str(option or "").strip()
    if "=" in cleaned:
        key, value = cleaned.split("=", 1)
        return key.strip(), value.strip() or None
    parts = cleaned.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip() or None
    return cleaned, None


def parse_int_option(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def shell_tokens_with_spans(command: str) -> list[tuple[str, int, int]]:
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: list[tuple[str, int, int]] = []
    while True:
        start = lexer.instream.tell()
        token = lexer.get_token()
        end = lexer.instream.tell()
        if token == lexer.eof:
            break
        tokens.append((token, start, end))
    return tokens


def is_shell_redirection_token(token: str) -> bool:
    stripped = str(token or "").strip()
    if not stripped:
        return False
    return (
        stripped.startswith((">", "<"))
        or stripped.startswith(("1>", "1<", "2>", "2<"))
        or stripped.startswith((">>", "<<", "&>", ">&", "<&"))
        or stripped.endswith((">&1", ">&2", "<&0", "<&1", "<&2"))
    )


def join_remote_shell_tokens(
    command: str,
    tokens: list[tuple[str, int, int]],
) -> str:
    return " ".join(
        (
            token
            if is_shell_redirection_token(token)
            else (
                token
                if idx == 0 and command[start:end].lstrip().startswith(("'", '"'))
                else shlex.quote(token)
            )
        )
        for idx, (token, start, end) in enumerate(tokens)
    )


def parse_ssh_exec_args_from_shell_command(command: str) -> dict[str, Any] | None:
    command_text = str(command or "").strip()
    if not command_text:
        return None

    try:
        token_spans = shell_tokens_with_spans(command_text)
    except ValueError:
        return None
    tokens = [token for token, _start, _end in token_spans]
    if not tokens:
        return None

    parsed: dict[str, Any] = {}
    target: str | None = None
    index = 0

    if tokens[index] == "sshpass":
        index += 1
        password: str | None = None
        while index < len(tokens):
            token = tokens[index]
            if token == "ssh":
                break
            if token == "-p":
                index += 1
                if index >= len(tokens):
                    return None
                password = tokens[index]
            elif token.startswith("-p") and len(token) > 2:
                password = token[2:]
            else:
                # Only rewrite the simple sshpass form we can preserve safely.
                return None
            index += 1
        if index >= len(tokens) or tokens[index] != "ssh" or not password:
            return None
        parsed["password"] = password

    if tokens[index] != "ssh":
        return None
    index += 1

    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            if index >= len(tokens):
                return None
            target = tokens[index]
            index += 1
            break
        if not token.startswith("-"):
            target = token
            index += 1
            break

        option_name = token
        option_value: str | None = None
        if token in {"-i", "-l", "-o", "-p"}:
            index += 1
            if index >= len(tokens):
                return None
            option_value = tokens[index]
        elif token.startswith("-i") and len(token) > 2:
            option_name = "-i"
            option_value = token[2:]
        elif token.startswith("-l") and len(token) > 2:
            option_name = "-l"
            option_value = token[2:]
        elif token.startswith("-o") and len(token) > 2:
            option_name = "-o"
            option_value = token[2:]
        elif token.startswith("-p") and len(token) > 2:
            option_name = "-p"
            option_value = token[2:]
        elif token in _IGNORABLE_SSH_FLAGS:
            index += 1
            continue
        else:
            return None

        if option_name == "-i":
            parsed["identity_file"] = option_value
        elif option_name == "-l":
            parsed["user"] = option_value
        elif option_name == "-p":
            port_value = parse_int_option(option_value)
            if port_value is None:
                return None
            parsed["port"] = port_value
        elif option_name == "-o":
            key, value = split_ssh_option_value(option_value or "")
            if key not in _SAFE_SSH_OPTION_KEYS:
                return None
            if key == "IdentityFile":
                parsed["identity_file"] = value
            elif key == "Port":
                port_value = parse_int_option(value)
                if port_value is None:
                    return None
                parsed["port"] = port_value
            elif key == "User":
                parsed["user"] = value
        index += 1

    if not target:
        return None

    remote_tokens = token_spans[index:]
    if not remote_tokens:
        parsed["host"] = target
        parsed["command"] = "whoami"
        return normalize_ssh_arguments(parsed)
    if any(token in _LOCAL_SHELL_CONTROL_TOKENS for token, _start, _end in remote_tokens):
        return None

    parsed["host"] = target
    parsed["command"] = join_remote_shell_tokens(command_text, remote_tokens)
    return normalize_ssh_arguments(parsed)
