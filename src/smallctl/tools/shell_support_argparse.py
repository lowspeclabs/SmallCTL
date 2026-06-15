from __future__ import annotations

from typing import Any

from .shell_support_constants import _ARGPARSE_REQUIRED_ARGS_PATTERN, _ARGPARSE_UNRECOGNIZED_ARGS_PATTERN


def _extract_missing_argparse_arguments(error_text: str) -> list[str]:
    match = _ARGPARSE_REQUIRED_ARGS_PATTERN.search(str(error_text or ""))
    if not match:
        return []

    missing = match.group(1).strip()
    if not missing:
        return []

    missing = missing.replace(" and ", ", ")
    values = [part.strip(" .`'\"") for part in missing.split(",")]
    return [value for value in values if value]


def _extract_unrecognized_argparse_arguments(error_text: str) -> list[str]:
    match = _ARGPARSE_UNRECOGNIZED_ARGS_PATTERN.search(str(error_text or ""))
    if not match:
        return []

    unrecognized = match.group(1).strip()
    if not unrecognized:
        return []

    # Argparse unrecognized arguments are space-separated (e.g., "--url --token")
    values = [part.strip(" .`'") for part in unrecognized.split()]
    return [value for value in values if value]


def _build_argparse_missing_args_question(command: str, missing_args: list[str]) -> str:
    missing_text = ", ".join(missing_args) if missing_args else "required arguments"
    return (
        f"The command `{command}` is missing required arguments: {missing_text}. "
        "What values should I use?"
    )


def _build_argparse_unrecognized_args_hint(command: str, unrecognized_args: list[str]) -> str | None:
    if not unrecognized_args:
        return None
    args_text = ", ".join(unrecognized_args)
    return (
        f"The command `{command}` has unrecognized arguments: {args_text}. "
        "If this is a CLI with subcommands, place global flags (like --url, --token) "
        "BEFORE the subcommand (e.g., `script.py --url X --token Y subcommand`)."
    )


def _detect_unsupported_shell_syntax(command: str) -> str | None:
    if "<<<" in command:
        return (
            "Command uses Bash-only here-string redirection (`<<<`), but smallctl runs shell "
            "commands through /bin/sh on Unix. Rewrite it with POSIX syntax (for example, "
            "use `printf` piped into the command) or wrap the whole command in `bash -lc`."
        )
    unbalanced = _detect_unbalanced_quotes(command)
    if unbalanced:
        return unbalanced
    return None


def _detect_unbalanced_quotes(command: str) -> str | None:
    """Return a diagnostic string when single/double quotes or backticks are unbalanced."""
    text = str(command or "")
    # Track quotes outside of simple ${...} expansions so we do not false-positive
    # on apostrophes inside words like "don't".
    in_single = False
    in_double = False
    in_backtick = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_double:
            if ch == '"':
                in_double = False
            continue
        if in_backtick:
            if ch == "`":
                in_backtick = False
            continue
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if ch == '"':
            in_double = True
        elif ch == "`":
            in_backtick = True
        elif ch == "'":
            in_single = True
    parts: list[str] = []
    if in_single:
        parts.append("unmatched single quote (`'`)")
    if in_double:
        parts.append("unmatched double quote (`\"`)")
    if in_backtick:
        parts.append(r"unmatched backtick (` `` `)")
    if not parts:
        return None
    return (
        f"Command has {' and '.join(parts)}. "
        "Close every opening quote/backtick or rewrite the command so it can be parsed by the shell."
    )
