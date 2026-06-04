from __future__ import annotations

from typing import Any

from .shell_support_constants import _ARGPARSE_REQUIRED_ARGS_PATTERN


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


def _build_argparse_missing_args_question(command: str, missing_args: list[str]) -> str:
    missing_text = ", ".join(missing_args) if missing_args else "required arguments"
    return (
        f"The command `{command}` is missing required arguments: {missing_text}. "
        "What values should I use?"
    )


def _detect_unsupported_shell_syntax(command: str) -> str | None:
    if "<<<" in command:
        return (
            "Command uses Bash-only here-string redirection (`<<<`), but smallctl runs shell "
            "commands through /bin/sh on Unix. Rewrite it with POSIX syntax (for example, "
            "use `printf` piped into the command) or wrap the whole command in `bash -lc`."
        )
    return None
