from __future__ import annotations

import shlex


def _split_shell_words(command: str) -> list[str]:
    try:
        return shlex.split(str(command or ""))
    except ValueError:
        return str(command or "").split()


def _simple_shell_command_segments(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        return []
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return [raw]
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in {"&&", "||", ";", "|"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [" ".join(part) for part in segments if part]


def _split_shell_command_segments(command: str) -> list[str]:
    words = _split_shell_words(command)
    if not words:
        return []
    segments: list[list[str]] = [[]]
    for word in words:
        if word in {"&&", "||", ";", "|"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(word)
    return [" ".join(shlex.quote(part) for part in segment) for segment in segments if segment]


def _strip_environment_and_wrappers(words: list[str]) -> list[str]:
    import re
    from pathlib import Path
    stripped = list(words)
    while stripped:
        removed = False
        while stripped and "=" in stripped[0] and not stripped[0].startswith("="):
            key, _value = stripped[0].split("=", 1)
            if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                break
            stripped.pop(0)
            removed = True
        if stripped and Path(stripped[0]).name.lower() in {"sudo", "doas", "env", "command"}:
            stripped.pop(0)
            removed = True
            while stripped and stripped[0].startswith("-"):
                stripped.pop(0)
        if not removed:
            break
    return stripped
