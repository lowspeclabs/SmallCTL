from __future__ import annotations

from typing import Any


def parse_directory_empty_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
    """Parse and deduplicate directory_empty_checks from a requirement dict."""
    raw_checks = requirement.get("directory_empty_checks")
    if not isinstance(raw_checks, list):
        return []
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_checks:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().rstrip("/")
        if not path or path in seen:
            continue
        seen.add(path)
        glob = str(item.get("glob") or "").strip()
        checks.append({"path": path, "glob": glob})
    return checks


def match_directory_empty_check(command: str, checks: list[dict[str, str]]) -> dict[str, str] | None:
    """Match a shell command against a list of directory empty checks."""
    command_text = str(command or "")
    if not command_text or not checks:
        return None
    lowered = command_text.lower()
    if not any(marker in lowered for marker in ("find ", "ls ", "test ", "compgen ")):
        return None
    if not any(marker in lowered for marker in ("-mindepth", "-maxdepth", "-print", "-quit", "-a", "-z", "compgen")):
        return None
    for check in checks:
        path = str(check.get("path") or "").strip().rstrip("/")
        glob = str(check.get("glob") or "").strip()
        if path and (path in command_text or glob and glob in command_text):
            return check
    return None


_REMOTE_SHELL_INTERPRETER_PATHS = {
    "/bin/bash",
    "/bin/sh",
    "/bin/zsh",
    "/usr/bin/bash",
    "/usr/bin/sh",
    "/usr/bin/env",
    "/usr/bin/python3",
    "/usr/bin/python",
}


_BENIGN_REMOTE_MUTATION_PATHS = {
    "/dev/null",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/urandom",
    "/dev/zero",
    "/proc/self",
    "/proc/mounts",
    "/etc/passwd",
    "/etc/group",
    "/etc/hosts",
    "/etc/resolv.conf",
    "/etc/localtime",
}


def guess_deletion_directory_empty_checks(command: str) -> list[dict[str, str]]:
    """Guess directory empty checks from a deletion command string."""
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    tokens = command.replace(";", " ").replace("&&", " ").replace("||", " ").split()
    for token in tokens:
        token = token.strip().rstrip("/")
        if not token or token in seen:
            continue
        if token.startswith("-") or token.startswith("#"):
            continue
        if "/" not in token:
            continue
        if token in _REMOTE_SHELL_INTERPRETER_PATHS or token in _BENIGN_REMOTE_MUTATION_PATHS:
            continue
        check = _deletion_glob_empty_check(token)
        if check:
            seen.add(token)
            checks.append(check)
    return checks


def _deletion_glob_empty_check(token: str) -> dict[str, str] | None:
    """Build a directory-empty check from a deletion path token."""
    if "*" in token or "?" in token:
        parent = token.rsplit("/", 1)[0]
        if parent and "/" in parent:
            return {"path": parent, "glob": token}
    return None
