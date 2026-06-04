from __future__ import annotations

import re
from typing import Any

_REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"
_STALE_VERIFIER_KEY = "_last_verifier_stale_after_mutation"
_SSH_FILE_VERIFIER_TOOLS = {"ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}
_REMOTE_MUTATING_COMMAND_RE = re.compile(
    r"\bsed\s+-i\b"
    r"|"
    r"\bperl\s+-p(?:i|[^A-Za-z0-9_]*-i)\b"
    r"|"
    r"\bpython3?\s+-c\b.*\bopen\s*\([^)]*['\"]w"
    r"|"
    r"(?:^|\s)(?:\d?>|\d?>>|>>|>)\s*(?!/dev/(?:null|stdout|stderr|fd/\d+)\b)\S+"
    r"|"
    r"\btee(?:\s+-a)?\s+/\S+"
    r"|"
    r"\bcat\s*>\s*/\S+"
    r"|"
    r"\b(?:rm|truncate|install\s+-m)\b"
    r"|"
    r"\b(?:mv|cp)\b.+\s+/(?:etc|var|usr|opt|srv|root)/\S+",
    re.IGNORECASE | re.DOTALL,
)
_REMOTE_SED_MUTATION_RE = re.compile(r"\bsed\s+-i\b", re.IGNORECASE)
_SED_SUBSTITUTION_RE = re.compile(
    r"""s([/|#])((?:\\.|(?!\1).)+)\1((?:\\.|(?!\1).)+)\1[gim]*""",
    re.IGNORECASE,
)
_REMOTE_MULTILINE_REPLACEMENT_RE = re.compile(
    r"<[A-Za-z][^>]*>.*</[A-Za-z][^>]*>"
    r"|"
    r"\\n|\[\\s\\S\]|\.\*"
    r"|"
    r"\.(?:html|xml|conf|service|ya?ml|json)\b",
    re.IGNORECASE | re.DOTALL,
)


def remote_mutation_requirement_satisfied(requirement: dict[str, Any]) -> bool:
    guessed_paths = [str(item).strip() for item in requirement.get("guessed_paths", []) if str(item).strip()]
    verified_paths = {
        str(item).strip()
        for item in requirement.get("verified_paths", [])
        if str(item).strip()
    }
    pending_paths = [path for path in guessed_paths if path not in verified_paths]

    directory_checks = remote_mutation_directory_checks(requirement)
    verified_directories = {
        str(item).strip().rstrip("/")
        for item in requirement.get("verified_directory_empty_checks", [])
        if str(item).strip()
    }
    pending_directories = [
        check["path"] for check in directory_checks if check["path"] not in verified_directories
    ]
    return not pending_paths and not pending_directories


def remote_mutation_directory_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
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


def mark_remote_mutation_path_verified(requirement: dict[str, Any], path: str) -> None:
    verified = set(requirement.get("verified_paths", []))
    verified.add(path)
    requirement["verified_paths"] = list(verified)


def mark_remote_mutation_directory_verified(requirement: dict[str, Any], path: str) -> None:
    verified = set(requirement.get("verified_directory_empty_checks", []))
    verified.add(path)
    requirement["verified_directory_empty_checks"] = list(verified)
