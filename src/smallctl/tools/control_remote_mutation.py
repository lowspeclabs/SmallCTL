from __future__ import annotations

import shlex
from typing import Any

_REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"

_REMOTE_BINARYISH_SUFFIXES = (
    ".asc",
    ".bin",
    ".crt",
    ".deb",
    ".der",
    ".gpg",
    ".gz",
    ".key",
    ".pem",
    ".pfx",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
)


def remote_mutation_verification_requirement(state: Any) -> dict[str, Any] | None:
    payload = state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(payload, dict) or not payload:
        return None
    if payload.get("failed_verification_attempts", 0) >= 3:
        return None
    if not remote_mutation_has_pending_verifier(payload):
        return None
    return payload


def remote_mutation_has_pending_verifier(requirement: dict[str, Any]) -> bool:
    guessed_paths = requirement.get("guessed_paths")
    if not isinstance(guessed_paths, list):
        guessed_paths = []
    verified_paths = {
        str(path).strip()
        for path in requirement.get("verified_paths", [])
        if str(path).strip()
    }
    if any(str(path).strip() and str(path).strip() not in verified_paths for path in guessed_paths):
        return True

    verified_directories = {
        str(path).strip().rstrip("/")
        for path in requirement.get("verified_directory_empty_checks", [])
        if str(path).strip()
    }
    return any(
        check["path"] not in verified_directories
        for check in _remote_mutation_directory_checks(requirement)
    )


def _remote_mutation_directory_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
    from ..harness.remote_mutation_helpers import remote_mutation_directory_checks

    return remote_mutation_directory_checks(requirement)


def remote_path_needs_presence_probe(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    return bool(lowered) and lowered.endswith(_REMOTE_BINARYISH_SUFFIXES)


def remote_presence_probe_command(path: str) -> str:
    quoted = shlex.quote(str(path or "").strip())
    return f"test -s {quoted} && sha256sum {quoted}"


def remote_mutation_block_payload(requirement: dict[str, Any]) -> dict[str, Any]:
    guessed_paths = requirement.get("guessed_paths")
    if not isinstance(guessed_paths, list):
        guessed_paths = []
    verified_paths = {
        str(path).strip()
        for path in requirement.get("verified_paths", [])
        if str(path).strip()
    }
    pending_paths = [
        str(path).strip()
        for path in guessed_paths
        if str(path).strip() and str(path).strip() not in verified_paths
    ]
    directory_checks = _remote_mutation_directory_checks(requirement)
    verified_directories = {
        str(path).strip().rstrip("/")
        for path in requirement.get("verified_directory_empty_checks", [])
        if str(path).strip()
    }
    pending_directory_checks = [
        check for check in directory_checks if check["path"] not in verified_directories
    ]
    path_hint = ", ".join(str(path) for path in pending_paths[:3] if str(path).strip())
    first_path = next((str(path).strip() for path in pending_paths if str(path).strip()), "")
    first_directory_check = pending_directory_checks[0] if pending_directory_checks else {}
    host = str(requirement.get("host") or "").strip()
    user = str(requirement.get("user") or "").strip()
    mutation_type = str(requirement.get("mutation_type") or "").strip().lower()
    required_arguments: dict[str, Any] = {}
    first_path_needs_presence_probe = mutation_type != "deletion" and remote_path_needs_presence_probe(first_path)
    if first_path:
        if first_path_needs_presence_probe:
            required_arguments["command"] = remote_presence_probe_command(first_path)
        else:
            required_arguments["path"] = first_path
    elif first_directory_check:
        directory_path = str(first_directory_check.get("path") or "").strip()
        if directory_path:
            required_arguments["command"] = (
                f"find {directory_path} -mindepth 1 -maxdepth 1 -print -quit"
            )
    if host:
        required_arguments["host"] = host
    if user:
        required_arguments["user"] = user
    if host and "@" in host:
        required_arguments.pop("host", None)
        required_arguments["target"] = host

    if mutation_type == "deletion":
        if first_path:
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote file deletion still needs meaningful verification. "
                "Verify the target is gone with `ssh_file_read`; a `not found` / `no such file` result counts as successful verification."
            )
            next_required_action = {
                "tool_names": ["ssh_file_read"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    "Read the deleted path directly.",
                    "A missing-file result is valid proof for deletion tasks and will clear the requirement.",
                ],
            }
        else:
            directory_path = str(first_directory_check.get("path") or "").strip()
            glob = str(first_directory_check.get("glob") or "").strip()
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote glob deletion still needs meaningful verification. "
                "Verify the parent directory is empty with a read-only `ssh_exec` check."
            )
            next_required_action = {
                "tool_names": ["ssh_exec"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    f"Check that `{directory_path}` has no remaining entries from `{glob or directory_path + '/*'}`.",
                    "Empty stdout from the suggested find command is valid proof for glob deletion tasks.",
                ],
            }
    else:
        if first_path_needs_presence_probe:
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote file mutation still needs meaningful verification. "
                "Verify the changed binary or key file exists and has content with a read-only `ssh_exec` presence/hash check."
            )
            next_required_action = {
                "tool_names": ["ssh_exec"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    "Binary/key files should be verified with metadata or hashes, not decoded as text.",
                    "A successful `test -s ... && sha256sum ...` check is valid proof for this file.",
                ],
            }
        else:
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote file mutation still needs meaningful verification. "
                "Read back the changed file with `ssh_file_read`, or redo the edit with `ssh_file_patch` / "
                "`ssh_file_replace_between` so the harness can verify the readback hash."
            )
            next_required_action = {
                "tool_names": ["ssh_file_read", "ssh_file_patch", "ssh_file_replace_between"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    "A grep-only positive match is not enough for replacement tasks.",
                    "Verify that the replacement exists and the old target is gone.",
                ],
            }
    if path_hint:
        error += f" Suspected path(s): {path_hint}."
    if first_directory_check:
        error += f" Directory check: {first_directory_check.get('path')}."
    if required_arguments:
        tool_name = "ssh_file_read" if "path" in required_arguments else "ssh_exec"
        verifier_call = tool_name + "(" + ", ".join(
            f"{key}={required_arguments[key]!r}"
            for key in ("target", "host", "user", "path", "command")
            if key in required_arguments
        ) + ")"
        error += f" Next required verifier: `{verifier_call}`."
    return {
        "error": error,
        "next_required_action": next_required_action,
    }
