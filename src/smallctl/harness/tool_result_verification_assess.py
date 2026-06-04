from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from ..shell_utils import strip_benign_shell_redirections as _strip_benign_shell_redirections
from .tool_result_verification_constants import (
    _REMOTE_FILE_PRESENCE_PROBE_RE,
    _REMOTE_MUTATING_COMMAND_RE,
)
from .tool_result_verification_helpers import snip_text as _snip_text


def assess_remote_mutation_verification(
    *,
    requirement: dict[str, Any] | None,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:

    if tool_name != "ssh_exec" or not isinstance(requirement, dict) or not isinstance(arguments, dict):
        return {"is_verifier_attempt": False}
    command = str(arguments.get("command") or "").strip()
    mutation_command = _strip_benign_shell_redirections(command, preserve_newlines=True)
    if not mutation_command or _REMOTE_MUTATING_COMMAND_RE.search(mutation_command):
        return {"is_verifier_attempt": False}

    output = result.output if isinstance(result.output, dict) else {}
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    stdout = str(output.get("stdout") or "")
    stderr = str(output.get("stderr") or "")
    exit_code = output.get("exit_code")

    host = str(arguments.get("host") or metadata.get("host") or "").strip().lower()
    requirement_host = str(requirement.get("host") or "").strip().lower()
    if requirement_host and host and requirement_host != host:
        return {"is_verifier_attempt": False}

    guessed_paths = [str(item).strip() for item in requirement.get("guessed_paths", []) if str(item).strip()]
    path_match = any(path in command for path in guessed_paths) if guessed_paths else False

    # Directory checks
    from .directory_empty_checks import parse_directory_empty_checks, match_directory_empty_check
    directory_checks = parse_directory_empty_checks(requirement)
    directory_check = match_directory_empty_check(command, directory_checks)

    profile = str(requirement.get("verification_profile") or "").strip().lower()
    patterns = requirement.get("verification_patterns")
    new_patterns = patterns.get("new_present", []) if isinstance(patterns, dict) else []
    old_patterns = patterns.get("old_absent", []) if isinstance(patterns, dict) else []

    mentions_new = any(str(pattern) and str(pattern) in command for pattern in new_patterns)
    mentions_old = any(str(pattern) and str(pattern) in command for pattern in old_patterns)
    is_verifier_attempt = path_match or directory_check is not None or mentions_new or mentions_old
    if not is_verifier_attempt:
        return {"is_verifier_attempt": False}

    assessment: dict[str, Any] = {
        "is_verifier_attempt": True,
        "host": host,
        "command": command,
        "exit_code": exit_code,
        "path_match": path_match,
        "verification_profile": profile,
    }

    if directory_check is not None:
        check_path = str(directory_check.get("path") or "").strip().rstrip("/")
        if result.success and exit_code in (0, None) and not stdout.strip():
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": True,
                    "reason": "directory_empty_check_passed",
                    "verified_directory_empty_check": check_path,
                }
            )
            return assessment
        assessment.update(
            {
                "verification_strength": "strong",
                "clears_requirement": False,
                "reason": "directory_empty_check_failed",
                "verified_directory_empty_check": check_path,
                "message": _snip_text(stderr or stdout or "Directory-empty verifier did not return empty output.", limit=240),
            }
        )
        return assessment

    if not isinstance(patterns, dict):
        from .tool_result_verification_readback import _simple_remote_readback_path
        readback_path = _simple_remote_readback_path(command)
        if readback_path and readback_path in guessed_paths:
            if result.success and exit_code in (0, None) and stdout.strip():
                assessment.update(
                    {
                        "verification_strength": "strong",
                        "clears_requirement": True,
                        "reason": "simple_readback_path_verified",
                        "verified_path": readback_path,
                    }
                )
                return assessment
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": False,
                    "reason": "simple_readback_path_empty_or_failed",
                    "verified_path": readback_path,
                    "message": _snip_text(stderr or stdout or "Readback command did not return file content.", limit=240),
                }
            )
            return assessment
        if path_match and _REMOTE_FILE_PRESENCE_PROBE_RE.search(command):
            if result.success and exit_code in (0, None):
                assessment.update(
                    {
                        "verification_strength": "strong",
                        "clears_requirement": True,
                        "reason": "file_presence_probe_passed",
                    }
                )
                return assessment
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": False,
                    "reason": "file_presence_probe_failed",
                    "message": _snip_text(stderr or stdout or "Presence/hash verifier did not confirm the file.", limit=240),
                }
            )
            return assessment

    if profile == "html_stylesheet_swap":
        command_lower = command.lower()
        stdout_lower = stdout.lower()
        has_positive_check = "theme" in command_lower or "stylesheet" in command_lower or any(
            str(pattern).lower() in command_lower for pattern in new_patterns
        )
        has_negative_check = "<style>" in command_lower and (
            "no_style" in command_lower
            or "! grep" in command_lower
            or "grep -l" not in command_lower
            or "grep -l" in command_lower and "has_style" in command_lower
        )
        weak_positive_only = "grep -l" in command_lower and has_positive_check and "<style>" not in command_lower
        if weak_positive_only:
            assessment.update(
                {
                    "verification_strength": "weak_positive_only",
                    "clears_requirement": False,
                    "reason": "replacement_positive_only",
                    "message": "Positive-only grep verifier does not prove the old target is gone.",
                }
            )
            return assessment

        if has_positive_check and has_negative_check:
            style_present = "has_style" in stdout_lower
            link_missing = "no_link" in stdout_lower
            link_present = "has_link" in stdout_lower or any(
                str(pattern).lower() in stdout_lower for pattern in new_patterns
            )
            if result.success and exit_code in (0, None) and not style_present and not link_missing and link_present:
                assessment.update(
                    {
                        "verification_strength": "strong",
                        "clears_requirement": True,
                        "reason": "strong_remote_replacement_verifier",
                    }
                )
                return assessment
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": False,
                    "reason": "strong_verifier_failed",
                    "message": _snip_text(stderr or stdout or "Strong replacement verifier did not show the expected NO_STYLE/HAS_LINK outcome.", limit=240),
                }
            )
            return assessment

    content_based_positive = [str(pattern) for pattern in new_patterns if str(pattern)]
    content_based_negative = [str(pattern) for pattern in old_patterns if str(pattern)]
    if result.success and exit_code in (0, None) and content_based_positive:
        positive_seen = all(pattern.lower() in stdout.lower() for pattern in content_based_positive)
        negative_absent = all(pattern.lower() not in stdout.lower() for pattern in content_based_negative)
        if positive_seen and negative_absent:
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": True,
                    "reason": "stdout_verifier_patterns_matched",
                }
            )
            return assessment

    assessment.update(
        {
            "verification_strength": "unknown",
            "clears_requirement": False,
        }
    )
    return assessment
