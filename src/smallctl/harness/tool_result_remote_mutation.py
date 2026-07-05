from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..models.conversation import ConversationMessage
from ..normalization import dedupe_keep_tail
from ..shell_utils import strip_benign_shell_redirections
from ..tools.shell_support import _REMOTE_INSTALLER_PREFLIGHT_KEY
from .directory_empty_checks import guess_deletion_directory_empty_checks
from .remote_mutation_helpers import (
    _REMOTE_MULTILINE_REPLACEMENT_RE,
    _REMOTE_MUTATING_COMMAND_RE,
    _REMOTE_MUTATION_VERIFICATION_KEY,
    _REMOTE_SED_MUTATION_RE,
    _SED_SUBSTITUTION_RE,
    _SSH_FILE_VERIFIER_TOOLS,
    bounded_region_not_found,
    mark_remote_mutation_directory_verified,
    mark_remote_mutation_path_verified,
    readback_content_satisfies_requirement,
    remote_missing_file_markers,
    remote_mutation_guessed_paths,
    remote_mutation_requirement_satisfied,
    remote_mutation_target_matches,
    should_emit_small_file_rewrite_nudge,
    tool_result_path_host,
)
from .remote_mutation_parsing import (
    _REMOTE_DELETION_RE,
    guess_remote_mutation_paths,
)
from .tool_result_verification import assess_remote_mutation_verification


def _observe_remote_installer_preflight_check(
    service: Any,
    *,
    result: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Observe successful ssh_exec results and transition preflight status
    from 'required' to 'clean' when all integrity checks have passed.

    The preflight guard in shell_support._remote_installer_preflight_guard()
    blocks high-risk installer commands (e.g. installfog.sh) and records a
    set of required check commands in the scratchpad.  This observer watches
    for successful execution of those check commands and marks the preflight
    as clean once every required check has been observed passing.
    """
    if not result.success:
        return
    if not isinstance(arguments, dict):
        return

    command = str(arguments.get("command") or "").strip()
    host = str(arguments.get("host") or "").strip().lower()
    if not command or not host:
        return

    # Check exit code — only exit_code == 0 counts as a passing check.
    output = result.output if isinstance(result.output, dict) else {}
    exit_code = output.get("exit_code")
    if exit_code is not None and int(exit_code) != 0:
        return

    scratchpad = getattr(service.harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    preflights = scratchpad.get(_REMOTE_INSTALLER_PREFLIGHT_KEY)
    if not isinstance(preflights, dict):
        return

    for key, entry in preflights.items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "").strip() != "required":
            continue
        if str(entry.get("host") or "").strip().lower() != host:
            continue

        checks = entry.get("checks")
        if not isinstance(checks, list) or not checks:
            continue

        completed = entry.get("completed_checks")
        if not isinstance(completed, list):
            completed = []
            entry["completed_checks"] = completed

        # Match the executed command against required checks.  The agent
        # may run individual checks or combine them with '&&', so we
        # check both exact match and substring containment.
        for check in checks:
            if check in completed:
                continue
            if command == check or check in command:
                completed.append(check)

        if all(check in completed for check in checks):
            entry["status"] = "clean"
            entry["created_at_step"] = int(
                getattr(service.harness.state, "step_count", 0) or 0
            )
            runlog = getattr(service.harness, "_runlog", None)
            if callable(runlog):
                runlog(
                    "remote_installer_preflight_cleared",
                    "remote installer preflight checks passed",
                    host=host,
                    key=key,
                    checks=checks,
                    completed_checks=completed,
                )
            break


def _record_remote_mutation_requirement(
    service: Any,
    *,
    result: Any,
    arguments: dict[str, Any] | None,
) -> None:
    if not result.success or not isinstance(arguments, dict):
        return
    command = str(arguments.get("command") or "").strip()
    mutation_command = strip_benign_shell_redirections(command, preserve_newlines=True)
    if not mutation_command or _REMOTE_MUTATING_COMMAND_RE.search(mutation_command) is None:
        return

    host = str(arguments.get("host") or arguments.get("target") or "").strip()
    is_deletion = _REMOTE_DELETION_RE.search(mutation_command) is not None
    guessed_paths = guess_remote_mutation_paths(mutation_command, deletion=is_deletion)
    directory_empty_checks = guess_deletion_directory_empty_checks(mutation_command) if is_deletion else []
    if is_deletion and not guessed_paths and not directory_empty_checks:
        return
    requirement: dict[str, Any] = {
        "tool_name": "ssh_exec",
        "host": host,
        "user": str(arguments.get("user") or "").strip(),
        "command": command,
        "guessed_paths": guessed_paths,
        "directory_empty_checks": directory_empty_checks,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suggested_verifier": (
            "You used ssh_exec to mutate a remote file. Before calling task_complete "
            "or attempting further edits, verify the change by reading the file with "
            "ssh_file_read. Do NOT try to re-apply the same change with "
            "ssh_file_replace_between; if the file is already fixed, re-application will fail."
        ),
    }
    if is_deletion:
        requirement["mutation_type"] = "deletion"
        requirement["suggested_verifier"] = (
            "You used ssh_exec to delete or truncate a remote file. Before calling "
            "task_complete, verify concrete file paths are gone with ssh_file_read; "
            "a missing-file result counts as successful verification. For glob "
            "deletions such as /var/www/*, verify the parent directory is empty "
            "with a read-only ssh_exec check."
        )
    if _REMOTE_SED_MUTATION_RE.search(mutation_command) is not None:
        sed_matches = _SED_SUBSTITUTION_RE.findall(mutation_command)
        if sed_matches:
            old_absent: list[str] = []
            new_present: list[str] = []
            for delim, old, new in sed_matches:
                old_unesc = old.replace("\\" + delim, delim)
                new_unesc = new.replace("\\" + delim, delim)
                if old_unesc and new_unesc and old_unesc != new_unesc:
                    old_absent.append(old_unesc)
                    new_present.append(new_unesc)
            if old_absent or new_present:
                requirement["verification_patterns"] = {
                    "old_absent": old_absent,
                    "new_present": new_present,
                }
    if "llm-explainer-theme.css" in command and "<style>" in command:
        requirement["verification_profile"] = "html_stylesheet_swap"
        requirement["verification_patterns"] = {
            "old_absent": ["<style>"],
            "new_present": ["llm-explainer-theme.css"],
        }
    if (
        not requirement.get("guessed_paths")
        and not requirement.get("directory_empty_checks")
        and not requirement.get("verification_patterns")
        and not requirement.get("verification_profile")
    ):
        return
    service.harness.state.scratchpad[_REMOTE_MUTATION_VERIFICATION_KEY] = requirement

    if _REMOTE_SED_MUTATION_RE.search(mutation_command) is not None:
        _emit_remote_mutation_nudge(
            service,
            command=command,
            guessed_paths=guessed_paths,
            multiline=bool(_REMOTE_MULTILINE_REPLACEMENT_RE.search(command)),
        )
    elif guessed_paths:
        _emit_remote_redirection_mutation_nudge(
            service,
            command=command,
            guessed_paths=guessed_paths,
        )


def _emit_remote_redirection_mutation_nudge(
    service: Any,
    *,
    command: str,
    guessed_paths: list[str],
) -> None:
    signature = f"redirection:{hash(command)}:{','.join(guessed_paths[:4])}"
    scratchpad = service.harness.state.scratchpad
    prior = scratchpad.get("_remote_mutation_nudges", [])
    if isinstance(prior, list) and signature in prior:
        return
    scratchpad["_remote_mutation_nudges"] = dedupe_keep_tail(
        ([str(item) for item in prior] if isinstance(prior, list) else []) + [signature],
        limit=12,
    )
    paths_text = ", ".join(f"`{path}`" for path in guessed_paths[:4])
    content = (
        f"Remote `ssh_exec` wrote files via redirection/echo ({paths_text}). "
        "The harness cannot verify those writes until you read the files back with "
        "`ssh_file_read`. Run the required `ssh_file_read` call(s) before calling "
        "`task_complete`; directory listings or `ls` output are not sufficient proof."
    )
    service.harness.state.append_message(
        ConversationMessage(
            role="system",
            content=content,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "remote_redirection_mutation",
                "guessed_paths": guessed_paths,
            },
        )
    )


def _emit_remote_mutation_nudge(
    service: Any,
    *,
    command: str,
    guessed_paths: list[str],
    multiline: bool,
) -> None:
    signature = f"{hash(command)}:{','.join(guessed_paths[:4])}"
    scratchpad = service.harness.state.scratchpad
    prior = scratchpad.get("_remote_mutation_nudges", [])
    if isinstance(prior, list) and signature in prior:
        return
    scratchpad["_remote_mutation_nudges"] = dedupe_keep_tail(
        ([str(item) for item in prior] if isinstance(prior, list) else []) + [signature],
        limit=12,
    )
    if multiline:
        content = (
            "This looks like a bounded multiline remote replacement. "
            "Read back the result with `ssh_file_read` before calling `task_complete`. "
            "If the readback shows the change is missing, only then use `ssh_file_replace_between` "
            "with exact `start_text` and `end_text`."
        )
        recovery_kind = "remote_multiline_mutation"
    else:
        content = (
            "Remote `sed -i` exited 0, but the harness cannot verify the change yet. "
            "Read back the changed file with `ssh_file_read` first. "
            "Do not use `ssh_file_replace_between` to redo the edit unless the readback "
            "shows the change is missing."
        )
        recovery_kind = "remote_sed_mutation"
    service.harness.state.append_message(
        ConversationMessage(
            role="system",
            content=content,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": recovery_kind,
                "guessed_paths": guessed_paths,
            },
        )
    )


def _clear_remote_mutation_requirement_from_tool(
    service: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None,
) -> None:
    if tool_name not in _SSH_FILE_VERIFIER_TOOLS:
        return
    requirement = service.harness.state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(requirement, dict):
        return

    # For deletion mutations, a failed ssh_file_read on the deleted path
    # (file not found) is positive verification that the file is gone.
    if not result.success and tool_name == "ssh_file_read":
        if str(requirement.get("mutation_type") or "").strip().lower() == "deletion":
            path, host = tool_result_path_host(result, arguments)
            if not remote_mutation_target_matches(requirement, path=path, host=host):
                return
            failure_markers = remote_missing_file_markers(result)
            if (
                "no such file" in failure_markers
                or "not found" in failure_markers
                or "file_not_found" in failure_markers
            ):
                mark_remote_mutation_path_verified(requirement, path)
                if remote_mutation_requirement_satisfied(requirement):
                    service.harness.state.scratchpad.pop(_REMOTE_MUTATION_VERIFICATION_KEY, None)
        return

    if not result.success:
        return

    path, host = tool_result_path_host(result, arguments)
    if not remote_mutation_target_matches(requirement, path=path, host=host):
        return
    guessed_paths = remote_mutation_guessed_paths(requirement)
    if tool_name == "ssh_file_read":
        output = result.output if isinstance(result.output, dict) else {}
        content = str(output.get("content") or "")
        if isinstance(requirement.get("verification_patterns"), dict):
            if not readback_content_satisfies_requirement(requirement, content):
                return
        elif not (guessed_paths and path and path in guessed_paths):
            return
    mark_remote_mutation_path_verified(requirement, path)
    if remote_mutation_requirement_satisfied(requirement):
        service.harness.state.scratchpad.pop(_REMOTE_MUTATION_VERIFICATION_KEY, None)


def _record_failed_verification_attempt(
    service: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None,
) -> None:
    if result.success:
        return
    if tool_name == "ssh_exec":
        requirement = service.harness.state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
        if not isinstance(requirement, dict):
            return
        assessment = assess_remote_mutation_verification(
            requirement=requirement,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
        )
        if not assessment.get("is_verifier_attempt"):
            return
        requirement["failed_verification_attempts"] = requirement.get("failed_verification_attempts", 0) + 1
        return
    if tool_name not in _SSH_FILE_VERIFIER_TOOLS:
        return
    requirement = service.harness.state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(requirement, dict):
        return
    path, host = tool_result_path_host(result, arguments)
    if not remote_mutation_target_matches(requirement, path=path, host=host):
        return
    requirement["failed_verification_attempts"] = requirement.get("failed_verification_attempts", 0) + 1


def _maybe_emit_bounded_region_trap_nudge(
    service: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Inject a nudge when ssh_file_replace_between fails because the region is gone,
    suggesting the model read back the file instead of retrying the same bounds."""
    if tool_name != "ssh_file_replace_between":
        return
    if result.success:
        return
    if not bounded_region_not_found(result):
        return

    path, host = tool_result_path_host(result, arguments)

    # First, check for the small-file anti-pattern: file is small and was recently read.
    # In that case, suggest ssh_file_write instead of retrying bounds.
    _maybe_emit_small_file_rewrite_nudge(service, path=path, host=host, arguments=arguments)

    # Then, the original bounded-region trap nudge for ssh_exec mutations.
    requirement = service.harness.state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(requirement, dict):
        return

    if not remote_mutation_target_matches(requirement, path=path, host=host):
        return

    scratchpad = service.harness.state.scratchpad
    prior = scratchpad.get("_bounded_region_trap_nudges", [])
    signature = f"{path}:{host}"
    if isinstance(prior, list) and signature in prior:
        return
    scratchpad["_bounded_region_trap_nudges"] = dedupe_keep_tail(
        ([str(item) for item in prior] if isinstance(prior, list) else []) + [signature],
        limit=12,
    )

    service.harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"The replacement bounds were not found in `{path}`. "
                "If you already applied this change with `ssh_exec`, the file may already be fixed. "
                "Use `ssh_file_read` to check the current content before trying again."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "remote_bounded_region_trap",
                "path": path,
                "host": host,
            },
        )
    )


def _maybe_emit_small_file_rewrite_nudge(
    service: Any,
    *,
    path: str,
    host: str,
    arguments: dict[str, Any] | None,
) -> None:
    """If a bounded replace failed on a small file that was recently read, nudge toward ssh_file_write."""
    if not path:
        return

    recent_read_size = _recent_ssh_file_read_size(service, path=path, host=host)
    replacement_text = ""
    if isinstance(arguments, dict):
        replacement_text = str(arguments.get("replacement_text") or "").strip()
    if not should_emit_small_file_rewrite_nudge(
        path=path,
        recent_read_size=recent_read_size,
        replacement_text=replacement_text,
    ):
        return

    scratchpad = service.harness.state.scratchpad
    prior = scratchpad.get("_small_file_rewrite_nudges", [])
    signature = f"{path}:{host}"
    if isinstance(prior, list) and signature in prior:
        return
    scratchpad["_small_file_rewrite_nudges"] = dedupe_keep_tail(
        ([str(item) for item in prior] if isinstance(prior, list) else []) + [signature],
        limit=12,
    )

    service.harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"`ssh_file_replace_between` failed on `{path}`. "
                f"This file is small (~{recent_read_size} bytes) and was recently read. "
                "Because the replacement covers most of the file, use `ssh_file_write` to overwrite the entire file instead. "
                "This avoids boundary-matching errors."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "small_file_prefer_rewrite",
                "path": path,
                "host": host,
            },
        )
    )


def _recent_ssh_file_read_size(service: Any, *, path: str, host: str) -> int | None:
    """Look back through recent tool results for an ssh_file_read on the same path+host. Return its size in bytes."""
    try:
        from ...state import ConversationMessage
    except Exception:
        return None

    messages = getattr(service.harness.state, "messages", [])
    if not isinstance(messages, list):
        return None

    # Scan last ~20 messages for a recent ssh_file_read on this path+host
    for msg in reversed(messages[-20:]):
        if not isinstance(msg, ConversationMessage):
            continue
        if msg.role != "tool":
            continue
        tool_name = msg.metadata.get("tool_name") if isinstance(msg.metadata, dict) else None
        if tool_name != "ssh_file_read":
            continue
        # Extract args from metadata if available
        args = msg.metadata.get("arguments") if isinstance(msg.metadata, dict) else None
        if not isinstance(args, dict):
            # Fall back to parsing the message content for FILE READ STATUS
            content = str(msg.content or "")
            if f"path={path}" in content:
                # Rough heuristic: use content length if we can't find exact size
                return len(content.encode("utf-8"))
            continue
        read_path = str(args.get("path") or "").strip()
        read_host = str(args.get("host") or args.get("target") or "").strip().lower()
        if read_path == path and read_host == host:
            # Try to get the file size from the artifact metadata or content length
            artifact_id = msg.metadata.get("artifact_id") if isinstance(msg.metadata, dict) else None
            if artifact_id:
                artifact = service.harness.state.artifacts.get(artifact_id)
                if artifact and isinstance(artifact, dict):
                    size = artifact.get("size_bytes")
                    if isinstance(size, int) and size > 0:
                        return size
            # Fallback: estimate from the message content itself
            return len(str(msg.content or "").encode("utf-8"))
    return None


def _handle_remote_mutation_verifier_result(
    service: Any,
    *,
    result: Any,
    arguments: dict[str, Any] | None,
) -> None:
    requirement = service.harness.state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(requirement, dict):
        return
    assessment = assess_remote_mutation_verification(
        requirement=requirement,
        tool_name="ssh_exec",
        result=result,
        arguments=arguments,
    )
    if not assessment.get("is_verifier_attempt"):
        return
    if assessment.get("clears_requirement"):
        verified_directory = str(assessment.get("verified_directory_empty_check") or "").strip()
        if verified_directory:
            mark_remote_mutation_directory_verified(requirement, verified_directory)
            if remote_mutation_requirement_satisfied(requirement):
                service.harness.state.scratchpad.pop(_REMOTE_MUTATION_VERIFICATION_KEY, None)
            return
        service.harness.state.scratchpad.pop(_REMOTE_MUTATION_VERIFICATION_KEY, None)
        return
    if assessment.get("verification_strength") == "weak_positive_only":
        service.harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    "This verifier only proves the replacement text exists somewhere. "
                    "It does not prove the old target is gone. Prefer `ssh_file_read` for the changed file "
                    "or a stronger verifier that checks both `NO_STYLE` and `HAS_LINK`."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "remote_weak_replacement_verifier",
                    "remote_verification_assessment": assessment,
                },
            )
        )
