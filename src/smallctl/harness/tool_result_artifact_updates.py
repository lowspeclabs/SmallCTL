from __future__ import annotations

import re
import shlex
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.conversation import ConversationMessage
from .directory_empty_checks import (
    guess_deletion_directory_empty_checks,
    parse_directory_empty_checks,
)
from .remote_mutation_helpers import (
    _REMOTE_MULTILINE_REPLACEMENT_RE,
    _REMOTE_MUTATING_COMMAND_RE,
    _REMOTE_MUTATION_VERIFICATION_KEY,
    _REMOTE_SED_MUTATION_RE,
    _SED_SUBSTITUTION_RE,
    _SSH_FILE_VERIFIER_TOOLS,
    _STALE_VERIFIER_KEY,
    mark_remote_mutation_directory_verified,
    mark_remote_mutation_path_verified,
    remote_mutation_directory_checks,
    remote_mutation_requirement_satisfied,
)
from .remote_mutation_parsing import (
    _REMOTE_DELETION_RE,
    append_remote_mutation_path as _append_remote_mutation_path,
    collect_remote_mutator_operands as _collect_remote_mutator_operands,
    collect_remote_redirection_targets as _collect_remote_redirection_targets,
    guess_remote_deletion_directory_empty_checks as _guess_remote_deletion_directory_empty_checks,
    guess_remote_deletion_paths as _guess_remote_deletion_paths,
    guess_remote_mutation_paths as _guess_remote_mutation_paths,
    normalize_remote_deletion_operand as _normalize_remote_deletion_operand,
    normalize_remote_mutation_operand as _normalize_remote_mutation_operand,
    python_open_write_mutation as _python_open_write_mutation,
    remote_deletion_glob_empty_check as _remote_deletion_glob_empty_check,
    remote_path_is_known_directory as _remote_path_is_known_directory,
    remote_path_should_be_ignored as _remote_path_should_be_ignored,
    remote_shell_command_lines as _remote_shell_command_lines,
    shell_command_name as _shell_command_name,
)
from ..models.tool_result import ToolEnvelope
from ..challenge_progress import record_code_change
from ..normalization import dedupe_keep_tail
from ..recovery_metrics import increment_metric
from ..state_schema import MemoryEntry
from .artifact_tracking import consolidate_shell_attempt_family as _consolidate_shell_attempt_family
from .artifact_read_ledger import record_artifact_read_ledger as _record_artifact_read_ledger
from ..tools.shell_support import _REMOTE_INSTALLER_PREFLIGHT_KEY
from .tool_result_evidence import record_evidence as _record_evidence
from .tool_result_support import (
    auto_mirror_session_anchor as _auto_mirror_session_anchor,
    invalidate_file_read_cache as _invalidate_file_read_cache,
    is_small_model as _is_small_model,
    maybe_support_claim_from_evidence as _maybe_support_claim_from_evidence,
)
from .touched_symbols import (
    SYMBOL_CAPTURE_LIMIT,
    extract_symbol_candidates_from_file,
    extract_symbol_candidates_from_path,
    extract_symbol_candidates_from_text,
)
from .tool_result_verification import (
    _annotate_verifier_artifact,
    _store_verifier_verdict,
    assess_remote_mutation_verification,
)
from .verifier_monitor import track_verifier_rejection
from ..shell_utils import (
    file_read_cache_key as _file_read_cache_key,
    ssh_file_read_cache_key as _ssh_file_read_cache_key,
    strip_benign_shell_redirections as _strip_benign_shell_redirections,
)
from ..tools.fs import is_file_mutating_tool
from ..tools.ssh_files import SSH_FILE_MUTATING_TOOLS

def _should_auto_record_known_fact(tool_name: str, result: ToolEnvelope) -> bool:
    if tool_name == "shell_exec":
        return False
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("skip_auto_fact_record"):
        return False
    if not result.success and (is_file_mutating_tool(tool_name) or tool_name in SSH_FILE_MUTATING_TOOLS):
        return False
    return True

def _supersede_prior_read_artifacts(
    service: Any,
    *,
    new_artifact_id: str,
    tool_name: str,
    path: str,
    host: str | None = None,
) -> None:
    """Mark older read artifacts for the same path as superseded by the new one."""
    if not new_artifact_id or not path:
        return
    normalized_path = Path(path).as_posix().lower()
    normalized_host = str(host or "").strip().lower()
    for artifact_id, artifact in list(service.harness.state.artifacts.items()):
        if artifact_id == new_artifact_id:
            continue
        art_tool = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if art_tool != tool_name:
            continue
        metadata = getattr(artifact, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        art_path = str(metadata.get("path") or "").strip()
        if not art_path:
            args = metadata.get("arguments")
            if isinstance(args, dict):
                art_path = str(args.get("path") or "").strip()
        if not art_path:
            art_path = str(getattr(artifact, "source", "") or "").strip()
        art_host = str(metadata.get("host") or "").strip().lower()
        if not art_host:
            args = metadata.get("arguments")
            if isinstance(args, dict):
                art_host = str(args.get("host") or "").strip().lower()
        if Path(art_path).as_posix().lower() == normalized_path:
            if normalized_host and art_host and art_host != normalized_host:
                continue
            metadata["superseded_by"] = new_artifact_id


def _mark_prior_read_artifacts_stale(
    service: Any,
    *,
    path: str,
    reason: str = "file_mutated",
) -> None:
    """Mark prior file_read artifacts for the same path as stale after a mutation.

    Unlike superseding (which happens when a newer read artifact replaces an
    older one), staleness is used when the live file has been modified by a
    patch or write and the old snapshot no longer reflects reality.
    """
    if not path:
        return
    normalized_path = Path(path).as_posix().lower()
    for artifact_id, artifact in list(service.harness.state.artifacts.items()):
        art_tool = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if art_tool not in {"file_read", "ssh_file_read"}:
            continue
        metadata = getattr(artifact, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        art_path = str(metadata.get("path") or "").strip()
        if not art_path:
            args = metadata.get("arguments")
            if isinstance(args, dict):
                art_path = str(args.get("path") or "").strip()
        if not art_path:
            art_path = str(getattr(artifact, "source", "") or "").strip()
        if Path(art_path).as_posix().lower() == normalized_path:
            if metadata.get("superseded_by"):
                continue
            metadata["stale"] = True
            metadata["artifact_stale_reason"] = reason
            metadata["authoritative_path"] = art_path
            staleness_index = service.harness.state.scratchpad.setdefault("_artifact_staleness", {})
            if isinstance(staleness_index, dict) and artifact_id:
                staleness_index[artifact_id] = {
                    "stale": True,
                    "reason": reason,
                    "paths": [art_path],
                }


def _maybe_emit_artifact_read_eof_overread_nudge(
    service: Any,
    *,
    result: ToolEnvelope,
    artifact: Any,
) -> None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not bool(metadata.get("eof_overread")):
        return

    artifact_id = str(metadata.get("artifact_id") or getattr(artifact, "artifact_id", "") or "").strip()
    requested_start = int(metadata.get("requested_start_line") or metadata.get("line_start") or 0)
    total_lines = int(metadata.get("artifact_total_lines") or metadata.get("total_lines") or 0)
    if not artifact_id or requested_start <= 0 or total_lines <= 0:
        return

    signature = f"{artifact_id}:{requested_start}:{total_lines}"
    scratchpad = getattr(service.harness.state, "scratchpad", {})
    prior = scratchpad.get("_artifact_read_eof_overread_nudges", [])
    if isinstance(prior, list) and signature in prior:
        return

    if isinstance(prior, list):
        scratchpad["_artifact_read_eof_overread_nudges"] = dedupe_keep_tail(prior + [signature], limit=16)
    else:
        scratchpad["_artifact_read_eof_overread_nudges"] = [signature]

    model_note = (
        " This is a strong hallucination signal for the current small model: trust the reported EOF instead of inventing more lines."
        if _is_small_model(service.harness)
        else ""
    )
    content = (
        f"`artifact_read` asked for unseen lines past EOF on artifact `{artifact_id}`: "
        f"requested `start_line={requested_start}` but the artifact only has `{total_lines}` lines."
        f"{model_note} Do not call `artifact_read` again past EOF. "
        "Use the evidence already in context, synthesize the answer, or choose a different tool."
    )
    service.harness.state.append_message(
        ConversationMessage(
            role="system",
            content=content,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "artifact_read_eof_overread",
                "artifact_id": artifact_id,
                "requested_start_line": requested_start,
                "artifact_total_lines": total_lines,
            },
        )
    )
    service.harness._runlog(
        "artifact_read_eof_overread_nudge",
        "nudged model after reading past artifact EOF",
        artifact_id=artifact_id,
        requested_start_line=requested_start,
        artifact_total_lines=total_lines,
    )


def _mark_verifier_stale_after_file_change(
    service: Any,
    *,
    tool_name: str,
    paths: list[str],
) -> None:
    state = getattr(service.harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if state is None or not isinstance(scratchpad, dict):
        return
    current_verifier = getattr(state, "current_verifier_verdict", None)
    verifier = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    if not isinstance(verifier, dict) or not verifier:
        return
    clean_paths = [str(path).strip() for path in paths if str(path).strip()]
    scratchpad[_STALE_VERIFIER_KEY] = {
        "reason": "file_changed_after_verifier",
        "tool_name": tool_name,
        "paths": clean_paths,
        "prior_verdict": dict(verifier),
    }


def _observe_remote_installer_preflight_check(
    service: Any,
    *,
    result: ToolEnvelope,
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


def _remember_session_ssh_target(
    service: Any,
    *,
    tool_name: str = "",
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    if not isinstance(arguments, dict):
        return
    host = str(arguments.get("host") or "").strip().lower()
    user = str(arguments.get("user") or "").strip()
    if not host or not user:
        return

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not (
        result.success
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    ):
        return
    reached_remote_host = (
        result.success
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    )

    targets = service.harness.state.scratchpad.setdefault("_session_ssh_targets", {})
    if not isinstance(targets, dict):
        targets = {}
        service.harness.state.scratchpad["_session_ssh_targets"] = targets

    remembered: dict[str, Any] = {
        "host": host,
        "user": user,
    }
    existing = targets.get(host)
    if isinstance(existing, dict) and existing.get("confirmed"):
        remembered["confirmed"] = True
    elif reached_remote_host:
        remembered["confirmed"] = True
    password = str(arguments.get("password") or "").strip()
    if password:
        remembered["password"] = password
    identity_file = str(arguments.get("identity_file") or "").strip()
    if identity_file:
        remembered["identity_file"] = identity_file
    port = arguments.get("port")
    if isinstance(port, int):
        remembered["port"] = port
    validated_tool = str(tool_name or "").strip()
    validated_path = str(
        arguments.get("path")
        or metadata.get("path")
        or (
            result.output.get("path")
            if isinstance(result.output, dict)
            else ""
        )
        or ""
    ).strip()
    if reached_remote_host and validated_tool:
        existing_tools = existing.get("validated_tools") if isinstance(existing, dict) else []
        if not isinstance(existing_tools, list):
            existing_tools = []
        remembered["validated_tools"] = dedupe_keep_tail(
            [str(item).strip() for item in existing_tools if str(item).strip()] + [validated_tool],
            limit=6,
        )
        remembered["last_success_tool"] = validated_tool
        prior_success_count = 0
        if isinstance(existing, dict):
            try:
                prior_success_count = max(0, int(existing.get("success_count") or 0))
            except (TypeError, ValueError):
                prior_success_count = 0
        remembered["success_count"] = prior_success_count + 1
    if reached_remote_host and validated_path:
        existing_paths = existing.get("validated_paths") if isinstance(existing, dict) else []
        if not isinstance(existing_paths, list):
            existing_paths = []
        remembered["validated_paths"] = dedupe_keep_tail(
            [str(item).strip() for item in existing_paths if str(item).strip()] + [validated_path],
            limit=8,
        )
        remembered["last_validated_path"] = validated_path
    targets[host] = remembered


def _remember_web_search_results(
    service: Any,
    *,
    result: ToolEnvelope,
    artifact: Any,
) -> None:
    output = result.output if isinstance(result.output, dict) else {}
    results = output.get("results")
    if not isinstance(results, list):
        return

    result_ids = [
        str(item.get("result_id") or "").strip()
        for item in results
        if isinstance(item, dict) and str(item.get("result_id") or "").strip()
    ]
    fetch_ids = [
        str(item.get("fetch_id") or "").strip()
        for item in results
        if isinstance(item, dict) and str(item.get("fetch_id") or "").strip()
    ]
    if not result_ids:
        return

    scratchpad = getattr(service.harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        service.harness.state.scratchpad = scratchpad

    by_artifact = scratchpad.get("_web_search_artifact_results")
    if not isinstance(by_artifact, dict):
        by_artifact = {}
        scratchpad["_web_search_artifact_results"] = by_artifact

    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip()
    if artifact_id:
        by_artifact[artifact_id] = list(result_ids)
        scratchpad["_web_last_search_artifact_id"] = artifact_id
        if isinstance(getattr(artifact, "metadata", None), dict):
            artifact.metadata["web_result_ids"] = list(result_ids)
            artifact.metadata["web_fetch_ids"] = list(fetch_ids)
            artifact.metadata["web_result_count"] = len(result_ids)

    scratchpad["_web_last_search_result_ids"] = list(result_ids)
    scratchpad["_web_last_search_fetch_ids"] = list(fetch_ids)


def _record_remote_mutation_requirement(
    service: Any,
    *,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    if not result.success or not isinstance(arguments, dict):
        return
    command = str(arguments.get("command") or "").strip()
    mutation_command = _strip_benign_shell_redirections(command, preserve_newlines=True)
    if not mutation_command or _REMOTE_MUTATING_COMMAND_RE.search(mutation_command) is None:
        return

    host = str(arguments.get("host") or arguments.get("target") or "").strip()
    is_deletion = _REMOTE_DELETION_RE.search(mutation_command) is not None
    guessed_paths = _guess_remote_mutation_paths(mutation_command, deletion=is_deletion)
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
    result: ToolEnvelope,
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
            path = ""
            host = ""
            if isinstance(result.metadata, dict):
                path = str(result.metadata.get("path") or "").strip()
                host = str(result.metadata.get("host") or "").strip().lower()
            if not path and isinstance(arguments, dict):
                path = str(arguments.get("path") or "").strip()
            if not host and isinstance(arguments, dict):
                host = str(arguments.get("host") or arguments.get("target") or "").strip().lower()
            requirement_host = str(requirement.get("host") or "").strip().lower()
            if requirement_host and host and host != requirement_host:
                return
            guessed_paths = [str(item) for item in requirement.get("guessed_paths", []) if str(item).strip()]
            if guessed_paths and path and path not in guessed_paths:
                return
            failure_markers = " ".join(
                [
                    str(result.error or ""),
                    str((result.metadata or {}).get("message") or "") if isinstance(result.metadata, dict) else "",
                    str((result.metadata or {}).get("error_kind") or "") if isinstance(result.metadata, dict) else "",
                ]
            ).lower()
            if (
                "no such file" in failure_markers
                or "not found" in failure_markers
                or "file_not_found" in failure_markers
            ):
                _mark_remote_mutation_path_verified(requirement, path)
                if _remote_mutation_requirement_satisfied(requirement):
                    service.harness.state.scratchpad.pop(_REMOTE_MUTATION_VERIFICATION_KEY, None)
        return

    if not result.success:
        return

    path = ""
    host = ""
    if isinstance(result.metadata, dict):
        path = str(result.metadata.get("path") or "").strip()
        host = str(result.metadata.get("host") or "").strip().lower()
    if not path and isinstance(arguments, dict):
        path = str(arguments.get("path") or "").strip()
    if not host and isinstance(arguments, dict):
        host = str(arguments.get("host") or arguments.get("target") or "").strip().lower()
    requirement_host = str(requirement.get("host") or "").strip().lower()
    if requirement_host and host and host != requirement_host:
        return
    guessed_paths = [str(item) for item in requirement.get("guessed_paths", []) if str(item).strip()]
    if guessed_paths and path and path not in guessed_paths:
        return
    if tool_name == "ssh_file_read":
        output = result.output if isinstance(result.output, dict) else {}
        content = str(output.get("content") or "")
        if isinstance(requirement.get("verification_patterns"), dict):
            if not _readback_content_satisfies_requirement(requirement, content):
                return
        elif not (guessed_paths and path and path in guessed_paths):
            return
    _mark_remote_mutation_path_verified(requirement, path)
    if _remote_mutation_requirement_satisfied(requirement):
        service.harness.state.scratchpad.pop(_REMOTE_MUTATION_VERIFICATION_KEY, None)


def _mark_remote_mutation_path_verified(requirement: dict[str, Any], path: str) -> None:
    mark_remote_mutation_path_verified(requirement, path)


def _mark_remote_mutation_directory_verified(requirement: dict[str, Any], path: str) -> None:
    mark_remote_mutation_directory_verified(requirement, path)


def _remote_mutation_requirement_satisfied(requirement: dict[str, Any]) -> bool:
    return remote_mutation_requirement_satisfied(requirement)


def _remote_mutation_directory_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
    return remote_mutation_directory_checks(requirement)


def _record_failed_verification_attempt(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
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
    path = ""
    host = ""
    if isinstance(result.metadata, dict):
        path = str(result.metadata.get("path") or "").strip()
        host = str(result.metadata.get("host") or "").strip().lower()
    if not path and isinstance(arguments, dict):
        path = str(arguments.get("path") or "").strip()
    if not host and isinstance(arguments, dict):
        host = str(arguments.get("host") or arguments.get("target") or "").strip().lower()
    requirement_host = str(requirement.get("host") or "").strip().lower()
    if requirement_host and host and host != requirement_host:
        return
    guessed_paths = [str(item) for item in requirement.get("guessed_paths", []) if str(item).strip()]
    if guessed_paths and path and path not in guessed_paths:
        return
    requirement["failed_verification_attempts"] = requirement.get("failed_verification_attempts", 0) + 1


def _maybe_emit_bounded_region_trap_nudge(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    """Inject a nudge when ssh_file_replace_between fails because the region is gone,
    suggesting the model read back the file instead of retrying the same bounds."""
    if tool_name != "ssh_file_replace_between":
        return
    if result.success:
        return
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    error_kind = str(metadata.get("error_kind") or "").strip()
    error_message = str(result.error or "").strip()
    is_bounded_region_not_found = (
        error_kind == "bounded_region_not_found"
        or "Remote bounded region was not found" in error_message
    )
    if not is_bounded_region_not_found:
        return

    path = ""
    host = ""
    if isinstance(result.metadata, dict):
        path = str(result.metadata.get("path") or "").strip()
        host = str(result.metadata.get("host") or "").strip().lower()
    if not path and isinstance(arguments, dict):
        path = str(arguments.get("path") or "").strip()
    if not host and isinstance(arguments, dict):
        host = str(arguments.get("host") or arguments.get("target") or "").strip().lower()

    # First, check for the small-file anti-pattern: file is small and was recently read.
    # In that case, suggest ssh_file_write instead of retrying bounds.
    _maybe_emit_small_file_rewrite_nudge(service, path=path, host=host, arguments=arguments)

    # Then, the original bounded-region trap nudge for ssh_exec mutations.
    requirement = service.harness.state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(requirement, dict):
        return

    requirement_host = str(requirement.get("host") or "").strip().lower()
    if requirement_host and host and host != requirement_host:
        return
    guessed_paths = [str(item) for item in requirement.get("guessed_paths", []) if str(item).strip()]
    if guessed_paths and path and path not in guessed_paths:
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

    # Check if we have a recent ssh_file_read on this exact path+host
    recent_read_size = _recent_ssh_file_read_size(service, path=path, host=host)
    if recent_read_size is None or recent_read_size == 0:
        return

    # Only nudge for small files (< 1KB)
    SMALL_FILE_THRESHOLD = 1024
    if recent_read_size >= SMALL_FILE_THRESHOLD:
        return

    # Check if replacement_text is large relative to file size
    replacement_text = ""
    if isinstance(arguments, dict):
        replacement_text = str(arguments.get("replacement_text") or "").strip()
    if not replacement_text:
        return

    # If replacement is >50% of the file, it's a rewrite masquerading as a patch
    if len(replacement_text.encode("utf-8")) <= recent_read_size * 0.5:
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


def _readback_content_satisfies_requirement(requirement: dict[str, Any], content: str) -> bool:
    patterns = requirement.get("verification_patterns")
    if not isinstance(patterns, dict):
        return False
    old_absent = [str(item) for item in patterns.get("old_absent", []) if str(item)]
    new_present = [str(item) for item in patterns.get("new_present", []) if str(item)]
    if not old_absent and not new_present:
        return False
    if any(marker in content for marker in old_absent):
        return False
    if any(marker not in content for marker in new_present):
        return False
    return True


def _handle_remote_mutation_verifier_result(
    service: Any,
    *,
    result: ToolEnvelope,
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
            _mark_remote_mutation_directory_verified(requirement, verified_directory)
            if _remote_mutation_requirement_satisfied(requirement):
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


def _emit_context_invalidation(
    service: Any,
    *,
    reason: str,
    paths: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if reason == "file_changed" and paths:
        from ..graph.tool_call_parser import allow_repeated_tool_call_once
        for path_entry in paths:
            path_str = str(path_entry or "").strip()
            if not path_str:
                continue
            if ":" in path_str:
                host, sep, remote_path = path_str.partition(":")
                if sep and host and remote_path:
                    allow_repeated_tool_call_once(
                        service.harness,
                        "ssh_file_read",
                        {"host": host, "path": remote_path},
                    )
            else:
                allow_repeated_tool_call_once(
                    service.harness,
                    "file_read",
                    {"path": path_str},
                )
    event = service.harness.state.invalidate_context(
        reason=reason,
        paths=paths,
        details=details,
    )
    runlog = getattr(service.harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "context_invalidated",
            "context invalidation applied",
            reason=event.get("reason", reason),
            paths=event.get("paths", []),
            invalidated_fact_count=event.get("invalidated_fact_count", 0),
            invalidated_memory_count=event.get("invalidated_memory_count", 0),
            invalidated_facts=event.get("invalidated_facts", []),
            invalidated_memory_ids=event.get("invalidated_memory_ids", []),
            invalidated_turn_bundle_count=event.get("invalidated_turn_bundle_count", 0),
            invalidated_turn_bundle_ids=event.get("invalidated_turn_bundle_ids", []),
            invalidated_brief_count=event.get("invalidated_brief_count", 0),
            invalidated_brief_ids=event.get("invalidated_brief_ids", []),
            invalidated_summary_count=event.get("invalidated_summary_count", 0),
            invalidated_summary_ids=event.get("invalidated_summary_ids", []),
            invalidated_artifact_count=event.get("invalidated_artifact_count", 0),
            invalidated_artifact_ids=event.get("invalidated_artifact_ids", []),
            invalidated_observation_count=event.get("invalidated_observation_count", 0),
            invalidated_observation_ids=event.get("invalidated_observation_ids", []),
            details=event.get("details", {}),
        )


def _record_touched_symbols_from_mutation(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    artifact: Any,
    mutated_path: str,
) -> None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    text_chunks: list[str] = []
    if isinstance(arguments, dict):
        for key in ("content", "replacement_text", "target_text", "patch", "diff"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                text_chunks.append(value)
        for key in ("section_name", "section_id", "next_section_name"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                text_chunks.append(f"def {value.strip()}(): pass")
    for key in ("content", "replacement_text", "target_text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            text_chunks.append(value)
    touched_symbols = metadata.get("touched_symbols")
    if isinstance(touched_symbols, list):
        existing_symbols = [str(item).strip() for item in touched_symbols if str(item).strip()]
    else:
        existing_symbols = []
    for key in ("replacement_text_preview", "target_text_preview"):
        preview_payload = metadata.get(key)
        if isinstance(preview_payload, dict):
            preview = str(preview_payload.get("preview") or "").replace("\\n", "\n").strip()
            if preview:
                text_chunks.append(preview)

    candidates: list[str] = []
    for chunk in text_chunks:
        candidates.extend(extract_symbol_candidates_from_text(chunk))

    path_candidate = str(mutated_path or metadata.get("path") or "").strip()
    if not path_candidate and isinstance(arguments, dict):
        path_candidate = str(arguments.get("path") or "").strip()
    if not path_candidate and artifact is not None:
        path_candidate = str(getattr(artifact, "source", "") or "").strip()
    candidates.extend(extract_symbol_candidates_from_path(path_candidate))
    candidates.extend(extract_symbol_candidates_from_file(path_candidate, cwd=str(service.harness.state.cwd or "")))

    deduped = dedupe_keep_tail(existing_symbols + [token for token in candidates if token], limit=SYMBOL_CAPTURE_LIMIT)
    if not deduped:
        return
    existing = service.harness.state.scratchpad.get("_touched_symbols")
    existing_list = [str(item).strip() for item in existing] if isinstance(existing, list) else []
    merged = dedupe_keep_tail(existing_list + deduped, limit=SYMBOL_CAPTURE_LIMIT)
    if not merged:
        return
    service.harness.state.scratchpad["_touched_symbols"] = merged
    runlog = getattr(service.harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "coding_symbols_captured",
            "captured touched symbols from file mutation",
            tool_name=tool_name,
            path=path_candidate,
            symbol_count=len(deduped),
            symbols=deduped[:8],
        )
_CRITICAL_ERROR_RE = re.compile(
    r"^(?:\s*[-*]+\s*)?(?:Error|ERROR|error|fatal|FATAL|Failed|FAILED|FAILURE|failure|Exception|EXCEPTION)\b.*",
    re.MULTILINE,
)


def _extract_and_pin_critical_errors(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
) -> None:
    """Extract critical error lines from tool output and pin them in working memory."""
    text = ""
    if not result.success and result.error:
        text = str(result.error)
    elif isinstance(result.output, dict):
        if tool_name in {"shell_exec", "ssh_exec"}:
            stdout = str(result.output.get("stdout") or "")
            stderr = str(result.output.get("stderr") or "")
            text = stdout + "\n" + stderr
        else:
            text = str(result.output.get("content") or result.output.get("stdout") or result.output.get("stderr") or "")
    elif isinstance(result.output, str):
        text = result.output

    if not text:
        return

    matches = _CRITICAL_ERROR_RE.findall(text)
    if not matches:
        return

    # Deduplicate and limit
    seen: set[str] = set()
    errors: list[str] = []
    for line in matches:
        normalized = line.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        clipped = line.strip()[:280]
        if clipped:
            errors.append(clipped)
        if len(errors) >= 2:
            break

    if not errors:
        return

    wm = service.harness.state.working_memory
    current_step = service.harness.state.step_count
    current_phase = service.harness.state.current_phase
    new_facts: list[str] = []
    new_meta: list[MemoryEntry] = []

    for err in errors:
        prefix = f"CRITICAL: {err}"
        if prefix not in wm.known_facts:
            new_facts.append(prefix)
            new_meta.append(
                MemoryEntry(
                    content=prefix,
                    created_at_step=current_step,
                    created_phase=current_phase,
                    freshness="pinned",
                )
            )

    if new_facts:
        wm.known_facts = dedupe_keep_tail(wm.known_facts + new_facts, limit=12)
        # Align meta so pinned entries survive invalidation
        existing_lookup = {m.content: m for m in wm.known_fact_meta}
        aligned: list[MemoryEntry] = []
        for fact in wm.known_facts:
            if fact in existing_lookup:
                aligned.append(existing_lookup[fact])
            else:
                # Find matching new meta
                match = next((m for m in new_meta if m.content == fact), None)
                if match:
                    aligned.append(match)
                else:
                    aligned.append(
                        MemoryEntry(
                            content=fact,
                            created_at_step=current_step,
                            created_phase=current_phase,
                        )
                    )
        wm.known_fact_meta = aligned


def _record_stderr_signature_circuit_breaker(service: Any, *, tool_name: str, result: ToolEnvelope) -> None:
    if tool_name not in {"shell_exec", "ssh_exec"}:
        return
    stderr = ""
    if isinstance(result.output, dict):
        stderr = str(result.output.get("stderr") or "")
    if not stderr and result.error:
        stderr = str(result.error)
    signature_line = ""
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped:
            signature_line = re.sub(r"\s+", " ", stripped)[:240]
            break
    if not signature_line:
        return
    key = hashlib.sha1(signature_line.lower().encode("utf-8", errors="replace")).hexdigest()[:12]
    state = service.harness.state
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    counts = scratchpad.setdefault("_stderr_signature_counts", {})
    if not isinstance(counts, dict):
        counts = {}
        scratchpad["_stderr_signature_counts"] = counts
    count = int(counts.get(key, 0) or 0) + 1
    counts[key] = count
    if count < 2:
        return
    scratchpad["_stderr_signature_circuit_breaker"] = {
        "signature": signature_line,
        "count": count,
        "tool_name": tool_name,
        "next_required_action": "Use a different repair strategy; do not retry the same command/fix against this stderr.",
    }
    state.recent_errors.append(f"stderr_signature_circuit_breaker: {signature_line}")


def _update_subtask_ledger_from_verifier(service: Any, verifier_verdict: dict[str, Any] | None) -> None:
    if not isinstance(verifier_verdict, dict) or not verifier_verdict:
        return
    harness = getattr(service, "harness", None)
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "subtask_ledger_enabled", True)):
        return
    ledger_service = getattr(harness, "subtask_ledger", None)
    if ledger_service is None:
        return
    try:
        ledger_service.import_plan_if_needed()
        active = ledger_service.infer_or_create_active_subtask()
        command = str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip()
        verdict = str(verifier_verdict.get("verdict") or "").strip().lower()
        if command:
            ledger_service.attach_evidence(active.subtask_id, f"verifier {verdict or 'unknown'}: {command}")
        if verdict == "pass":
            if any(item in {"verifier_failed", "test_failed"} for item in getattr(active, "failure_classes", [])):
                increment_metric(harness.state, "verifier_fail_then_success_count")
            ledger_service.mark_done_if_verified(active.subtask_id, verifier_verdict)
    except Exception:
        return


def _is_dry_run_invariant_violation(tool_name: str, result: ToolEnvelope) -> bool:
    if not result.success:
        return False
    if tool_name not in SSH_FILE_MUTATING_TOOLS:
        return False
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    dry_run = bool(metadata.get("dry_run"))
    changed = metadata.get("changed")
    return dry_run and bool(changed)


def _apply_shell_exec_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for shell_exec and ssh_exec results."""
    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip() if artifact else ""
    if artifact_id:
        _consolidate_shell_attempt_family(state=service.harness.state, artifact_id=artifact_id, result=result)
        _record_stderr_signature_circuit_breaker(service, tool_name=tool_name, result=result)
    if tool_name == "ssh_exec":
        _remember_session_ssh_target(service, tool_name=tool_name, result=result, arguments=arguments)
        _observe_remote_installer_preflight_check(service, result=result, arguments=arguments)
        _handle_remote_mutation_verifier_result(service, result=result, arguments=arguments)
        if not result.success:
            _record_failed_verification_attempt(service, tool_name=tool_name, result=result, arguments=arguments)
        _record_remote_mutation_requirement(service, result=result, arguments=arguments)


def _apply_ssh_file_verifier_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for SSH file verifier tools."""
    _remember_session_ssh_target(service, tool_name=tool_name, result=result, arguments=arguments)
    if not result.success:
        _record_failed_verification_attempt(service, tool_name=tool_name, result=result, arguments=arguments)
        _maybe_emit_bounded_region_trap_nudge(service, tool_name=tool_name, result=result, arguments=arguments)
    _clear_remote_mutation_requirement_from_tool(service, tool_name=tool_name, result=result, arguments=arguments)


def _apply_web_search_updates(
    service: Any,
    *,
    result: ToolEnvelope,
    artifact: Any,
) -> None:
    """Apply side-effects for web_search results."""
    _remember_web_search_results(service, result=result, artifact=artifact)


def _apply_file_read_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for file_read and ssh_file_read results."""
    if tool_name == "file_read":
        cache_key = _file_read_cache_key(service.harness.state.cwd, result.metadata)
        if cache_key:
            cache = service.harness.state.scratchpad.setdefault("file_read_cache", {})
            if isinstance(cache, dict):
                cache[cache_key] = artifact.artifact_id
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        read_path = str(metadata.get("path") or "").strip()
        if not read_path and isinstance(arguments, dict):
            read_path = str(arguments.get("path") or "").strip()
        if read_path:
            _supersede_prior_read_artifacts(
                service,
                new_artifact_id=artifact.artifact_id,
                tool_name="file_read",
                path=read_path,
            )
    elif tool_name == "ssh_file_read":
        cache_key = _ssh_file_read_cache_key(result.metadata)
        if cache_key:
            cache = service.harness.state.scratchpad.setdefault("ssh_file_read_cache", {})
            if isinstance(cache, dict):
                cache[cache_key] = artifact.artifact_id
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        read_path = str(metadata.get("path") or "").strip()
        read_host = str(metadata.get("host") or "").strip()
        if not read_path and isinstance(arguments, dict):
            read_path = str(arguments.get("path") or "").strip()
            read_host = str(arguments.get("host") or "").strip()
        if read_path:
            _supersede_prior_read_artifacts(
                service,
                new_artifact_id=artifact.artifact_id,
                tool_name="ssh_file_read",
                path=read_path,
                host=read_host,
            )


def _apply_file_mutation_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    artifact: Any,
) -> None:
    """Apply side-effects for local file mutating tools."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if bool(metadata.get("dry_run")) or metadata.get("changed") is False:
        return
    mutated_path = ""
    if isinstance(result.metadata, dict):
        mutated_path = str(result.metadata.get("path") or "").strip()
    if not mutated_path and isinstance(arguments, dict):
        mutated_path = str(arguments.get("path") or "").strip()
    if mutated_path:
        _invalidate_file_read_cache(service.harness, mutated_path)
        _mark_prior_read_artifacts_stale(
            service,
            path=mutated_path,
            reason=f"{tool_name}_applied",
        )
        _emit_context_invalidation(
            service,
            reason="file_changed",
            paths=[mutated_path],
            details={
                "tool_name": tool_name,
                "state_change": f"File changed: {mutated_path}",
            },
        )
        _mark_verifier_stale_after_file_change(
            service,
            tool_name=tool_name,
            paths=[mutated_path],
        )
    record_code_change(
        service.harness.state,
        tool_name=tool_name,
        path=mutated_path,
        changed=True,
    )
    _record_touched_symbols_from_mutation(
        service,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
        artifact=artifact,
        mutated_path=mutated_path,
    )


def _apply_ssh_file_mutation_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    artifact: Any,
) -> None:
    """Apply side-effects for SSH file mutating tools."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if bool(metadata.get("dry_run")) or metadata.get("changed") is False:
        return
    mutated_path = ""
    host = ""
    if isinstance(result.metadata, dict):
        mutated_path = str(result.metadata.get("path") or "").strip()
        host = str(result.metadata.get("host") or "").strip().lower()
    if not mutated_path and isinstance(arguments, dict):
        mutated_path = str(arguments.get("path") or "").strip()
        host = str(arguments.get("host") or "").strip().lower()
    if mutated_path and host:
        cache = service.harness.state.scratchpad.get("ssh_file_read_cache")
        if isinstance(cache, dict):
            prefix = f"ssh://{host}{mutated_path}"
            for key in list(cache.keys()):
                if key.startswith(prefix):
                    cache.pop(key, None)
        _emit_context_invalidation(
            service,
            reason="file_changed",
            paths=[f"{host}:{mutated_path}"],
            details={
                "tool_name": tool_name,
                "state_change": f"Remote file changed: {host}:{mutated_path}",
            },
        )
        _mark_verifier_stale_after_file_change(
            service,
            tool_name=tool_name,
            paths=[f"{host}:{mutated_path}"],
        )
        from ..graph.tool_call_parser import allow_repeated_tool_call_once

        one_shot_args: dict[str, Any] = {"path": mutated_path, "host": host}
        if isinstance(arguments, dict):
            for conn_key in ("user", "password", "target", "port", "identity_file"):
                val = arguments.get(conn_key)
                if val is not None:
                    one_shot_args[conn_key] = val
        allow_repeated_tool_call_once(
            service.harness,
            "ssh_file_read",
            one_shot_args,
        )
    _record_touched_symbols_from_mutation(
        service,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
        artifact=artifact,
        mutated_path=mutated_path,
    )


def _apply_plan_and_read_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> None:
    """Apply side-effects for plan tools, memory_update, and artifact_read."""
    if tool_name in {"plan_set", "plan_step_update", "plan_request_execution", "plan_export"}:
        playbook_artifact_id = str(result.metadata.get("artifact_id", "") or "").strip()
        if playbook_artifact_id:
            service.harness.state.plan_artifact_id = playbook_artifact_id
            service.harness.state.plan_resolved = True
            service.harness.state.retrieval_cache = [playbook_artifact_id]

    if tool_name == "memory_update" and result.success:
        section = str(result.metadata.get("section", "")).strip().lower()
        if section == "plan":
            service.harness.state.plan_artifact_id = artifact.artifact_id
            service.harness.state.plan_resolved = True
    elif tool_name == "artifact_read" and result.success and artifact:
        artifact_id = str(result.metadata.get("artifact_id", "")).strip()
        if artifact_id:
            if artifact_id == service.harness.state.plan_artifact_id:
                service.harness.state.plan_resolved = True
            elif (
                not service.harness.state.plan_artifact_id
                and artifact.tool_name == "memory_update"
                and str(artifact.metadata.get("section", "")).strip().lower() == "plan"
            ):
                service.harness.state.plan_artifact_id = artifact_id
                service.harness.state.plan_resolved = True
            if result.metadata.get("truncated"):
                suppressed = service.harness.state.scratchpad.get("suppressed_truncated_artifact_ids", [])
                if isinstance(suppressed, list):
                    service.harness.state.scratchpad["suppressed_truncated_artifact_ids"] = dedupe_keep_tail(
                        suppressed + [artifact_id],
                        limit=12,
                    )
                else:
                    service.harness.state.scratchpad["suppressed_truncated_artifact_ids"] = [artifact_id]
                service.harness._runlog(
                    "artifact_read_truncated",
                    "artifact_read returned a truncated slice",
                    artifact_id=artifact_id,
                    total_lines=result.metadata.get("total_lines"),
                    start_line=result.metadata.get("start_line"),
                    end_line=result.metadata.get("end_line"),
                    marker_included=True,
                )
        _record_artifact_read_ledger(
            service,
            result=result,
            arguments=arguments,
            artifact=artifact,
        )
        _maybe_emit_artifact_read_eof_overread_nudge(
            service,
            result=result,
            artifact=artifact,
        )


def _apply_verifier_and_evidence_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
) -> Any:
    """Store verifier verdict, update ledger, emit invalidations, and record evidence."""
    verifier_verdict = _store_verifier_verdict(
        service.harness.state,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
    )
    loop_info = track_verifier_rejection(service.harness.state, verifier_verdict)
    if loop_info.get("is_loop"):
        service.harness._runlog(
            "verifier_loop_detected",
            "Verifier rejecting task_complete repeatedly",
            rejection_count=loop_info["rejection_count"],
            last_verdict=loop_info["verdict"],
        )
    _update_subtask_ledger_from_verifier(service, verifier_verdict)
    if isinstance(verifier_verdict, dict):
        verdict_label = str(verifier_verdict.get("verdict") or "").strip().lower()
        if verdict_label == "fail":
            _emit_context_invalidation(
                service,
                reason="verifier_failed",
                details={
                    "command": str(verifier_verdict.get("command") or ""),
                    "target": str(verifier_verdict.get("target") or ""),
                    "failure_mode": str(getattr(service.harness.state, "last_failure_class", "") or ""),
                    "state_change": "Verifier failure invalidated optimistic context",
                },
            )
    if artifact and isinstance(verifier_verdict, dict) and verifier_verdict:
        _annotate_verifier_artifact(artifact, verifier_verdict=verifier_verdict)
    return verifier_verdict


def _apply_finalization_updates(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
    verifier_verdict: Any,
) -> Any:
    """Record evidence, update known facts, and return the evidence object."""
    evidence = _record_evidence(
        service,
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        operation_id=operation_id,
    )
    _maybe_support_claim_from_evidence(
        state=service.harness.state,
        tool_name=tool_name,
        result=result,
        evidence=evidence,
        harness=service.harness,
        operation_id=operation_id,
    )
    _auto_mirror_session_anchor(service.harness, tool_name=tool_name, result=result, arguments=arguments)

    if _should_auto_record_known_fact(tool_name, result):
        fact_label = evidence.statement or (artifact.summary if artifact else "") or tool_name
        prefix = f"{tool_name}: "
        if fact_label.startswith(prefix):
            fact_label = fact_label[len(prefix):]
        service.harness.state.working_memory.known_facts = dedupe_keep_tail(
            service.harness.state.working_memory.known_facts + [f"{tool_name}: {fact_label}"],
            limit=12,
        )
    _extract_and_pin_critical_errors(service, tool_name=tool_name, result=result, artifact=artifact)
    service.harness.state.recent_errors = []
    return evidence


def apply_artifact_success_outcome(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
) -> Any:
    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip() if artifact else ""
    if artifact:
        service.harness.state.artifacts[artifact.artifact_id] = artifact
        service.harness.state.retrieval_cache = [artifact.artifact_id]
    if _is_dry_run_invariant_violation(tool_name, result):
        result.success = False
        result.status = "failed"
        if not result.error:
            result.error = "Tool returned contradictory metadata: dry_run=true with changed=yes. This is an invariant violation."
        result.metadata = {**(result.metadata if isinstance(result.metadata, dict) else {}), "dry_run_invariant_violation": True}
        service.harness._runlog(
            "dry_run_invariant_violation",
            "SSH mutating tool returned dry_run=true and changed=yes; treating as failure",
            tool_name=tool_name,
        )
    if tool_name in {"shell_exec", "ssh_exec"}:
        _apply_shell_exec_updates(service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments)
    elif tool_name in _SSH_FILE_VERIFIER_TOOLS:
        _apply_ssh_file_verifier_updates(service, tool_name=tool_name, result=result, arguments=arguments)
    elif tool_name == "web_search" and result.success:
        _apply_web_search_updates(service, result=result, artifact=artifact)

    verifier_verdict = _apply_verifier_and_evidence_updates(
        service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments
    )

    if tool_name in {"file_read", "ssh_file_read"} and result.success and artifact:
        _apply_file_read_updates(service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments)
    elif is_file_mutating_tool(tool_name) and result.success:
        _apply_file_mutation_updates(service, tool_name=tool_name, result=result, arguments=arguments, artifact=artifact)
    elif tool_name in SSH_FILE_MUTATING_TOOLS and result.success:
        _apply_ssh_file_mutation_updates(service, tool_name=tool_name, result=result, arguments=arguments, artifact=artifact)

    _apply_plan_and_read_updates(service, tool_name=tool_name, result=result, artifact=artifact, arguments=arguments)

    return _apply_finalization_updates(
        service,
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        arguments=arguments,
        operation_id=operation_id,
        verifier_verdict=verifier_verdict,
    )
