from __future__ import annotations

import re
import shlex
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..challenge_progress import record_code_change
from ..normalization import dedupe_keep_tail
from ..recovery_metrics import increment_metric
from ..state_schema import MemoryEntry
from .artifact_tracking import consolidate_shell_attempt_family as _consolidate_shell_attempt_family
from ..tools.shell_support import _REMOTE_INSTALLER_PREFLIGHT_KEY
from .tool_result_evidence import record_evidence as _record_evidence
from .tool_result_support import (
    auto_mirror_session_anchor as _auto_mirror_session_anchor,
    invalidate_file_read_cache as _invalidate_file_read_cache,
    is_small_model as _is_small_model,
    maybe_support_claim_from_evidence as _maybe_support_claim_from_evidence,
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

_ARTIFACT_COVERAGE_SCRATCHPAD_KEY = "_artifact_read_coverage"


def _should_auto_record_known_fact(tool_name: str, result: ToolEnvelope) -> bool:
    if tool_name == "shell_exec":
        return False
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("skip_auto_fact_record"):
        return False
    if not result.success and (is_file_mutating_tool(tool_name) or tool_name in SSH_FILE_MUTATING_TOOLS):
        return False
    return True


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_artifact_read_ranges(ranges: Any) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    if not isinstance(ranges, list):
        return normalized
    for item in ranges:
        if isinstance(item, dict):
            start_line = _coerce_int_or_none(item.get("start_line"))
            end_line = _coerce_int_or_none(item.get("end_line"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start_line = _coerce_int_or_none(item[0])
            end_line = _coerce_int_or_none(item[1])
        else:
            continue
        if start_line is None or end_line is None or start_line < 1 or end_line < start_line:
            continue
        normalized.append((start_line, end_line))
    normalized.sort()
    merged: list[tuple[int, int]] = []
    for start_line, end_line in normalized:
        if not merged or start_line > merged[-1][1] + 1:
            merged.append((start_line, end_line))
            continue
        prior_start, prior_end = merged[-1]
        merged[-1] = (prior_start, max(prior_end, end_line))
    return merged


def _artifact_read_coverage_is_complete(*, ranges: Any, total_lines: int) -> bool:
    if total_lines < 1:
        return False
    normalized = _normalize_artifact_read_ranges(ranges)
    return bool(normalized and normalized[0][0] <= 1 and normalized[0][1] >= total_lines)


def _record_artifact_read_ledger(
    service: Any,
    *,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    artifact: Any,
) -> None:
    state = getattr(service.harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None) if state is not None else None
    if not isinstance(scratchpad, dict):
        return
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    artifact_id = str(metadata.get("artifact_id") or getattr(artifact, "artifact_id", "") or "").strip()
    if not artifact_id:
        return

    args = arguments if isinstance(arguments, dict) else {}
    requested_start = _coerce_int_or_none(args.get("start_line"))
    requested_end = _coerce_int_or_none(args.get("end_line"))
    start_line = _coerce_int_or_none(metadata.get("line_start", metadata.get("requested_start_line", requested_start)))
    end_line = _coerce_int_or_none(metadata.get("line_end", metadata.get("requested_end_line", requested_end)))
    total_lines = _coerce_int_or_none(metadata.get("total_lines", metadata.get("artifact_total_lines")))
    if start_line is None:
        return
    if end_line is None and total_lines is not None and not bool(metadata.get("truncated")):
        end_line = total_lines
    if end_line is None:
        return
    if total_lines is not None:
        end_line = min(end_line, total_lines)
    if start_line < 1 or end_line < start_line:
        return

    coverage = scratchpad.setdefault(_ARTIFACT_COVERAGE_SCRATCHPAD_KEY, {})
    if not isinstance(coverage, dict):
        coverage = {}
        scratchpad[_ARTIFACT_COVERAGE_SCRATCHPAD_KEY] = coverage
    entry = coverage.setdefault(artifact_id, {"ranges": []})
    if not isinstance(entry, dict):
        entry = {"ranges": []}
        coverage[artifact_id] = entry

    if total_lines is not None:
        entry["total_lines"] = total_lines
    entry["last_read_step"] = int(getattr(state, "step_count", 0) or 0)
    entry["truncated"] = bool(metadata.get("truncated"))
    if getattr(artifact, "source", ""):
        entry["source"] = str(getattr(artifact, "source", ""))
    output = result.output if isinstance(result.output, str) else ""
    if output:
        entry["preview"] = output[:1200]

    if bool(metadata.get("eof_overread")):
        entry["eof_overread"] = True
        return

    ranges = _normalize_artifact_read_ranges(entry.get("ranges", []))
    ranges.append((start_line, end_line))
    entry["ranges"] = [
        {"start_line": merged_start, "end_line": merged_end}
        for merged_start, merged_end in _normalize_artifact_read_ranges(ranges)
    ]
    total = _coerce_int_or_none(entry.get("total_lines"))
    if total is not None and total > 0:
        entry["complete"] = _artifact_read_coverage_is_complete(
            ranges=entry["ranges"],
            total_lines=total,
        )


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


_SYMBOL_TOKEN_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]{0,80}$")
_SYMBOL_LINE_PATTERNS = (
    re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*="),
)
_SYMBOL_CAPTURE_LIMIT = 24
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
# Extracts simple `s|old|new|` / `s/old/new/` / `s#old#new#` substitutions from sed commands.
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
_REMOTE_PATH_RE = re.compile(r"(?<![\w/])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:\.[A-Za-z0-9._-]+)?")
_REMOTE_DELETION_RE = re.compile(r"\b(?:rm|truncate)\b", re.IGNORECASE)
_BENIGN_REMOTE_MUTATION_PATHS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
_BENIGN_REMOTE_MUTATION_PATH_RE = re.compile(r"^/dev/fd/\d+$")
_REMOTE_SHELL_INTERPRETER_PATHS = frozenset(
    {"/bin/bash", "/bin/sh", "/usr/bin/bash", "/usr/bin/sh", "/usr/bin/env"}
)
_REMOTE_PATH_GLOB_CHARS = frozenset("*?[")
_REMOTE_CONTROL_TOKENS = frozenset({"&&", "||", "|", ";", "\n", "&"})
_REMOTE_REDIRECT_TOKENS = frozenset({">", ">>", "<", "<<", "<<<", "<>", ">|"})
_REMOTE_OUTPUT_REDIRECT_TOKENS = frozenset({">", ">>", "<>", ">|"})
_REMOTE_DELETION_COMMANDS = frozenset({"rm", "truncate"})
_REMOTE_PATH_MUTATOR_COMMANDS = frozenset({"sed", "perl", "tee", "cp", "mv", "install"})


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
    directory_empty_checks = _guess_remote_deletion_directory_empty_checks(mutation_command) if is_deletion else []
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


def _guess_remote_mutation_paths(command: str, *, deletion: bool = False) -> list[str]:
    if deletion:
        return _guess_remote_deletion_paths(command)
    paths: list[str] = []
    _collect_remote_redirection_targets(command, paths)
    _collect_remote_mutator_operands(command, paths)
    if _python_open_write_mutation(command):
        for match in _REMOTE_PATH_RE.finditer(str(command or "")):
            _append_remote_mutation_path(paths, match.group(0))
    return paths[:12]


def _collect_remote_redirection_targets(command: str, paths: list[str]) -> None:
    for tokens in _remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            target_index = _redirection_target_index(tokens, index, output_only=True)
            if target_index is None:
                index += 1
                continue
            if target_index < len(tokens):
                target = _redirection_target_token(tokens[target_index])
                _append_remote_mutation_path(paths, target)
            index = target_index + 1


def _collect_remote_mutator_operands(command: str, paths: list[str]) -> None:
    for tokens in _remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            token = _shell_command_name(tokens[index])
            if token not in _REMOTE_PATH_MUTATOR_COMMANDS:
                index += 1
                continue
            index = _collect_mutator_path_operands(tokens, index, paths)


def _collect_mutator_path_operands(tokens: list[str], command_index: int, paths: list[str]) -> int:
    command = _shell_command_name(tokens[command_index])
    index = command_index + 1
    operands: list[str] = []
    while index < len(tokens):
        token = tokens[index]
        if token in _REMOTE_CONTROL_TOKENS or token == "+":
            break
        target_index = _redirection_target_index(tokens, index, output_only=False)
        if target_index is not None:
            index = target_index + 1
            continue
        if token == "--" or token.startswith("-") or token in {"{}", r"\;"}:
            index += 1
            continue
        normalized = _normalize_remote_mutation_operand(token)
        if normalized:
            operands.append(normalized)
        index += 1

    if command in {"cp", "install"} and operands:
        _append_remote_mutation_path(paths, operands[-1])
    else:
        for operand in operands:
            _append_remote_mutation_path(paths, operand)
    return max(index, command_index + 1)


def _python_open_write_mutation(command: str) -> bool:
    return bool(
        re.search(
            r"\bpython3?\s+-c\b.*\bopen\s*\([^)]*['\"]w",
            str(command or ""),
            re.IGNORECASE | re.DOTALL,
        )
    )


def _guess_remote_deletion_paths(command: str) -> list[str]:
    paths: list[str] = []
    for tokens in _remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            token = _shell_command_name(tokens[index])
            if token in _REMOTE_DELETION_COMMANDS:
                index = _collect_deletion_operands(tokens, index + 1, paths)
                continue
            index += 1
    return paths[:12]


def _guess_remote_deletion_directory_empty_checks(command: str) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    for tokens in _remote_shell_command_lines(command):
        index = 0
        while index < len(tokens):
            token = _shell_command_name(tokens[index])
            if token in _REMOTE_DELETION_COMMANDS:
                index = _collect_deletion_glob_checks(tokens, index + 1, checks, seen)
                continue
            index += 1
    return checks[:12]


def _remote_shell_command_lines(command: str) -> list[list[str]]:
    lines: list[list[str]] = []
    for raw_line in str(command or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            tokens = shlex.split(line, comments=True, posix=True)
        if tokens:
            lines.append(tokens)
    return lines


def _collect_deletion_operands(tokens: list[str], start_index: int, paths: list[str]) -> int:
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if token in _REMOTE_CONTROL_TOKENS or token == "+":
            return index + 1
        if _token_starts_redirection(tokens, index):
            index = _skip_redirection(tokens, index)
            continue
        if token == "--" or token.startswith("-") or token in {"{}", r"\;"}:
            index += 1
            continue
        path = _normalize_remote_deletion_operand(token)
        if path and path not in paths:
            paths.append(path)
        index += 1
    return index


def _collect_deletion_glob_checks(
    tokens: list[str],
    start_index: int,
    checks: list[dict[str, str]],
    seen: set[str],
) -> int:
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if token in _REMOTE_CONTROL_TOKENS or token == "+":
            return index + 1
        if _token_starts_redirection(tokens, index):
            index = _skip_redirection(tokens, index)
            continue
        if token == "--" or token.startswith("-") or token in {"{}", r"\;"}:
            index += 1
            continue
        check = _remote_deletion_glob_empty_check(token)
        path = check.get("path", "") if check else ""
        if check and path not in seen:
            seen.add(path)
            checks.append(check)
        index += 1
    return index


def _shell_command_name(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    return Path(value).name.lower()


def _token_starts_redirection(tokens: list[str], index: int) -> bool:
    token = str(tokens[index] or "").strip()
    if token in _REMOTE_REDIRECT_TOKENS:
        return True
    if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _REMOTE_REDIRECT_TOKENS:
        return True
    return bool(re.match(r"^\d*(?:>>?|<<?|<>)", token))


def _skip_redirection(tokens: list[str], index: int) -> int:
    token = str(tokens[index] or "").strip()
    if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _REMOTE_REDIRECT_TOKENS:
        index += 2
    elif token in _REMOTE_REDIRECT_TOKENS:
        index += 1
    else:
        index += 1
        return index
    if index < len(tokens) and tokens[index] not in _REMOTE_CONTROL_TOKENS:
        index += 1
    return index


def _redirection_target_index(tokens: list[str], index: int, *, output_only: bool) -> int | None:
    token = str(tokens[index] or "").strip()
    redirect_token = ""
    target_index = index + 1
    if token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _REMOTE_REDIRECT_TOKENS:
        redirect_token = str(tokens[index + 1])
        target_index = index + 2
    elif token in _REMOTE_REDIRECT_TOKENS:
        redirect_token = token
    else:
        compact = re.match(r"^\d*(>>?|<>|>\|)(/\S+)$", token)
        if not compact:
            return None
        redirect_token = compact.group(1)
        if output_only and redirect_token not in _REMOTE_OUTPUT_REDIRECT_TOKENS:
            return None
        return index

    if output_only and redirect_token not in _REMOTE_OUTPUT_REDIRECT_TOKENS:
        return None
    if redirect_token in {"<<", "<<<"}:
        return None
    if target_index < len(tokens) and tokens[target_index] not in _REMOTE_CONTROL_TOKENS:
        return target_index
    return None


def _redirection_target_token(token: str) -> str:
    value = str(token or "").strip()
    compact = re.match(r"^\d*(?:>>?|<>|>\|)(/\S+)$", value)
    if compact:
        return compact.group(1)
    return value


def _normalize_remote_deletion_operand(token: str) -> str:
    candidate = str(token or "").strip().strip("`'\"")
    candidate = candidate.rstrip(";,")
    if not candidate.startswith("/"):
        return ""
    if any(char in candidate for char in _REMOTE_PATH_GLOB_CHARS):
        return ""
    if candidate.endswith("/"):
        candidate = candidate.rstrip("/")
    if not _REMOTE_PATH_RE.fullmatch(candidate):
        return ""
    if _remote_path_should_be_ignored(candidate):
        return ""
    return candidate


def _normalize_remote_mutation_operand(token: str) -> str:
    candidate = str(token or "").strip().strip("`'\"")
    candidate = candidate.rstrip(";,")
    if not candidate.startswith("/"):
        return ""
    if any(char in candidate for char in _REMOTE_PATH_GLOB_CHARS):
        return ""
    if candidate.endswith("/"):
        candidate = candidate.rstrip("/")
    if not _REMOTE_PATH_RE.fullmatch(candidate):
        return ""
    if _remote_path_should_be_ignored(candidate) or _remote_path_is_known_directory(candidate):
        return ""
    return candidate


def _append_remote_mutation_path(paths: list[str], path: str) -> None:
    normalized = _normalize_remote_mutation_operand(path)
    if normalized and normalized not in paths:
        paths.append(normalized)


def _remote_path_is_known_directory(path: str) -> bool:
    return str(path or "").strip().rstrip("/") in {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
    }


def _remote_deletion_glob_empty_check(token: str) -> dict[str, str] | None:
    candidate = str(token or "").strip().strip("`'\"").rstrip(";,")
    if not candidate.startswith("/") or not any(char in candidate for char in _REMOTE_PATH_GLOB_CHARS):
        return None
    if not candidate.endswith("/*"):
        return None
    parent = candidate[:-2].rstrip("/")
    if not parent or not _REMOTE_PATH_RE.fullmatch(parent):
        return None
    if _remote_path_should_be_ignored(parent):
        return None
    return {"path": parent, "glob": candidate}


def _remote_path_should_be_ignored(path: str) -> bool:
    normalized = str(path or "").strip()
    if (
        normalized in _BENIGN_REMOTE_MUTATION_PATHS
        or normalized in _REMOTE_SHELL_INTERPRETER_PATHS
        or bool(_BENIGN_REMOTE_MUTATION_PATH_RE.match(normalized))
    ):
        return True
    # Ignore ephemeral temp files that are not task deliverables
    return any(
        normalized.startswith(prefix)
        for prefix in ("/tmp/", "/var/tmp/", "/dev/shm/")
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
    normalized = str(path or "").strip()
    if not normalized:
        return
    verified = requirement.get("verified_paths")
    if not isinstance(verified, list):
        verified = []
    cleaned = [str(item).strip() for item in verified if str(item).strip()]
    if normalized not in cleaned:
        cleaned.append(normalized)
    requirement["verified_paths"] = cleaned[-24:]


def _mark_remote_mutation_directory_verified(requirement: dict[str, Any], path: str) -> None:
    normalized = str(path or "").strip().rstrip("/")
    if not normalized:
        return
    verified = requirement.get("verified_directory_empty_checks")
    if not isinstance(verified, list):
        verified = []
    cleaned = [str(item).strip().rstrip("/") for item in verified if str(item).strip()]
    if normalized not in cleaned:
        cleaned.append(normalized)
    requirement["verified_directory_empty_checks"] = cleaned[-24:]


def _remote_mutation_requirement_satisfied(requirement: dict[str, Any]) -> bool:
    guessed_paths = [str(item).strip() for item in requirement.get("guessed_paths", []) if str(item).strip()]
    verified_paths = {
        str(item).strip()
        for item in requirement.get("verified_paths", [])
        if str(item).strip()
    }
    pending_paths = [path for path in guessed_paths if path not in verified_paths]

    directory_checks = _remote_mutation_directory_checks(requirement)
    verified_directories = {
        str(item).strip().rstrip("/")
        for item in requirement.get("verified_directory_empty_checks", [])
        if str(item).strip()
    }
    pending_directories = [
        check["path"] for check in directory_checks if check["path"] not in verified_directories
    ]
    return not pending_paths and not pending_directories


def _remote_mutation_directory_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
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
        candidates.extend(_extract_symbol_candidates_from_text(chunk))

    path_candidate = str(mutated_path or metadata.get("path") or "").strip()
    if not path_candidate and isinstance(arguments, dict):
        path_candidate = str(arguments.get("path") or "").strip()
    if not path_candidate and artifact is not None:
        path_candidate = str(getattr(artifact, "source", "") or "").strip()
    candidates.extend(_extract_symbol_candidates_from_path(path_candidate))
    candidates.extend(_extract_symbol_candidates_from_file(path_candidate, cwd=str(service.harness.state.cwd or "")))

    deduped = dedupe_keep_tail(existing_symbols + [token for token in candidates if token], limit=_SYMBOL_CAPTURE_LIMIT)
    if not deduped:
        return
    existing = service.harness.state.scratchpad.get("_touched_symbols")
    existing_list = [str(item).strip() for item in existing] if isinstance(existing, list) else []
    merged = dedupe_keep_tail(existing_list + deduped, limit=_SYMBOL_CAPTURE_LIMIT)
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


def _extract_symbol_candidates_from_text(text: str) -> list[str]:
    symbols: list[str] = []
    if not text:
        return symbols
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("+++", "---")):
            continue
        if line[0] in {"+", "-"} and len(line) > 1:
            line = line[1:].lstrip()
        for pattern in _SYMBOL_LINE_PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            token = _normalize_symbol_token(match.group(1))
            if token and token not in symbols:
                symbols.append(token)
    return symbols


def _extract_symbol_candidates_from_path(path: str) -> list[str]:
    normalized = str(path or "").strip()
    if not normalized:
        return []
    stem = Path(normalized).stem.strip()
    if stem.lower() in {"", "__init__", "__main__", "index", "main"}:
        return []
    token = _normalize_symbol_token(stem)
    return [token] if token else []


def _extract_symbol_candidates_from_file(path: str, *, cwd: str) -> list[str]:
    normalized = str(path or "").strip()
    if not normalized:
        return []
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = Path(cwd or ".") / candidate
    try:
        if not candidate.exists() or not candidate.is_file():
            return []
        content = candidate.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    # We only need top-level-ish anchors for prompt stability; avoid scanning huge files.
    lines = content.splitlines()[:500]
    return _extract_symbol_candidates_from_text("\n".join(lines))


def _normalize_symbol_token(value: str) -> str:
    token = str(value or "").strip().strip("`'\".,:;()[]{}<>")
    if not token:
        return ""
    if token.lower() in {"path", "file", "content", "target", "replacement"}:
        return ""
    if not _SYMBOL_TOKEN_RE.match(token):
        return ""
    return token


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
    if tool_name in {"shell_exec", "ssh_exec"} and artifact_id:
        _consolidate_shell_attempt_family(state=service.harness.state, artifact_id=artifact_id, result=result)
        _record_stderr_signature_circuit_breaker(service, tool_name=tool_name, result=result)
    if tool_name == "ssh_exec":
        _remember_session_ssh_target(
            service,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
        )
        _observe_remote_installer_preflight_check(
            service,
            result=result,
            arguments=arguments,
        )
        _handle_remote_mutation_verifier_result(
            service,
            result=result,
            arguments=arguments,
        )
        # Count failed ssh_exec verifier attempts so the requirement can be
        # auto-disabled after repeated SSH failures (e.g. auth unavailable).
        if not result.success:
            _record_failed_verification_attempt(
                service,
                tool_name=tool_name,
                result=result,
                arguments=arguments,
            )
        _record_remote_mutation_requirement(
            service,
            result=result,
            arguments=arguments,
        )
    elif tool_name in _SSH_FILE_VERIFIER_TOOLS:
        _remember_session_ssh_target(
            service,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
        )
        if not result.success:
            _record_failed_verification_attempt(
                service,
                tool_name=tool_name,
                result=result,
                arguments=arguments,
            )
            _maybe_emit_bounded_region_trap_nudge(
                service,
                tool_name=tool_name,
                result=result,
                arguments=arguments,
            )
        _clear_remote_mutation_requirement_from_tool(
            service,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
        )
    elif tool_name == "web_search" and result.success:
        _remember_web_search_results(
            service,
            result=result,
            artifact=artifact,
        )
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
    if tool_name == "file_read" and result.success and artifact:
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
    elif tool_name == "ssh_file_read" and result.success and artifact:
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
    elif is_file_mutating_tool(tool_name) and result.success:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if bool(metadata.get("dry_run")) or metadata.get("changed") is False:
            pass
        else:
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
    elif tool_name in SSH_FILE_MUTATING_TOOLS and result.success:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if bool(metadata.get("dry_run")) or metadata.get("changed") is False:
            pass
        else:
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

                # Grant a one-shot pass for ssh_file_read using the same connection
                # parameters the model will likely emit (host/user/password/path).
                # We copy them from the original patch arguments so the fingerprint
                # matches the model's next read attempt.
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
