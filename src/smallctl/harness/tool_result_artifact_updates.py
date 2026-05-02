from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..normalization import dedupe_keep_tail
from .artifact_tracking import consolidate_shell_attempt_family as _consolidate_shell_attempt_family
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
from ..shell_utils import file_read_cache_key as _file_read_cache_key
from ..tools.fs import is_file_mutating_tool

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
    if not command or _REMOTE_MUTATING_COMMAND_RE.search(command) is None:
        return

    host = str(arguments.get("host") or arguments.get("target") or "").strip()
    guessed_paths = _guess_remote_mutation_paths(command)
    requirement: dict[str, Any] = {
        "tool_name": "ssh_exec",
        "host": host,
        "user": str(arguments.get("user") or "").strip(),
        "command": command,
        "guessed_paths": guessed_paths,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suggested_verifier": (
            "You used ssh_exec to mutate a remote file. Before calling task_complete "
            "or attempting further edits, verify the change by reading the file with "
            "ssh_file_read. Do NOT try to re-apply the same change with "
            "ssh_file_replace_between; if the file is already fixed, re-application will fail."
        ),
    }
    if _REMOTE_DELETION_RE.search(command) is not None:
        requirement["mutation_type"] = "deletion"
        requirement["suggested_verifier"] = (
            "You used ssh_exec to delete or truncate a remote file. Before calling "
            "task_complete, verify the file is gone with ssh_file_read; a missing-file "
            "result counts as successful verification."
        )
    if _REMOTE_SED_MUTATION_RE.search(command) is not None:
        sed_matches = _SED_SUBSTITUTION_RE.findall(command)
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
    service.harness.state.scratchpad[_REMOTE_MUTATION_VERIFICATION_KEY] = requirement

    if _REMOTE_SED_MUTATION_RE.search(command) is not None:
        _emit_remote_mutation_nudge(
            service,
            command=command,
            guessed_paths=guessed_paths,
            multiline=bool(_REMOTE_MULTILINE_REPLACEMENT_RE.search(command)),
        )


def _guess_remote_mutation_paths(command: str) -> list[str]:
    paths: list[str] = []
    for match in _REMOTE_PATH_RE.finditer(str(command or "")):
        path = match.group(0)
        if path in _BENIGN_REMOTE_MUTATION_PATHS or _BENIGN_REMOTE_MUTATION_PATH_RE.match(path):
            continue
        if path not in paths:
            paths.append(path)
    return paths[:12]


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
    service.harness.state.scratchpad.pop(_REMOTE_MUTATION_VERIFICATION_KEY, None)


def _record_failed_verification_attempt(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
) -> None:
    if tool_name not in _SSH_FILE_VERIFIER_TOOLS or result.success:
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


def apply_artifact_success_outcome(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
) -> Any:
    service.harness.state.artifacts[artifact.artifact_id] = artifact
    service.harness.state.retrieval_cache = [artifact.artifact_id]
    if tool_name in {"shell_exec", "ssh_exec"}:
        _consolidate_shell_attempt_family(state=service.harness.state, artifact_id=artifact.artifact_id, result=result)
    if tool_name == "ssh_exec":
        _remember_session_ssh_target(
            service,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
        )
        _handle_remote_mutation_verifier_result(
            service,
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
    if tool_name == "file_read" and result.success:
        cache_key = _file_read_cache_key(service.harness.state.cwd, result.metadata)
        if cache_key:
            cache = service.harness.state.scratchpad.setdefault("file_read_cache", {})
            if isinstance(cache, dict):
                cache[cache_key] = artifact.artifact_id
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
                _emit_context_invalidation(
                    service,
                    reason="file_changed",
                    paths=[mutated_path],
                    details={
                        "tool_name": tool_name,
                        "state_change": f"File changed: {mutated_path}",
                    },
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
    elif tool_name == "artifact_read" and result.success:
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

    if tool_name != "shell_exec" and not result.metadata.get("skip_auto_fact_record"):
        fact_label = evidence.statement or artifact.summary or tool_name
        prefix = f"{tool_name}: "
        if fact_label.startswith(prefix):
            fact_label = fact_label[len(prefix):]
        service.harness.state.working_memory.known_facts = dedupe_keep_tail(
            service.harness.state.working_memory.known_facts + [f"{tool_name}: {fact_label}"],
            limit=12,
        )
    service.harness.state.recent_errors = []
    return evidence
