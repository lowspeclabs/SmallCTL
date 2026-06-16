from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from ..state import LoopState
from ..risk_policy import evaluate_risk_policy
from ..write_session_fsm import record_write_session_event
from .fs_sessions import (
    _append_unique_section,
    _clone_section_ranges,
    _looks_like_complete_html_document,
    _normalize_replace_strategy,
    _normalize_section_name,
    _repair_cycle_session_id_failure,
    _repair_cycle_allows_patch,
    _repair_cycle_read_required_metadata,
    _record_file_change,
    _record_repair_cycle_read,
    _same_target_path,
    _mark_repeat_patch,
    _mark_repeat_command,
    _write_session_can_finalize,
    infer_write_session_intent,
    new_write_session_id,
)
from .fs_patching import (
    _apply_exact_patch,
    _build_patch_ambiguity_hint,
    _build_patch_failure_metadata,
    _build_patch_failure_message,
    _build_patch_metadata,
    _build_patch_text_preview,
    _count_exact_occurrences,
)
from .fs_listing import (
    _active_session_staging_path,
    _build_dir_tree,
    _missing_dir_error,
    _missing_path_error,
    _workspace_relative_hint,
    active_write_session_source_path,
    dir_list,
    dir_tree,
    file_read,
)
from .fs_write_sessions import (
    _content_hash,
    _ensure_write_session_files,
    _read_text_file,
    _replace_known_section,
    _resolve_patch_source,
    _resolve,
    _session_attempt_snapshot_path,
    _session_original_snapshot_path,
    _session_stage_path,
    _write_session_dir,
    _write_text_file,
    format_write_session_status_block,
    maybe_create_implicit_write_session,
    promote_write_session_target,
    resolve_write_session_for_path,
    restore_write_session_snapshot,
    write_session_status_snapshot,
    write_session_verify_path,
)
from .fs_write_session_policy import (
    _guard_suspicious_temp_root_path,
    _guard_write_session_staging_mutation,
    _looks_like_system_repair_cycle_id,
)
from .fs_mutations import file_append, file_delete
from .fs_write_flow import handle_file_write_session
from .ast_patch import handle_ast_patch
from .fs_patch_flow import handle_file_patch
from .common import fail, ok


def _looks_like_sensitive_known_hosts_path(path: str, cwd: str | None = None) -> bool:
    try:
        resolved = _resolve(path, cwd)
    except Exception:
        resolved = Path(str(path or "")).expanduser()
    normalized = resolved.as_posix().lower()
    return normalized.endswith("/.ssh/known_hosts") or normalized.endswith("/.ssh/known_hosts2")

FILE_MUTATING_TOOLS = {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"}


def _looks_like_truncated_write_payload(content: str, path: str | Path) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    if len(text) < 80 and text.startswith('{"tool_name'):
        return True
    suffix = str(path or "").strip().lower()
    if suffix.endswith((".html", ".htm")) and len(text) < 120:
        lowered = text.lower()
        html_starters = ("<!doctype", "<html", "<head", "<body", "<script", "<style")
        if lowered.startswith(html_starters) and "</html>" not in lowered:
            return True
    return False


def _looks_like_poisoned_truncated_file(content: str) -> bool:
    text = str(content or "").strip()
    if not text or len(text) >= 120:
        return False
    if text.startswith('{"tool_name'):
        return True
    lowered = text.lower()
    html_starters = ("<!doctype", "<html", "<head", "<body", "<script", "<style")
    return lowered.startswith(html_starters) and "</html>" not in lowered


def _infer_implicit_section_name(
    session: Any,
    content: str,
    replace_strategy: str | None,
) -> str | None:
    """Best-effort section label when the model omits section_name during a session."""
    next_section = str(getattr(session, "write_next_section", "") or "").strip()
    if next_section:
        return next_section
    current_section = str(getattr(session, "write_current_section", "") or "").strip()
    if current_section:
        return current_section
    strategy = _normalize_replace_strategy(replace_strategy)
    if strategy == "overwrite" or _looks_like_complete_html_document(content):
        return "full_file"
    if session.write_sections_completed:
        # Ambiguous: previous sections exist but no next/current section was named.
        return None
    return None


def is_file_mutating_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip() in FILE_MUTATING_TOOLS

async def file_write(
    path: str,
    content: str,
    cwd: str | None = None,
    encoding: str = "utf-8",
    state: LoopState | None = None,
    session_id: str | None = None,
    write_session_id: str | None = None,
    section_name: str | None = None,
    section_id: str | None = None,
    section_role: str | None = None,
    next_section_name: str | None = None,
    replace_strategy: str | None = None,
    expected_followup_verifier: str | None = None,
) -> dict[str, Any]:
    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    target = _resolve(path, cwd)
    if not write_session_id and session_id:
        write_session_id = session_id

    if not content and not write_session_id:
        return fail(
            "File write received empty content. If you intended to create an empty file, please use `\"\"` (or use shell commands like touch). "
            "If you forgot the content, please retry with the full content payload.",
            metadata={
                "error_kind": "empty_content_without_session",
                "next_required_tool": {
                    "tool_name": "file_write",
                    "required_fields": ["path", "content"],
                    "required_arguments": {"path": path},
                },
            },
        )

    if not write_session_id and _looks_like_truncated_write_payload(content, path):
        return fail(
            f"file_write to `{path}` was rejected because the content looks like a truncated or malformed tool payload, not file contents. "
            "Retry with the complete file content in the `content` argument.",
            metadata={
                "path": str(target),
                "error_kind": "truncated_write_payload",
                "content_chars": len(str(content or "")),
                "next_required_tool": {
                    "tool_name": "file_write",
                    "required_fields": ["path", "content"],
                    "required_arguments": {"path": path},
                    "notes": ["Provide the complete target file contents, not a partial tool-call fragment."],
                },
            },
        )
    
    if write_session_id and state is not None:
        _terminal_session = getattr(state, "write_session", None)
        if (
            _terminal_session is not None
            and str(getattr(_terminal_session, "status", "") or "").strip().lower() == "complete"
            and str(getattr(_terminal_session, "write_session_id", "") or "").strip() == str(write_session_id).strip()
            and _same_target_path(
                str(getattr(_terminal_session, "write_target_path", "") or ""), path, cwd
            )
            and _normalize_replace_strategy(replace_strategy) == "overwrite"
        ):
            record_write_session_event(
                state,
                event="terminal_session_direct_overwrite_repaired",
                session=_terminal_session,
                details={"path": str(target), "reason": "replace_strategy_overwrite"},
            )
            write_session_id = None

    # Path-based implicit session resolution takes precedence over explicit IDs.
    # If an active non-terminal session owns the target path, route the write into
    # the session regardless of whether the model supplied a matching, stale, or
    # missing write_session_id.
    implicit_session = resolve_write_session_for_path(state, path, cwd)
    if implicit_session is not None:
        if write_session_id:
            provided_id = str(write_session_id).strip()
            active_id = str(getattr(implicit_session, "write_session_id", "") or "").strip()
            if provided_id != active_id:
                record_write_session_event(
                    state,
                    event="implicit_session_continued",
                    session=implicit_session,
                    details={
                        "path": str(target),
                        "provided_write_session_id": provided_id,
                        "active_write_session_id": active_id,
                        "reason": "path_match_overrides_unknown_id",
                    },
                )
            else:
                record_write_session_event(
                    state,
                    event="implicit_session_continued",
                    session=implicit_session,
                    details={
                        "path": str(target),
                        "provided_write_session_id": provided_id,
                        "reason": "path_and_id_match",
                    },
                )
        else:
            record_write_session_event(
                state,
                event="implicit_session_resolved",
                session=implicit_session,
                details={"path": str(target), "reason": "path_match_no_id"},
            )

        if not section_name and not section_id:
            inferred = _infer_implicit_section_name(
                implicit_session, content, replace_strategy
            )
            if inferred is None:
                return fail(
                    f"file_write to `{path}` continues an active Write Session, but the section name is ambiguous. "
                    "Provide `section_name` to choose which section to write or replace.",
                    metadata={
                        "path": str(target),
                        "error_kind": "implicit_session_section_ambiguous",
                        "write_session_id": str(getattr(implicit_session, "write_session_id", "") or "").strip(),
                        "write_sections_completed": list(getattr(implicit_session, "write_sections_completed", []) or []),
                        "write_next_section": str(getattr(implicit_session, "write_next_section", "") or "").strip(),
                        "write_current_section": str(getattr(implicit_session, "write_current_section", "") or "").strip(),
                        "staged_only": True,
                        "next_required_tool": {
                            "tool_name": "file_write",
                            "required_fields": ["path", "content", "section_name"],
                            "required_arguments": {"path": path},
                            "optional_fields": ["next_section_name", "replace_strategy"],
                            "notes": [
                                "Use the target path and section_name to continue the session.",
                                "Use section_name='full_file' with replace_strategy='overwrite' for a complete staged-file replacement.",
                            ],
                        },
                    },
                )
            section_name = inferred

        return handle_file_write_session(
            path=path,
            content=content,
            cwd=cwd,
            encoding=encoding,
            state=state,
            session_id=session_id,
            write_session_id=None,
            section_name=section_name,
            section_id=section_id,
            section_role=section_role,
            next_section_name=next_section_name,
            replace_strategy=replace_strategy,
            expected_followup_verifier=expected_followup_verifier,
            session=implicit_session,
        )

    if write_session_id:
        _active_session = getattr(state, "write_session", None) if state is not None else None
        _active_session_id = str(getattr(_active_session, "write_session_id", "") or "").strip()
        if _active_session_id and _active_session_id == str(write_session_id).strip():
            return handle_file_write_session(
                path=path,
                content=content,
                cwd=cwd,
                encoding=encoding,
                state=state,
                session_id=session_id,
                write_session_id=write_session_id,
                section_name=section_name,
                section_id=section_id,
                section_role=section_role,
                next_section_name=next_section_name,
                replace_strategy=replace_strategy,
                expected_followup_verifier=expected_followup_verifier,
            )
        record_write_session_event(
            state,
            event="unknown_write_session_id_fallback_to_direct_write",
            session=_active_session,
            details={
                "path": str(target),
                "provided_write_session_id": str(write_session_id).strip(),
                "active_write_session_id": _active_session_id,
                "reason": "unknown_write_session_id",
            },
        )
        write_session_id = None

    if _looks_like_sensitive_known_hosts_path(path, cwd) and _normalize_replace_strategy(replace_strategy) == "overwrite":
        return fail(
            f"Refusing to overwrite sensitive SSH trust-store file `{path}`. "
            "Use approved `shell_exec(command='ssh-keygen -R <host> -f ~/.ssh/known_hosts')` "
            "or a narrow, explicit file_patch after human approval.",
            metadata={
                "path": str(target),
                "error_kind": "sensitive_known_hosts_overwrite_blocked",
                "requires_approval": True,
                "next_required_tool": {
                    "tool_name": "shell_exec",
                    "required_arguments": {"command": "ssh-keygen -R <host> -f ~/.ssh/known_hosts"},
                    "notes": [
                        "Do not synthesize full-file overwrites for SSH trust stores.",
                        "Ask the user before changing host-key trust entries.",
                    ],
                },
            },
        )

    # Implicit session creation for path-based chunked authoring.
    implicit_session = maybe_create_implicit_write_session(
        state,
        path,
        content,
        section_name=section_name,
        next_section_name=next_section_name,
        replace_strategy=replace_strategy,
        cwd=cwd,
    )
    if implicit_session is not None:
        if not section_name and not section_id:
            section_name = str(getattr(implicit_session, "write_current_section", "") or "").strip()
        return handle_file_write_session(
            path=path,
            content=content,
            cwd=cwd,
            encoding=encoding,
            state=state,
            session_id=session_id,
            write_session_id=None,
            section_name=section_name,
            section_id=section_id,
            section_role=section_role,
            next_section_name=next_section_name,
            replace_strategy=replace_strategy,
            expected_followup_verifier=expected_followup_verifier,
            session=implicit_session,
        )

    if not _repair_cycle_allows_patch(state, target):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata=_repair_cycle_read_required_metadata(state, target, requested_path=path),
        )

    # Fix 5: Guard against bare overwrites of existing files that should be patched.
    # If the file exists, has content, and the new content is very different,
    # require an explicit patch or overwrite flag to prevent accidental full rewrites.
    # Skip the guard if a completed write session exists for this target (legitimate overwrite).
    _has_completed_session = False
    if state is not None:
        _terminal_session = getattr(state, "write_session", None)
        if (
            _terminal_session is not None
            and str(getattr(_terminal_session, "status", "") or "").strip().lower() == "complete"
            and _same_target_path(str(getattr(_terminal_session, "write_target_path", "") or ""), path, cwd)
        ):
            _has_completed_session = True

    if target.exists() and target.is_file() and target.stat().st_size > 0 and not write_session_id and not _has_completed_session:
        try:
            existing_content = target.read_text(encoding=encoding)
            if _looks_like_poisoned_truncated_file(existing_content):
                pass
            elif existing_content.strip() and content.strip():
                existing_lines = set(existing_content.strip().splitlines())
                new_lines = set(content.strip().splitlines())
                if existing_lines and new_lines:
                    overlap = len(existing_lines & new_lines)
                    similarity = overlap / max(len(existing_lines), len(new_lines))
                    if similarity < 0.3:
                        return fail(
                            f"file_write to `{path}` was rejected because the new content is very "
                            f"different from the existing file ({int(similarity * 100)}% line overlap). "
                            f"This looks like an unintended full rewrite. Use `file_patch` to make "
                            f"targeted changes, or if you truly intend to overwrite the entire file, "
                            f"use `file_write` with `replace_strategy='overwrite'`.",
                            metadata={
                                "path": str(target),
                                "error_kind": "patch_over_rewrite_guard",
                                "line_overlap_pct": int(similarity * 100),
                                "next_required_tool": {
                                    "tool_name": "file_patch",
                                    "notes": [
                                        "Use file_patch with target_text and replacement_text for targeted changes.",
                                        "Or use file_write with replace_strategy='overwrite' for intentional full rewrite.",
                                    ],
                                },
                            },
                        )
        except Exception:
            pass  # Best-effort guard; don't block if we can't read the file

    risk_decision = evaluate_risk_policy(
        state if state is not None else LoopState(cwd=str(Path.cwd())),
        tool_name="file_write",
        tool_risk="high",
        phase=str((state.current_phase if state is not None else "") or ""),
        action=f"Write file {path}",
        expected_effect="Update the target file with the requested content.",
        rollback="Restore the previous file contents from the snapshot or staging file.",
        verification="Read back the file and run the smallest useful verifier.",
    )
    if not risk_decision.allowed:
        return fail(
            risk_decision.reason,
            metadata={
                "path": path,
                "reason": "missing_supported_claim",
                "proof_bundle": risk_decision.proof_bundle,
            },
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
    except Exception as exc:
        return fail(f"Unable to write file: {exc}")

    # Keep active write-session staging file in sync so subsequent reads don't drift.
    # Fix 2: emit a structured warning when this sync runs — it means a bare write landed on
    # a session-owned path and bypassed the FSM.  This should never happen after Fix 1 is
    # active, but log it defensively so it surfaces in harness.log if it ever does.
    if state is not None:
        session = getattr(state, "write_session", None)
        if session is not None and str(getattr(session, "status", "") or "").strip().lower() != "complete":
            try:
                session_target = _resolve(str(getattr(session, "write_target_path", "") or ""), cwd)
                if target == session_target:
                    _ws_id_sync = str(getattr(session, "write_session_id", "") or "").strip()
                    _parent_logger = getattr(state, "log", logging.getLogger("smallctl.tools.fs"))
                    _parent_logger.warning(
                        "write_session_stall bare_write_bypassed_fsm path=%s session_id=%s "
                        "session_status=%s — staging synced but session state NOT advanced; "
                        "session remains open and will block task_complete",
                        str(target),
                        _ws_id_sync,
                        str(getattr(session, "status", "open") or "open"),
                    )
                    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
                    if staging_path:
                        _write_text_file(Path(staging_path), content, encoding=encoding)
            except Exception:
                pass  # staging sync is best-effort; the target write already succeeded

    _record_file_change(state, target)
    return ok("written", metadata={"path": str(target), "bytes": len(content.encode(encoding)), "changed": True})


async def file_patch(
    path: str,
    target_text: str | None = None,
    replacement_text: str | None = None,
    cwd: str | None = None,
    encoding: str = "utf-8",
    state: LoopState | None = None,
    session_id: str | None = None,
    write_session_id: str | None = None,
    expected_occurrences: int = 1,
    expected_followup_verifier: str | None = None,
    dry_run: bool = False,
    occurrence_index: int | None = None,
    regex: bool = False,
    case_insensitive: bool = False,
    multiline: bool = False,
    dotall: bool = False,
) -> dict[str, Any]:
    return await handle_file_patch(
        path=path,
        target_text=target_text,
        replacement_text=replacement_text,
        cwd=cwd,
        encoding=encoding,
        state=state,
        session_id=session_id,
        write_session_id=write_session_id,
        expected_occurrences=expected_occurrences,
        expected_followup_verifier=expected_followup_verifier,
        dry_run=dry_run,
        occurrence_index=occurrence_index,
        regex=regex,
        case_insensitive=case_insensitive,
        multiline=multiline,
        dotall=dotall,
    )


async def ast_patch(
    path: str,
    language: str | None = "python",
    operation: str | None = None,
    target: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    cwd: str | None = None,
    encoding: str = "utf-8",
    state: LoopState | None = None,
    session_id: str | None = None,
    write_session_id: str | None = None,
    expected_followup_verifier: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return await handle_ast_patch(
        path=path,
        language=language,
        operation=operation,
        target=target,
        payload=payload,
        cwd=cwd,
        encoding=encoding,
        state=state,
        session_id=session_id,
        write_session_id=write_session_id,
        expected_followup_verifier=expected_followup_verifier,
        dry_run=dry_run,
    )
