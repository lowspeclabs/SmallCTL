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
    promote_write_session_target,
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

FILE_MUTATING_TOOLS = {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"}


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

    if write_session_id:
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

    # Fix 1: Intercept bare file_write to a session-owned canonical path.
    # A model that omits write_session_id while a session is open gets a silent
    # direct-write success, the FSM never advances, and task_complete is blocked
    # later with no actionable context.  Block it here instead.
    if state is not None:
        _active_session = getattr(state, "write_session", None)
        if (
            _active_session is not None
            and str(getattr(_active_session, "status", "") or "").strip().lower() not in {"complete"}
            and _same_target_path(
                str(getattr(_active_session, "write_target_path", "") or ""), path, cwd
            )
        ):
            from .control import _write_session_resume_action
            _ws_id = str(getattr(_active_session, "write_session_id", "") or "").strip()
            _next_section = str(getattr(_active_session, "write_next_section", "") or "").strip() or "imports"
            record_write_session_event(
                state,
                event="bare_write_intercepted",
                session=_active_session,
                details={"path": str(target), "reason": "write_session_id_missing"},
            )
            return fail(
                f"file_write to `{path}` was rejected because Write Session `{_ws_id}` is "
                f"still open for that path. You must continue the session: provide "
                f"`write_session_id='{_ws_id}'` and `section_name='{_next_section}'` in this "
                f"call. A bare file_write without `write_session_id` bypasses the session FSM "
                f"and will leave the session permanently open. Use `loop_status` to see the "
                f"current session state and the required next section.",
                metadata={
                    "path": str(target),
                    "error_kind": "bare_write_to_session_owned_path",
                    "write_session_id": _ws_id,
                    "write_next_section": _next_section,
                    "staged_only": False,
                    "next_required_tool": _write_session_resume_action(state, None),
                },
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
            if existing_content.strip() and content.strip():
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
