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
    _repair_cycle_reads,
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
            "If you forgot the content, please retry with the full content payload."
        )
    
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

    if not _repair_cycle_allows_patch(state, target):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata={
                "path": str(target),
                "system_repair_cycle_id": getattr(state, "repair_cycle_id", ""),
                "required_read_paths": _repair_cycle_reads(state),
            },
        )
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
    if state is not None:
        session = getattr(state, "write_session", None)
        if session is not None and str(getattr(session, "status", "") or "").strip().lower() != "complete":
            try:
                session_target = _resolve(str(getattr(session, "write_target_path", "") or ""), cwd)
                if target == session_target:
                    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
                    if staging_path:
                        _write_text_file(Path(staging_path), content, encoding=encoding)
            except Exception:
                pass  # staging sync is best-effort; the target write already succeeded

    _record_file_change(state, target)
    return ok("written", metadata={"path": str(target), "bytes": len(content.encode(encoding))})


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
