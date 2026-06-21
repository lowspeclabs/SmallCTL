from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..state import LoopState
from ..write_session_fsm import record_write_session_event
from ..risk_policy import evaluate_risk_policy
from .common import fail, ok
from .fs_paths import _same_target_path
from .fs_sessions import (
    _append_unique_section,
    _clone_section_ranges,
    _normalize_replace_strategy,
    _normalize_section_name,
    _repair_cycle_session_id_failure,
    _mark_repeat_patch,
    _looks_like_complete_html_document,
    _looks_like_full_script_content,
    _record_file_change,
    _section_matches_any_suggestion,
    _section_name_allows_full_file_finalization,
    _infer_next_suggested_section,
    _write_session_should_finalize,
)
from .fs_write_sessions import (
    _content_hash,
    _ensure_write_session_files,
    _read_text_file,
    _replace_known_section,
    _resolve,
    _write_text_file,
    format_write_session_status_block,
    write_session_contract,
    write_session_status_snapshot,
)
from .fs_write_session_policy import (
    _guard_write_session_staging_mutation,
    _looks_like_system_repair_cycle_id,
)
from .fs_loop_guard import (
    mark_chunked_write_success,
    maybe_block_chunked_write,
)


def handle_file_write_session(
    *,
    path: str,
    content: str,
    cwd: str | None,
    encoding: str,
    state: LoopState | None,
    session_id: str | None,
    write_session_id: str | None,
    section_name: str | None,
    section_id: str | None,
    section_role: str | None,
    next_section_name: str | None,
    replace_strategy: str | None,
    expected_followup_verifier: str | None,
    session: Any | None = None,
) -> dict[str, Any]:
    target = _resolve(path, cwd)
    if session is not None:
        effective_session_id = str(getattr(session, "write_session_id", "") or "").strip()
    else:
        effective_session_id = str(write_session_id or "").strip()
    if _looks_like_system_repair_cycle_id(effective_session_id):
        return _repair_cycle_session_id_failure(
            supplied_id=effective_session_id,
            path=path,
            state=state,
        )
    if session is not None:
        pass
    elif state is None or state.write_session is None:
        return fail(
            f"No active write session found for session ID `{write_session_id}`. "
            "If you want to write the file directly, omit `write_session_id`. "
            "If you want staged/chunked authoring, start a write session explicitly.",
            metadata={
                "path": str(target),
                "error_kind": "missing_active_write_session",
                "write_session_id": str(write_session_id or "").strip(),
                "session_id": str(session_id or "").strip(),
                "section_name": _normalize_section_name(section_name, section_id),
                "section_id": str(section_id or ""),
                "replace_strategy": _normalize_replace_strategy(replace_strategy),
                "staged_only": True,
                "next_required_tool": {
                    "tool_name": "file_write",
                    "required_fields": ["path", "content"],
                    "required_arguments": {"path": path},
                    "optional_fields": ["replace_strategy"],
                    "notes": [
                        "Omit `write_session_id` to write directly to the target path.",
                        "Do not reuse a session ID that is not currently active.",
                    ],
                },
            },
        )
    else:
        session = state.write_session
        if write_session_id and session.write_session_id != write_session_id:
            return fail(
                f"Session ID mismatch: expected `{session.write_session_id}`, got `{write_session_id}`.",
                metadata={
                    "error_kind": "write_session_id_mismatch",
                    "expected_session_id": session.write_session_id,
                    "provided_session_id": write_session_id,
                    "next_required_tool": {
                        "tool_name": "file_write",
                        "required_fields": ["path", "content", "write_session_id"],
                        "required_arguments": {"path": path, "write_session_id": session.write_session_id},
                    },
                },
            )

    write_session_id = effective_session_id

    session_status = str(getattr(session, "status", "") or "open").strip().lower() or "open"
    if session_status not in {"open", "local_repair", "fallback"}:
        record_write_session_event(
            state,
            event="terminal_session_write_rejected",
            session=session,
            details={
                "path": str(target),
                "section_name": _normalize_section_name(section_name, section_id),
                "status": session_status,
            },
        )
        return fail(
            f"Write Session `{write_session_id}` is already `{session_status}` and cannot accept more file_write chunks. "
            "Start a new write session or omit `write_session_id` for a direct write.",
            metadata={
                "path": str(target),
                "error_kind": "write_session_already_terminal",
                "write_session_id": write_session_id,
                "write_session_status": session_status,
                "target_path": str(getattr(session, "write_target_path", "") or path),
                "staged_only": True,
                "next_required_tool": {
                    "tool_name": "file_write",
                    "required_fields": ["path", "content"],
                    "required_arguments": {"path": path},
                    "optional_fields": ["write_session_id"],
                    "notes": [
                        "Do not reuse a completed write_session_id.",
                        "Start a new write session for staged chunked authoring, or omit write_session_id for a direct write.",
                    ],
                },
            },
        )

    staging_guard = _guard_write_session_staging_mutation(
        tool_name="file_write",
        path=path,
        state=state,
        cwd=cwd,
        session=session,
        write_session_id=write_session_id,
        encoding=encoding,
        section_name=_normalize_section_name(section_name, section_id),
    )
    if staging_guard is not None:
        return staging_guard

    if not _same_target_path(session.write_target_path, path, cwd):
        if not session.write_sections_completed:
            parent_logger = getattr(state, "log", logging.getLogger("smallctl.tools.fs"))
            parent_logger.info(f"Correcting session target path from `{session.write_target_path}` to `{path}`")
            session.write_target_path = path
            session.write_staging_path = ""
            session.write_original_snapshot_path = ""
            session.write_last_attempt_snapshot_path = ""
            session.write_target_existed_at_start = False
        else:
            return fail(
                f"Session target path mismatch: expected `{session.write_target_path}`, got `{path}`.",
                metadata={
                    "error_kind": "write_session_target_mismatch",
                    "expected_path": session.write_target_path,
                    "provided_path": path,
                    "next_required_tool": {
                        "tool_name": "file_write",
                        "required_fields": ["path", "content"],
                        "required_arguments": {"path": session.write_target_path},
                    },
                },
            )

    normalized_section_name = _normalize_section_name(section_name, section_id)
    normalized_next_section = str(next_section_name or "").strip()
    strategy = _normalize_replace_strategy(replace_strategy)
    if (
        strategy == "overwrite"
        and target.suffix.lower() in {".html", ".htm"}
        and _looks_like_complete_html_document(content)
    ):
        normalized_section_name = "full_file"
        normalized_next_section = ""
    staging_path = _ensure_write_session_files(session, target, cwd=cwd, encoding=encoding)
    staged_content = _read_text_file(staging_path, encoding=encoding)
    previous_sections = list(session.write_sections_completed)
    previous_ranges = _clone_section_ranges(session.write_section_ranges)
    _write_text_file(Path(session.write_last_attempt_snapshot_path), staged_content, encoding=encoding)
    session.write_last_attempt_sections = previous_sections
    session.write_last_attempt_ranges = previous_ranges

    from .fs_loop_guard_utils import is_substantially_new_content

    current_range = previous_ranges.get(normalized_section_name)
    full_staged_overwrite_requested = (
        strategy == "overwrite" and _section_name_allows_full_file_finalization(normalized_section_name)
    )
    existing_section_content = ""
    if current_range:
        existing_section_content = staged_content[
            int(current_range.get("start", 0)) : int(current_range.get("end", 0))
        ]
    already_checkpointed_rewrite = (
        normalized_section_name in previous_sections
        and not full_staged_overwrite_requested
        and not (
            strategy in {"auto", "append"}
            and is_substantially_new_content(
                content,
                existing_section_content,
                similarity_threshold=0.85,
                min_new_chars=50,
            )
        )
    )
    if current_range:
        updated_content, updated_ranges = _replace_known_section(
            staged_content,
            previous_ranges,
            normalized_section_name,
            content,
        )
        effective_strategy = "replace_section"
    elif strategy == "overwrite":
        if previous_sections and not _section_name_allows_full_file_finalization(normalized_section_name):
            return fail(
                "Refusing to overwrite the entire staged file for a new chunk section after prior sections were recorded. "
                "Use `file_write` without `replace_strategy='overwrite'` to append the next section, use the same section name "
                "to replace a known section, or use `section_name='full_file'` with `replace_strategy='overwrite'` for an "
                "intentional full-file replacement.",
                metadata={
                    "path": str(target),
                    "staging_path": str(staging_path),
                    "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
                    "section_name": normalized_section_name,
                    "replace_strategy": strategy,
                    "error_kind": "chunked_write_overwrite_new_section_after_progress",
                    "write_sections_completed": previous_sections,
                    "write_next_section": str(getattr(session, "write_next_section", "") or "").strip(),
                    "staged_only": True,
                    "next_required_tool": {
                        "tool_name": "file_write",
                        "required_fields": ["path", "content", "section_name"],
                        "required_arguments": {
                            "path": path,
                            "section_name": str(getattr(session, "write_next_section", "") or normalized_section_name).strip(),
                        },
                        "optional_fields": ["next_section_name", "replace_strategy"],
                        "notes": [
                            "Omit replace_strategy or use append for the next new chunk section.",
                            "Use section_name='full_file' plus replace_strategy='overwrite' only for an intentional complete staged-file replacement.",
                        ],
                    },
                },
            )
        updated_content = content
        updated_ranges = {
            normalized_section_name: {"start": 0, "end": len(content)}
        }
        effective_strategy = "overwrite"
        session.write_sections_completed = []
        session.write_section_ranges = {}
    else:
        if (
            session.write_session_intent == "patch_existing"
            and not previous_sections
            and strategy in {"auto", "append"}
        ):
            return fail(
                "Patch-existing write sessions need an explicit first-chunk choice. "
                "The target path is still the canonical destination, but the staged copy is the active read/verify source. "
                "Use `file_patch` for a narrow exact edit inside the staged copy, or `ast_patch` for a narrow structural edit, "
                "or `file_write` with `replace_strategy='overwrite'` to replace the staged file. "
                "If earlier chunks are not fully visible in local context, call "
                f"`file_read(path='{path}')` first; during an active write session that reads from the staged copy. "
                "No sections are committed yet, so do not send another implicit first-chunk `file_write`/`file_append` "
                "with `replace_strategy='auto'` or `replace_strategy='append'`.",
                metadata={
                    "path": str(target),
                    "staging_path": str(staging_path),
                    "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
                    "write_session_intent": session.write_session_intent,
                    "replace_strategy": strategy,
                    "staged_only": True,
                    "error_kind": "patch_existing_requires_explicit_replace_strategy",
                    "next_required_tool": {
                        "tool_name": "file_write",
                        "required_fields": ["path", "content", "replace_strategy"],
                        "required_arguments": {
                            "path": path,
                            "replace_strategy": "overwrite",
                        },
                        "notes": [
                            "Alternatively use file_patch or ast_patch for narrow edits, or file_read to inspect the staged copy.",
                        ],
                    },
                },
            )
        start = len(staged_content)
        updated_content = staged_content + content
        updated_ranges = _clone_section_ranges(previous_ranges)
        updated_ranges[normalized_section_name] = {
            "start": start,
            "end": start + len(content),
        }
        effective_strategy = "append"

    full_script_needs_explicit_finalization = (
        not normalized_next_section
        and strategy != "overwrite"
        and not _section_name_allows_full_file_finalization(normalized_section_name)
        and _looks_like_full_script_content(content, normalized_section_name)
    )
    if full_script_needs_explicit_finalization:
        record_write_session_event(
            state,
            event="ambiguous_full_script_section_rejected",
            session=session,
            details={
                "path": str(target),
                "section_name": normalized_section_name,
                "replace_strategy": strategy,
            },
        )
        return fail(
            "This `file_write` looks like a full script, but it was sent as "
            f"section `{normalized_section_name}` with no `next_section_name`. "
            "To finalize a one-shot full file, retry with `section_name='full_file'` "
            "or `section_name='final_content'`, or set `replace_strategy='overwrite'`. "
            "For ordinary chunked authoring, keep the current section small and provide "
            "`next_section_name` for the next logical section.",
            metadata={
                "path": str(target),
                "error_kind": "ambiguous_full_script_section_finalization",
                "write_session_id": write_session_id,
                "section_name": normalized_section_name,
                "replace_strategy": strategy,
                "next_required_tool": {
                    "tool_name": "file_write",
                    "required_arguments": {"path": path},
                    "required_fields": ["path", "content", "section_name"],
                    "notes": [
                        "Use section_name='full_file' or section_name='final_content' for a complete one-shot script.",
                        "Or use replace_strategy='overwrite' when intentionally replacing the staged file.",
                        "Otherwise provide next_section_name and send only the current logical section.",
                    ],
                },
                "staged_only": True,
            },
        )

    if (
        normalized_next_section
        and strategy == "overwrite"
        and _looks_like_full_script_content(content, normalized_section_name)
    ):
        normalized_next_section = ""

    inferred_next_section = False
    final_chunk = _write_session_should_finalize(
        session,
        section_name=normalized_section_name,
        next_section_name=normalized_next_section,
        replace_strategy=strategy,
        content=content,
    )
    if (
        not normalized_next_section
        and not final_chunk
        and target.suffix.lower() in {".html", ".htm"}
        and _looks_like_complete_html_document(updated_content)
    ):
        final_chunk = True
    infer_next_section = not (
        target.suffix.lower() in {".html", ".htm"}
        and not _section_matches_any_suggestion(session, normalized_section_name)
    )
    if not normalized_next_section and not final_chunk and infer_next_section:
        inferred = _infer_next_suggested_section(session, normalized_section_name)
        if inferred:
            normalized_next_section = inferred
            inferred_next_section = True
    if not final_chunk:
        final_chunk = _write_session_should_finalize(
            session,
            section_name=normalized_section_name,
            next_section_name=normalized_next_section,
            replace_strategy=strategy,
            content=content,
        )

    append_overlap_ratio = 0.0
    if effective_strategy == "append" and staged_content and len(content) >= 0.5 * len(staged_content):
        import difflib
        append_overlap_ratio = difflib.SequenceMatcher(None, content, staged_content, autojunk=False).ratio()

    loop_guard_block = maybe_block_chunked_write(
        state=state,
        session=session,
        path=path,
        cwd=cwd,
        content=content,
        section_name=normalized_section_name,
        next_section_name=normalized_next_section,
        replace_strategy=effective_strategy,
        staged_content=staged_content,
        updated_content=updated_content,
        append_overlap_ratio=append_overlap_ratio,
    )
    if loop_guard_block is not None:
        if state is not None:
            record_write_session_event(
                state,
                event="loop_guard_blocked_write",
                session=session,
                details={
                    "path": str(target),
                    "section_name": normalized_section_name,
                    "next_section_name": normalized_next_section,
                    "error_kind": str(loop_guard_block.get("metadata", {}).get("error_kind", "") or ""),
                    "stagnation_score": int(loop_guard_block.get("metadata", {}).get("loop_guard_score", 0) or 0),
                    "escalation_level": int(
                        loop_guard_block.get("metadata", {}).get("loop_guard_escalation_level", 0) or 0
                    ),
                },
            )
        return loop_guard_block

    if already_checkpointed_rewrite:
        contract = write_session_contract(session)
        next_section = str(getattr(session, "write_next_section", "") or "").strip()
        return fail(
            f"Refusing to rewrite already-checkpointed section `{normalized_section_name}` in Write Session `{write_session_id}`. "
            "Advance to the next legal operation, or explicitly request a full staged overwrite with "
            "`section_name='full_file'` and `replace_strategy='overwrite'`.",
            metadata={
                "path": str(target),
                "staging_path": str(staging_path),
                "error_kind": "chunked_write_already_checkpointed_section",
                "write_session_contract": contract,
                "active_write_session_id": contract["active_write_session_id"],
                "checkpointed_sections": contract["checkpointed_sections"],
                "next_legal_operation": contract["next_legal_operation"],
                "section_name": normalized_section_name,
                "write_sections_completed": previous_sections,
                "staged_only": True,
                "next_required_tool": {
                    "tool_name": "file_write" if next_section else "finalize_write_session",
                    "required_fields": ["path", "content", "section_name"] if next_section else [],
                    "required_arguments": (
                        {"path": path, "section_name": next_section}
                        if next_section
                        else {}
                    ),
                    "notes": [
                        "Do not rewrite checkpointed sections.",
                        "Use section_name='full_file' with replace_strategy='overwrite' only for an intentional full staged overwrite.",
                    ],
                },
            },
        )

    try:
        _write_text_file(staging_path, updated_content, encoding=encoding)
    except Exception as exc:
        return fail(
            f"Unable to write section `{normalized_section_name}` to `{path}`: {exc}",
            metadata={
                "error_kind": "staging_write_io_error",
                "path": str(target),
                "staging_path": str(staging_path),
                "section_name": normalized_section_name,
                "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
                "next_required_tool": {
                    "tool_name": "file_write",
                    "required_fields": ["path", "content"],
                    "required_arguments": {
                        "path": path,
                    },
                    "notes": ["Retry the same section write; if the error persists, call file_read on the staged copy first."],
                },
            },
        )

    session.write_last_staged_hash = _content_hash(updated_content)
    session.write_section_ranges = updated_ranges
    session.write_current_section = normalized_section_name
    session.write_next_section = normalized_next_section
    section_added = _append_unique_section(session.write_sections_completed, normalized_section_name)
    if section_added and session.write_first_chunk_at <= 0:
        session.write_first_chunk_at = time.time()
        record_write_session_event(
            state,
            event="first_chunk_written",
            session=session,
            details={"section_name": normalized_section_name},
        )

    _record_file_change(state, target)
    mark_chunked_write_success(
        state=state,
        path=path,
        cwd=cwd,
        section_name=normalized_section_name,
    )

    status_snapshot = write_session_status_snapshot(
        session,
        cwd=cwd,
        finalized=False,
        encoding=encoding,
    )
    if final_chunk:
        status_snapshot["finalized"] = "pending"
    status_block = format_write_session_status_block(status_snapshot)

    msg = f"Section `{normalized_section_name}` written to `{path}`."
    if normalized_next_section:
        if inferred_next_section:
            msg += f" Next section inferred: `{normalized_next_section}`."
        else:
            msg += f" Waiting for next section: `{normalized_next_section}`."
    elif final_chunk:
        msg += " Final section candidate recorded. Awaiting verifier."
    else:
        msg += " Session remains active for local repair."
    msg += f" Staged copy: `{staging_path}`."
    msg += f"\n{status_block}"

    return ok(
        msg,
        metadata={
            "path": str(target),
            "bytes": len(content.encode(encoding)),
            "staging_path": str(staging_path),
            "write_session_intent": session.write_session_intent,
            "write_session_handle_type": "write_session",
            "write_session_id": write_session_id,
            "write_current_section": normalized_section_name,
            "write_next_section": normalized_next_section,
            "write_sections_completed": session.write_sections_completed,
            "write_section_ranges": session.write_section_ranges,
            "write_session_contract": write_session_contract(session),
            "active_write_session_id": write_session_id,
            "checkpointed_sections": list(session.write_sections_completed),
            "next_legal_operation": write_session_contract(session)["next_legal_operation"],
            "write_session_staged_hash": session.write_last_staged_hash,
            "write_session_status_block": status_block,
            "write_session_finalized": False,
            "write_session_final_chunk": final_chunk,
            "write_next_section_inferred": inferred_next_section,
            "section_name": normalized_section_name,
            "section_id": str(section_id or normalized_section_name),
            "section_role": str(section_role or ""),
            "section_added": section_added,
            "replace_strategy": effective_strategy,
            "expected_followup_verifier": str(expected_followup_verifier or ""),
            "staged_only": True,
        },
    )
