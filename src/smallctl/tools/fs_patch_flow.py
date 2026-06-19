from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail, ok
from .fs_patching import (
    _apply_exact_patch_plan,
    _apply_regex_patch_plan,
    _build_patch_best_match,
    _build_patch_ambiguity_hint,
    _build_patch_failure_metadata,
    _build_patch_failure_message,
    _build_patch_metadata,
    _build_patch_text_preview,
    _count_exact_occurrences,
)
from .fs_sessions import (
    _append_unique_section,
    _clone_section_ranges,
    _repair_cycle_allows_patch,
    _repair_cycle_read_required_metadata,
    _record_file_change,
)
from .fs_write_sessions import (
    _content_hash,
    _read_text_file,
    _resolve_patch_source,
    _write_text_file,
    format_write_session_status_block,
    write_session_status_snapshot,
)
from .fs_write_session_policy import _guard_write_session_staging_mutation
from .fs_patch_flow_support import (
    _empty_target_patch_file_write_metadata,
    _repeat_sensitive_patch_records,
    _repeat_sensitive_patch_signature,
    _verifier_traceback_focus,
)


async def handle_file_patch(
    *,
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
    from .fs import _guard_suspicious_temp_root_path, _mark_repeat_patch, _resolve

    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    target = _resolve(path, cwd)
    if not write_session_id and session_id:
        write_session_id = session_id

    normalized_target_text = str(target_text or "")
    normalized_replacement_text = str(replacement_text or "")

    if normalized_target_text and normalized_target_text == normalized_replacement_text:
        return fail(
            "No changes needed — text already matches",
            metadata={
                "path": str(target),
                "requested_path": path,
                "error_kind": "patch_noop_identical_text",
                "changed": False,
                "expected_occurrences": expected_occurrences,
                "target_text_preview": _build_patch_text_preview(normalized_target_text),
                "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                "suppress_failure_persistence": True,
            },
        )

    if normalized_target_text == "":
        file_write_metadata = _empty_target_patch_file_write_metadata(
            path=path,
            target=target,
            cwd=cwd,
            encoding=encoding,
            state=state,
            write_session_id=write_session_id,
            replacement_text=normalized_replacement_text,
        )
        if file_write_metadata is not None:
            return fail(
                "Patch target text is empty, but the target appears to be new or empty. "
                "Use `file_write` with `replace_strategy='overwrite'` to create the initial content.",
                metadata=file_write_metadata,
            )
        return fail(
            "Patch target text cannot be empty. Provide a non-empty exact anchor from the current file, or use `ast_patch` for Python structural edits.",
            metadata={
                "path": str(target),
                "requested_path": path,
                "error_kind": "patch_target_empty",
                "expected_occurrences": expected_occurrences,
                "recovery_hint": (
                    "Read the current file content and retry with a non-empty `target_text`. "
                    "For Python edits that are easier to locate by function, class, import, call, or field, use `ast_patch`."
                ),
                "suggested_tools": ["ast_patch"],
            },
        )

    try:
        normalized_expected_occurrences = int(expected_occurrences)
    except (TypeError, ValueError):
        return fail(
            "Patch received an invalid `expected_occurrences` value.",
            metadata={
                "path": str(target),
                "requested_path": path,
                "error_kind": "patch_occurrence_mismatch",
                "expected_occurrences": expected_occurrences,
            },
        )
    if normalized_expected_occurrences < 1:
        return fail(
            "`expected_occurrences` must be at least 1.",
            metadata={
                "path": str(target),
                "requested_path": path,
                "error_kind": "patch_occurrence_mismatch",
                "expected_occurrences": normalized_expected_occurrences,
            },
        )

    normalized_occurrence_index: int | None = None
    if occurrence_index is not None:
        try:
            normalized_occurrence_index = int(occurrence_index)
        except (TypeError, ValueError):
            return fail(
                "`occurrence_index` must be a one-based integer.",
                metadata={
                    "path": str(target),
                    "requested_path": path,
                    "error_kind": "invalid_occurrence_index",
                    "occurrence_index": occurrence_index,
                    "expected_occurrences": normalized_expected_occurrences,
                },
            )
        if normalized_occurrence_index < 1:
            return fail(
                "`occurrence_index` must be at least 1.",
                metadata={
                    "path": str(target),
                    "requested_path": path,
                    "error_kind": "invalid_occurrence_index",
                    "occurrence_index": normalized_occurrence_index,
                    "expected_occurrences": normalized_expected_occurrences,
                },
            )

    staging_guard = _guard_write_session_staging_mutation(
        tool_name="file_patch",
        path=path,
        state=state,
        cwd=cwd,
        write_session_id=write_session_id,
        encoding=encoding,
    )
    if staging_guard is not None:
        return staging_guard

    if write_session_id:
        try:
            source_path, _, session, staged_only = _resolve_patch_source(
                state,
                path,
                cwd=cwd,
                encoding=encoding,
                write_session_id=write_session_id,
            )
        except LookupError as exc:
            return fail(
                str(exc),
                metadata={
                    "path": str(target),
                    "requested_path": path,
                    "error_kind": "session_id_mismatch",
                    "write_session_id": write_session_id,
                },
            )
        except ValueError as exc:
            return fail(
                str(exc),
                metadata={
                    "path": str(target),
                    "requested_path": path,
                    "error_kind": "session_id_mismatch",
                    "write_session_id": write_session_id,
                },
            )
    else:
        session = getattr(state, "write_session", None) if state is not None else None
        source_path, _, session, staged_only = _resolve_patch_source(
            state,
            path,
            cwd=cwd,
            encoding=encoding,
        )

    if not source_path.exists():
        message = (
            f"Active staged copy `{source_path}` is missing for target `{path}`."
            if staged_only
            else f"Patch target text was not found in `{path}`."
        )
        return fail(
            message,
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="patch_target_not_found",
                extra={
                    "actual_occurrences": 0,
                    "expected_occurrences": normalized_expected_occurrences,
                },
            ),
        )

    try:
        source_text = _read_text_file(source_path, encoding=encoding)
    except Exception as exc:
        return fail(f"Unable to read file for patching: {exc}")

    patch_signature = _repeat_sensitive_patch_signature(
        source_path=source_path,
        target_text=normalized_target_text,
        replacement_text=normalized_replacement_text,
        expected_occurrences=normalized_expected_occurrences,
    )
    repeat_records = _repeat_sensitive_patch_records(state)
    prior_repeat_sensitive_patch = repeat_records.get(patch_signature)
    if (
        isinstance(prior_repeat_sensitive_patch, dict)
        and normalized_replacement_text
        and normalized_replacement_text in source_text
    ):
        _mark_repeat_patch(state)
        traceback_focus = _verifier_traceback_focus(
            state,
            source_path=source_path,
            requested_path=path,
        )
        next_required_tool = {
            "tool_name": "file_read",
            "required_arguments": {"path": path},
            "reason": "already_applied_patch_requires_current_file_snapshot",
        }
        if isinstance(traceback_focus, dict):
            next_required_tool = traceback_focus.get("next_required_tool", next_required_tool)
        return fail(
            (
                "This exact patch already landed; the old target text is gone and the replacement text "
                "is present. Do not retry the prior patch; read the current verifier traceback slice "
                "and patch the live failing line."
            ),
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="repeat_sensitive_patch_already_applied",
                extra={
                    "prior_patch": prior_repeat_sensitive_patch,
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                    "suggested_tools": ["file_read", "ast_patch"],
                    "next_required_tool": next_required_tool,
                    "verifier_traceback_focus": traceback_focus,
                },
            ),
        )

    repair_allows = _repair_cycle_allows_patch(state, target)
    print(f"DEBUG fs_patch_flow: staged_only={staged_only}, repair_allows={repair_allows}, state.repair_cycle_id={getattr(state, 'repair_cycle_id', None)}")
    if not staged_only and not repair_allows:
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata=_repair_cycle_read_required_metadata(state, target, requested_path=path),
        )

    if not staged_only:
        from ..risk_policy import evaluate_risk_policy

        risk_decision = evaluate_risk_policy(
            state if state is not None else LoopState(cwd=str(Path.cwd())),
            tool_name="file_patch",
            tool_risk="high",
            phase=str((state.current_phase if state is not None else "") or ""),
            action=f"Patch file {path}",
            expected_effect="Replace exact text in the target file.",
            rollback="Restore the previous file contents from the snapshot or revert the patch.",
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

    actual_occurrences = _count_exact_occurrences(source_text, normalized_target_text) if not regex else None
    if actual_occurrences == 0:
        ambiguity_hint = _build_patch_ambiguity_hint(
            actual_occurrences=actual_occurrences,
            expected_occurrences=normalized_expected_occurrences,
        )
        return fail(
            _build_patch_failure_message(
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                error_kind="patch_target_not_found",
                actual_occurrences=actual_occurrences,
                expected_occurrences=normalized_expected_occurrences,
                regex=bool(regex),
            ),
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="patch_target_not_found",
                extra={
                    "actual_occurrences": 0,
                    "expected_occurrences": normalized_expected_occurrences,
                    "ambiguity_hint": ambiguity_hint,
                    "best_match": _build_patch_best_match(source_text, normalized_target_text),
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                },
            ),
        )
    if actual_occurrences is not None and actual_occurrences != normalized_expected_occurrences:
        ambiguity_hint = _build_patch_ambiguity_hint(
            actual_occurrences=actual_occurrences,
            expected_occurrences=normalized_expected_occurrences,
        )
        return fail(
            _build_patch_failure_message(
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                error_kind="patch_occurrence_mismatch",
                actual_occurrences=actual_occurrences,
                expected_occurrences=normalized_expected_occurrences,
                regex=bool(regex),
            ),
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="patch_occurrence_mismatch",
                extra={
                    "actual_occurrences": actual_occurrences,
                    "expected_occurrences": normalized_expected_occurrences,
                    "ambiguity_hint": ambiguity_hint,
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                },
            ),
        )

    if isinstance(prior_repeat_sensitive_patch, dict):
        return fail(
            (
                "This exact insertion-style patch already succeeded and the target text is still present. "
                "Read the current file and retry with a target that includes the newly intended context, "
                "or use `ast_patch` for a structural Python edit."
            ),
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="repeat_sensitive_patch_already_applied",
                extra={
                    "actual_occurrences": actual_occurrences,
                    "expected_occurrences": normalized_expected_occurrences,
                    "prior_patch": prior_repeat_sensitive_patch,
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                    "suggested_tools": ["file_read", "ast_patch"],
                },
            ),
        )

    try:
        if regex:
            plan = _apply_regex_patch_plan(
                source_text,
                normalized_target_text,
                normalized_replacement_text,
                expected_occurrences=normalized_expected_occurrences,
                occurrence_index=normalized_occurrence_index,
                case_insensitive=bool(case_insensitive),
                multiline=bool(multiline),
                dotall=bool(dotall),
                fromfile=str(source_path),
                tofile=str(source_path),
                encoding=encoding,
            )
        else:
            plan = _apply_exact_patch_plan(
                source_text,
                normalized_target_text,
                normalized_replacement_text,
                expected_occurrences=normalized_expected_occurrences,
                occurrence_index=normalized_occurrence_index,
                fromfile=str(source_path),
                tofile=str(source_path),
                encoding=encoding,
            )
    except re.error as exc:
        return fail(
            f"Invalid regex for file_patch: {exc}",
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="invalid_regex",
                extra={
                    "expected_occurrences": normalized_expected_occurrences,
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                },
            ),
        )
    except ValueError as exc:
        reason = str(exc)
        if reason == "empty_regex_match":
            return fail(
                "Regex patch produced an empty match; use a pattern that consumes at least one character.",
                metadata=_build_patch_failure_metadata(
                    path=target,
                    requested_path=path,
                    source_path=source_path,
                    staged_only=staged_only,
                    session=session,
                    error_kind="invalid_regex",
                    extra={
                        "expected_occurrences": normalized_expected_occurrences,
                        "target_text_preview": _build_patch_text_preview(normalized_target_text),
                        "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                    },
                ),
            )
        if reason == "invalid_occurrence_index":
            return fail(
                f"`occurrence_index` {normalized_occurrence_index} did not select an observed occurrence.",
                metadata=_build_patch_failure_metadata(
                    path=target,
                    requested_path=path,
                    source_path=source_path,
                    staged_only=staged_only,
                    session=session,
                    error_kind="invalid_occurrence_index",
                    extra={
                        "occurrence_index": normalized_occurrence_index,
                        "actual_occurrences": actual_occurrences,
                        "expected_occurrences": normalized_expected_occurrences,
                        "target_text_preview": _build_patch_text_preview(normalized_target_text),
                        "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                    },
                ),
            )
        if actual_occurrences is None:
            try:
                actual_occurrences = int(reason)
            except (TypeError, ValueError):
                actual_occurrences = 0
        return fail(
            _build_patch_failure_message(
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                error_kind="patch_occurrence_mismatch",
                actual_occurrences=actual_occurrences,
                expected_occurrences=normalized_expected_occurrences,
                regex=bool(regex),
            ),
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="patch_occurrence_mismatch",
                extra={
                    "actual_occurrences": actual_occurrences,
                    "expected_occurrences": normalized_expected_occurrences,
                    "ambiguity_hint": _build_patch_ambiguity_hint(
                        actual_occurrences=actual_occurrences,
                        expected_occurrences=normalized_expected_occurrences,
                    ),
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                },
            ),
        )

    updated_text = plan.new_text
    occurrence_count = plan.replacement_count
    if updated_text == source_text:
        return fail(
            "Patch would not change file content.",
            metadata=_build_patch_failure_metadata(
                path=target,
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                session=session,
                error_kind="patch_noop",
                extra={
                    "changed": False,
                    "actual_occurrences": actual_occurrences,
                    "expected_occurrences": normalized_expected_occurrences,
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                    "suggested_tools": ["file_read", "ast_patch"],
                },
            ),
        )

    if not dry_run:
        try:
            _write_text_file(source_path, updated_text, encoding=encoding)
        except Exception as exc:
            return fail(f"Unable to patch file: {exc}")

    if not dry_run and normalized_target_text:
        repeat_records[patch_signature] = {
            "path": str(target),
            "source_path": str(source_path),
            "requested_path": path,
            "target_text_preview": _build_patch_text_preview(normalized_target_text),
            "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
            "occurrence_count": occurrence_count,
            "expected_occurrences": normalized_expected_occurrences,
        }

    section_added = False
    write_session_final_chunk = False
    if session is not None and not dry_run:
        if (
            staged_only
            and str(getattr(session, "write_session_intent", "") or "").strip().lower() == "patch_existing"
            and not list(getattr(session, "write_sections_completed", []) or [])
        ):
            section_name = str(
                getattr(session, "write_next_section", "")
                or getattr(session, "write_current_section", "")
                or "patch"
            ).strip() or "patch"
            session.write_current_section = section_name
            section_added = _append_unique_section(session.write_sections_completed, section_name)
            session.write_next_section = ""
            session.write_pending_finalize = True
            write_session_final_chunk = True
        session.write_last_staged_hash = _content_hash(updated_text)
        session.write_last_attempt_sections = list(getattr(session, "write_sections_completed", []) or [])
        session.write_last_attempt_ranges = _clone_section_ranges(getattr(session, "write_section_ranges", {}) or {})
        status_snapshot = write_session_status_snapshot(
            session,
            cwd=cwd,
            finalized=False,
            encoding=encoding,
        )
        status_block = format_write_session_status_block(status_snapshot)
    else:
        status_block = None

    if not dry_run:
        _record_file_change(state, target)
    metadata = _build_patch_metadata(
        path=target,
        requested_path=path,
        target_text=normalized_target_text,
        replacement_text=normalized_replacement_text,
        occurrence_count=occurrence_count,
        expected_occurrences=normalized_expected_occurrences,
        source_path=source_path,
        staged_only=staged_only,
        encoding=encoding,
        session=session,
        staging_path=source_path if staged_only else None,
        status_block=status_block,
        plan=plan,
        dry_run=bool(dry_run),
    )
    if session is not None:
        metadata.update(
            {
                "write_current_section": str(getattr(session, "write_current_section", "") or ""),
                "write_next_section": str(getattr(session, "write_next_section", "") or ""),
                "write_sections_completed": list(getattr(session, "write_sections_completed", []) or []),
                "write_session_final_chunk": write_session_final_chunk,
                "section_added": section_added,
            }
        )
    metadata["expected_followup_verifier"] = str(expected_followup_verifier or "")
    if dry_run:
        message = f"Dry run: patch would replace {occurrence_count} occurrence(s) in `{path}`."
    else:
        message = f"Patched {occurrence_count} occurrence(s) in `{path}`."
    if staged_only:
        message += f" Staged copy: `{source_path}`."
        if status_block:
            message += f"\n{status_block}"
    return ok(message, metadata=metadata)
