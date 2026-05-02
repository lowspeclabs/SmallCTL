from __future__ import annotations

from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail, ok
from .fs_patching import (
    _apply_exact_patch,
    _build_patch_ambiguity_hint,
    _build_patch_failure_metadata,
    _build_patch_failure_message,
    _build_patch_metadata,
    _build_patch_text_preview,
    _count_exact_occurrences,
)
from .fs_sessions import (
    _clone_section_ranges,
    _repair_cycle_allows_patch,
    _repair_cycle_reads,
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


_REPEAT_SENSITIVE_PATCHES_KEY = "_repeat_sensitive_file_patches"


def _repeat_sensitive_patch_signature(
    *,
    source_path: Path,
    target_text: str,
    replacement_text: str,
    expected_occurrences: int,
) -> str:
    return "|".join(
        [
            str(source_path),
            _content_hash(target_text),
            _content_hash(replacement_text),
            str(expected_occurrences),
        ]
    )


def _repeat_sensitive_patch_records(state: LoopState | None) -> dict[str, dict[str, Any]]:
    if state is None:
        return {}
    records = state.scratchpad.get(_REPEAT_SENSITIVE_PATCHES_KEY)
    if not isinstance(records, dict):
        records = {}
        state.scratchpad[_REPEAT_SENSITIVE_PATCHES_KEY] = records
    return records


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
) -> dict[str, Any]:
    from .fs import _guard_suspicious_temp_root_path, _mark_repeat_patch, _repair_cycle_allows_patch, _resolve, _repair_cycle_reads

    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    target = _resolve(path, cwd)
    if not write_session_id and session_id:
        write_session_id = session_id

    normalized_target_text = str(target_text or "")
    normalized_replacement_text = str(replacement_text or "")

    if normalized_target_text == "":
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

    if not staged_only and not _repair_cycle_allows_patch(state, target):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata={
                "path": str(target),
                "requested_path": path,
                "system_repair_cycle_id": getattr(state, "repair_cycle_id", ""),
                "required_read_paths": _repair_cycle_reads(state),
                "error_kind": "repair_cycle_read_required",
            },
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

    try:
        source_text = _read_text_file(source_path, encoding=encoding)
    except Exception as exc:
        return fail(f"Unable to read file for patching: {exc}")

    actual_occurrences = _count_exact_occurrences(source_text, normalized_target_text)
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
                    "target_text_preview": _build_patch_text_preview(normalized_target_text),
                    "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
                },
            ),
        )
    if actual_occurrences != normalized_expected_occurrences:
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

    patch_signature = _repeat_sensitive_patch_signature(
        source_path=source_path,
        target_text=normalized_target_text,
        replacement_text=normalized_replacement_text,
        expected_occurrences=normalized_expected_occurrences,
    )
    repeat_records = _repeat_sensitive_patch_records(state)
    prior_repeat_sensitive_patch = repeat_records.get(patch_signature)
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
        updated_text, occurrence_count = _apply_exact_patch(
            source_text,
            normalized_target_text,
            normalized_replacement_text,
            expected_occurrences=normalized_expected_occurrences,
        )
    except ValueError:
        return fail(
            _build_patch_failure_message(
                requested_path=path,
                source_path=source_path,
                staged_only=staged_only,
                error_kind="patch_occurrence_mismatch",
                actual_occurrences=actual_occurrences,
                expected_occurrences=normalized_expected_occurrences,
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

    try:
        _write_text_file(source_path, updated_text, encoding=encoding)
    except Exception as exc:
        return fail(f"Unable to patch file: {exc}")

    if normalized_target_text and normalized_target_text in normalized_replacement_text and normalized_target_text in updated_text:
        repeat_records[patch_signature] = {
            "path": str(target),
            "source_path": str(source_path),
            "requested_path": path,
            "target_text_preview": _build_patch_text_preview(normalized_target_text),
            "replacement_text_preview": _build_patch_text_preview(normalized_replacement_text),
            "occurrence_count": occurrence_count,
            "expected_occurrences": normalized_expected_occurrences,
        }

    if session is not None:
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
    )
    metadata["expected_followup_verifier"] = str(expected_followup_verifier or "")
    message = f"Patched {occurrence_count} occurrence(s) in `{path}`."
    if staged_only:
        message += f" Staged copy: `{source_path}`."
        if status_block:
            message += f"\n{status_block}"
    return ok(message, metadata=metadata)
