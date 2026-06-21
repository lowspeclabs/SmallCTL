from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import libcst as cst
except Exception:  # pragma: no cover - import fallback is only for degraded envs
    cst = None

from ..state import LoopState
from .common import fail, ok
from .ast_patch_results import (
    build_ast_patch_metadata,
    unsupported_language_failure,
)
from .ast_patch_operations import apply_python_ast_patch
from .fs_sessions import (
    _clone_section_ranges,
    _record_file_change,
)
from .fs_write_session_policy import _guard_write_session_staging_mutation
from .fs_write_sessions import (
    _content_hash,
    _read_text_file,
    _resolve_patch_source,
    _write_text_file,
    format_write_session_status_block,
    write_session_status_snapshot,
)
from .ast_cst_transformers import AstPatchError, PythonAstPatchOutcome


def _resolve_ast_patch_source(
    *,
    path: str,
    cwd: str | None,
    encoding: str,
    state: LoopState | None,
    write_session_id: str | None,
    session_id: str | None,
) -> tuple[Path, Any, bool, dict[str, Any] | None]:
    from .fs import _resolve

    if not write_session_id and session_id:
        write_session_id = session_id

    target_path = _resolve(path, cwd)

    staging_guard = _guard_write_session_staging_mutation(
        tool_name="ast_patch",
        path=path,
        state=state,
        cwd=cwd,
        write_session_id=write_session_id,
        encoding=encoding,
    )
    if staging_guard is not None:
        return target_path, None, False, staging_guard

    if write_session_id:
        try:
            source_path, _, session, staged_only = _resolve_patch_source(
                state,
                path,
                cwd=cwd,
                encoding=encoding,
                write_session_id=write_session_id,
            )
        except (LookupError, ValueError) as exc:
            return target_path, None, False, fail(
                str(exc),
                metadata={
                    "path": str(target_path),
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

    return source_path, session, staged_only, None


def _authorize_ast_patch(
    *,
    path: str,
    target_path: Path,
    state: LoopState | None,
    staged_only: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    if staged_only or dry_run:
        return None
    from ..risk_policy import evaluate_risk_policy
    risk_decision = evaluate_risk_policy(
        state if state is not None else LoopState(cwd=str(Path.cwd())),
        tool_name="ast_patch",
        tool_risk="high",
        phase=str((state.current_phase if state is not None else "") or ""),
        action=f"Structurally patch file {path}",
        expected_effect="Apply a targeted structural edit to the target file.",
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
    return None


def _persist_ast_patch(
    *,
    source_path: Path,
    outcome: PythonAstPatchOutcome,
    state: LoopState | None,
    target_path: Path,
    cwd: str | None,
    encoding: str,
    session: Any,
    staged_only: bool,
    dry_run: bool,
) -> str | None:
    if not outcome.changed or dry_run:
        return None
    _write_text_file(source_path, outcome.updated_text, encoding=encoding)
    if session is not None:
        session.write_last_staged_hash = _content_hash(outcome.updated_text)
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
    _record_file_change(state, target_path)
    return status_block


def _build_ast_patch_result(
    *,
    path: str,
    target_path: Path,
    source_path: Path,
    session: Any,
    staged_only: bool,
    language: str,
    operation: str,
    target: dict[str, Any],
    payload: dict[str, Any],
    outcome: PythonAstPatchOutcome,
    source_text: str,
    dry_run: bool,
    expected_followup_verifier: str | None,
    status_block: str | None,
) -> dict[str, Any]:
    metadata = build_ast_patch_metadata(
        path=target_path,
        requested_path=path,
        source_path=source_path,
        session=session,
        staged_only=staged_only,
        language=language,
        operation=operation,
        target=target,
        payload=payload,
        changed=outcome.changed,
        updated_text=outcome.updated_text,
        original_text=source_text,
        matched_node_count=outcome.matched_node_count,
        touched_symbols=outcome.touched_symbols,
        dry_run=dry_run,
        expected_followup_verifier=expected_followup_verifier,
        staging_path=source_path if staged_only else None,
        status_block=status_block,
    )

    if dry_run:
        if outcome.changed:
            return ok(
                f"Dry run prepared structural patch for `{path}`.",
                metadata=metadata,
            )
        return ok(
            f"Dry run found no structural change needed for `{path}`.",
            metadata=metadata,
        )

    if not outcome.changed:
        return ok(
            f"No structural change was needed for `{path}`.",
            metadata=metadata,
        )

    message = f"Structurally patched `{path}` with `{operation}`."
    if staged_only:
        message += f" Staged copy: `{source_path}`."
        if status_block:
            message += f"\n{status_block}"
    return ok(message, metadata=metadata)


async def handle_ast_patch(
    *,
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
    from .fs import (
        _guard_suspicious_temp_root_path,
        _mark_repeat_patch,
        _repair_cycle_allows_patch,
        _repair_cycle_read_required_metadata,
        _resolve,
    )

    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    from .fs_listing import _guard_workspace_containment
    workspace = cwd or (state.cwd if state is not None else None)
    if workspace:
        containment = _guard_workspace_containment(path, workspace, operation="ast_patch")
        if containment is not None:
            return containment

    normalized_language = str(language or "python").strip().lower() or "python"
    normalized_operation = str(operation or "").strip()
    normalized_target = dict(target or {})
    normalized_payload = dict(payload or {})
    normalized_dry_run = bool(dry_run)
    target_path = _resolve(path, cwd)

    if not normalized_operation:
        return fail(
            "AST patch operation is required.",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "error_kind": "ast_operation_invalid",
                "language": normalized_language,
            },
        )

    if normalized_language != "python":
        return unsupported_language_failure(
            path=target_path,
            requested_path=path,
            language=normalized_language,
            operation=normalized_operation,
        )
    if cst is None:
        return fail(
            "`ast_patch` requires `libcst` for Python structural edits, but the dependency is not available in this environment.",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "error_kind": "ast_operation_invalid",
                "language": normalized_language,
                "operation": normalized_operation,
                "next_action_hint": "Install `libcst` in the active environment, then retry the structural patch.",
            },
        )

    source_path, session, staged_only, guard = _resolve_ast_patch_source(
        path=path,
        cwd=cwd,
        encoding=encoding,
        state=state,
        write_session_id=write_session_id,
        session_id=session_id,
    )
    if guard is not None:
        return guard

    if not staged_only and not normalized_dry_run and not _repair_cycle_allows_patch(state, target_path):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata=_repair_cycle_read_required_metadata(
                state,
                target_path,
                requested_path=path,
                extra={
                    "language": normalized_language,
                    "operation": normalized_operation,
                },
            ),
        )

    if staged_only and not source_path.exists():
        return fail(
            f"Active staged copy `{source_path}` is missing for target `{path}`.",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "source_path": str(source_path),
                "staged_only": True,
                "error_kind": "ast_target_not_found",
                "language": normalized_language,
                "operation": normalized_operation,
                "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
            },
        )

    auth_guard = _authorize_ast_patch(
        path=path,
        target_path=target_path,
        state=state,
        staged_only=staged_only,
        dry_run=normalized_dry_run,
    )
    if auth_guard is not None:
        return auth_guard

    try:
        source_text = _read_text_file(source_path, encoding=encoding)
    except Exception as exc:
        return fail(
            f"Unable to read file for AST patching: {exc}",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "source_path": str(source_path),
                "error_kind": "parse_failed",
                "language": normalized_language,
                "operation": normalized_operation,
            },
        )

    try:
        outcome = apply_python_ast_patch(
            source_text=source_text,
            operation=normalized_operation,
            target=normalized_target,
            payload=normalized_payload,
        )
    except AstPatchError as exc:
        return fail(
            exc.message,
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "source_path": str(source_path),
                "staged_only": staged_only,
                "error_kind": exc.error_kind,
                "language": normalized_language,
                "operation": normalized_operation,
                "target": normalized_target,
                **exc.extra,
                **(
                    {
                        "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
                    }
                    if session is not None
                    else {}
                ),
            },
        )

    status_block = _persist_ast_patch(
        source_path=source_path,
        outcome=outcome,
        state=state,
        target_path=target_path,
        cwd=cwd,
        encoding=encoding,
        session=session,
        staged_only=staged_only,
        dry_run=normalized_dry_run,
    )

    return _build_ast_patch_result(
        path=path,
        target_path=target_path,
        source_path=source_path,
        session=session,
        staged_only=staged_only,
        language=normalized_language,
        operation=normalized_operation,
        target=normalized_target,
        payload=normalized_payload,
        outcome=outcome,
        source_text=source_text,
        dry_run=normalized_dry_run,
        expected_followup_verifier=expected_followup_verifier,
        status_block=status_block,
    )
