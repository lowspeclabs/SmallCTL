from __future__ import annotations

from pathlib import Path
from typing import Any

from ..state import LoopState
from ..risk_policy import evaluate_risk_policy
from .common import fail, ok
from .fs_sessions import _mark_repeat_patch, _repair_cycle_allows_patch, _repair_cycle_reads, _record_file_change, _normalize_section_name
from .fs_write_session_policy import _guard_suspicious_temp_root_path, _guard_write_session_staging_mutation
from .fs_write_sessions import _resolve


async def file_append(
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

    if not write_session_id and session_id:
        write_session_id = session_id
    if write_session_id:
        from .fs import file_write

        return await file_write(
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
            replace_strategy=replace_strategy or "append",
            expected_followup_verifier=expected_followup_verifier,
        )
    staging_guard = _guard_write_session_staging_mutation(
        tool_name="file_append",
        path=path,
        state=state,
        cwd=cwd,
        encoding=encoding,
        section_name=_normalize_section_name(section_name, section_id),
    )
    if staging_guard is not None:
        return staging_guard
    target = _resolve(path, cwd)
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
        tool_name="file_append",
        tool_risk="high",
        phase=str((state.current_phase if state is not None else "") or ""),
        action=f"Append to file {path}",
        expected_effect="Append the requested content to the target file.",
        rollback="Remove the appended content or restore from the snapshot if needed.",
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
        with target.open("a", encoding=encoding) as fh:
            fh.write(content)
    except Exception as exc:
        return fail(f"Unable to append file: {exc}")
    _record_file_change(state, target)
    return ok("appended", metadata={"path": str(target)})


async def file_delete(
    path: str,
    cwd: str | None = None,
    state: LoopState | None = None,
) -> dict[str, Any]:
    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    staging_guard = _guard_write_session_staging_mutation(
        tool_name="file_delete",
        path=path,
        state=state,
        cwd=cwd,
    )
    if staging_guard is not None:
        return staging_guard

    target = _resolve(path, cwd)
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
        tool_name="file_delete",
        tool_risk="high",
        phase=str((state.current_phase if state is not None else "") or ""),
        action=f"Delete file {path}",
        expected_effect="Remove the target file.",
        rollback="Restore the file from version control or backup if needed.",
        verification="Confirm the file is gone and the task state still matches expectations.",
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
        if not target.exists():
            return fail(f"File does not exist: {target}")
        target.unlink()
    except Exception as exc:
        return fail(f"Unable to delete file: {exc}")
    _record_file_change(state, target)
    return ok("deleted", metadata={"path": str(target)})
