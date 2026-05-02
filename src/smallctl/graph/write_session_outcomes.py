from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import clip_text_value
from ..tools.fs import (
    format_write_session_status_block,
    promote_write_session_target,
    write_session_status_snapshot,
    write_session_verify_path,
)
from ..write_session_fsm import (
    record_write_session_event as record_write_session_event_alias,
    recent_write_session_events,
    transition_write_session,
)
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord
from .tool_execution_support import _tool_envelope_from_dict
from .write_session_plan_support import (
    _auto_update_active_plan_step,
    _build_plan_export_recovery_message,
    _is_plan_export_validation_error,
)
from . import write_session_recovery as _write_session_recovery

_WRITE_SESSION_SCHEMA_FAILURE_KEY = "_last_write_session_schema_failure"
_WRITE_SESSION_REPLAY_COUNTS_KEY = "_write_session_stranded_replay_counts"
_WRITE_SESSION_REPLAY_GUARD_THRESHOLD = 2
_maybe_schedule_write_recovery_readback = _write_session_recovery._maybe_schedule_write_recovery_readback
_maybe_record_write_session_first_chunk_metric = _write_session_recovery._maybe_record_write_session_first_chunk_metric
_register_write_session_stage_artifact = _write_session_recovery._register_write_session_stage_artifact
_invalidate_write_session_stage_artifacts = _write_session_recovery._invalidate_write_session_stage_artifacts
_maybe_emit_patch_existing_first_choice_nudge = _write_session_recovery._maybe_emit_patch_existing_first_choice_nudge
_recover_patch_existing_recovery_session = _write_session_recovery._recover_patch_existing_recovery_session
_maybe_schedule_patch_existing_stage_read_recovery = _write_session_recovery._maybe_schedule_patch_existing_stage_read_recovery
_maybe_schedule_file_patch_read_recovery = _write_session_recovery._maybe_schedule_file_patch_read_recovery
_maybe_emit_write_session_target_path_redirect_nudge = _write_session_recovery._maybe_emit_write_session_target_path_redirect_nudge
_clear_patch_existing_stage_read_autocontinue_count_after_success = (
    _write_session_recovery._clear_patch_existing_stage_read_autocontinue_count_after_success
)


def _increment_stranded_replay_count(
    harness: Any,
    *,
    session_id: str,
    operation_id: str,
) -> int:
    key = f"{session_id}:{operation_id}"
    counts = harness.state.scratchpad.setdefault(_WRITE_SESSION_REPLAY_COUNTS_KEY, {})
    if not isinstance(counts, dict):
        counts = {}
    count = int(counts.get(key, 0) or 0) + 1
    counts[key] = count
    harness.state.scratchpad[_WRITE_SESSION_REPLAY_COUNTS_KEY] = counts
    return count


def _preserve_unverified_section(harness: Any, session: Any, record: ToolExecutionRecord) -> None:
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    current_section = str(record.result.metadata.get("write_current_section") or session.write_current_section or "").strip()
    record_write_session_event_alias(
        harness.state,
        event="failed_stage_preserved",
        session=session,
        details={
            "section": current_section,
            "tool_name": record.tool_name,
            "staging_path": staging_path,
        },
    )
    harness._runlog(
        "write_session_failed_stage_preserved",
        "preserved staged content after verifier failure",
        session_id=session.write_session_id,
        detail=staging_path,
    )


def _maybe_trigger_write_session_fallback(harness: Any, session: Any) -> bool:
    config = getattr(harness, "config", None)
    limit = config.failed_local_patch_limit if config else 3
    if session.write_failed_local_patches < limit:
        return False

    transition_write_session(
        session,
        next_mode=session.write_session_fallback_mode or "stub_and_fill",
        next_status="fallback",
    )
    msg = (
        f"Write Session `{session.write_session_id}` has encountered {session.write_failed_local_patches} failures "
        f"during verification/repair. Transitioning to `{session.write_session_mode}` mode. "
        "Write a minimal stable scaffold first, then fill in one section at a time."
    )
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=msg,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "write_session_fallback",
                "session_id": session.write_session_id,
                "failures": session.write_failed_local_patches,
            },
        )
    )
    harness._runlog(
        "write_session_fallback_triggered",
        "too many failures during verification, suggesting stub_and_fill",
        session_id=session.write_session_id,
        failures=session.write_failed_local_patches,
    )
    record_write_session_event_alias(
        harness.state,
        event="fallback_triggered",
        session=session,
        details={"failures": session.write_failed_local_patches},
    )
    return True


async def _handle_write_session_outcome(harness: Any, record: ToolExecutionRecord) -> None:
    session = getattr(harness.state, "write_session", None)
    if not session:
        return

    if record.tool_name in {"file_write", "file_append", "file_patch", "ast_patch"}:
        res_session_id = record.result.metadata.get("write_session_id")
        if not res_session_id or res_session_id != session.write_session_id:
            return

        if record.result.success:
            harness.state.scratchpad.pop(_WRITE_SESSION_SCHEMA_FAILURE_KEY, None)
            _clear_patch_existing_stage_read_autocontinue_count_after_success(harness, session, record)
            current_section = str(record.result.metadata.get("write_current_section") or session.write_current_section or "").strip()
            next_section = str(record.result.metadata.get("write_next_section") or "").strip()
            final_chunk = bool(record.result.metadata.get("write_session_final_chunk"))
            verdict = await _run_syntax_check(
                harness,
                write_session_verify_path(session, getattr(harness.state, "cwd", None)),
            )
            if verdict:
                session.write_last_verifier = verdict
                if verdict.get("verdict") == "fail":
                    session.write_failed_local_patches += 1
                    transition_write_session(
                        session,
                        next_mode="local_repair",
                        next_status="local_repair",
                        pending_finalize=final_chunk,
                    )
                    verify_path = write_session_verify_path(session, getattr(harness.state, "cwd", None))
                    if current_section:
                        transition_write_session(
                            session,
                            current_section=current_section,
                            next_section=current_section,
                        )
                    record_write_session_event_alias(
                        harness.state,
                        event="verifier_fail",
                        session=session,
                        details={"section": current_section or "", "output": str(verdict.get("output") or "")[:240]},
                    )
                    _preserve_unverified_section(harness, session, record)
                    verifier_output, clipped = clip_text_value(str(verdict.get("output") or "").strip(), limit=500)
                    clipped_note = "\n[truncated]" if clipped else ""
                    harness.state.append_message(
                        ConversationMessage(
                            role="system",
                            content=(
                                f"SYNTAX ERROR detected in `{session.write_target_path}` after writing section "
                                f"`{current_section or 'unnamed'}`:\n```\n{verifier_output}\n```{clipped_note}\n"
                                f"Keep the write session open and repair this active section locally before moving on. "
                                f"Use the staged file at `{verify_path}` for compile/read checks until the session is finalized.\n"
                                + format_write_session_status_block(
                                    write_session_status_snapshot(
                                        session,
                                        cwd=getattr(harness.state, "cwd", None),
                                        finalized=False,
                                    )
                                )
                            ),
                            metadata={
                                "is_recovery_nudge": True,
                                "recovery_kind": "syntax_error",
                                "session_id": session.write_session_id,
                                "active_section": current_section,
                            },
                        )
                    )
                    _maybe_trigger_write_session_fallback(harness, session)
                    return
                session.write_failed_local_patches = 0
                if session.write_session_mode == "local_repair":
                    transition_write_session(
                        session,
                        next_mode="chunked_author",
                        next_status="open",
                    )
                    record_write_session_event_alias(
                        harness.state,
                        event="recovered_from_local_repair",
                        session=session,
                        details={"section": current_section or ""},
                    )
                if next_section:
                    transition_write_session(session, pending_finalize=False)

                _register_write_session_stage_artifact(harness, session)

            if (final_chunk or session.write_pending_finalize) and not next_section:
                record_write_session_event_alias(
                    harness.state,
                    event="finalize_attempted",
                    session=session,
                    details={"final_chunk": bool(final_chunk)},
                )
                promoted, promote_detail = promote_write_session_target(
                    session,
                    cwd=getattr(harness.state, "cwd", None),
                )
                if not promoted:
                    session.write_failed_local_patches += 1
                    transition_write_session(
                        session,
                        next_mode="local_repair",
                        next_status="local_repair",
                        pending_finalize=True,
                    )
                    record_write_session_event_alias(
                        harness.state,
                        event="finalize_failed",
                        session=session,
                        details={"reason": str(promote_detail)},
                    )
                    verify_path = write_session_verify_path(session, getattr(harness.state, "cwd", None))
                    harness.state.append_message(
                        ConversationMessage(
                            role="system",
                            content=(
                                f"Write Session `{session.write_session_id}` could not finalize "
                                f"`{session.write_target_path}`: {promote_detail} "
                                f"Keep repairing the staged file at `{verify_path}` and retry the final chunk.\n"
                                + format_write_session_status_block(
                                    write_session_status_snapshot(
                                        session,
                                        cwd=getattr(harness.state, "cwd", None),
                                        finalized=False,
                                    )
                                )
                            ),
                            metadata={
                                "is_recovery_nudge": True,
                                "recovery_kind": "write_session_finalize_error",
                                "session_id": session.write_session_id,
                                "target_path": session.write_target_path,
                            },
                        )
                    )
                    _maybe_trigger_write_session_fallback(harness, session)
                    return

                target_path = session.write_target_path
                harness._runlog(
                    "write_session_finalized",
                    "chunked authoring session complete",
                    session_id=session.write_session_id,
                    path=promote_detail,
                    sections=session.write_sections_completed,
                )
                transition_write_session(
                    session,
                    next_status="complete",
                    pending_finalize=False,
                )
                record_write_session_event_alias(
                    harness.state,
                    event="finalize_succeeded",
                    session=session,
                    details={"path": str(promote_detail)},
                )
                record_write_session_event_alias(
                    harness.state,
                    event="session_completed",
                    session=session,
                    details={"target_path": target_path},
                )
                _invalidate_write_session_stage_artifacts(harness, session, target_path=target_path)
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=(
                            f"Write Session `{session.write_session_id}` for `{target_path}` is complete. "
                            f"Please VERIFY the promoted file at `{target_path}` now (e.g. run a linter, test, or `file_read`). "
                            "If errors are found, you may continue making small repairs. "
                            "If you hit a loop of errors, I will suggest a fallback strategy.\n"
                            + format_write_session_status_block(
                                write_session_status_snapshot(
                                    session,
                                    cwd=getattr(harness.state, "cwd", None),
                                    finalized=True,
                                )
                            )
                        ),
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "write_session_complete",
                            "session_id": session.write_session_id,
                            "target_path": target_path,
                        },
                    )
                )
            return

        _maybe_emit_patch_existing_first_choice_nudge(harness, session, record)
        session.write_failed_local_patches += 1
        _maybe_trigger_write_session_fallback(harness, session)
        return

    if not record.result.success and session.status in {"complete", "verifying", "local_repair", "fallback"}:
        session.write_failed_local_patches += 1
        _maybe_trigger_write_session_fallback(harness, session)


async def _run_syntax_check(harness: Any, path: str) -> dict[str, Any] | None:
    ext = Path(path).suffix.lower()
    command = ""
    if ext == ".py":
        command = f"python3 -m py_compile {path}"
    elif ext in {".js", ".ts"}:
        command = f"node --check {path}"
    elif ext == ".json":
        command = f"python3 -c \"import json, sys; json.load(open('{path}'))\""
    elif ext in {".yaml", ".yml"}:
        command = f"python3 -c \"import yaml, sys; yaml.safe_load(open('{path}'))\""

    if not command:
        return None

    harness._runlog("write_session_auto_verify", f"Running automated syntax check: {command}")

    import subprocess

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=getattr(harness.state, "cwd", None),
        )
        success = proc.returncode == 0
        return {
            "verdict": "pass" if success else "fail",
            "command": command,
            "output": proc.stdout + proc.stderr,
            "exit_code": proc.returncode,
            "timestamp": time.time(),
        }
    except Exception as exc:
        harness.log.error("Internal verifier failed: %s", exc)
        return None


def _find_stranded_write_session_record(
    harness: Any, session: Any
) -> ToolExecutionRecord | None:
    """Locate the most recent unprocessed final-chunk tool record for a session."""
    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if not session_id:
        return None

    records = getattr(harness.state, "tool_execution_records", {})
    if not isinstance(records, dict):
        return None

    candidates: list[dict[str, Any]] = []
    for record in records.values():
        if not isinstance(record, dict):
            continue
        if record.get("tool_name") not in {"file_write", "file_append", "file_patch", "ast_patch"}:
            continue
        result = record.get("result") or {}
        metadata = result.get("metadata") or {}
        if str(metadata.get("write_session_id") or "").strip() != session_id:
            continue
        if not bool(metadata.get("write_session_final_chunk")):
            continue
        candidates.append(record)

    if not candidates:
        return None

    # If the session was already finalized in a later event, don't replay.
    events = recent_write_session_events(harness.state, limit=5)
    for evt in events:
        if evt.get("event") in {"finalize_succeeded", "session_completed"}:
            return None

    best = max(candidates, key=lambda r: int(r.get("step_count", 0) or 0))
    return ToolExecutionRecord(
        operation_id=str(best.get("operation_id", "")),
        tool_name=str(best.get("tool_name", "")),
        args=dict(best.get("args") or {}),
        tool_call_id=str(best.get("tool_call_id", ""))
        if best.get("tool_call_id") is not None
        else None,
        result=_tool_envelope_from_dict(dict(best.get("result") or {})),
        replayed=True,
    )


def maybe_replay_stranded_write_session_record(
    harness: Any, graph_state: GraphRunState
) -> bool:
    """Replay a stranded final-chunk record into last_tool_results so the normal
    outcome handler can finalize it.
    """
    session = getattr(harness.state, "write_session", None)
    if not session or str(session.status or "").strip().lower() == "complete":
        return False

    record = _find_stranded_write_session_record(harness, session)
    if record is None:
        return False

    replay_count = _increment_stranded_replay_count(
        harness,
        session_id=str(session.write_session_id or "").strip(),
        operation_id=str(record.operation_id or "").strip(),
    )
    if replay_count > _WRITE_SESSION_REPLAY_GUARD_THRESHOLD:
        from ..tools.fs_sessions import _write_session_can_finalize

        can_finalize_directly = (
            not str(getattr(session, "write_next_section", "") or "").strip()
            and bool(getattr(session, "write_sections_completed", []))
            and _write_session_can_finalize(session)
        )
        if can_finalize_directly:
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=(
                        f"Replay circuit breaker triggered for Write Session `{session.write_session_id}` "
                        f"(operation `{record.operation_id}`). "
                        "Skipping repeated replay and attempting direct finalization on this turn."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "write_session_stranded_replay_circuit_breaker",
                        "session_id": session.write_session_id,
                        "operation_id": record.operation_id,
                        "replay_count": replay_count,
                        "next_action": "direct_finalize_attempt",
                    },
                )
            )
        else:
            transition_write_session(
                session,
                next_mode="local_repair",
                next_status="local_repair",
                pending_finalize=True,
            )
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=(
                        f"Replay circuit breaker triggered for Write Session `{session.write_session_id}` "
                        f"(operation `{record.operation_id}`). "
                        "Do not replay the same final chunk again. Repair the staged file, then retry finalization."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "write_session_stranded_replay_circuit_breaker",
                        "session_id": session.write_session_id,
                        "operation_id": record.operation_id,
                        "replay_count": replay_count,
                        "next_action": "local_repair_then_finalize",
                    },
                )
            )
        harness._runlog(
            "write_session_stranded_replay_circuit_breaker",
            "skipped repeated stranded replay",
            session_id=session.write_session_id,
            operation_id=record.operation_id,
            replay_count=replay_count,
        )
        return False

    graph_state.last_tool_results.append(record)
    harness._runlog(
        "write_session_stranded_replay",
        "replaying stranded final-chunk record into last_tool_results",
        session_id=session.write_session_id,
        operation_id=record.operation_id,
        replay_count=replay_count,
    )
    return True


async def _attempt_write_session_finalize(
    harness: Any, session: Any
) -> tuple[bool, str]:
    """Run verifier and promote a write session that is ready to finalize.
    Returns (success, detail_message).
    """
    from ..tools.fs_sessions import _write_session_can_finalize

    if str(session.status or "").strip().lower() == "complete":
        return True, "Write session is already complete."

    if not _write_session_can_finalize(session):
        return False, f"Write session mode `{session.write_session_mode}` does not allow finalization."

    # Run syntax verifier if applicable
    verdict = await _run_syntax_check(
        harness, write_session_verify_path(session, getattr(harness.state, "cwd", None))
    )
    if verdict:
        session.write_last_verifier = verdict
        if verdict.get("verdict") == "fail":
            session.write_failed_local_patches += 1
            transition_write_session(
                session,
                next_mode="local_repair",
                next_status="local_repair",
                pending_finalize=True,
            )
            current = str(session.write_current_section or "").strip()
            if current:
                transition_write_session(
                    session,
                    current_section=current,
                    next_section=current,
                )
            record_write_session_event_alias(
                harness.state,
                event="verifier_fail",
                session=session,
                details={
                    "section": current or "",
                    "output": str(verdict.get("output") or "")[:240],
                },
            )
            return False, f"Syntax check failed: {verdict.get('output', '')}"
        session.write_failed_local_patches = 0
        if session.write_session_mode == "local_repair":
            transition_write_session(
                session,
                next_mode="chunked_author",
                next_status="open",
            )
            record_write_session_event_alias(
                harness.state,
                event="recovered_from_local_repair",
                session=session,
                details={"section": str(session.write_current_section or "").strip()},
            )

    # Promote staged file to target
    promoted, promote_detail = promote_write_session_target(
        session,
        cwd=getattr(harness.state, "cwd", None),
    )
    if not promoted:
        session.write_failed_local_patches += 1
        transition_write_session(
            session,
            next_mode="local_repair",
            next_status="local_repair",
            pending_finalize=True,
        )
        record_write_session_event_alias(
            harness.state,
            event="finalize_failed",
            session=session,
            details={"reason": str(promote_detail)},
        )
        return False, str(promote_detail)

    target_path = session.write_target_path
    harness._runlog(
        "write_session_finalized",
        "chunked authoring session complete",
        session_id=session.write_session_id,
        path=promote_detail,
        sections=session.write_sections_completed,
    )
    transition_write_session(
        session,
        next_status="complete",
        pending_finalize=False,
    )
    record_write_session_event_alias(
        harness.state,
        event="finalize_succeeded",
        session=session,
        details={"path": str(promote_detail)},
    )
    record_write_session_event_alias(
        harness.state,
        event="session_completed",
        session=session,
        details={"target_path": target_path},
    )
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Write Session `{session.write_session_id}` for `{target_path}` is complete. "
                f"Please VERIFY the promoted file at `{target_path}` now (e.g. run a linter, test, or `file_read`). "
                "If errors are found, you may continue making small repairs. "
                "If you hit a loop of errors, I will suggest a fallback strategy.\n"
                + format_write_session_status_block(
                    write_session_status_snapshot(
                        session,
                        cwd=getattr(harness.state, "cwd", None),
                        finalized=True,
                    )
                )
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "write_session_complete",
                "session_id": session.write_session_id,
                "target_path": target_path,
            },
        )
    )
    return True, str(promote_detail)


async def maybe_finalize_stranded_write_session(
    harness: Any, graph_state: GraphRunState
) -> bool:
    """Safety-net: directly finalize a session that is ready but was never processed."""
    from ..tools.fs_sessions import _write_session_can_finalize

    session = getattr(harness.state, "write_session", None)
    if not session or str(session.status or "").strip().lower() == "complete":
        return False

    # If there's already a replayed record in last_tool_results, let the normal handler do it.
    for record in graph_state.last_tool_results:
        if record.tool_name in {"file_write", "file_append", "file_patch", "ast_patch"}:
            metadata = record.result.metadata or {}
            if str(metadata.get("write_session_id") or "").strip() == str(
                getattr(session, "write_session_id", "") or ""
            ).strip() and bool(metadata.get("write_session_final_chunk")):
                return False

    if session.write_next_section:
        return False
    if not session.write_sections_completed:
        return False
    if not _write_session_can_finalize(session):
        return False

    harness._runlog(
        "write_session_stranded_finalize",
        "attempting direct finalization of stranded write session",
        session_id=session.write_session_id,
    )
    success, detail = await _attempt_write_session_finalize(harness, session)
    return success
