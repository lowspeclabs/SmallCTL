from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..models.conversation import ConversationMessage
from ..recovery_metrics import record_failure_event_metric
from ..recovery_schema import FailureEvent
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
_WRITE_SESSION_FAILURE_EVENT_LIMIT = 40
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


def _record_write_session_recovery_failure(
    harness: Any,
    session: Any,
    *,
    failure_class: str,
    message: str,
    evidence: list[str] | None = None,
    next_safe_action: str | None = None,
    severity: str = "recoverable",
    operation_id: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FailureEvent | None:
    state = getattr(harness, "state", None)
    if state is None:
        return None

    normalized_class = str(failure_class or "").strip() or "write_session_stall"
    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    evidence_items = [str(item).strip()[:500] for item in (evidence or []) if str(item).strip()]
    raw = "|".join(
        [
            normalized_class,
            "write_session",
            session_id,
            str(operation_id or ""),
            str(tool_name or ""),
            str(message or ""),
            "|".join(evidence_items),
        ]
    )
    subtask_id = _infer_active_subtask_id(harness, state)
    event = FailureEvent(
        event_id="write-session-" + hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16],
        timestamp=time.time(),
        failure_class=normalized_class,
        severity=severity if severity in {"info", "warning", "recoverable", "hard"} else "recoverable",
        source="write_session",
        message=f"{normalized_class}: {str(message or '').strip()[:240]}",
        evidence=evidence_items,
        fama_kind="write_session_stall",
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        operation_id=operation_id,
        subtask_id=subtask_id,
        suggested_next_action=next_safe_action,
        metadata={
            "session_id": session_id,
            "target_path": str(getattr(session, "write_target_path", "") or ""),
            "mode": str(getattr(session, "write_session_mode", "") or ""),
            "status": str(getattr(session, "status", "") or ""),
            **(metadata or {}),
        },
    )

    events = getattr(state, "failure_events", None)
    if not isinstance(events, list):
        events = []
        try:
            state.failure_events = events
        except Exception:
            return None
    if any(isinstance(item, FailureEvent) and item.event_id == event.event_id for item in events[-8:]):
        return None
    events.append(event)
    del events[:-_WRITE_SESSION_FAILURE_EVENT_LIMIT]
    record_failure_event_metric(state, event)

    try:
        state.last_failure_class = normalized_class
    except Exception:
        pass
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad["_last_failure_class"] = normalized_class

    if not _attach_write_session_failure_via_service(harness, event):
        _attach_write_session_failure_direct(state, event)
    _maybe_create_write_session_reflection(harness, state, event)
    _mirror_write_session_working_memory_failure(state, event)

    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "recovery_failure_event_recorded",
            "Write-session recovery failure event recorded",
            event_id=event.event_id,
            failure_class=event.failure_class,
            source=event.source,
            session_id=session_id,
            tool_name=tool_name,
        )
    return event


def _infer_active_subtask_id(harness: Any, state: Any) -> str | None:
    service = getattr(harness, "subtask_ledger", None)
    infer = getattr(service, "infer_or_create_active_subtask", None)
    if callable(infer) and bool(getattr(getattr(harness, "config", None), "subtask_ledger_enabled", True)):
        try:
            active = infer()
            subtask_id = str(getattr(active, "subtask_id", "") or "").strip()
            if subtask_id:
                return subtask_id
        except Exception:
            pass
    ledger = getattr(state, "subtask_ledger", None)
    active = ledger.active() if ledger is not None and callable(getattr(ledger, "active", None)) else None
    subtask_id = str(getattr(active, "subtask_id", "") or "").strip()
    return subtask_id or None


def _attach_write_session_failure_via_service(harness: Any, event: FailureEvent) -> bool:
    service = getattr(harness, "subtask_ledger", None)
    attach_failure = getattr(service, "attach_failure", None)
    if not event.subtask_id or not callable(attach_failure):
        return False
    try:
        attach_failure(event.subtask_id, event)
        return True
    except Exception:
        return False


def _attach_write_session_failure_direct(state: Any, event: FailureEvent) -> None:
    ledger = getattr(state, "subtask_ledger", None)
    active = ledger.active() if ledger is not None and callable(getattr(ledger, "active", None)) else None
    if active is None:
        return
    active.attempts = int(getattr(active, "attempts", 0) or 0) + 1
    if event.failure_class not in active.failure_classes:
        active.failure_classes.append(event.failure_class)
    if event.message and event.message not in active.blockers:
        active.blockers.append(event.message)
        active.blockers = active.blockers[-5:]
    if event.suggested_next_action:
        active.next_action = event.suggested_next_action
    active.updated_at = event.timestamp


def _maybe_create_write_session_reflection(harness: Any, state: Any, event: FailureEvent) -> None:
    service = getattr(harness, "reflexion", None)
    maybe_create = getattr(service, "maybe_create_reflection", None)
    if not callable(maybe_create):
        return
    try:
        maybe_create(event, getattr(state, "subtask_ledger", None))
    except Exception:
        return


def _mirror_write_session_working_memory_failure(state: Any, event: FailureEvent) -> None:
    working_memory = getattr(state, "working_memory", None)
    failures = getattr(working_memory, "failures", None)
    if not isinstance(failures, list):
        return
    line = event.message
    if event.suggested_next_action:
        line = f"{line}; next={event.suggested_next_action}"
    if line not in failures:
        failures.append(line)
        del failures[:-5]


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
    _record_write_session_recovery_failure(
        harness,
        session,
        failure_class="write_session_stall",
        message=(
            f"Write session entered fallback after {session.write_failed_local_patches} "
            "verification/repair failures."
        ),
        evidence=[f"failures={session.write_failed_local_patches}", f"mode={session.write_session_mode}"],
        next_safe_action="Write a minimal stable scaffold, then fill one verified section at a time.",
        severity="recoverable",
        metadata={"recovery_kind": "write_session_fallback"},
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
                    _record_write_session_recovery_failure(
                        harness,
                        session,
                        failure_class="verifier_failed",
                        message=(
                            f"syntax check failed for `{session.write_target_path}` "
                            f"after section `{current_section or 'unnamed'}`"
                        ),
                        evidence=[
                            str(verdict.get("command") or ""),
                            str(verdict.get("output") or ""),
                        ],
                        next_safe_action=(
                            "Repair the active staged section locally, then rerun the smallest syntax check "
                            "before finalizing."
                        ),
                        operation_id=record.operation_id,
                        tool_name=record.tool_name,
                        tool_call_id=record.tool_call_id,
                        metadata={
                            "recovery_kind": "syntax_error",
                            "section": current_section or "",
                            "exit_code": verdict.get("exit_code"),
                        },
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
                    _record_write_session_recovery_failure(
                        harness,
                        session,
                        failure_class="write_session_stall",
                        message=f"could not finalize `{session.write_target_path}`: {promote_detail}",
                        evidence=[str(promote_detail)],
                        next_safe_action="Repair the staged file and retry the final chunk/finalization once.",
                        operation_id=record.operation_id,
                        tool_name=record.tool_name,
                        tool_call_id=record.tool_call_id,
                        metadata={"recovery_kind": "write_session_finalize_error"},
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
        _record_write_session_recovery_failure(
            harness,
            session,
            failure_class="write_session_stall",
            message=f"{record.tool_name} failed while write session `{session.write_session_id}` was active",
            evidence=[str(record.result.error or record.result.output or "")],
            next_safe_action="Read the staged file/target state, then retry the smallest valid write-session repair.",
            operation_id=record.operation_id,
            tool_name=record.tool_name,
            tool_call_id=record.tool_call_id,
            metadata={"recovery_kind": "write_session_tool_failure"},
        )
        _maybe_trigger_write_session_fallback(harness, session)
        return

    if not record.result.success and session.status in {"complete", "verifying", "local_repair", "fallback"}:
        session.write_failed_local_patches += 1
        _record_write_session_recovery_failure(
            harness,
            session,
            failure_class="write_session_stall",
            message=f"{record.tool_name} failed during write-session recovery status `{session.status}`",
            evidence=[str(record.result.error or record.result.output or "")],
            next_safe_action="Use the latest verifier or staged-file evidence before retrying the write-session step.",
            operation_id=record.operation_id,
            tool_name=record.tool_name,
            tool_call_id=record.tool_call_id,
            metadata={"recovery_kind": "write_session_recovery_tool_failure"},
        )
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
            _record_write_session_recovery_failure(
                harness,
                session,
                failure_class="verifier_failed",
                message=(
                    f"syntax check failed while directly finalizing `{session.write_target_path}` "
                    f"at section `{current or 'unnamed'}`"
                ),
                evidence=[
                    str(verdict.get("command") or ""),
                    str(verdict.get("output") or ""),
                ],
                next_safe_action=(
                    "Repair the staged file locally, then retry direct finalization after the syntax check passes."
                ),
                metadata={
                    "recovery_kind": "write_session_direct_finalize_syntax_error",
                    "section": current or "",
                    "exit_code": verdict.get("exit_code"),
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
        _record_write_session_recovery_failure(
            harness,
            session,
            failure_class="write_session_stall",
            message=f"direct finalization could not promote `{session.write_target_path}`: {promote_detail}",
            evidence=[str(promote_detail)],
            next_safe_action="Repair the staged file or target path, then retry finalization once.",
            metadata={"recovery_kind": "write_session_direct_finalize_error"},
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
