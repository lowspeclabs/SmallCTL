from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..state import clip_text_value
from ..tools.fs import (
    format_write_session_status_block,
    write_session_status_snapshot,
    write_session_verify_path,
)
from ..write_session_fsm import (
    record_write_session_event as record_write_session_event_alias,
    transition_write_session,
)
from .state import ToolExecutionRecord


async def _handle_write_session_syntax_failure(
    harness: Any,
    session: Any,
    record: ToolExecutionRecord,
    verdict: dict[str, Any],
    current_section: str,
    final_chunk: bool,
    *,
    _record_write_session_recovery_failure,
    _preserve_unverified_section,
    _maybe_trigger_write_session_fallback,
) -> None:
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


async def _handle_write_session_finalize_failure(
    harness: Any,
    session: Any,
    record: ToolExecutionRecord,
    promote_detail: str,
    *,
    _record_write_session_recovery_failure,
    _maybe_trigger_write_session_fallback,
) -> None:
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


async def _handle_write_session_finalize_success(
    harness: Any,
    session: Any,
    promote_detail: str,
    *,
    _invalidate_write_session_stage_artifacts,
) -> None:
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
