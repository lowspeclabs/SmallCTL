from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..task_targets import primary_task_target_path
from ..tools.fs import infer_write_session_intent, new_write_session_id
from ..write_session_fsm import new_write_session, record_write_session_event
from .state import PendingToolCall


async def _validate_pending_tool_calls(harness: Any, graph_state: Any, deps: Any) -> bool:
    """Validate pending tool calls, handle repairs, chunk mode, oversize.

    Returns True if the step should short-circuit (e.g., validation failed
    or chunk mode was triggered).
    """
    from . import node_support as _nodes
    from .node_support import get_suggested_sections as _get_suggested_sections
    from .tool_call_parser_support import (
        _detect_empty_file_write_payload,
        _detect_missing_required_tool_arguments,
        _detect_patch_existing_stage_read_contract_violation,
        _repair_empty_target_file_patch_to_file_write,
    )
    from .tool_loop_guards import _detect_placeholder_tool_call
    from .tool_write_session_policy import _ensure_chunk_write_session, _should_enter_chunk_mode
    from .tool_write_session_support import (
        _detect_oversize_patch_payload,
        _detect_oversize_write_payload,
        _repair_active_write_session_args,
    )

    for pending in graph_state.pending_tool_calls:
        _repair_active_write_session_args(
            harness,
            pending,
            assistant_text=str(graph_state.last_assistant_text or ""),
        )
        _repair_empty_target_file_patch_to_file_write(harness, pending)

        missing_args = _detect_placeholder_tool_call(harness, pending)
        if missing_args is None:
            missing_args = _detect_empty_file_write_payload(harness, pending)
        if missing_args is None:
            missing_args = _detect_missing_required_tool_arguments(harness, pending)
        if missing_args is None:
            missing_args = _detect_patch_existing_stage_read_contract_violation(harness, pending)
        if missing_args is not None:
            err_msg, details = missing_args
            target_path = None
            if pending.tool_name in {"file_write", "file_patch", "ast_patch"}:
                target_path = primary_task_target_path(harness)
                if pending.tool_name == "file_write" and target_path:
                    _ensure_chunk_write_session(harness, target_path)
                if target_path:
                    details = dict(details)
                    details["target_path"] = target_path

            repair_decision = _nodes.schema_validation_repair_decision(
                harness,
                pending,
                err_msg,
                details,
                target_path=target_path,
            )
            if repair_decision.status == "fail":
                harness.state.recent_errors.append(err_msg)
                harness._runlog(
                    "tool_call_validation_error",
                    "tool call missing required arguments",
                    **repair_decision.runlog_data,
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ERROR,
                        content=err_msg,
                        data=repair_decision.alert_data,
                    ),
                )
                graph_state.pending_tool_calls = []
                graph_state.final_result = harness._failure(
                    err_msg,
                    error_type="schema_validation_error",
                    details=repair_decision.details,
                )
                graph_state.error = graph_state.final_result["error"]
                return True

            harness.state.recent_errors.append(err_msg)
            if repair_decision.conversation_message is not None:
                harness.state.append_message(repair_decision.conversation_message)
            harness._runlog(
                "tool_call_repair",
                "injected schema repair nudge",
                **repair_decision.runlog_data,
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=repair_decision.repair_message,
                    data=repair_decision.alert_data,
                ),
            )
            graph_state.pending_tool_calls = []
            return True

        if _should_enter_chunk_mode(harness, pending):
            target_path = str(pending.args.get("path") or "")
            content = str(pending.args.get("content") or "")

            suggestions = _get_suggested_sections(target_path)
            session_id = new_write_session_id()

            from ..tools.fs_write_sessions import _store_active_write_session
            session = new_write_session(
                session_id=session_id,
                target_path=target_path,
                intent=infer_write_session_intent(
                    target_path,
                    getattr(harness.state, "cwd", None),
                ),
                mode="chunked_author",
                suggested_sections=suggestions,
                next_section=suggestions[0] if suggestions else "",
            )
            _store_active_write_session(harness.state, session)
            _register_write_session_stage_artifact(harness, session)
            record_write_session_event(
                harness.state,
                event="session_opened",
                session=session,
                details={"source": "chunk_mode_trigger", "size": len(content)},
            )

            msg = (
                f"Writing `{target_path}` requires chunked authoring for this model/task ({len(content)} chars in the attempted write). "
                f"I have initialized a Write Session `{session_id}`. "
                "Please break this file into logical sections (e.g. imports, classes, functions) and write them one by one. "
                f"Use `file_write` with `write_session_id='{session_id}'`, `section_name='...'`, and `next_section_name='...'`. "
                "When finished, call `file_write` for the last chunk without `next_section_name` to finalize."
            )
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=msg,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "chunk_mode_trigger",
                        "session_id": session_id,
                        "target_path": target_path,
                    },
                )
            )
            harness._runlog(
                "chunk_mode_triggered",
                "intercepted large write, suggesting chunk mode",
                session_id=session_id,
                target_path=target_path,
                size=len(content),
            )
            graph_state.pending_tool_calls = []
            return True

        oversize = _detect_oversize_write_payload(harness, pending)
        if oversize:
            err_msg, details = oversize
            harness.state.recent_errors.append(err_msg)
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=err_msg,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "oversize_write",
                        **details,
                    },
                )
            )
            harness._runlog(
                "oversize_write_intercepted",
                "rejected one-shot write exceeding hard threshold",
                **details,
            )
            graph_state.pending_tool_calls = []
            return True

        oversize_patch = _detect_oversize_patch_payload(harness, pending)
        if oversize_patch:
            err_msg, details = oversize_patch
            harness.state.recent_errors.append(err_msg)
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=err_msg,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "oversize_patch",
                        **details,
                    },
                )
            )
            harness._runlog(
                "oversize_patch_intercepted",
                "rejected large patch exceeding hard threshold",
                **details,
            )
            graph_state.pending_tool_calls = []
            return True

    return False


def _register_write_session_stage_artifact(harness: Any, session: Any) -> None:
    """Register a write session stage artifact for plan tracking."""
    from ..tools.planning import _refresh_plan_playbook_artifact
    _refresh_plan_playbook_artifact(harness.state)
