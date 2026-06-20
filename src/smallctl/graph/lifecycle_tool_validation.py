from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..recovery_metrics import (
    record_tool_call_repair_metrics,
    record_tool_call_repair_next_call_signal,
    remember_tool_call_repair_hint,
)
from ..task_targets import primary_task_target_path
from ..state import json_safe_value
from ..tools.fs import infer_write_session_intent, new_write_session_id
from ..tools.tool_call_repair import ToolCallRepairResult, repair_pending_tool_call_args
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

        repair_result = _apply_tool_call_schema_repair(harness, pending)
        if repair_result is not None and repair_result.repaired and not _tool_call_repair_log_only(harness):
            _record_tool_call_schema_repair(harness, pending, repair_result)

        missing_args = _schema_validation_repair_failure(pending, repair_result)
        if missing_args is None:
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


def _schema_validation_repair_failure(
    pending: PendingToolCall,
    result: ToolCallRepairResult | None,
) -> tuple[str, dict[str, Any]] | None:
    if result is None or result.valid_initially or result.valid_after_repair:
        return None
    issue = result.issues[0] if result.issues else None
    if issue is None:
        message = f"Invalid arguments for {pending.tool_name}."
    else:
        path = ".".join(str(part) for part in issue.path) or "arguments"
        if issue.kind == "required":
            message = f"Missing required field: {path}"
        elif issue.kind == "type":
            message = f"Field {path} expected {issue.expected} but got {issue.actual}."
        elif issue.kind == "additional_property":
            message = f"Unknown field for {pending.tool_name}: {path}"
        elif issue.message:
            message = issue.message
        else:
            message = f"Invalid field for {pending.tool_name}: {path}"
    return message, {
        "tool_name": pending.tool_name,
        "validation_error": "schema_validation",
        "required_fields": [
            str(issue.path[-1]) for issue in result.issues if issue.kind == "required" and issue.path
        ],
        "validation_issues": [
            {
                "path": [str(part) for part in issue.path],
                "kind": issue.kind,
                "expected": issue.expected,
                "actual": issue.actual,
                "message": issue.message,
            }
            for issue in result.issues
        ],
    }


def _apply_tool_call_schema_repair(harness: Any, pending: PendingToolCall) -> ToolCallRepairResult | None:
    if not _tool_call_repair_enabled(harness):
        return None
    result = repair_pending_tool_call_args(harness, pending)
    if result is None:
        return None
    state = getattr(harness, "state", None)
    next_signal = ""
    if str(getattr(pending, "source", "model") or "model").strip().lower() == "model" and state is not None:
        next_signal = record_tool_call_repair_next_call_signal(
            state,
            tool_name=pending.tool_name,
            issue_kinds=[issue.kind for issue in result.issues],
            repair_kinds=[action.kind for action in result.actions],
        )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        if next_signal in {"improved", "repeated"}:
            runlog(
                "tool_call_repair_next_call_improved" if next_signal == "improved" else "tool_call_repair_next_call_repeated",
                "tool call repair next-call signal recorded",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                issue_kinds=[issue.kind for issue in result.issues],
                repair_kinds=[action.kind for action in result.actions],
            )
        runlog(
            "tool_call_validation_passed" if result.valid_initially else "tool_call_validation_failed",
            "tool call schema validation completed",
            tool_name=pending.tool_name,
            tool_call_id=pending.tool_call_id,
            valid_initially=result.valid_initially,
            valid_after_repair=result.valid_after_repair,
            issue_kinds=[issue.kind for issue in result.issues],
            issue_paths=[".".join(str(part) for part in issue.path) for issue in result.issues],
        )
    if not result.repaired:
        return result
    if _tool_call_repair_log_only(harness):
        if callable(runlog):
            runlog(
                "tool_call_repair_log_only",
                "tool call repair candidate observed without mutation",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                repair_kinds=[action.kind for action in result.actions],
            )
        return result
    max_actions = _tool_call_repair_max_actions(harness)
    if max_actions >= 0 and len(result.actions) > max_actions:
        if callable(runlog):
            runlog(
                "tool_call_repair_failed",
                "tool call repair exceeded max action count",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                repair_kinds=[action.kind for action in result.actions],
                action_count=len(result.actions),
                max_actions=max_actions,
            )
        return ToolCallRepairResult(
            valid_initially=result.valid_initially,
            valid_after_repair=False,
            repaired=False,
            args=getattr(pending, "args", {}),
            issues=result.issues,
            actions=[],
            stripped_extra_fields=[],
            hint="",
        )

    pending.args = result.args
    metadata = dict(getattr(pending, "parser_metadata", {}) or {})
    metadata["tool_call_repaired"] = True
    metadata["tool_call_repair_kinds"] = [action.kind for action in result.actions]
    metadata["tool_call_repair_actions"] = [
        {
            "kind": action.kind,
            "path": [str(part) for part in action.path],
            "message": action.message,
        }
        for action in result.actions
    ]
    pending.parser_metadata = metadata
    return result


def _tool_call_repair_enabled(harness: Any) -> bool:
    return bool(getattr(getattr(harness, "config", None), "tool_call_repair_enabled", True))


def _tool_call_repair_log_only(harness: Any) -> bool:
    return bool(getattr(getattr(harness, "config", None), "tool_call_repair_log_only", False))


def _tool_call_repair_max_actions(harness: Any) -> int:
    raw = getattr(getattr(harness, "config", None), "tool_call_repair_max_actions_per_call", 4)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 4


def _record_tool_call_schema_repair(harness: Any, pending: PendingToolCall, result: ToolCallRepairResult) -> None:
    state = getattr(harness, "state", None)
    repair_kinds = [action.kind for action in result.actions]
    issue_kinds = [issue.kind for issue in result.issues]
    if state is not None:
        record_tool_call_repair_metrics(state, repair_kinds=repair_kinds)

    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "tool_call_repair_applied",
            "tool call arguments repaired before dispatch",
            tool_name=pending.tool_name,
            tool_call_id=pending.tool_call_id,
            repair_kinds=repair_kinds,
            issue_kinds=issue_kinds,
            stripped_extra_fields=list(result.stripped_extra_fields),
            repaired_arg_keys=sorted(str(key) for key in result.args.keys()),
        )

    if str(getattr(pending, "source", "model") or "model").strip().lower() != "model":
        return
    append_message = getattr(state, "append_message", None)
    if callable(append_message) and result.hint:
        append_message(
            ConversationMessage(
                role="system",
                content=result.hint,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "tool_call_repair",
                    "tool_name": pending.tool_name,
                    "tool_call_id": pending.tool_call_id,
                    "repair_kinds": repair_kinds,
                    "stripped_extra_fields": list(result.stripped_extra_fields),
                },
            )
        )
        if state is not None:
            record_tool_call_repair_metrics(state, repair_kinds=[], hint_injected=True)
            remember_tool_call_repair_hint(
                state,
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                step_count=int(getattr(state, "step_count", 0) or 0),
                issue_kinds=issue_kinds,
                repair_kinds=repair_kinds,
                repaired_args_preview=json_safe_value(result.args),
            )
        if callable(runlog):
            runlog(
                "tool_call_repair_hint_injected",
                "tool call repair hint injected",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                repair_kinds=repair_kinds,
            )


def _register_write_session_stage_artifact(harness: Any, session: Any) -> None:
    """Register a write session stage artifact for plan tracking."""
    from ..tools.planning import _refresh_plan_playbook_artifact
    _refresh_plan_playbook_artifact(harness.state)
