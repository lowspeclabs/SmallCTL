from __future__ import annotations

import json
import logging
from typing import Any

from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from ..write_session_fsm import record_write_session_event
from .chat_progress import (
    artifact_evidence_is_unavailable,
    artifact_prefers_summary_synthesis,
    build_artifact_evidence_unavailable_message,
    build_artifact_summary_exit_message,
    build_file_read_recovery_message,
    build_repeated_tool_loop_interrupt_payload,
    should_pause_repeated_tool_loop,
)
from .shell_outcomes import (
    _shell_human_retry_hint,
    _shell_ssh_retry_hint,
    _shell_workspace_relative_retry_hint,
)
from .state import GraphRunState, PendingToolCall
from .tool_call_parser import (
    _artifact_read_synthesis_hint,
    _extract_artifact_id_from_args,
    _fallback_repeated_artifact_read,
    _fallback_repeated_file_read,
)
from .state import ToolExecutionRecord
from .tool_execution_recovery_helpers import (
    _maybe_emit_repair_recovery_nudge,
)
from . import write_session_outcomes as _write_session_outcomes

_CHUNK_WRITE_LOOP_GUARD_TOOLS = {"file_write", "file_append"}


def _artifact_read_loop_exceeded_limit(
    harness: Any,
    pending: PendingToolCall,
    limit: int = 5,
) -> bool:
    """
    Returns True if there have been `limit` or more artifact_read calls
    on a stage artifact without an intervening write operation.
    """
    if pending.tool_name != "artifact_read":
        return False
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if not artifact_id or not artifact_id.endswith("__stage"):
        return False

    from .tool_loop_guards import _tool_attempt_history

    history = _tool_attempt_history(harness)
    read_count = 0
    for item in reversed(history):
        tool_name = str(item.get("tool_name", ""))
        if tool_name in {"file_write", "file_append", "file_patch", "ast_patch"}:
            break
        if tool_name == "artifact_read":
            fingerprint = str(item.get("fingerprint", ""))
            try:
                payload = json.loads(fingerprint)
            except Exception:
                continue
            args = payload.get("args", {}) if isinstance(payload, dict) else {}
            if str(args.get("artifact_id", "")).strip() == artifact_id:
                read_count += 1
                if read_count >= limit:
                    return True
    return False


def _maybe_schedule_chunked_write_loop_guard_read(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in _CHUNK_WRITE_LOOP_GUARD_TOOLS or record.result.success:
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if not bool(metadata.get("loop_guard_schedule_read")):
        return False

    path = str(metadata.get("path") or record.args.get("path") or "").strip()
    if not path:
        return False

    signature = "|".join(
        [
            path,
            str(metadata.get("error_kind") or ""),
            str(metadata.get("loop_guard_escalation_level") or ""),
            str(metadata.get("loop_guard_score") or ""),
            str(metadata.get("section_name") or ""),
        ]
    )
    if harness.state.scratchpad.get("_chunk_write_loop_guard_read_scheduled") == signature:
        return False
    harness.state.scratchpad["_chunk_write_loop_guard_read_scheduled"] = signature

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": path},
            raw_arguments=json.dumps({"path": path}, ensure_ascii=True, sort_keys=True),
            source="system",
        )
    ]
    from .tool_call_parser import allow_repeated_tool_call_once

    allow_repeated_tool_call_once(harness, "file_read", {"path": path})

    escalation_level = int(metadata.get("loop_guard_escalation_level", 0) or 0)
    base_message = (
        f"LoopGuard blocked a repeated chunk write for `{path}`. "
        "Reading the current staged content before another write."
    )
    if escalation_level >= 2:
        base_message = (
            f"LoopGuard blocked a repeated chunk write for `{path}` again. "
            "Reading the current staged content now; then send a 3-bullet outline with `ask_human(...)` before the next `file_write`."
        )
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=base_message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "chunked_write_loop_guard",
                "target_path": path,
                "escalation_level": escalation_level,
            },
        )
    )
    harness._runlog(
        "chunked_write_loop_guard_read_scheduled",
        "scheduled mandatory file_read after chunked write loop guard block",
        path=path,
        error_kind=str(metadata.get("error_kind") or ""),
        escalation_level=escalation_level,
        stagnation_score=int(metadata.get("loop_guard_score", 0) or 0),
    )
    return True


async def _maybe_finalize_chunked_write_loop_guard_abort(
    graph_state: GraphRunState,
    harness: Any,
    deps: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in _CHUNK_WRITE_LOOP_GUARD_TOOLS or record.result.success:
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if not bool(metadata.get("loop_guard_hard_abort")):
        return False

    postmortem = str(
        metadata.get("loop_guard_postmortem")
        or record.result.error
        or "Model stuck in a repeated chunk-write loop."
    ).strip()
    harness.state.recent_errors.append(postmortem)
    session = getattr(harness.state, "write_session", None)
    if session is not None and str(getattr(session, "status", "") or "").strip().lower() != "complete":
        record_write_session_event(
            harness.state,
            event="loop_guard_hard_abort",
            session=session,
            details={
                "path": str(metadata.get("path") or record.args.get("path") or ""),
                "section_name": str(metadata.get("section_name") or ""),
                "trigger_kind": str(metadata.get("loop_guard_trigger_kind") or ""),
                "attempts": int(metadata.get("loop_guard_attempts", 0) or 0),
            },
        )

    graph_state.pending_tool_calls = []
    graph_state.interrupt_payload = None
    harness.state.pending_interrupt = None
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ERROR,
            content=postmortem,
            data={
                "status_activity": "loop-guard hard abort",
                "path": str(metadata.get("path") or record.args.get("path") or ""),
                "section_name": str(metadata.get("section_name") or ""),
                "attempts": int(metadata.get("loop_guard_attempts", 0) or 0),
                "trigger_kind": str(metadata.get("loop_guard_trigger_kind") or ""),
            },
        ),
    )
    harness._runlog(
        "chunked_write_loop_guard_hard_abort",
        "aborted task after repeated chunked write loop recovery failures",
        path=str(metadata.get("path") or record.args.get("path") or ""),
        section_name=str(metadata.get("section_name") or ""),
        attempts=int(metadata.get("loop_guard_attempts", 0) or 0),
        trigger_kind=str(metadata.get("loop_guard_trigger_kind") or ""),
        stagnation_score=int(metadata.get("loop_guard_score", 0) or 0),
    )
    graph_state.final_result = harness._failure(
        postmortem,
        error_type="guard",
        details={
            "tool_name": record.tool_name,
            "arguments": json_safe_value(record.args),
            "guard": "chunked_write_loop_guard",
            "path": str(metadata.get("path") or record.args.get("path") or ""),
            "section_name": str(metadata.get("section_name") or ""),
            "trigger_kind": str(metadata.get("loop_guard_trigger_kind") or ""),
            "attempts": int(metadata.get("loop_guard_attempts", 0) or 0),
        },
    )
    graph_state.error = graph_state.final_result["error"]
    return True


async def handle_repeated_tool_loop(
    *,
    harness: Any,
    graph_state: GraphRunState,
    deps: Any,
    pending: PendingToolCall,
    repeat_error: str,
) -> PendingToolCall | None:
    shell_human_hint = _shell_human_retry_hint(harness, pending)
    if shell_human_hint is not None:
        if harness.state.scratchpad.get("_shell_human_retry_nudged") != shell_human_hint:
            harness.state.scratchpad["_shell_human_retry_nudged"] = shell_human_hint
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=shell_human_hint,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "shell_exec",
                        "recovery_mode": "human_input",
                    },
                )
            )
            harness._runlog(
                "shell_exec_human_retry_nudge",
                "nudged model away from retrying a human-gated shell command",
                step=harness.state.step_count,
                tool_name=pending.tool_name,
                arguments=json_safe_value(pending.args),
                guard_error=repeat_error,
            )
        graph_state.pending_tool_calls = []
        graph_state.last_tool_results = []
        return None

    ssh_shell_hint = _shell_ssh_retry_hint(harness, pending)
    if ssh_shell_hint is not None:
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=ssh_shell_hint,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "ssh_exec",
                    "recovery_mode": "routing",
                },
            )
        )
        harness._runlog(
            "shell_exec_ssh_routing_nudge",
            "nudged model to use ssh_exec for SSH commands",
            step=harness.state.step_count,
            tool_name=pending.tool_name,
            arguments=json_safe_value(pending.args),
        )
        graph_state.pending_tool_calls = []
        graph_state.last_tool_results = []
        return None

    workspace_relative_shell_hint = _shell_workspace_relative_retry_hint(harness, pending)
    if workspace_relative_shell_hint is not None:
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=workspace_relative_shell_hint,
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "shell_exec",
                    "recovery_mode": "workspace_relative_path",
                },
            )
        )
        harness._runlog(
            "shell_exec_workspace_relative_nudge",
            "nudged model away from retrying a root-level /temp path",
            step=harness.state.step_count,
            tool_name=pending.tool_name,
            arguments=json_safe_value(pending.args),
            guard_error=repeat_error,
        )
        graph_state.pending_tool_calls = []
        graph_state.last_tool_results = []
        return None

    if pending.tool_name == "file_read":
        recovered = _fallback_repeated_file_read(harness, pending)
        if recovered is not None:
            log_kv(
                harness.log,
                logging.INFO,
                "harness_repeated_tool_loop_recovered",
                step=harness.state.step_count,
                original_tool_name=pending.tool_name,
                recovered_tool_name=recovered.tool_name,
                recovered_args=recovered.args,
            )
            pending = recovered
            repeat_error = None
        else:
            file_read_fingerprint = json.dumps(json_safe_value(pending.args), sort_keys=True)
            if harness.state.scratchpad.get("_file_read_recovery_nudged") != file_read_fingerprint:
                harness.state.scratchpad["_file_read_recovery_nudged"] = file_read_fingerprint
                file_read_hint = build_file_read_recovery_message(harness, pending)
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=file_read_hint,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "file_read",
                            "recovery_mode": "read_once_then_act",
                            "path": str(pending.args.get("path", "") or ""),
                        },
                    )
                )
                harness._runlog(
                    "file_read_recovery_nudge",
                    "nudged model away from rereading the same file",
                    step=harness.state.step_count,
                    tool_name=pending.tool_name,
                    arguments=json_safe_value(pending.args),
                    guard_error=repeat_error,
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                return None

    if repeat_error is not None:
        summary_exit_artifact_id = _extract_artifact_id_from_args(pending.args)
        if artifact_evidence_is_unavailable(harness, pending):
            nudge_key = f"{pending.tool_name}:{summary_exit_artifact_id or 'missing_artifact'}"
            if harness.state.scratchpad.get("_artifact_evidence_unavailable_nudged") != nudge_key:
                harness.state.scratchpad["_artifact_evidence_unavailable_nudged"] = nudge_key
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=build_artifact_evidence_unavailable_message(
                            harness,
                            artifact_id=summary_exit_artifact_id,
                        ),
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "artifact_missing_evidence",
                            "artifact_id": summary_exit_artifact_id,
                            "recovery_mode": "rerun_or_admit_unverified",
                        },
                    )
                )
                harness._runlog(
                    "artifact_missing_evidence_nudge",
                    "nudged model away from synthesizing missing artifact evidence",
                    step=harness.state.step_count,
                    artifact_id=summary_exit_artifact_id,
                    guard_error=repeat_error,
                )
            graph_state.pending_tool_calls = []
            graph_state.last_tool_results = []
            return None

        if artifact_prefers_summary_synthesis(harness, pending):
            nudge_key = f"{pending.tool_name}:{summary_exit_artifact_id or 'summary_exit'}"
            if harness.state.scratchpad.get("_artifact_summary_exit_nudged") != nudge_key:
                harness.state.scratchpad["_artifact_summary_exit_nudged"] = nudge_key
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=build_artifact_summary_exit_message(harness, artifact_id=summary_exit_artifact_id),
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "artifact_summary_exit",
                            "artifact_id": summary_exit_artifact_id,
                            "recovery_mode": "synthesis",
                        },
                    )
                )
                harness._runlog(
                    "artifact_summary_exit_nudge",
                    "nudged model to answer from current artifact evidence",
                    step=harness.state.step_count,
                    artifact_id=summary_exit_artifact_id,
                    guard_error=repeat_error,
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                return None

        synthesis_artifact_id = _artifact_read_synthesis_hint(harness, repeat_error)
        if synthesis_artifact_id is not None and pending.tool_name == "artifact_read":
            if harness.state.scratchpad.get("_artifact_read_synthesis_nudged") != synthesis_artifact_id:
                harness.state.scratchpad["_artifact_read_synthesis_nudged"] = synthesis_artifact_id
                synth_msg = (
                    f"You already tried `artifact_read` and `artifact_grep` on artifact {synthesis_artifact_id}. "
                    "Synthesize the answer from the evidence you already have instead of reading the same artifact again."
                )
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=synth_msg,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "artifact_read",
                            "artifact_id": synthesis_artifact_id,
                            "recovery_mode": "synthesis",
                        },
                    )
                )
                harness._runlog(
                    "artifact_read_synthesis_nudge",
                    "nudged model to synthesize from existing artifact evidence",
                    step=harness.state.step_count,
                    artifact_id=synthesis_artifact_id,
                    guard_error=repeat_error,
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                return None

        recovered = _fallback_repeated_artifact_read(harness, pending)
        if recovered is not None:
            log_kv(
                harness.log,
                logging.INFO,
                "harness_repeated_tool_loop_recovered",
                step=harness.state.step_count,
                original_tool_name=pending.tool_name,
                recovered_tool_name=recovered.tool_name,
                recovered_args=recovered.args,
            )
            return recovered

        if _artifact_read_loop_exceeded_limit(harness, pending):
            artifact_id = _extract_artifact_id_from_args(pending.args)
            stage_path = ""
            write_session = getattr(harness.state, "write_session", None)
            if write_session is not None:
                from ..tools.fs_write_sessions import write_session_verify_path
                stage_path = write_session_verify_path(write_session, getattr(harness.state, "cwd", None))
                harness.state.write_session = None

            if stage_path:
                from .write_session_health import _abandon_staged_artifact
                _abandon_staged_artifact(
                    harness,
                    stage_path,
                    "Artifact read loop detected; clearing staged artifact and rewriting.",
                )

            if artifact_id:
                harness.state.artifacts.pop(artifact_id, None)

            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content="You have read the staged artifact too many times without writing. The staged copy has been cleared. Please write the complete file from scratch.",
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "artifact_read_loop_circuit_breaker",
                        "artifact_id": artifact_id,
                    },
                )
            )
            harness._runlog(
                "artifact_read_loop_circuit_breaker",
                "cleared staged artifact and write session after repeated reads",
                step=harness.state.step_count,
                artifact_id=artifact_id,
                guard_error=repeat_error,
            )
            graph_state.pending_tool_calls = []
            graph_state.last_tool_results = []
            return None

        payload = build_repeated_tool_loop_interrupt_payload(
            harness=harness,
            graph_state=graph_state,
            pending=pending,
            repeat_error=repeat_error,
        )
        guidance = str(payload.get("guidance", "") or "").strip()

        nudge_key = f"generic_loop:{pending.tool_name}:{json.dumps(json_safe_value(pending.args), sort_keys=True)}"
        if not should_pause_repeated_tool_loop(harness, pending) and harness.state.scratchpad.get("_generic_loop_nudged") != nudge_key:
            harness.state.scratchpad["_generic_loop_nudged"] = nudge_key
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=repeat_error,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "generic_tool_loop",
                        "guard": "repeated_tool_loop",
                        "tool_name": pending.tool_name,
                    },
                )
            )
            harness._runlog(
                "generic_tool_loop_nudge",
                "nudged model away from repeating identical tool calls",
                step=harness.state.step_count,
                tool_name=pending.tool_name,
                arguments=json_safe_value(pending.args),
                guard_error=repeat_error,
            )
            graph_state.pending_tool_calls = []
            graph_state.last_tool_results = []
            return None

        if should_pause_repeated_tool_loop(harness, pending):
            if guidance:
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=guidance,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "repeated_tool_loop",
                            "guard": "repeated_tool_loop",
                            "tool_name": pending.tool_name,
                        },
                    )
                )
            harness.state.pending_interrupt = payload
            graph_state.interrupt_payload = payload
            harness._runlog(
                "repeated_tool_loop_interrupt",
                "paused repeated tool loop for human-guided resume",
                step=harness.state.step_count,
                tool_name=pending.tool_name,
                arguments=json_safe_value(pending.args),
                error=repeat_error,
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=payload["question"],
                    data={"interrupt": payload},
                ),
            )
            graph_state.pending_tool_calls = []
            return None

        harness.state.recent_errors.append(repeat_error)
        log_kv(
            harness.log,
            logging.WARNING,
            "harness_repeated_tool_loop",
            step=harness.state.step_count,
            tool_name=pending.tool_name,
            arguments=pending.args,
            error=repeat_error,
        )
        await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.ERROR, content=repeat_error))
        graph_state.pending_tool_calls = []
        graph_state.final_result = harness._failure(
            repeat_error,
            error_type="guard",
            details={
                "tool_name": pending.tool_name,
                "arguments": json_safe_value(pending.args),
                "guard": "repeated_tool_loop",
            },
        )
        graph_state.error = graph_state.final_result["error"]
        return None

    return pending


async def handle_failed_file_write_outcome(
    *,
    graph_state: GraphRunState,
    harness: Any,
    deps: Any,
    record: ToolExecutionRecord,
) -> None:
    if record.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"} or record.result.success:
        return

    if await _maybe_finalize_chunked_write_loop_guard_abort(graph_state, harness, deps, record):
        return

    _maybe_emit_repair_recovery_nudge(harness, record, deps)
    _write_session_outcomes._maybe_emit_write_session_target_path_redirect_nudge(harness, record)
    scheduled_stage_read = _write_session_outcomes._maybe_schedule_patch_existing_stage_read_recovery(
        graph_state,
        harness,
        record,
    )
    scheduled_patch_read = _write_session_outcomes._maybe_schedule_file_patch_read_recovery(
        graph_state,
        harness,
        record,
    )
    if scheduled_stage_read:
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Auto-continuing recovery by reading the current staged content.",
                data={"status_activity": "auto-continuing staged read"},
            ),
        )
    elif _maybe_schedule_chunked_write_loop_guard_read(graph_state, harness, record):
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Auto-continuing loop-guard recovery by reading the current staged content.",
                data={"status_activity": "auto-continuing loop-guard read"},
            ),
        )
    elif scheduled_patch_read:
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Auto-continuing patch recovery by reading the current file before another patch.",
                data={"status_activity": "auto-continuing patch read"},
            ),
        )
