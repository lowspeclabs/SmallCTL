from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
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
from .tool_loop_guards import _model_name_for_loop_guard, _tool_call_fingerprint
from ..guards import is_seven_b_or_under_model_name
from .state import ToolExecutionRecord
from .tool_execution_recovery_helpers import (
    _maybe_emit_repair_recovery_nudge,
)
from . import write_session_outcomes as _write_session_outcomes

_CHUNK_WRITE_LOOP_GUARD_TOOLS = {"file_write", "file_append"}


def _has_active_verifier_failure(harness: Any) -> bool:
    """Return True if the latest verifier verdict is a failure."""
    verifier = getattr(getattr(harness, "state", None), "last_verifier_verdict", None)
    if not isinstance(verifier, dict):
        return False
    return str(verifier.get("verdict") or "").strip().lower() in {"fail", "failed", "error"}


def _consecutive_verifier_failure_class(state: Any, threshold: int = 3) -> str | None:
    """Return the failure class if the last `threshold` failure events are all verifier failures of the same class."""
    failure_events = getattr(state, "failure_events", None)
    if not isinstance(failure_events, list) or len(failure_events) < threshold:
        return None
    # Only look at the last `threshold` events
    recent = failure_events[-threshold:]
    # All must have a failure_class
    classes = [str(getattr(e, "failure_class", "") or "").strip() for e in recent]
    if not all(classes):
        return None
    # All must be the same class
    first = classes[0]
    if not all(c == first for c in classes):
        return None
    # Must be a verifier-related failure class
    if first not in {
        "verifier_failed",
        "verifier_timeout",
        "infinite_loop_suspected",
        "test_failed",
        "syntax_error",
        "import_error",
    }:
        return None
    return first
_TERMINAL_WRITE_SESSION_REPAIR_KEY = "_terminal_write_session_repair_signatures"
_MISSING_FIRST_WRITE_SESSION_RECOVERY_KEY = "_missing_first_write_session_recovery_signatures"


def _artifact_read_loop_exceeded_limit(
    harness: Any,
    pending: PendingToolCall,
    limit: int = 5,
) -> bool:
    """
    Returns True if there have been `limit` or more artifact_read calls
    on the same artifact without an intervening write operation.
    """
    if pending.tool_name != "artifact_read":
        return False
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if not artifact_id:
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


def _maybe_emit_terminal_write_session_reuse_nudge(harness: Any, record: ToolExecutionRecord) -> bool:
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("error_kind") or "").strip() != "write_session_already_terminal":
        return False
    write_session_id = str(metadata.get("write_session_id") or record.args.get("write_session_id") or "").strip()
    if not write_session_id:
        return False
    path = str(record.args.get("path") or metadata.get("target_path") or metadata.get("path") or "").strip()
    signature = "|".join([record.tool_name, write_session_id, path])
    seen = harness.state.scratchpad.setdefault(_TERMINAL_WRITE_SESSION_REPAIR_KEY, [])
    if not isinstance(seen, list):
        seen = []
    if signature in seen:
        return False
    seen.append(signature)
    harness.state.scratchpad[_TERMINAL_WRITE_SESSION_REPAIR_KEY] = seen[-20:]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Write Session `{write_session_id}` is terminal and cannot be reused. "
                f"Do not reuse write_session_id={write_session_id}; omit `write_session_id` "
                "for direct overwrite or start a fresh session. If this is a narrow repair, "
                "prefer `file_patch`/`ast_patch` without the stale write_session_id."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "terminal_write_session_reuse",
                "write_session_id": write_session_id,
                "target_path": path,
                "tool_name": record.tool_name,
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "terminal_write_session_reuse_nudge",
            "nudged model away from reusing terminal write_session_id",
            tool_name=record.tool_name,
            write_session_id=write_session_id,
            target_path=path,
        )
    return True


async def _maybe_recover_missing_first_write_session(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in {"file_write", "file_append"} or record.result.success:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("error_kind") or "").strip() != "missing_active_write_session":
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False
    active_session = getattr(state, "write_session", None)
    active_status = str(getattr(active_session, "status", "") or "").strip().lower()
    if active_session is not None and active_status not in {"complete", "finalized", "aborted"}:
        return False

    args = dict(record.args or {})
    session_id = str(
        metadata.get("write_session_id")
        or args.get("write_session_id")
        or args.get("session_id")
        or ""
    ).strip()
    path = str(args.get("path") or metadata.get("path") or "").strip()
    content = args.get("content")
    if not session_id or not path or content is None:
        return False

    content_text = str(content)
    try:
        base = Path(getattr(state, "cwd", None) or Path.cwd()).resolve()
        resolved_target = Path(path)
        if not resolved_target.is_absolute():
            resolved_target = base / resolved_target
        target_exists_nonempty = resolved_target.exists() and resolved_target.is_file() and resolved_target.stat().st_size > 0
    except Exception:
        target_exists_nonempty = False
    replace_strategy = str(args.get("replace_strategy") or metadata.get("replace_strategy") or "").strip().lower()
    if target_exists_nonempty and replace_strategy != "overwrite":
        return False

    content_hash = hashlib.sha256(content_text.encode("utf-8", errors="replace")).hexdigest()[:12]
    signature = "|".join([record.tool_name, session_id, path, content_hash])
    seen = state.scratchpad.get(_MISSING_FIRST_WRITE_SESSION_RECOVERY_KEY)
    if not isinstance(seen, list):
        seen = []
    if signature in seen:
        return False
    seen.append(signature)
    state.scratchpad[_MISSING_FIRST_WRITE_SESSION_RECOVERY_KEY] = seen[-20:]

    from ..tools.fs import file_write, infer_write_session_intent
    from ..tools.fs_write_sessions import promote_write_session_target
    from ..write_session_fsm import new_write_session, transition_write_session
    from .tool_write_session_policy import _suggested_chunk_sections

    suggestions = _suggested_chunk_sections(path)
    section_name = str(
        args.get("section_name")
        or args.get("section_id")
        or metadata.get("section_name")
        or ""
    ).strip()
    next_section = section_name or (suggestions[0] if suggestions else "initial_content")
    intent = infer_write_session_intent(path, getattr(state, "cwd", None))
    session = new_write_session(
        session_id=session_id,
        target_path=path,
        intent=intent,
        mode="chunked_author",
        suggested_sections=suggestions,
        next_section=next_section,
    )
    state.write_session = session

    replay_args = dict(args)
    replay_args.pop("session_id", None)
    replay_args["path"] = path
    replay_args["write_session_id"] = session.write_session_id
    # Treat the captured first payload as an intentional one-shot file creation.
    # The model should not need to reason about staging/session IDs just because
    # it supplied a synthetic write_session_id on the first write.
    replay_args["section_name"] = "full_file"
    replay_args["replace_strategy"] = "overwrite"

    replay_result_dict = await file_write(
        path=path,
        content=content_text,
        cwd=getattr(state, "cwd", None),
        state=state,
        write_session_id=session.write_session_id,
        section_name="full_file",
        replace_strategy="overwrite",
    )
    replay_metadata = dict(replay_result_dict.get("metadata") or {})
    replay_success = bool(replay_result_dict.get("success"))
    promote_success = False
    promote_detail = ""
    if replay_success:
        promote_success, promote_detail = promote_write_session_target(
            session,
            cwd=getattr(state, "cwd", None),
        )
        replay_success = promote_success
        replay_metadata.update(
            {
                "missing_active_write_session_recovered": True,
                "auto_repaired_from_error": "missing_active_write_session",
                "replayed_failed_payload": True,
                "promoted_to_target": promote_success,
                "promoted_path": promote_detail,
                "staged_only": False,
                "changed": True,
            }
        )
        if promote_success:
            transition_write_session(
                session,
                next_status="complete",
                pending_finalize=False,
                current_section="full_file",
                next_section="",
            )
            record_write_session_event(
                state,
                event="missing_first_write_session_replay_promoted",
                session=session,
                details={"path": path, "promoted_path": promote_detail, "content_hash": content_hash},
            )
    else:
        replay_metadata.update(
            {
                "missing_active_write_session_recovered": False,
                "auto_repaired_from_error": "missing_active_write_session",
                "replayed_failed_payload": True,
            }
        )

    if replay_success:
        output = f"Recovered missing write session and wrote `{path}`. Local file written: {promote_detail}"
        recovered_result = ToolEnvelope(success=True, output=output, metadata=replay_metadata)
    else:
        error = str(replay_result_dict.get("error") or promote_detail or "Recovered write-session replay failed.")
        recovered_result = ToolEnvelope(success=False, error=error, metadata=replay_metadata)
    record.result = recovered_result
    record.args = replay_args
    for idx, item in enumerate(list(getattr(graph_state, "last_tool_results", []) or [])):
        if item is record or str(getattr(item, "operation_id", "") or "") == str(record.operation_id or ""):
            graph_state.last_tool_results[idx] = record
            break

    record_write_session_event(
        state,
        event="missing_first_write_session_recovered",
        session=session,
        details={
            "path": path,
            "tool_name": record.tool_name,
            "tool_call_id": str(record.tool_call_id or ""),
            "content_hash": content_hash,
        },
    )
    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Recovered Write Session `{session.write_session_id}` for `{path}` from the failed first write and "
                + (
                    "applied the captured payload to the target file. Verify the target file now."
                    if replay_success
                    else "attempted to apply the captured payload, but replay failed. Retry without inventing a write_session_id."
                )
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "missing_first_write_session_recovered",
                "write_session_id": session.write_session_id,
                "target_path": path,
                "tool_name": record.tool_name,
                "content_hash": content_hash,
                "replay_success": replay_success,
                "promoted_path": promote_detail,
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "missing_first_write_session_recovered",
            "recovered missing active write session and applied failed payload replay",
            tool_name=record.tool_name,
            session_id=session.write_session_id,
            target_path=path,
            content_hash=content_hash,
            replay_success=replay_success,
            promoted_path=promote_detail,
        )
    return True


def _shell_exec_success_record_for_pending(graph_state: GraphRunState, pending: PendingToolCall) -> ToolExecutionRecord | None:
    if pending.tool_name != "shell_exec":
        return None
    pending_fingerprint = _tool_call_fingerprint(pending.tool_name, pending.args)
    for record in reversed(list(getattr(graph_state, "last_tool_results", []) or [])):
        if not isinstance(record, ToolExecutionRecord):
            continue
        if record.tool_name != "shell_exec" or not record.result.success:
            continue
        if _tool_call_fingerprint(record.tool_name, record.args) == pending_fingerprint:
            return record
    return None


def _current_verifier_already_passed(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "shell_exec":
        return False
    command = str(pending.args.get("command") or "").strip()
    if not command:
        return False
    state = getattr(harness, "state", None)
    verifier_fn = getattr(state, "current_verifier_verdict", None)
    verifier = verifier_fn() if callable(verifier_fn) else getattr(state, "last_verifier_verdict", None)
    if not isinstance(verifier, dict):
        return False
    verdict = str(verifier.get("verdict") or verifier.get("status") or "").strip().lower()
    verifier_command = str(verifier.get("command") or verifier.get("target") or "").strip()
    return verdict == "pass" and verifier_command == command


def _suppress_repeated_successful_shell_exec(
    *,
    harness: Any,
    graph_state: GraphRunState,
    pending: PendingToolCall,
    repeat_error: str,
) -> bool:
    if pending.tool_name != "shell_exec":
        return False
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    if isinstance(scratchpad, dict) and isinstance(scratchpad.get("_last_verifier_stale_after_mutation"), dict):
        return False
    if _shell_exec_success_record_for_pending(graph_state, pending) is None and not _current_verifier_already_passed(harness, pending):
        return False

    command = str(pending.args.get("command") or "").strip()
    signature = _tool_call_fingerprint(pending.tool_name, pending.args)
    if isinstance(scratchpad, dict) and scratchpad.get("_shell_exec_success_reuse_nudged") != signature:
        scratchpad["_shell_exec_success_reuse_nudged"] = signature
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    "The exact `shell_exec` command already succeeded and no later mutation invalidated it. "
                    f"Do not rerun `{command}`. Use that verifier evidence and call `task_complete(message=...)` now; "
                    "if the task is not actually satisfied, explain the remaining blocker instead."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "shell_exec_already_succeeded",
                    "guard": "repeated_tool_loop",
                    "tool_name": pending.tool_name,
                    "command": command,
                },
            )
        )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "shell_exec_repeated_success_suppressed",
            "suppressed repeated successful shell_exec and nudged terminal completion",
            step=harness.state.step_count,
            tool_name=pending.tool_name,
            arguments=json_safe_value(pending.args),
            guard_error=repeat_error,
        )
    graph_state.pending_tool_calls = []
    graph_state.last_tool_results = []
    return True


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

    if _suppress_repeated_successful_shell_exec(
        harness=harness,
        graph_state=graph_state,
        pending=pending,
        repeat_error=repeat_error,
    ):
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
            else:
                # HARD BLOCK: model was already nudged but is still calling file_read on the same path.
                raw_path = str(pending.args.get("path", "") or "").strip()
                hard_block = (
                    f"STOP. You were already warned not to call `file_read` on `{raw_path}` again. "
                    f"The file does not exist. Create it with `file_write(path='{raw_path}', content='...')` "
                    f"or verify the path with `dir_list`. Do NOT call `file_read` on this path again."
                )
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=hard_block,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "file_read_hard_block",
                            "path": raw_path,
                        },
                    )
                )
                harness.state.recent_errors.append(repeat_error)
                harness._runlog(
                    "file_read_hard_block",
                    "blocked repeated file_read after recovery nudge was ignored",
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
            if summary_exit_artifact_id:
                existing_cache = list(getattr(harness.state, "retrieval_cache", []) or [])
                harness.state.retrieval_cache = [
                    summary_exit_artifact_id,
                    *[item for item in existing_cache if item != summary_exit_artifact_id],
                ][:8]
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

        # Option C: Hybrid escalation logic
        model_name = _model_name_for_loop_guard(harness)
        is_small = is_seven_b_or_under_model_name(model_name)
        nudge_key = f"generic_loop:{pending.tool_name}:{json.dumps(json_safe_value(pending.args), sort_keys=True)}"
        nudge_already_given = harness.state.scratchpad.get("_generic_loop_nudged") == nudge_key
        fingerprint = _tool_call_fingerprint(pending.tool_name, pending.args)
        escalation_seen = harness.state.scratchpad.get("_escalation_auto_tool_loop_fingerprints")
        if not isinstance(escalation_seen, list):
            escalation_seen = []
        escalation_already_tried = fingerprint in escalation_seen

        should_force_escalation = False
        if is_small and not escalation_already_tried:
            # Small model first trip: escalate immediately, skip nudge
            should_force_escalation = True
        elif not is_small and nudge_already_given and not escalation_already_tried:
            # Large model second trip (nudge ignored): escalate instead of abort
            should_force_escalation = True

        if await _maybe_auto_trigger_escalation_for_tool_loop(
            harness=harness,
            pending=pending,
            repeat_error=repeat_error,
            force=should_force_escalation,
        ):
            graph_state.pending_tool_calls = []
            graph_state.last_tool_results = []
            return None

        if not should_pause_repeated_tool_loop(harness, pending) and harness.state.scratchpad.get("_generic_loop_nudged") != nudge_key:
            harness.state.scratchpad["_generic_loop_nudged"] = nudge_key
            nudge_content = repeat_error
            config = getattr(harness, "config", None)
            if bool(getattr(config, "escalation_enabled", False)) and bool(getattr(config, "escalation_expose_tool", True)):
                nudge_content += (
                    "\n\nIf you are stuck and have already gathered evidence, you may call "
                    "`escalate_to_bigger_model(reason=..., question=..., requested_output='next_action')` "
                    "for bounded recovery advice."
                )
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=nudge_content,
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
            fingerprint = _tool_call_fingerprint(pending.tool_name, pending.args)
            pause_key = f"_repeated_tool_loop_pause_count:{fingerprint}"
            pause_count = harness.state.scratchpad.get(pause_key, 0)
            if pause_count >= 1:
                postmortem = (
                    f"Guard tripped: repeated {pending.tool_name} loop (same arguments repeated after prior pause). "
                    "Task auto-failed after multiple loop guard trips."
                )
                harness.state.recent_errors.append(postmortem)
                await harness._emit(
                    deps.event_handler,
                    UIEvent(event_type=UIEventType.ERROR, content=postmortem),
                )
                graph_state.pending_tool_calls = []
                graph_state.final_result = harness._failure(
                    postmortem,
                    error_type="guard",
                    details={
                        "tool_name": pending.tool_name,
                        "arguments": json_safe_value(pending.args),
                        "guard": "repeated_tool_loop",
                        "auto_fail": True,
                    },
                )
                graph_state.error = graph_state.final_result["error"]
                harness._runlog(
                    "repeated_tool_loop_auto_fail",
                    "auto-failed task after repeated tool loop guard re-tripped",
                    step=harness.state.step_count,
                    tool_name=pending.tool_name,
                    arguments=json_safe_value(pending.args),
                    error=postmortem,
                )
                return None
            harness.state.scratchpad[pause_key] = pause_count + 1
            harness.state.scratchpad["_repeated_tool_loop_suppressed_tool"] = pending.tool_name
            harness.state.scratchpad["_repeated_tool_loop_suppressed_ttl"] = 2
            # Ensure guard trips are counted in session summary even when we pause for human interrupt
            if repeat_error and repeat_error not in (harness.state.recent_errors or []):
                harness.state.recent_errors.append(repeat_error)
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


async def _maybe_auto_trigger_escalation_for_tool_loop(
    *,
    harness: Any,
    pending: PendingToolCall,
    repeat_error: str,
    force: bool = False,
) -> bool:
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not force and not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    fingerprint = _tool_call_fingerprint(pending.tool_name, pending.args)
    seen = harness.state.scratchpad.get("_escalation_auto_tool_loop_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    harness.state.scratchpad["_tool_loop_suppression"] = {
        "tool_name": pending.tool_name,
        "arguments": json_safe_value(pending.args),
        "error": repeat_error,
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=f"Repeated `{pending.tool_name}` call was blocked by the tool-loop guard.",
        question="What is the smallest safe next evidence-gathering or repair step?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    harness.state.scratchpad["_escalation_auto_tool_loop_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice. Treat this as advice only; "
                "choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_tool_loop",
                "tool_name": pending.tool_name,
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_tool_loop_advisory",
            "injected escalation advisory after repeated tool loop",
            tool_name=pending.tool_name,
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


async def _maybe_auto_trigger_escalation_for_same_tool_failures(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate when the same tool fails multiple times in one turn with different arguments.

    Only triggers when there is an active verifier failure."""
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    failure_counts: dict[str, int] = {}
    for record in graph_state.last_tool_results:
        if not record.result.success:
            failure_counts[record.tool_name] = failure_counts.get(record.tool_name, 0) + 1

    for tool_name, count in failure_counts.items():
        if count < 3:
            continue

        seen_key = f"_escalation_auto_same_tool_fingerprints:{tool_name}"
        seen = harness.state.scratchpad.get(seen_key)
        if isinstance(seen, list) and harness.state.step_count in seen:
            continue

        harness.state.scratchpad["_tool_loop_suppression"] = {
            "tool_name": tool_name,
            "error": f"Same tool failed {count} times in one turn with different arguments.",
        }
        from ..harness.escalation_service import EscalationService

        result = await EscalationService(harness).run(
            reason=f"`{tool_name}` failed {count} times in one turn with different arguments.",
            question="What is the smallest safe next evidence-gathering or repair step?",
            requested_output="next_action",
            risk_level="medium",
            source="auto",
        )
        if not isinstance(seen, list):
            seen = []
        seen.append(harness.state.step_count)
        harness.state.scratchpad[seen_key] = seen[-10:]

        if not bool(result.get("success")):
            continue

        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    "Escalation advisor returned bounded recovery advice. Treat this as advice only; "
                    "choose any next action through normal tool policy.\n"
                    f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "escalation_advisory",
                    "source": "auto_same_tool_failures",
                    "tool_name": tool_name,
                    "escalation_id": result.get("escalation_id"),
                },
            )
        )
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "escalation_auto_same_tool_failures_advisory",
                "injected escalation advisory after same-tool repeated failures",
                tool_name=tool_name,
                escalation_id=result.get("escalation_id"),
                verdict=result.get("verdict"),
            )
        return True

    return False


async def _maybe_auto_trigger_escalation_for_patch_stall(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate after patch/write-session stalls once policy evidence exists.

    Only triggers when there is an active verifier failure (i.e. the patch
    is failing to fix a verifier/test failure)."""
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False

    threshold = max(3, _safe_int(getattr(config, "escalation_repeated_failure_threshold", 3), 3))
    trigger = _patch_stall_trigger(state, graph_state, threshold=threshold)
    if not trigger:
        return False

    fingerprint = "|".join(
        [
            trigger,
            str(getattr(state, "step_count", 0) or 0),
            _latest_patch_stall_path(state, graph_state),
        ]
    )
    seen = scratchpad.get("_escalation_auto_patch_stall_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "file_patch",
        "error": f"Patch/write-session stall detected: {trigger}.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=f"Patch/write-session stall detected after file mutation: {trigger}.",
        question="What is the smallest safe next evidence-gathering or repair step?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad["_escalation_auto_patch_stall_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for a patch stall. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_patch_stall",
                "trigger": trigger,
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_patch_stall_advisory",
            "injected escalation advisory after patch/write-session stall",
            trigger=trigger,
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


async def _maybe_auto_trigger_escalation_for_completion_block(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Handle explicit escalation requests stuck behind repeated task_complete blocks."""
    state = getattr(harness, "state", None)
    if state is None:
        return False
    if not _user_requested_escalation(state):
        return False

    threshold = max(2, _safe_int(getattr(getattr(harness, "config", None), "escalation_repeated_failure_threshold", 2), 2))
    if _consecutive_task_complete_verification_blocks(state, threshold=threshold) < threshold:
        return False

    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    fingerprint = f"completion_block|{getattr(state, 'step_count', 0)}"
    seen = scratchpad.get("_escalation_auto_completion_block_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return _emit_escalation_config_blocker(
            state,
            scratchpad,
            seen,
            fingerprint,
            "Escalation was explicitly requested, but escalation is disabled. Configure escalation_enabled, escalation_endpoint, and escalation_model.",
        )
    if not str(getattr(config, "escalation_endpoint", "") or "").strip() or not str(getattr(config, "escalation_model", "") or "").strip():
        return _emit_escalation_config_blocker(
            state,
            scratchpad,
            seen,
            fingerprint,
            "Escalation was explicitly requested, but escalation_endpoint or escalation_model is missing.",
        )
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return _emit_escalation_config_blocker(
            state,
            scratchpad,
            seen,
            fingerprint,
            "Escalation was explicitly requested, but escalation_auto_trigger is disabled. Call escalate_to_bigger_model or enable escalation_auto_trigger.",
        )

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "task_complete",
        "error": "Repeated task_complete calls were blocked by post-change verification while escalation was requested.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason="User explicitly requested escalation after repeated task_complete verification blocks.",
        question="What is the smallest safe next step to resolve the stuck completion/verification loop?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad["_escalation_auto_completion_block_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for a repeated task_complete block. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_completion_block",
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_completion_block_advisory",
            "injected escalation advisory after repeated task_complete block",
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


def _emit_escalation_config_blocker(
    state: Any,
    scratchpad: dict[str, Any],
    seen: list[str],
    fingerprint: str,
    message: str,
) -> bool:
    seen.append(fingerprint)
    scratchpad["_escalation_auto_completion_block_fingerprints"] = seen[-20:]
    scratchpad["_last_escalation"] = {
        "step_count": getattr(state, "step_count", 0),
        "trigger": "completion_block_config_error",
        "verdict": "config_error",
    }
    state.append_message(
        ConversationMessage(
            role="system",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_config_blocker",
                "source": "auto_completion_block",
            },
        )
    )
    recent_errors = getattr(state, "recent_errors", None)
    if isinstance(recent_errors, list):
        recent_errors.append(message)
    return True


def _consecutive_task_complete_verification_blocks(state: Any, *, threshold: int) -> int:
    failure_events = getattr(state, "failure_events", None)
    if not isinstance(failure_events, list):
        return 0
    count = 0
    for event in reversed(failure_events):
        tool_name = str(getattr(event, "tool_name", "") or "").strip()
        failure_class = str(getattr(event, "failure_class", "") or "").strip()
        if tool_name == "task_complete" and failure_class == "post_change_verification_required":
            count += 1
            if count >= threshold:
                return count
            continue
        if count:
            break
    return count


def _user_requested_escalation(state: Any) -> bool:
    for messages_attr in ("conversation_history", "transcript_messages", "recent_messages"):
        messages = getattr(state, messages_attr, None)
        if not isinstance(messages, list):
            continue
        for message in reversed(messages[-20:]):
            if not isinstance(message, dict):
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
            else:
                role = message.get("role")
                content = message.get("content")
            if str(role or "").strip().lower() != "user":
                continue
            lowered = str(content or "").lower()
            if any(marker in lowered for marker in ("escalate", "bigger model", "larger model", "stronger model")):
                return True
    return False


def _patch_stall_trigger(state: Any, graph_state: GraphRunState, *, threshold: int) -> str:
    counters = getattr(state, "stagnation_counters", None)
    counters = counters if isinstance(counters, dict) else {}
    if _safe_int(counters.get("repeat_patch"), 0) >= threshold:
        return "repeat_patch"
    if _safe_int(counters.get("no_actionable_progress"), 0) >= threshold and _last_turn_touched_patch_path(graph_state):
        return "no_actionable_progress_after_patch"

    failure_events = getattr(state, "failure_events", None)
    if isinstance(failure_events, list):
        for event in reversed(failure_events[-8:]):
            failure_class = str(getattr(event, "failure_class", "") or "").strip()
            fama_kind = str(getattr(event, "fama_kind", "") or "").strip()
            if failure_class == "write_session_stall" or fama_kind == "write_session_stall":
                return "write_session_stall"

    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    for signal in _fama_signal_classes_from_scratchpad(scratchpad):
        if signal == "write_session_stall":
            return "write_session_stall"

    # Mandatory escalation after 2 repair cycles on the same target
    if scratchpad.get("_repair_cycle_escalation_ready") and _last_turn_touched_patch_path(graph_state):
        return "repair_cycle_exhausted"
    return ""


def _latest_patch_stall_path(state: Any, graph_state: GraphRunState) -> str:
    for record in reversed(getattr(graph_state, "last_tool_results", []) or []):
        args = record.args if isinstance(getattr(record, "args", None), dict) else {}
        path = str(args.get("path") or "").strip()
        if path:
            return path
    changed = getattr(state, "files_changed_this_cycle", None)
    if isinstance(changed, list) and changed:
        return str(changed[-1] or "").strip()
    return ""


def _last_turn_touched_patch_path(graph_state: GraphRunState) -> bool:
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name in {"file_patch", "ast_patch", "file_write", "file_append"}:
            return True
    return False


async def _maybe_auto_trigger_escalation_for_verifier_stall(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate after 3 consecutive verifier failures of the same class.

    This forces escalation when the small model is stuck in a verifier
    failure loop (e.g., same test/import/runtime error repeated)."""
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False

    threshold = max(3, _safe_int(getattr(config, "escalation_repeated_failure_threshold", 3), 3))
    failure_class = _consecutive_verifier_failure_class(state, threshold=threshold)
    if not failure_class:
        return False

    fingerprint = f"verifier_stall|{failure_class}|{getattr(state, 'step_count', 0)}"
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    seen = scratchpad.get("_escalation_auto_verifier_stall_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "verifier",
        "error": f"Verifier stalled with {threshold} consecutive {failure_class} failures.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=f"{threshold} consecutive verifier failures of class '{failure_class}'. Small model is stuck in a repair loop.",
        question="What is the smallest safe next evidence-gathering or repair step?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad["_escalation_auto_verifier_stall_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for repeated verifier failures. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_verifier_stall",
                "failure_class": failure_class,
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_verifier_stall_advisory",
            "injected escalation advisory after repeated verifier failures",
            failure_class=failure_class,
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


def _fama_signal_classes_from_scratchpad(scratchpad: dict[str, Any]) -> list[str]:
    fama = scratchpad.get("_fama")
    if not isinstance(fama, dict):
        return []
    signals = fama.get("signals")
    if not isinstance(signals, list):
        return []
    classes: list[str] = []
    for item in signals[-8:]:
        if not isinstance(item, dict):
            continue
        for key in ("failure_class", "kind"):
            value = str(item.get(key) or "").strip()
            if value:
                classes.append(value)
    return classes


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_APT_SOURCES_FAILURE_CLASSES = {"apt_sources_malformed", "apt_sources_deb822"}
_APT_TIP_TAG = "{apt_sources_tip}"
_APT_CONFIRMED_TAG = "{apt_sources_confirmed}"


def _has_apt_sources_tip(state: Any) -> str:
    """Return existing confirmed apt-sources tip from working memory if present."""
    wm = getattr(state, "working_memory", None)
    if wm is None:
        return ""
    for fact in getattr(wm, "known_facts", []) or []:
        text = str(fact or "").strip()
        if _APT_TIP_TAG in text and _APT_CONFIRMED_TAG in text:
            return text
    return ""


def _store_apt_sources_tip(state: Any, tip: str, classification: str = "") -> None:
    """Store a confirmed apt-sources tip in working memory.

    Format: {apt_sources_tip} {classification} <tip_text> {apt_sources_confirmed}
    """
    wm = getattr(state, "working_memory", None)
    if wm is None:
        return
    if _APT_TIP_TAG not in tip:
        tip = f"{_APT_TIP_TAG} {tip}"
    if classification and f"class:{classification}" not in tip:
        tip = f"{tip} class:{classification}"
    if _APT_CONFIRMED_TAG not in tip:
        tip = f"{tip} {_APT_CONFIRMED_TAG}"
    existing = list(getattr(wm, "known_facts", []) or [])
    if tip not in existing:
        existing.append(tip)
        # Keep last 12 facts
        wm.known_facts = existing[-12:]
        state.touch()


def _get_pending_apt_sources_tip(state: Any) -> dict[str, Any]:
    """Retrieve pending apt-sources escalation result from scratchpad."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    return scratchpad.get("_pending_apt_sources_tip") or {}


def _clear_pending_apt_sources_tip(state: Any) -> None:
    """Clear pending apt-sources tip from scratchpad."""
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict) and "_pending_apt_sources_tip" in scratchpad:
        del scratchpad["_pending_apt_sources_tip"]
        state.touch()


def _maybe_confirm_apt_sources_tip(state: Any, record: ToolExecutionRecord) -> str:
    """Check if a successful apt command confirms a pending tip.

    If a pending tip exists in scratchpad and apt succeeds, extract the tip
    and classification from the escalation response, save to working memory,
    and clear the pending tip.
    """
    if not isinstance(record, ToolExecutionRecord):
        return ""
    if not record.result.success:
        return ""
    tool_name = str(getattr(record, "tool_name", "") or "").strip()
    if tool_name not in {"ssh_exec", "shell_exec"}:
        return ""
    command = str(record.args.get("command") or "").strip().lower()
    if "apt" not in command:
        return ""

    # Verify apt actually succeeded (no errors in stderr)
    stderr = str(record.result.error or "").lower()
    stdout = str(getattr(record.result, "stdout", "") or "").lower()
    combined = f"{stderr} {stdout}"
    if (
        "malformed" in combined
        and "sources" in combined
        and ("e:" in combined or "apt" in combined)
    ):
        # Apt still failing - don't confirm yet
        return ""
    if "the list of sources could not be read" in combined:
        # Apt still failing - don't confirm yet
        return ""

    pending = _get_pending_apt_sources_tip(state)
    if not pending:
        return ""

    # Extract tip from escalation response
    next_action = pending.get("recommended_next_action", {})
    tip_text = ""
    if isinstance(next_action, dict):
        tip_text = str(next_action.get("reason") or "").strip()
    if not tip_text:
        repair_plan = pending.get("repair_plan", "")
        tip_text = str(repair_plan or "").strip()

    # Extract classification from response
    classification = str(pending.get("classification") or pending.get("failure_diagnosis") or "").strip()
    if not classification:
        classification = "apt_sources_malformed"

    if tip_text and len(tip_text) > 10:
        _store_apt_sources_tip(state, tip_text, classification)
        _clear_pending_apt_sources_tip(state)
        runlog = getattr(state, "_runlog", None)
        if callable(runlog):
            runlog(
                "apt_sources_tip_confirmed",
                "apt command succeeded; saved confirmed tip to working memory",
                classification=classification,
                tip_preview=tip_text[:120],
            )
        return f"{_APT_TIP_TAG} class:{classification} {tip_text} {_APT_CONFIRMED_TAG}"

    _clear_pending_apt_sources_tip(state)
    return ""


async def _maybe_auto_trigger_escalation_for_apt_sources_failure(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate when apt commands fail due to malformed sources.list files.

    If a tip already exists in working memory, nudge the model to try it
    instead of escalating again.
    """
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False

    # Look for apt sources failure in recent tool results.
    # IMPORTANT: apt commands often return exit_code=0 (SSH success) but write
    # errors to stderr (e.g., "E: Malformed entry 1 in sources file"). We must
    # inspect stderr even when record.result.success is True.
    failure_count = 0
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if not isinstance(record, ToolExecutionRecord):
            continue
        tool_name = str(getattr(record, "tool_name", "") or "").strip()
        if tool_name not in {"ssh_exec", "shell_exec"}:
            continue
        command = str(record.args.get("command") or "").strip().lower()
        if "apt" not in command:
            continue
        failure_class = str(
            getattr(record.result, "failure_class", "")
            or getattr(record, "failure_class", "")
            or ""
        ).strip()
        stderr = str(record.result.error or "").lower()
        stdout = str(getattr(record.result, "stdout", "") or "").lower()
        combined = f"{stderr} {stdout}"
        if failure_class in _APT_SOURCES_FAILURE_CLASSES:
            failure_count += 1
            continue
        # Heuristic: apt errors in stderr even with exit code 0
        if (
            "malformed" in combined
            and "sources" in combined
            and ("e:" in combined or "apt" in combined)
        ):
            failure_count += 1
            continue
        # Broader catch-all for apt sources failures
        if "the list of sources could not be read" in combined:
            failure_count += 1

    if failure_count == 0:
        return False

    # Check if we already have a tip
    existing_tip = _has_apt_sources_tip(state)
    if existing_tip:
        # Nudge the model to use the existing tip instead of escalating
        state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    "APT SOURCES FAILURE RECOVERY: A previous escalation provided a tip for this exact failure. "
                    f"Try this first: {existing_tip}\n"
                    "Apply this tip before calling escalate_to_bigger_model again."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "apt_sources_tip",
                    "source": "working_memory",
                },
            )
        )
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "apt_sources_tip_nudge",
                "nudged model to use existing apt-sources tip from working memory",
                tip=existing_tip[:120],
            )
        return True

    # Check if we already escalated for this failure mode recently
    seen_key = "_escalation_auto_apt_sources_fingerprints"
    seen = scratchpad.get(seen_key)
    if not isinstance(seen, list):
        seen = []
    fingerprint = f"apt_sources_failure:{state.step_count}"
    if fingerprint in seen:
        return False

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "ssh_exec",
        "error": "apt sources malformed; escalating to bigger model for deb822 guidance.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason="apt command reports malformed sources file error in stderr (exit code may be 0 but apt failed).",
        question=(
            "How should I fix a malformed apt sources file on a modern Debian/Ubuntu system that uses deb822 .sources format? "
            "Include a 'classification' field in your response describing the type of fix (e.g., deb822_format_correction, sources_list_replacement, etc.)."
        ),
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad[seen_key] = seen[-10:]

    if not bool(result.get("success")):
        return False

    # Store the escalation result as pending; only save to working memory after verification
    scratchpad["_pending_apt_sources_tip"] = result
    state.touch()

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for apt sources failure. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_apt_sources_failure",
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_apt_sources_advisory",
            "injected escalation advisory after apt sources failure",
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


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

    recovered_missing_session = await _maybe_recover_missing_first_write_session(graph_state, harness, record)
    if recovered_missing_session:
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Recovered the missing write session and queued the failed first write for replay.",
                data={"status_activity": "replaying recovered first write"},
            ),
        )
        return

    _maybe_emit_terminal_write_session_reuse_nudge(harness, record)
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
