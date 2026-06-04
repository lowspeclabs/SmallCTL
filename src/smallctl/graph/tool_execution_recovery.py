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
from .tool_loop_guards import _tool_call_fingerprint
from .tool_loop_guards_support import _model_name_for_loop_guard
from ..guards import is_seven_b_or_under_model_name
from .state import ToolExecutionRecord
from .tool_execution_recovery_constants import (
    CHUNK_WRITE_LOOP_GUARD_TOOLS,
    MISSING_FIRST_WRITE_SESSION_RECOVERY_KEY,
    TERMINAL_WRITE_SESSION_REPAIR_KEY,
    WRITE_TOOLS,
)
from .tool_execution_recovery_helpers import (
    _maybe_emit_repair_recovery_nudge,
    current_verifier_already_passed,
    shell_exec_success_record_for_pending,
)
from .tool_execution_recovery_support import (
    _artifact_read_loop_exceeded_limit,
    _maybe_schedule_chunked_write_loop_guard_read,
    _maybe_emit_terminal_write_session_reuse_nudge,
    _suppress_repeated_successful_shell_exec,
)
from . import write_session_outcomes as _write_session_outcomes
from .escalation_triggers import _maybe_auto_trigger_escalation_for_tool_loop

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
    seen = state.scratchpad.get(MISSING_FIRST_WRITE_SESSION_RECOVERY_KEY)
    if not isinstance(seen, list):
        seen = []
    if signature in seen:
        return False
    seen.append(signature)
    state.scratchpad[MISSING_FIRST_WRITE_SESSION_RECOVERY_KEY] = seen[-20:]

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


async def _maybe_finalize_chunked_write_loop_guard_abort(
    graph_state: GraphRunState,
    harness: Any,
    deps: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in CHUNK_WRITE_LOOP_GUARD_TOOLS or record.result.success:
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
