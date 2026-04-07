from __future__ import annotations

import json
import re
import time
from typing import Any

from ..harness.tool_results import _classify_execution_failure
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..plans import write_plan_file
from ..state import clip_text_value, json_safe_value
from .deps import GraphRuntimeDeps
from .display import format_tool_result_display
from .interrupts import build_interrupt_payload, pause_for_plan_approval
from .routing import LoopRoute
from .state import GraphRunState, ToolExecutionRecord


_WRITE_SESSION_SCHEMA_FAILURE_KEY = "_last_write_session_schema_failure"
_CHAT_PROGRESS_GUARD_KEY = "_chat_progress_guard"
_CHAT_STALL_REPEAT_LIMIT = 2
_CHAT_PROGRESS_CONTROL_TOOLS = {"task_complete", "task_fail", "ask_human"}


def _is_plan_export_validation_error(error: str | None) -> bool:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return False
    markers = (
        "refusing to write a plan",
        "plan export targets must use .md, .txt, or .text",
        "plan export path ending in .md requires markdown format",
        "plan export path ending in .txt/.text requires text format",
        "unsupported plan export format",
    )
    return any(marker in normalized for marker in markers)


def _build_plan_export_recovery_message(record: ToolExecutionRecord) -> str:
    requested_path = str(
        record.args.get("path")
        or record.args.get("output_path")
        or record.args.get("plan_output_path")
        or ""
    ).strip()
    if requested_path:
        from pathlib import Path
        suggested_path = str(Path(requested_path).with_suffix(".md"))
        return (
            "Plan export paths are only for plan documents (.md, .txt, .text). "
            f"Keep implementation targets like `{requested_path}` out of `plan_set` and `plan_export`; "
            f"continue planning without that export, or use `{suggested_path}` for the plan file instead."
        )
    return (
        "Plan exports only support markdown or text plan documents. "
        "Continue planning, and if you still want a plan file, use a `.md`, `.txt`, or `.text` path."
    )


def _auto_update_active_plan_step(harness: Any, *, status: str, note: str = "") -> None:
    plan = getattr(harness.state, "active_plan", None) or getattr(harness.state, "draft_plan", None)
    if plan is None:
        return
    active_step = plan.active_step()
    if active_step is None:
        return
    active_step.status = status
    if note.strip():
        active_step.notes.append(note.strip())
    plan.touch()
    harness.state.sync_plan_mirror()
    harness.state.touch()


def _clear_shell_human_retry_state(harness: Any) -> None:
    for key in (
        "_shell_human_blocked_command_fingerprint",
        "_shell_human_blocked_command",
        "_shell_human_blocked_reason",
        "_shell_human_blocked_question",
        "_shell_human_blocked_tool_call_id",
    ):
        harness.state.scratchpad.pop(key, None)


def _remember_shell_human_retry_state(harness: Any, record: ToolExecutionRecord) -> None:
    command = str(record.args.get("command", "") or record.result.metadata.get("command", "") or "").strip()
    if not command:
        return
    from .tool_call_parser import _tool_call_fingerprint

    harness.state.scratchpad["_shell_human_blocked_command_fingerprint"] = _tool_call_fingerprint(
        "shell_exec",
        {"command": command},
    )
    harness.state.scratchpad["_shell_human_blocked_command"] = command
    harness.state.scratchpad["_shell_human_blocked_reason"] = str(record.result.metadata.get("reason", "") or "").strip()
    harness.state.scratchpad["_shell_human_blocked_question"] = str(record.result.metadata.get("question", "") or "").strip()
    harness.state.scratchpad["_shell_human_blocked_tool_call_id"] = str(record.tool_call_id or "")


def _shell_human_retry_hint(harness: Any, pending: Any) -> str | None:
    if getattr(pending, "tool_name", "") != "shell_exec":
        return None
    blocked_fingerprint = harness.state.scratchpad.get("_shell_human_blocked_command_fingerprint")
    if not isinstance(blocked_fingerprint, str) or not blocked_fingerprint:
        return None

    from .tool_call_parser import _tool_call_fingerprint

    current_fingerprint = _tool_call_fingerprint("shell_exec", dict(getattr(pending, "args", {}) or {}))
    if current_fingerprint != blocked_fingerprint:
        return None

    command = str(harness.state.scratchpad.get("_shell_human_blocked_command", "") or "").strip()
    if not command:
        return None
    reason = str(harness.state.scratchpad.get("_shell_human_blocked_reason", "") or "").strip()
    question = str(harness.state.scratchpad.get("_shell_human_blocked_question", "") or "").strip()

    if reason == "unsupported_shell_syntax":
        return (
            f"You already got a shell-syntax prompt for `{command}`. "
            "Do not retry the same command; rewrite it using POSIX syntax or wrap it in `bash -lc`."
        )
    if reason in {"password_prompt_detected", "password_prompt_timeout"}:
        return (
            f"You already hit a password prompt for `{command}`. "
            "Do not retry the same command; ask the user for credentials or switch to a passwordless path."
        )
    if question:
        return (
            f"You already hit a human-input prompt for `{command}`: {question}. "
            "Do not retry the same command; ask the user or change strategy."
        )
    return (
        f"You already hit a human-input prompt for `{command}`. "
        "Do not retry the same command; ask the user or change strategy."
    )


def _shell_ssh_retry_hint(harness: Any, pending: Any) -> str | None:
    if getattr(pending, "tool_name", "") != "shell_exec":
        return None

    command = str(getattr(pending, "args", {}).get("command", "") or "").strip()
    if not command:
        return None
    if not re.search(r"\b(?:ssh|scp|sftp)\b", command):
        return None

    nudge_key = f"ssh_exec::{command}"
    if harness.state.scratchpad.get("_shell_ssh_routing_nudged") == nudge_key:
        return None
    harness.state.scratchpad["_shell_ssh_routing_nudged"] = nudge_key

    return (
        f"You are trying to run an SSH command through `shell_exec`: `{command}`. "
        "Use `ssh_exec` for remote SSH work and reserve `shell_exec` for local shell commands."
    )


def _has_completed_tool_backed_answer(harness: Any) -> bool:
    if harness.state.scratchpad.get("_task_complete"):
        return True
    return any(message.role == "tool" for message in harness.state.recent_messages)


def _clear_chat_progress_guard(harness: Any) -> None:
    harness.state.scratchpad.pop(_CHAT_PROGRESS_GUARD_KEY, None)


def _chat_failure_evidence_excerpt(record: ToolExecutionRecord) -> str:
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    output_payload = record.result.output if isinstance(record.result.output, dict) else {}
    metadata_output = metadata.get("output")
    if not output_payload and isinstance(metadata_output, dict):
        output_payload = metadata_output

    candidates: list[str] = [
        str(record.result.error or "").strip(),
        str(output_payload.get("stderr") or "").strip(),
        str(output_payload.get("stdout") or "").strip(),
    ]
    if isinstance(record.result.output, str):
        candidates.append(str(record.result.output).strip())
    elif isinstance(record.result.output, dict):
        candidates.append(json.dumps(json_safe_value(record.result.output), ensure_ascii=True, sort_keys=True))

    for candidate in candidates:
        if not candidate:
            continue
        clipped, _ = clip_text_value(candidate, limit=180)
        if clipped:
            return clipped
    return ""


def _chat_failure_signature(record: ToolExecutionRecord) -> dict[str, str] | None:
    if record.tool_name in _CHAT_PROGRESS_CONTROL_TOOLS:
        return None
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if record.result.success:
        return None
    if getattr(record.result, "status", None) == "needs_human" or metadata.get("status") == "needs_human":
        return None

    evidence_excerpt = _chat_failure_evidence_excerpt(record)
    failure_text = " | ".join(
        bit
        for bit in (
            str(record.result.error or "").strip(),
            evidence_excerpt,
        )
        if bit
    )
    failure_class = _classify_execution_failure(failure_text)
    return {
        "tool_name": str(record.tool_name or "").strip(),
        "failure_class": failure_class,
        "evidence_excerpt": evidence_excerpt,
    }


def _record_chat_progress_outcome(harness: Any, records: list[ToolExecutionRecord]) -> None:
    relevant_records = [record for record in records if record.tool_name not in _CHAT_PROGRESS_CONTROL_TOOLS]
    if not relevant_records:
        _clear_chat_progress_guard(harness)
        return

    if any(record.result.success for record in relevant_records):
        _clear_chat_progress_guard(harness)
        return

    failure_record = next(
        (
            record
            for record in reversed(relevant_records)
            if _chat_failure_signature(record) is not None
        ),
        None,
    )
    if failure_record is None:
        _clear_chat_progress_guard(harness)
        return

    signature_payload = _chat_failure_signature(failure_record)
    if signature_payload is None:
        _clear_chat_progress_guard(harness)
        return

    signature = json.dumps(signature_payload, ensure_ascii=True, sort_keys=True)
    prior = harness.state.scratchpad.get(_CHAT_PROGRESS_GUARD_KEY)
    prior_signature = ""
    prior_stall_count = 0
    if isinstance(prior, dict):
        prior_signature = str(prior.get("signature") or "").strip()
        prior_stall_count = int(prior.get("stall_count", 0) or 0)

    stall_count = prior_stall_count + 1 if signature == prior_signature else 1
    stored = _get_tool_execution_record(harness, failure_record.operation_id)
    artifact_id = str(stored.get("artifact_id") or "").strip()
    harness.state.scratchpad[_CHAT_PROGRESS_GUARD_KEY] = {
        **signature_payload,
        "signature": signature,
        "stall_count": stall_count,
        "artifact_id": artifact_id,
        "operation_id": str(failure_record.operation_id or ""),
        "tool_call_id": str(failure_record.tool_call_id or ""),
    }


def _chat_progress_guard_failure(harness: Any) -> dict[str, Any] | None:
    raw = harness.state.scratchpad.get(_CHAT_PROGRESS_GUARD_KEY)
    if not isinstance(raw, dict):
        return None

    stall_count = int(raw.get("stall_count", 0) or 0)
    if stall_count < _CHAT_STALL_REPEAT_LIMIT:
        return None

    tool_name = str(raw.get("tool_name") or "").strip() or "tool"
    failure_class = str(raw.get("failure_class") or "").strip()
    evidence_excerpt = str(raw.get("evidence_excerpt") or "").strip()
    artifact_id = str(raw.get("artifact_id") or "").strip()

    if failure_class:
        message = f"Chat mode stalled on repeated `{tool_name}` {failure_class} failures without new evidence."
    else:
        message = f"Chat mode stalled on repeated `{tool_name}` failures without new evidence."
    if evidence_excerpt:
        message = f"{message} Latest evidence: {evidence_excerpt}"

    return {
        "message": message,
        "details": {
            "guard": "chat_progress_loop",
            "tool_name": tool_name,
            "failure_class": failure_class,
            "stalled_rounds": stall_count,
            "artifact_id": artifact_id,
            "evidence_excerpt": evidence_excerpt,
        },
    }


def _build_repair_recovery_message(harness: Any, record: ToolExecutionRecord) -> str:
    tool_name = record.tool_name
    failure_class = str(getattr(harness.state, "last_failure_class", "") or "").strip()
    repair_cycle_id = str(getattr(harness.state, "repair_cycle_id", "") or "").strip()
    path = str(record.args.get("path") or record.args.get("command") or record.args.get("host") or "").strip()
    counters = getattr(harness.state, "stagnation_counters", {}) or {}
    counter_bits = ", ".join(
        f"{name}={value}"
        for name, value in sorted(counters.items())
        if int(value or 0) > 0
    )
    lead = "Repair loop stalled."
    if failure_class:
        lead = f"Repair loop stalled on {failure_class} failures."
    bits = [lead]
    if repair_cycle_id:
        bits.append(f"cycle {repair_cycle_id}")
    if counter_bits:
        bits.append(f"stagnation: {counter_bits}")
    if path:
        bits.append(f"focus target: {path}")
    if tool_name in {"file_write", "file_append", "file_delete"}:
        return (
            " | ".join(bits)
            + ". Do not broad-rewrite the file. Read the current file or failing evidence first, "
            "then make one narrow patch or one narrow verification step."
        )
    if tool_name in {"shell_exec", "ssh_exec"}:
        return (
            " | ".join(bits)
            + ". Do not repeat the same command blindly. Read the smallest relevant evidence first, "
            "classify the failure, then run one narrow check or one narrow patch."
        )
    return (
        " | ".join(bits)
        + ". Do not repeat the same recovery shape; read the smallest relevant evidence first, "
        "classify the failure, then make one narrow change."
    )


def _maybe_emit_task_complete_verifier_nudge(harness: Any, record: ToolExecutionRecord) -> bool:
    if record.tool_name != "task_complete" or record.result.success:
        return False

    error_text = str(record.result.error or "").strip().lower()
    if "latest verifier verdict is still failing" not in error_text:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    verifier = metadata.get("last_verifier_verdict")
    if not isinstance(verifier, dict) or not verifier:
        current_verifier = getattr(harness.state, "current_verifier_verdict", None)
        verifier = current_verifier() if callable(current_verifier) else None
    if not isinstance(verifier, dict) or not verifier:
        return False

    target_text, clipped = clip_text_value(
        str(verifier.get("command") or verifier.get("target") or "").strip(),
        limit=180,
    )
    note = ""
    acceptance_delta = verifier.get("acceptance_delta")
    if isinstance(acceptance_delta, dict):
        notes = acceptance_delta.get("notes")
        if isinstance(notes, list):
            note = next((str(item).strip() for item in notes if str(item).strip()), "")
    if not note:
        note = str(verifier.get("key_stderr") or verifier.get("key_stdout") or "").strip()
    note_text, note_clipped = clip_text_value(note, limit=180)

    signature_bits = [
        str(record.tool_call_id or ""),
        str(verifier.get("verdict") or ""),
        target_text,
        note_text,
    ]
    signature = "|".join(signature_bits)
    if harness.state.scratchpad.get("_task_complete_verifier_retry_nudge") == signature:
        return False
    harness.state.scratchpad["_task_complete_verifier_retry_nudge"] = signature

    message = "Do not repeat `task_complete` yet."
    verifier_bits: list[str] = []
    if target_text:
        suffix = " [truncated]" if clipped else ""
        verifier_bits.append(f"latest verifier: `{target_text}{suffix}`")
    if note_text:
        suffix = " [truncated]" if note_clipped else ""
        verifier_bits.append(f"result: {note_text}{suffix}")
    if verifier_bits:
        message += " " + " | ".join(verifier_bits) + "."
    message += (
        " Use `loop_status` to inspect the blocker, then either run one focused repair step "
        "or rerun the check in a zero-exit diagnostic form if the failure itself is the proof you need."
    )

    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "task_complete_verifier_retry",
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "verifier_verdict": str(verifier.get("verdict") or ""),
            },
        )
    )
    harness._runlog(
        "task_complete_verifier_retry_nudge",
        "injected recovery nudge after blocked task completion",
        tool_call_id=record.tool_call_id,
        verifier_verdict=str(verifier.get("verdict") or ""),
        verifier_target=str(verifier.get("command") or verifier.get("target") or ""),
    )
    return True


def _maybe_record_write_session_first_chunk_metric(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> float | None:
    session = getattr(harness.state, "write_session", None)
    if session is None:
        return None
    if record.tool_name not in {"file_write", "file_append"} or not record.result.success:
        return None
    if str(record.result.metadata.get("write_session_id") or "").strip() != session.write_session_id:
        return None
    if float(getattr(session, "write_first_chunk_at", 0.0) or 0.0) > 0:
        return None
    if not bool(record.result.metadata.get("section_added")):
        return None
    completed_sections = record.result.metadata.get("write_sections_completed")
    if not isinstance(completed_sections, list) or len(completed_sections) != 1:
        return None

    started_at = float(getattr(session, "write_session_started_at", 0.0) or 0.0)
    if started_at <= 0:
        return None

    now = time.time()
    session.write_first_chunk_at = now
    elapsed = round(max(0.0, now - started_at), 3)
    graph_state.latency_metrics["time_to_first_chunk_sec"] = elapsed
    harness._runlog(
        "write_session_first_chunk",
        "recorded first successful chunk for active write session",
        session_id=session.write_session_id,
        target_path=session.write_target_path,
        section_name=str(record.result.metadata.get("write_current_section") or session.write_current_section or ""),
        time_to_first_chunk_sec=elapsed,
    )
    return elapsed


def _maybe_emit_repair_recovery_nudge(
    harness: Any,
    record: ToolExecutionRecord,
    deps: GraphRuntimeDeps,
) -> bool:
    counters = getattr(harness.state, "stagnation_counters", {}) or {}
    repeat_patch = int(counters.get("repeat_patch", 0) or 0)
    no_progress = int(counters.get("no_progress", 0) or 0)
    repeat_command = int(counters.get("repeat_command", 0) or 0)
    if max(repeat_patch, no_progress, repeat_command) < 2:
        return False
    if record.result.success or getattr(record.result, "status", None) == "needs_human":
        return False

    failure_signature = "|".join(
        [
            str(record.tool_name),
            str(record.args.get("path") or ""),
            str(record.args.get("command") or ""),
            str(getattr(harness.state, "last_failure_class", "") or ""),
            str(record.result.error or ""),
            str(max(repeat_patch, no_progress, repeat_command)),
        ]
    )
    if harness.state.scratchpad.get("_repair_recovery_nudged") == failure_signature:
        return False
    harness.state.scratchpad["_repair_recovery_nudged"] = failure_signature

    message = _build_repair_recovery_message(harness, record)
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "repair_stall",
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "failure_class": getattr(harness.state, "last_failure_class", ""),
                "repair_cycle_id": getattr(harness.state, "repair_cycle_id", ""),
            },
        )
    )
    harness._runlog(
        "repair_stall_recovery",
        "injected repair recovery nudge after repeated failures",
        tool_name=record.tool_name,
        tool_call_id=record.tool_call_id,
        failure_class=getattr(harness.state, "last_failure_class", ""),
        repair_cycle_id=getattr(harness.state, "repair_cycle_id", ""),
        stagnation_counters=json_safe_value(counters),
    )
    return True


async def _run_syntax_check(harness: Any, path: str) -> dict[str, Any] | None:
    from pathlib import Path
    ext = Path(path).suffix.lower()
    command = ""
    if ext == ".py":
        command = f"python3 -m py_compile {path}"
    elif ext in {".js", ".ts"}:
        command = f"node --check {path}"
    elif ext == ".json":
        # Simple JSON parse check using python
        command = f"python3 -c \"import json, sys; json.load(open('{path}'))\""
    elif ext in {".yaml", ".yml"}:
        # Simple YAML parse check using python if pyyaml is installed
        # Fallback to a check that just tries to load it
        command = f"python3 -c \"import yaml, sys; yaml.safe_load(open('{path}'))\""
    
    if not command:
        return None
        
    harness._runlog("write_session_auto_verify", f"Running automated syntax check: {command}")
    
    import subprocess
    try:
        # Run locally and capture output
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
            "timestamp": time.time() if "time" in globals() else 0,
        }
    except Exception as exc:
        harness.log.error("Internal verifier failed: %s", exc)
        return None


def _revert_unverified_section(harness: Any, session: Any, record: ToolExecutionRecord) -> None:
    from ..tools.fs import restore_write_session_snapshot

    restored, detail = restore_write_session_snapshot(
        session,
        cwd=getattr(harness.state, "cwd", None),
    )
    if restored:
        harness._runlog(
            "write_session_reverted",
            "restored staged content after verifier failure",
            session_id=session.write_session_id,
            detail=detail,
        )
        return

    current_section = str(record.result.metadata.get("write_current_section") or session.write_current_section or "").strip()
    if current_section and session.write_sections_completed and session.write_sections_completed[-1] == current_section:
        session.write_sections_completed.pop()
    harness._runlog(
        "write_session_revert_partial",
        "fell back to metadata-only rollback after verifier failure",
        session_id=session.write_session_id,
        detail=detail,
    )


def _maybe_trigger_write_session_fallback(harness: Any, session: Any) -> bool:
    config = getattr(harness, "config", None)
    limit = config.failed_local_patch_limit if config else 3
    if session.write_failed_local_patches < limit:
        return False

    session.write_session_mode = session.write_session_fallback_mode or "stub_and_fill"
    session.status = "fallback"
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
    return True


def _register_write_session_stage_artifact(harness: Any, session: Any) -> str | None:
    from pathlib import Path
    from datetime import datetime, timezone
    from ..state import ArtifactRecord
    from ..tools.fs import write_session_verify_path

    cwd = getattr(harness.state, "cwd", None)
    stage_path = write_session_verify_path(session, cwd)
    if not stage_path or not Path(stage_path).exists():
        return None

    artifact_id = f"{session.write_session_id}__stage"
    # We also support the expanded name the model used in 96c06dcc
    basename = Path(session.write_target_path).name
    legacy_id = f"{session.write_session_id}__{basename}__stage.py"
    
    stat = Path(stage_path).stat()
    record = ArtifactRecord(
        artifact_id=artifact_id,
        kind="file",
        source=stage_path,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        size_bytes=stat.st_size,
        summary=f"Staged content for {session.write_target_path} (session {session.write_session_id})",
        tool_name="file_write",
        metadata={
            "write_session_id": session.write_session_id,
            "target_path": session.write_target_path,
            "is_stage": True,
            "mtime": stat.st_mtime,
        }
    )
    
    harness.state.artifacts[artifact_id] = record
    harness.state.artifacts[legacy_id] = record
    return artifact_id


async def _handle_write_session_outcome(harness: Any, record: ToolExecutionRecord) -> None:
    session = getattr(harness.state, "write_session", None)
    if not session:
        return

    # Success Handling for Writes
    if record.tool_name in {"file_write", "file_append"}:
        res_session_id = record.result.metadata.get("write_session_id")
        if not res_session_id or res_session_id != session.write_session_id:
            return
            
        if record.result.success:
            harness.state.scratchpad.pop(_WRITE_SESSION_SCHEMA_FAILURE_KEY, None)
            current_section = str(record.result.metadata.get("write_current_section") or session.write_current_section or "").strip()
            next_section = str(record.result.metadata.get("write_next_section") or "").strip()
            final_chunk = bool(record.result.metadata.get("write_session_final_chunk"))
            # Automatic Syntax Check
            from ..tools.fs import (
                format_write_session_status_block,
                promote_write_session_target,
                write_session_status_snapshot,
                write_session_verify_path,
            )

            verdict = await _run_syntax_check(
                harness,
                write_session_verify_path(session, getattr(harness.state, "cwd", None)),
            )
            if verdict:
                session.write_last_verifier = verdict
                if verdict.get("verdict") == "fail":
                    session.write_failed_local_patches += 1
                    session.write_session_mode = "local_repair"
                    session.status = "local_repair"
                    session.write_pending_finalize = final_chunk
                    verify_path = write_session_verify_path(session, getattr(harness.state, "cwd", None))
                    if current_section:
                        session.write_current_section = current_section
                        session.write_next_section = current_section
                    _revert_unverified_section(harness, session, record)
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
                    session.write_session_mode = "chunked_author"
                    session.status = "open"
                if next_section:
                    session.write_pending_finalize = False
                
                # Register the stage artifact so it can be read via artifact_read immediately
                _register_write_session_stage_artifact(harness, session)

            # Check if it was the final chunk only after verifier success.
            if (final_chunk or session.write_pending_finalize) and not next_section:
                promoted, promote_detail = promote_write_session_target(
                    session,
                    cwd=getattr(harness.state, "cwd", None),
                )
                if not promoted:
                    session.write_failed_local_patches += 1
                    session.write_session_mode = "local_repair"
                    session.status = "local_repair"
                    session.write_pending_finalize = True
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

                # Session Complete!
                target_path = session.write_target_path
                harness._runlog(
                    "write_session_finalized",
                    "chunked authoring session complete",
                    session_id=session.write_session_id,
                    path=promote_detail,
                    sections=session.write_sections_completed,
                )
                
                # Update status
                session.status = "complete"
                session.write_pending_finalize = False
                
                # Nudge for verification
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
        else:
            # Failed chunk write
            session.write_failed_local_patches += 1
            _maybe_trigger_write_session_fallback(harness, session)
            return

    # Failure Handling for any tool while in session (Potential verification failure)
    if not record.result.success:
        # If we are in 'complete' or 'verifying' and something fails, count it as a repair failure
        if session.status in {"complete", "verifying", "local_repair", "fallback"}:
            session.write_failed_local_patches += 1
            _maybe_trigger_write_session_fallback(harness, session)


async def apply_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        if record.tool_name == "task_complete" and record.result.success:
            _auto_update_active_plan_step(harness, status="completed", note=str(record.result.output or ""))
            await harness._emit(
                deps.event_handler,
                UIEvent(event_type=UIEventType.SYSTEM, content="Task marked complete."),
            )
            graph_state.final_result = {
                "status": "completed",
                "message": record.result.output,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
            return LoopRoute.FINALIZE
        if record.tool_name == "task_complete" and not record.result.success:
            _maybe_emit_task_complete_verifier_nudge(harness, record)
        if record.tool_name == "shell_exec":
            if record.result.success:
                _clear_shell_human_retry_state(harness)
            elif getattr(record.result, "status", None) != "needs_human" and record.result.metadata.get("status") != "needs_human":
                _clear_shell_human_retry_state(harness)
            _maybe_emit_repair_recovery_nudge(harness, record, deps)
        elif record.tool_name == "ssh_exec":
            _maybe_emit_repair_recovery_nudge(harness, record, deps)
        if record.tool_name == "task_fail" and record.result.success:
            _auto_update_active_plan_step(harness, status="blocked", note=str(record.result.output or ""))
            await harness._emit(
                deps.event_handler,
                UIEvent(event_type=UIEventType.ERROR, content="Task marked failed."),
            )
            graph_state.final_result = harness._failure(
                "Task marked failed.",
                error_type="tool",
                details={
                    "tool_name": record.tool_name,
                    "output": record.result.output,
                    "assistant": graph_state.last_assistant_text,
                    "thinking": graph_state.last_thinking_text,
                    "usage": graph_state.last_usage,
                },
            )
            graph_state.error = graph_state.final_result["error"]
            return LoopRoute.FINALIZE
        if record.tool_name == "ask_human" and record.result.success:
            payload = build_interrupt_payload(
                harness=harness,
                graph_state=graph_state,
                record=record,
            )
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content="Human input requested by model.",
                    data={"interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": payload.get("question", "Human input requested."),
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE

        # Generic handling for tool-initiated human interrupts (e.g. Sudo password prompts)
        if (
            getattr(record.result, "status", None) == "needs_human"
            or record.result.metadata.get("status") == "needs_human"
        ):
            if record.tool_name == "shell_exec":
                _remember_shell_human_retry_state(harness, record)
            question = record.result.metadata.get("question", "Human input required for tool.")
            payload = {
                "question": question,
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "metadata": {**record.result.metadata, "interrupt_type": "tool_request"},
                "current_phase": "explore",
                "active_profiles": list(harness.state.active_tool_profiles),
                "thread_id": graph_state.thread_id,
                "operation_id": record.operation_id,
                "recent_tool_outcomes": [r.to_summary_dict() for r in graph_state.last_tool_results]
            }
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content=f"Tool '{record.tool_name}' requires human input: {question}",
                    data={"interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": record.result.error,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE
        if record.tool_name in {"file_write", "file_append", "file_delete"} and not record.result.success:
            _maybe_emit_repair_recovery_nudge(harness, record, deps)

        _maybe_record_write_session_first_chunk_metric(graph_state, harness, record)
        # Handle Write Sessions
        await _handle_write_session_outcome(harness, record)

    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


async def apply_chat_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        if record.tool_name in {"file_write", "file_append", "file_delete"} and not record.result.success:
            _maybe_emit_repair_recovery_nudge(harness, record, deps)

        _maybe_record_write_session_first_chunk_metric(graph_state, harness, record)
        # Chat mode needs the same staged-write promotion/finalization path as loop mode.
        await _handle_write_session_outcome(harness, record)

        # Chat mode should also respect explicit task completion tools
        if record.tool_name == "task_complete" and record.result.success:
            _auto_update_active_plan_step(harness, status="completed", note=str(record.result.output or ""))
            _clear_chat_progress_guard(harness)
            message = str(record.result.output.get("message") if isinstance(record.result.output, dict) else record.result.output)
            graph_state.final_result = {
                "status": "chat_completed",
                "message": message,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
            return LoopRoute.FINALIZE
        if record.tool_name == "task_complete" and not record.result.success:
            _maybe_emit_task_complete_verifier_nudge(harness, record)
        if record.tool_name == "shell_exec":
            if record.result.success:
                _clear_shell_human_retry_state(harness)
            elif getattr(record.result, "status", None) != "needs_human" and record.result.metadata.get("status") != "needs_human":
                _clear_shell_human_retry_state(harness)

        if record.tool_name == "task_fail" and record.result.success:
            _auto_update_active_plan_step(harness, status="blocked", note=str(record.result.output or ""))
            _clear_chat_progress_guard(harness)
            message = str(record.result.output.get("message") if isinstance(record.result.output, dict) else record.result.output)
            graph_state.final_result = {
                "status": "chat_failed",
                "message": message,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
            return LoopRoute.FINALIZE

        if record.tool_name == "ask_human" and record.result.success:
            _clear_chat_progress_guard(harness)
            payload = build_interrupt_payload(
                harness=harness,
                graph_state=graph_state,
                record=record,
            )
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            graph_state.final_result = {
                "status": "needs_human",
                "message": payload.get("question", "Human input requested."),
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE

        # Generic handling for tool-initiated human interrupts (e.g. Sudo password prompts)
        if (
            getattr(record.result, "status", None) == "needs_human"
            or record.result.metadata.get("status") == "needs_human"
        ):
            _clear_chat_progress_guard(harness)
            if record.tool_name == "shell_exec":
                _remember_shell_human_retry_state(harness, record)
            question = record.result.metadata.get("question", "Human input required for tool.")
            payload = {
                "question": question,
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "metadata": {**record.result.metadata, "interrupt_type": "tool_request"},
                "current_phase": "explore",
                "active_profiles": list(harness.state.active_tool_profiles),
                "thread_id": graph_state.thread_id,
                "operation_id": record.operation_id,
                "recent_tool_outcomes": [r.to_summary_dict() for r in graph_state.last_tool_results]
            }
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content=f"Tool '{record.tool_name}' requires human input: {question}",
                    data={"interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": record.result.error,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE

    _record_chat_progress_outcome(harness, graph_state.last_tool_results)
    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


async def apply_planning_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    has_explicit_plan_request = any(
        record.tool_name == "plan_request_execution" and record.result.success
        for record in graph_state.last_tool_results
    )
    for record in graph_state.last_tool_results:
        if not record.result.success and _is_plan_export_validation_error(record.result.error):
            repair_attempts = int(harness.state.scratchpad.get("_plan_export_recovery_nudges", 0))
            if repair_attempts < 1:
                repair_message = _build_plan_export_recovery_message(record)
                harness.state.scratchpad["_plan_export_recovery_nudges"] = repair_attempts + 1
                harness.state.recent_errors.append(str(record.result.error or repair_message))
                harness.state.append_message(
                    ConversationMessage(
                        role="user",
                        content=repair_message,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "plan_export_validation",
                            "tool_name": record.tool_name,
                            "tool_call_id": record.tool_call_id,
                        },
                    )
                )
                harness._runlog(
                    "plan_export_repair",
                    "injected plan export repair nudge",
                    tool_name=record.tool_name,
                    tool_call_id=record.tool_call_id,
                    retry_count=repair_attempts + 1,
                    error=str(record.result.error or ""),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=repair_message,
                        data={
                            "repair_kind": "plan_export_validation",
                            "tool_name": record.tool_name,
                            "tool_call_id": record.tool_call_id,
                            "retry_count": repair_attempts + 1,
                        },
                    ),
                )
                graph_state.last_tool_results = []
                graph_state.last_assistant_text = ""
                graph_state.last_thinking_text = ""
                return LoopRoute.NEXT_STEP

        if record.tool_name == "plan_set" and record.result.success:
            plan = harness.state.draft_plan or harness.state.active_plan
            if plan is not None:
                plan.status = "draft"
                plan.touch()
                harness.state.draft_plan = plan
                harness.state.sync_plan_mirror()
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content="Draft plan created.",
                        data={"status_activity": "draft plan created"},
                    ),
                )
                export_warning = str(record.result.metadata.get("export_warning", "") or "").strip()
                if export_warning:
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(
                            event_type=UIEventType.ALERT,
                            content=f"Draft plan created; skipped invalid export hint: {export_warning}",
                            data={
                                "status_activity": "draft plan created",
                                "warning_type": "plan_export_validation",
                                "rejected_output_path": record.result.metadata.get("rejected_output_path", ""),
                                "suggested_output_path": record.result.metadata.get("suggested_output_path", ""),
                            },
                        ),
                    )
                if plan.requested_output_path:
                    try:
                        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                    except ValueError as exc:
                        harness.log.warning("skipping invalid plan export after plan_set: %s", exc)
                if not has_explicit_plan_request:
                    await pause_for_plan_approval(graph_state, deps)
                    return LoopRoute.FINALIZE
            continue
        if record.tool_name == "plan_step_update" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                plan.touch()
                harness.state.sync_plan_mirror()
                active_step = plan.find_step(str(record.args.get("step_id", "")).strip())
                step_label = active_step.step_id if active_step is not None else str(record.args.get("step_id", "")).strip()
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=f"Plan step updated: {step_label}",
                        data={"status_activity": f"step {step_label} updated"},
                    ),
                )
                if plan.requested_output_path:
                    try:
                        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                    except ValueError as exc:
                        harness.log.warning("skipping invalid plan export after step update: %s", exc)
            continue
        if record.tool_name == "plan_export" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content="Plan file exported.",
                        data={"status_activity": "plan file exported"},
                    ),
                )
            continue
        if record.tool_name == "plan_request_execution" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            payload = {
                "kind": "plan_execute_approval",
                "question": record.result.metadata.get("question", "Plan ready. Execute it now?"),
                "plan_id": plan.plan_id if plan is not None else "",
                "approved": False,
                "response_mode": "yes/no/revise",
                "current_phase": harness.state.current_phase,
                "thread_id": graph_state.thread_id,
            }
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=payload["question"],
                    data={"status_activity": "awaiting plan approval...", "interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": payload["question"],
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE
        if record.tool_name == "task_complete" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None and plan.approved:
                return LoopRoute.FINALIZE

    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


def _tool_envelope_from_dict(payload: dict[str, Any]) -> Any:
    from ..models.tool_result import ToolEnvelope

    metadata = json_safe_value(payload.get("metadata") or {})
    if not isinstance(metadata, dict):
        metadata = {}
    return ToolEnvelope(
        success=bool(payload.get("success")),
        output=json_safe_value(payload.get("output")),
        error=None if payload.get("error") is None else str(payload.get("error")),
        metadata=metadata,
    )


def _conversation_message_from_dict(payload: dict[str, Any]) -> Any:
    from ..models.conversation import ConversationMessage

    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    role = str(normalized.get("role", "tool"))
    content = normalized.get("content")
    if content is not None:
        content = str(content)
    name = normalized.get("name")
    if name is not None:
        name = str(name)
    tool_call_id = normalized.get("tool_call_id")
    if tool_call_id is not None:
        tool_call_id = str(tool_call_id)
    tool_calls = normalized.get("tool_calls")
    metadata = normalized.get("metadata")
    return ConversationMessage(
        role=role,
        content=content,
        name=name,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls if isinstance(tool_calls, list) else [],
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _get_tool_execution_record(harness: Any, operation_id: str) -> dict[str, Any]:
    records = getattr(harness.state, "tool_execution_records", None)
    if not isinstance(records, dict):
        harness.state.tool_execution_records = {}
        return {}
    record = records.get(operation_id)
    return dict(record) if isinstance(record, dict) else {}


def _store_tool_execution_record(
    harness: Any,
    *,
    operation_id: str,
    thread_id: str,
    step_count: int,
    pending: Any,
    result: Any,
) -> None:
    existing = _get_tool_execution_record(harness, operation_id)
    existing.update(
        {
            "operation_id": operation_id,
            "thread_id": thread_id,
            "step_count": step_count,
            "tool_name": pending.tool_name,
            "tool_call_id": pending.tool_call_id,
            "args": dict(pending.args),
            "result": result.to_dict(),
        }
    )
    harness.state.tool_execution_records[operation_id] = existing


def _has_matching_tool_message(harness: Any, message: ConversationMessage) -> bool:
    for existing in reversed(harness.state.recent_messages):
        if existing.role != "tool":
            continue
        if existing.name != message.name:
            continue
        if existing.tool_call_id != message.tool_call_id:
            continue
        if existing.content != message.content:
            continue
        if existing.metadata != message.metadata:
            continue
        return True
    return False
