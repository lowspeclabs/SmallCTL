from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..guards import is_seven_b_or_under_model_name
from ..state import clip_text_value, json_safe_value
from .recovery_context import build_goal_recap
from .state import GraphRunState, PendingToolCall
from .tool_call_parser import _extract_artifact_id_from_args
from ..harness.task_transactions import recovery_context_lines, transaction_from_scratchpad
from .chat_progress_guard import (
    _chat_failure_evidence_excerpt,
    _chat_failure_signature,
    _chat_progress_guard_failure,
    _clear_chat_progress_guard,
    _record_chat_progress_outcome,
)
from .chat_progress_support import (
    _has_prior_successful_evidence_for_output_write,
    _merged_task_text,
    _nearby_missing_input_candidates,
    _path_mentions_in_task_text,
    _pending_artifact_record,
    _task_requests_written_output_path,
    artifact_prefers_summary_synthesis,
    looks_like_freeze_or_hang,
    recent_assistant_texts,
    task_prefers_summary_synthesis,
)

_ARTIFACT_EVIDENCE_TOOLS = {
    "artifact_grep",
    "artifact_print",
    "artifact_read",
}
_ARTIFACT_EVIDENCE_UNAVAILABLE_MARKERS = (
    "not found in state",
    "content file not found",
    "has no stored content",
    "has no content to search",
)
def build_blank_message_nudge(harness: Any, *, repeated: bool) -> str:
    goal_recap = build_goal_recap(harness)
    goal_note = f" {goal_recap}" if goal_recap else ""
    if repeated:
        return (
            "Blank Message Nudge: the last assistant turn had no text and no tool calls."
            f"{goal_note} Provide a concrete next step, emit the JSON tool call, or call `task_complete(message='...')` if finished."
        )
    return (
        "The assistant turn was empty."
        f"{goal_note} Please respond with a concrete thought or tool call; if you are finished, call `task_complete(message='...')`."
    )


def build_small_model_continue_message(
    harness: Any,
    assistant_text: str,
    *,
    stream_halt_reason: str = "",
) -> str:
    model_name = str(getattr(getattr(harness, "client", None), "model", "") or "").strip()
    clipped_text, clipped = clip_text_value(str(assistant_text or "").strip(), limit=180)
    lead = "You may be frozen or hanging."
    if stream_halt_reason == "stream_ended_without_done":
        lead = "The response stream ended before a clean completion signal."
    if clipped_text:
        if stream_halt_reason == "stream_ended_without_done":
            lead = f"The response stream ended after: {clipped_text}."
        else:
            lead = f"You may be frozen or hanging after: {clipped_text}."
    if clipped:
        lead = f"{lead} [truncated]"
    model_note = f" Model: {model_name}." if model_name else ""
    goal_recap = build_goal_recap(harness)
    goal_note = f" {goal_recap}" if goal_recap else ""
    return (
        f"{lead}{model_note}{goal_note} Continue from the last concrete step within that objective. "
        "Do not restart the task; either call the next tool or emit the next JSON tool call immediately."
    )


def chat_turn_signature(graph_state: GraphRunState) -> str:
    thinking_text = re.sub(r"\s+", " ", str(graph_state.last_thinking_text or "").strip())
    if thinking_text:
        return thinking_text
    assistant_text = re.sub(r"\s+", " ", str(graph_state.last_assistant_text or "").strip())
    if not assistant_text:
        return ""
    return assistant_text


def build_repeated_chat_thinking_message(harness: Any, graph_state: GraphRunState) -> str:
    thinking_text = str(graph_state.last_thinking_text or "").strip()
    clipped_thinking, was_clipped = clip_text_value(thinking_text, limit=240)
    goal_recap = build_goal_recap(harness)
    goal_note = f" {goal_recap}" if goal_recap else ""
    repeat_note = f" Previous thinking: {clipped_thinking}{' [truncated]' if was_clipped else ''}." if clipped_thinking else ""
    return (
        "You repeated the same reasoning without making forward progress."
        f"{repeat_note}{goal_note} "
        "Do not restate the same thoughts. Continue from the last concrete step and either call the next tool "
        "or call `task_complete(message='...')` if you are actually finished."
    )


def build_repeated_reasoning_loop_message(harness: Any, graph_state: GraphRunState) -> str:
    thinking_text = str(graph_state.last_thinking_text or "").strip()
    clipped_thinking, was_clipped = clip_text_value(thinking_text, limit=240)
    goal_recap = build_goal_recap(harness)
    goal_note = f" {goal_recap}" if goal_recap else ""
    repeat_note = f" Previous thinking: {clipped_thinking}{' [truncated]' if was_clipped else ''}." if clipped_thinking else ""
    return (
        "You are repeating the same reasoning without taking action."
        f"{repeat_note}{goal_note} "
        "Do not restate the same thoughts. Either call the next tool, call `task_complete(message='...')` if you are finished, "
        "or call `ask_human(question='...')` to request clarification on what to do next."
    )


def chat_completion_recovery_guard(harness: Any) -> dict[str, str] | None:
    state = getattr(harness, "state", None)
    if state is None:
        return None

    session = getattr(state, "write_session", None)
    if session is not None and str(getattr(session, "status", "") or "").strip().lower() != "complete":
        session_id = str(getattr(session, "write_session_id", "") or "").strip()
        target_path = str(getattr(session, "write_target_path", "") or "").strip()
        next_section = str(getattr(session, "write_next_section", "") or "").strip()
        if next_section:
            detail = f" Next required section: `{next_section}`."
        elif bool(getattr(session, "write_pending_finalize", False)):
            detail = " The staged file still needs verification and finalization."
        else:
            detail = " The staged file has not been finalized to the target path yet."
        return {
            "kind": "write_session_guard",
            "signature": "|".join(
                [
                    "write_session",
                    session_id,
                    str(getattr(session, "status", "") or "").strip(),
                    next_section,
                    "pending_finalize=yes" if bool(getattr(session, "write_pending_finalize", False)) else "pending_finalize=no",
                ]
            ),
            "message": (
                f"Do not present the task as finished yet. Write Session `{session_id}` for `{target_path}` is still open."
                f"{detail} Continue the active write session or explicitly finalize it before ending the run."
            ),
        }

    current_verifier = getattr(state, "current_verifier_verdict", None)
    verifier = current_verifier() if callable(current_verifier) else None
    if not isinstance(verifier, dict) or not verifier:
        return None
    verdict = str(verifier.get("verdict") or "").strip().lower()
    if verdict in {"", "pass"} or bool(getattr(state, "acceptance_waived", False)):
        return None

    target_text, target_clipped = clip_text_value(
        str(verifier.get("command") or verifier.get("target") or "").strip(),
        limit=180,
    )
    note_text, note_clipped = clip_text_value(
        str(verifier.get("key_stderr") or verifier.get("key_stdout") or "").strip(),
        limit=180,
    )
    message = "Do not present the task as finished yet. The latest verifier is still failing."
    if target_text:
        message += f" Latest verifier: `{target_text}{' [truncated]' if target_clipped else ''}`."
    if note_text:
        message += f" Result: {note_text}{' [truncated]' if note_clipped else ''}."
    message += " Continue with one focused repair step or inspect `loop_status` before trying to finish."
    return {
        "kind": "verifier_guard",
        "signature": "|".join(["verifier", verdict, target_text, note_text]),
        "message": message,
    }


def build_terminal_state_nudge(harness: Any) -> str | None:
    from ..challenge_progress import terminal_readiness_state
    readiness = terminal_readiness_state(harness.state)
    if not readiness:
        return None
    return (
        "The required artifact exists and verification is complete. "
        "Next action must be `task_complete` or `task_fail` with a concrete blocker. "
        "Do not run exploratory read-only tools."
    )


def build_artifact_summary_exit_message(harness: Any, *, artifact_id: str = "") -> str:
    objective = str(getattr(getattr(harness, "state", None), "run_brief", None).original_task or "").strip()
    artifact_note = f" from artifact {artifact_id}" if artifact_id else ""
    objective_note = f" for `{objective}`" if objective else ""
    evidence = _small_artifact_evidence_block(harness, artifact_id=artifact_id)
    message = (
        f"You already have enough evidence{artifact_note}{objective_note}. "
        "Produce the requested table or summary now with `task_complete(message='...')` "
        "instead of rereading or printing the same artifact again."
    )
    if evidence:
        message = f"{message}\n\n{evidence}"
    return message


def _small_artifact_evidence_block(harness: Any, *, artifact_id: str = "") -> str:
    if not artifact_id:
        return ""
    state = getattr(harness, "state", None)
    artifacts = getattr(state, "artifacts", {}) if state is not None else {}
    artifact = artifacts.get(artifact_id) if isinstance(artifacts, dict) else None
    if artifact is None:
        return ""
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if bool(metadata.get("truncated")):
        return ""
    text = str(getattr(artifact, "inline_content", "") or "")
    content_path = str(getattr(artifact, "content_path", "") or "").strip()
    if not text and content_path:
        try:
            text = Path(content_path).read_text(encoding="utf-8")
        except OSError:
            text = ""
    text = text.rstrip("\n")
    if not text or len(text) > 4096 or len(text.splitlines()) > 100:
        return ""
    return (
        f"Full content of {artifact_id} is pinned for this recovery turn. "
        "Use this text directly:\n\n"
        f"```text\n{text}\n```"
    )


def artifact_evidence_is_unavailable(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name not in _ARTIFACT_EVIDENCE_TOOLS:
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False

    artifact_id = str(_extract_artifact_id_from_args(pending.args) or "").strip().lower()
    evidence_tokens = {artifact_id} if artifact_id else set()
    if artifact_id.startswith("a"):
        evidence_tokens.add(f"e-{artifact_id}")

    recent_errors = getattr(state, "recent_errors", None)
    failure_memory = getattr(getattr(state, "working_memory", None), "failures", None)
    candidate_texts: list[str] = []
    if isinstance(recent_errors, list):
        candidate_texts.extend(str(item or "") for item in recent_errors)
    if isinstance(failure_memory, list):
        candidate_texts.extend(str(item or "") for item in failure_memory)

    for text in reversed(candidate_texts):
        lowered = text.strip().lower()
        if not lowered:
            continue
        if pending.tool_name not in lowered and "artifact_" not in lowered:
            continue
        if not any(marker in lowered for marker in _ARTIFACT_EVIDENCE_UNAVAILABLE_MARKERS):
            continue
        if evidence_tokens and not any(token in lowered for token in evidence_tokens):
            continue
        return True
    return False


def build_artifact_evidence_unavailable_message(harness: Any, *, artifact_id: str = "") -> str:
    objective = str(getattr(getattr(harness, "state", None), "run_brief", None).original_task or "").strip()
    artifact_note = f" for artifact {artifact_id}" if artifact_id else ""
    objective_note = f" for `{objective}`" if objective else ""
    model_name = str(
        getattr(getattr(harness, "state", None), "scratchpad", {}).get("_model_name")
        or getattr(getattr(harness, "client", None), "model", "")
        or ""
    ).strip()
    small_model_nudge = ""
    if is_seven_b_or_under_model_name(model_name):
        requested_artifact = artifact_id or "that artifact"
        small_model_nudge = (
            f"The artifact you requested, `{requested_artifact}`, does not exist in this session. "
            f"Do not request `{requested_artifact}` again unless you rerun the original tool that creates it. "
        )
    return (
        small_model_nudge
        +
        f"The requested proof{artifact_note}{objective_note} is unavailable in the current session state. "
        "Do not claim or infer what the missing artifact contained from memory, summaries, or prior reasoning. "
        "Re-execute the original tool call to regenerate the evidence; if you cannot rerun it, explicitly say "
        "that you cannot verify the claim from the current session."
    )


def build_file_read_recovery_message(harness: Any, pending: PendingToolCall) -> str:
    raw_path = str(pending.args.get("path", "") or "").strip()
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
    tx_lines = recovery_context_lines(transaction)
    tx_note = (" " + " ".join(tx_lines)) if tx_lines else ""
    if not raw_path:
        return (
            "You already read this file. Do not reread it; use the evidence you already have "
            "to patch the file, run the focused test, or move on."
            f"{tx_note}"
        )

    path = Path(raw_path)
    cwd = getattr(harness.state, "cwd", None)
    if not path.is_absolute():
        base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
        try:
            path = (base / path).resolve()
        except Exception:
            path = base / path
    else:
        try:
            path = path.resolve()
        except Exception:
            pass

    if _task_requests_written_output_path(harness, raw_path):
        evidence_note = ""
        if _has_prior_successful_evidence_for_output_write(harness):
            evidence_note = " Use the prior successful tool output as the file content."
        return (
            f"`{path}` is the requested output file and it does not exist yet. "
            f"Do not call `file_read` on it again. Create it with "
            f"`file_write(path='{raw_path}', content='...')`."
            f"{evidence_note} Do not call `task_complete` until the `file_write` succeeds."
            f"{tx_note}"
        )

    if _path_mentions_in_task_text(harness, raw_path):
        candidates = _nearby_missing_input_candidates(harness, raw_path)
        candidate_note = ""
        if len(candidates) == 1:
            candidate_note = f" A nearby file exists: `{candidates[0]}`. If that is the intended input, read it next."
        elif candidates:
            formatted = ", ".join(f"`{candidate}`" for candidate in candidates)
            candidate_note = f" Nearby files exist: {formatted}. Ask which input to use before proceeding."
        return (
            f"`{path}` is a required input file from the task, but it does not exist. "
            "Do not claim the task is complete and do not infer the file contents from memory or directory listings."
            f"{candidate_note} If no nearby file is the intended input, ask the user for the correct path or call `task_fail`."
            f"{tx_note}"
        )

    return (
        f"You already read `{path}`. Do not reread the same file; use the evidence you already "
        "have to patch it, run the focused test, or move on."
        f"{tx_note}"
    )


def should_pause_repeated_tool_loop(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name in {"dir_list", "artifact_read", "artifact_print", "artifact_grep", "file_read", "loop_status"}:
        return True
    return artifact_prefers_summary_synthesis(harness, pending)


def build_repeated_tool_loop_interrupt_payload(
    *,
    harness: Any,
    graph_state: GraphRunState,
    pending: PendingToolCall,
    repeat_error: str,
) -> dict[str, Any]:
    goal_recap = build_goal_recap(harness)
    notepad = harness.state.scratchpad.get("_session_notepad", {})
    entries = []
    if isinstance(notepad, dict):
        raw_entries = notepad.get("entries", [])
        if isinstance(raw_entries, list):
            entries = raw_entries[-5:]
    progress = " | ".join(entries) if entries else "exploration in progress"
    scratchpad = harness.state.scratchpad
    transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
    tx_lines = recovery_context_lines(transaction)
    tx_note = (" " + " ".join(tx_lines)) if tx_lines else ""
    base_guidance = f"TASK ANCHOR: {goal_recap}. PROGRESS SO FAR: {progress}.{tx_note} "
    question = (
        f"PAUSED: You were looping on `{pending.tool_name}` while working on: "
        f"{goal_recap or 'the current task'}. "
        "Reply `continue` to resume THAT SAME TASK. Do NOT start a different task."
    )
    guidance = (
        f"{base_guidance}"
        f"FORBIDDEN: Do not call `{pending.tool_name}` again with these same arguments. "
        "Do not read any file already in your context. "
        "Do not switch to a different task, project, or codebase. "
        "REQUIRED NEXT ACTION: Make a state-changing tool call "
        "(write, patch, exec) or call `task_complete`."
    )
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if pending.tool_name == "dir_list":
        guidance = (
            f"{base_guidance}"
            "Repeated dir_list loop detected. The chat preview shows up to 50 directory items, "
            "so trust the visible listing if the target is present. "
            "Move to a targeted next step or a different path instead of repeating dir_list "
            "on the same directory. Do NOT switch tasks."
        )
    elif pending.tool_name == "loop_status":
        stale = ""
        scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {})
        if isinstance(scratchpad, dict) and isinstance(scratchpad.get("_last_verifier_stale_after_mutation"), dict):
            stale = (
                " The last verifier verdict is stale because a file changed after it was recorded; "
                "do not wait for it to update by polling."
            )
        guidance = (
            f"{base_guidance}"
            "Repeated loop_status detected."
            f"{stale} "
            "Do not call `loop_status` again. REQUIRED NEXT ACTION: rerun the focused verifier, "
            "make a different specific patch/write, or call `task_complete` only if the task is already proven."
        )
    if artifact_evidence_is_unavailable(harness, pending):
        guidance = build_artifact_evidence_unavailable_message(harness, artifact_id=artifact_id)
    elif artifact_prefers_summary_synthesis(harness, pending):
        guidance = build_artifact_summary_exit_message(harness, artifact_id=artifact_id)
    return {
        "kind": "repeated_tool_loop_resume",
        "question": question,
        "current_phase": harness.state.current_phase,
        "active_profiles": list(harness.state.active_tool_profiles),
        "thread_id": graph_state.thread_id,
        "tool_name": pending.tool_name,
        "arguments": json_safe_value(pending.args),
        "guard": "repeated_tool_loop",
        "guard_error": repeat_error,
        "guidance": guidance,
        "created_at": time.time(),
    }
