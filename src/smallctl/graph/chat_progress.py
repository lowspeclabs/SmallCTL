from __future__ import annotations

import json
import re
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


def recent_assistant_texts(harness: Any, *, limit: int = 2) -> list[str]:
    texts: list[str] = []
    for message in reversed(getattr(harness.state, "recent_messages", [])):
        if getattr(message, "role", "") != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        texts.append(content)
        if len(texts) >= limit:
            break
    return texts


def looks_like_freeze_or_hang(harness: Any, assistant_text: str) -> bool:
    text = str(assistant_text or "").strip()
    if not text:
        return False
    recent = recent_assistant_texts(harness, limit=3)
    if not recent:
        return False
    if text in recent:
        return True
    if len(recent) >= 2 and recent[0] == recent[1]:
        return True
    if len(recent) >= 3 and recent[0] == recent[1] == recent[2]:
        return True
    return False


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


def task_prefers_summary_synthesis(harness: Any) -> bool:
    merged = _merged_task_text(harness)
    if not merged:
        return False
    asks_for_summary = any(keyword in merged for keyword in ("table", "summary", "summarize", "report", "overview", "present"))
    asks_about_listing = any(keyword in merged for keyword in ("list", "listing", "files", "directories", "artifact", "results", "output", "current env"))
    return asks_for_summary and asks_about_listing


def _merged_task_text(harness: Any) -> str:
    texts = [
        str(getattr(getattr(harness, "state", None), "run_brief", None).original_task or "")
        if getattr(getattr(harness, "state", None), "run_brief", None) is not None
        else "",
        str(getattr(getattr(harness, "state", None), "working_memory", None).current_goal or "")
        if getattr(getattr(harness, "state", None), "working_memory", None) is not None
        else "",
    ]
    current_user_task = getattr(harness, "_current_user_task", None)
    if callable(current_user_task):
        texts.append(str(current_user_task() or ""))
    return " ".join(text.strip().lower() for text in texts if text and text.strip())


def _pending_artifact_record(harness: Any, pending: PendingToolCall) -> Any | None:
    artifact_id = str(_extract_artifact_id_from_args(pending.args) or "").strip()
    if not artifact_id:
        return None
    artifacts = getattr(getattr(harness, "state", None), "artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    return artifacts.get(artifact_id)


def artifact_prefers_summary_synthesis(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name not in {"artifact_read", "artifact_print"}:
        return False
    if task_prefers_summary_synthesis(harness):
        return True

    artifact = _pending_artifact_record(harness, pending)
    if artifact is None:
        return False

    artifact_kind = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip().lower()
    if artifact_kind not in {"web_search", "web_fetch"}:
        return False

    merged = _merged_task_text(harness)
    if not merged:
        return False

    asks_for_web_research_synthesis = any(
        keyword in merged
        for keyword in (
            "web search",
            "websearch",
            "research",
            "findings",
            "what it is",
            "what is ",
            "how it works",
            "how does it work",
            "explain",
            "detailed summary",
        )
    )
    return asks_for_web_research_synthesis


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


def build_artifact_summary_exit_message(harness: Any, *, artifact_id: str = "") -> str:
    objective = str(getattr(getattr(harness, "state", None), "run_brief", None).original_task or "").strip()
    artifact_note = f" from artifact {artifact_id}" if artifact_id else ""
    objective_note = f" for `{objective}`" if objective else ""
    return (
        f"You already have enough evidence{artifact_note}{objective_note}. "
        "Produce the requested table or summary now with `task_complete(message='...')` "
        "instead of rereading or printing the same artifact again."
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

    return (
        f"You already read `{path}`. Do not reread the same file; use the evidence you already "
        "have to patch it, run the focused test, or move on."
        f"{tx_note}"
    )


def should_pause_repeated_tool_loop(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name in {"dir_list", "artifact_read", "artifact_print", "artifact_grep", "file_read"}:
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
    }
