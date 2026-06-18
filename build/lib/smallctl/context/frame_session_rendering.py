from __future__ import annotations

from typing import Any

from ..state import LoopState, clip_string_list, clip_text_value


def render_session_notepad(state: LoopState) -> str:
    payload = state.scratchpad.get("_session_notepad")
    if not isinstance(payload, dict):
        return ""
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        return ""

    entries, clipped = clip_string_list(
        raw_entries,
        limit=5,
        item_char_limit=160,
    )
    if not entries:
        return ""

    bits = ["Session notepad: " + " | ".join(entries)]
    bits.append("Keep this brief; the notepad is durable session memory, not a transcript dump.")
    if clipped or len(entries) < len([item for item in raw_entries if str(item).strip()]):
        bits.append("Some entries were clipped to keep the session note concise.")
    return " ".join(bits)


def render_write_session(state: LoopState) -> str:
    session = state.write_session
    ws_bits = []
    if session.write_target_path:
        ws_bits.append(f"Target: {session.write_target_path}")
    if session.write_current_section:
        ws_bits.append(f"Current section: {session.write_current_section}")
    if session.write_next_section:
        ws_bits.append(f"Next section: {session.write_next_section}")
    if session.write_sections_completed:
        ws_bits.append(f"Completed sections: {', '.join(session.write_sections_completed)}")
    if session.write_staging_path:
        ws_bits.append(f"Staging: {session.write_staging_path}")
    if session.write_staging_path:
        ws_bits.append("Reminder: the staging path is for read/verify only; write to the target path.")
    else:
        ws_bits.append("Reminder: write to the target path; staged copies are for read/verify.")
    ws_bits.append(f"Mode: {session.write_session_mode}")
    ws_bits.append(f"Intent: {session.write_session_intent}")
    ws_bits.append(f"Status: {session.status}")
    if session.suggested_sections:
        ws_bits.append(f"Suggested sections: {', '.join(session.suggested_sections)}")
    if session.write_failed_local_patches:
        ws_bits.append(f"Local patch failures: {session.write_failed_local_patches}")
    if session.write_pending_finalize:
        ws_bits.append("Pending finalize: yes")
    else:
        ws_bits.append("Pending finalize: no")
    next_action = render_write_session_next_action(session)
    if next_action:
        ws_bits.append(f"Next action: {next_action}")
    ws_bits.append(f"Session ID: {session.write_session_id}")
    verifier = session.write_last_verifier or {}
    if verifier:
        ws_bits.append(f"Last verifier verdict: {verifier.get('verdict', 'unknown')}")
        command = str(verifier.get("command", "") or "").strip()
        if command:
            ws_bits.append(f"Verifier command: {command}")
        verifier_output, clipped = clip_text_value(str(verifier.get("output", "") or "").strip(), limit=180)
        if verifier_output:
            suffix = " [truncated]" if clipped else ""
            ws_bits.append(f"Verifier output: {verifier_output}{suffix}")
    return "Active Write Session: " + " | ".join(ws_bits)


def render_write_session_next_action(session: Any) -> str:
    if bool(getattr(session, "write_pending_finalize", False)):
        return "Finalize the staged copy after verification."
    next_section = str(getattr(session, "write_next_section", "") or "").strip()
    if next_section:
        return f"Continue with section {next_section}."
    current_section = str(getattr(session, "write_current_section", "") or "").strip()
    if current_section:
        return f"Complete section {current_section} or move to verification once it is ready."
    completed = list(getattr(session, "write_sections_completed", []) or [])
    if completed:
        return "Choose the next section or verify the completed staged file."
    return "Continue authoring the active target file."
