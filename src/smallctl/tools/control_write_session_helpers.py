from __future__ import annotations

from typing import Any


WRITE_SESSION_SCHEMA_FAILURE_KEY = "_last_write_session_schema_failure"


def write_session_schema_failure(state: Any) -> dict[str, Any] | None:
    payload = getattr(state, "scratchpad", {}).get(WRITE_SESSION_SCHEMA_FAILURE_KEY)
    if not isinstance(payload, dict) or not payload:
        return None
    return payload


def write_session_resume_action(
    state: Any,
    failure: dict[str, Any] | None,
) -> dict[str, Any] | None:
    session = getattr(state, "write_session", None)
    if session is None or str(getattr(session, "status", "") or "").strip().lower() == "complete":
        return None

    is_finalizable = (
        getattr(session, "write_sections_completed", None)
        and not str(getattr(session, "write_next_section", "") or "").strip()
        and str(getattr(session, "status", None) or "open").strip().lower() in {"open", "verifying"}
    )
    if is_finalizable:
        return {
            "tool_name": "finalize_write_session",
            "required_fields": [],
            "required_arguments": {},
            "optional_fields": [],
            "notes": ["The write session is ready to finalize. Call finalize_write_session to promote the staged file."],
        }

    section_name = str(
        (failure or {}).get("recommended_section_name")
        or getattr(session, "write_next_section", None)
        or getattr(session, "write_current_section", None)
        or "imports"
    ).strip() or "imports"
    required_arguments = {
        "path": str((failure or {}).get("target_path") or getattr(session, "write_target_path", None) or "").strip(),
        "section_name": section_name,
    }
    notes = ["Provide non-empty `content` for this section."]
    if getattr(session, "write_sections_completed", None) and not getattr(session, "write_next_section", None):
        notes.append("Omit `next_section_name` on the final chunk so the session can finalize after verification.")
    else:
        notes.append("Set `next_section_name` only if another section still needs to be written after this one.")
    if failure:
        missing = failure.get("required_fields")
        if isinstance(missing, list) and missing:
            notes.append(
                "Last schema failure was missing: "
                + ", ".join(str(field) for field in missing if str(field).strip())
            )
    return {
        "tool_name": "file_write",
        "required_fields": ["path", "content", "section_name"],
        "required_arguments": required_arguments,
        "optional_fields": ["next_section_name"],
        "notes": notes,
    }


def write_session_warning(state: Any) -> str | None:
    active_ws = getattr(state, "write_session", None)
    if active_ws is None:
        return None
    if str(getattr(active_ws, "status", "") or "").strip().lower() in {"complete"}:
        return None
    ws_id_hint = str(getattr(active_ws, "write_session_id", "") or "").strip()
    ws_path_hint = str(getattr(active_ws, "write_target_path", "") or "").strip()
    ws_next_hint = str(getattr(active_ws, "write_next_section", "") or "").strip() or "imports"
    if not ws_id_hint or not ws_path_hint:
        return None
    return (
        f"Write Session is open for `{ws_path_hint}`. "
        f"All file_write / file_patch / ast_patch calls to that path use the target path; "
        f"the harness will match them to the active session. "
        f"Next expected section: `{ws_next_hint}`. "
        f"task_complete will be blocked until the session is finalized."
    )
