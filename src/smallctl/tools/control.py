from __future__ import annotations

from typing import Any

from ..state import LoopState, clip_text_value
from .common import fail, ok


_WRITE_SESSION_SCHEMA_FAILURE_KEY = "_last_write_session_schema_failure"


def _normalized_verifier_verdict(state: LoopState) -> dict[str, Any] | None:
    verdict = state.current_verifier_verdict()
    if not isinstance(verdict, dict) or not verdict:
        return None
    return verdict


def _verifier_failure_summary(verifier_verdict: dict[str, Any] | None) -> str:
    if not isinstance(verifier_verdict, dict) or not verifier_verdict:
        return ""

    bits: list[str] = []
    target_text, clipped = clip_text_value(
        str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip(),
        limit=180,
    )
    if target_text:
        suffix = " [truncated]" if clipped else ""
        bits.append(f"check={target_text}{suffix}")

    detail = ""
    acceptance_delta = verifier_verdict.get("acceptance_delta")
    if isinstance(acceptance_delta, dict):
        notes = acceptance_delta.get("notes")
        if isinstance(notes, list):
            detail = next((str(note).strip() for note in notes if str(note).strip()), "")
    if not detail:
        detail = str(
            verifier_verdict.get("key_stderr")
            or verifier_verdict.get("key_stdout")
            or ""
        ).strip()
    detail_text, clipped = clip_text_value(detail, limit=180)
    if detail_text:
        suffix = " [truncated]" if clipped else ""
        bits.append(f"details={detail_text}{suffix}")

    return " | ".join(bits)


def _write_session_schema_failure(state: LoopState) -> dict[str, Any] | None:
    payload = state.scratchpad.get(_WRITE_SESSION_SCHEMA_FAILURE_KEY)
    if not isinstance(payload, dict) or not payload:
        return None
    return payload


def _write_session_resume_action(
    state: LoopState,
    failure: dict[str, Any] | None,
) -> dict[str, Any] | None:
    session = state.write_session
    if session is None or str(session.status or "").strip().lower() == "complete":
        return None

    section_name = str(
        (failure or {}).get("recommended_section_name")
        or session.write_next_section
        or session.write_current_section
        or "imports"
    ).strip() or "imports"
    required_arguments = {
        "path": str((failure or {}).get("target_path") or session.write_target_path or "").strip(),
        "write_session_id": str(session.write_session_id or ""),
        "section_name": section_name,
    }
    notes = ["Provide non-empty `content` for this section."]
    if session.write_sections_completed and not session.write_next_section:
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
        "required_fields": ["path", "content", "write_session_id", "section_name"],
        "required_arguments": required_arguments,
        "optional_fields": ["next_section_name"],
        "notes": notes,
    }


async def task_complete(message: str, state: LoopState) -> dict:
    verifier_verdict = _normalized_verifier_verdict(state)
    if verifier_verdict and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"} and not state.acceptance_waived:
        error = "Cannot complete the task while the latest verifier verdict is still failing."
        verifier_summary = _verifier_failure_summary(verifier_verdict)
        if verifier_summary:
            error = f"{error} Latest verifier: {verifier_summary}."
        return fail(
            error,
            metadata={
                "last_verifier_verdict": verifier_verdict,
                "acceptance_checklist": state.acceptance_checklist(),
            },
        )
    if not state.acceptance_ready():
        checklist = state.acceptance_checklist()
        pending = [item["criterion"] for item in checklist if not item["satisfied"]]
        return fail(
            "Cannot complete the task until acceptance criteria are satisfied or waived.",
            metadata={
                "pending_acceptance_criteria": pending,
                "acceptance_checklist": checklist,
                "last_verifier_verdict": _normalized_verifier_verdict(state),
            },
        )
    state.scratchpad["_task_complete"] = True
    state.scratchpad["_task_complete_message"] = message
    state.touch()
    return ok({"status": "complete", "message": message})


async def task_fail(message: str, state: LoopState) -> dict:
    state.scratchpad["_task_failed"] = True
    state.scratchpad["_task_failed_message"] = message
    state.recent_errors.append(message)
    state.touch()
    return ok({"status": "failed", "message": message})


async def ask_human(question: str, state: LoopState) -> dict:
    state.scratchpad["_ask_human"] = True
    state.scratchpad["_ask_human_question"] = question
    state.touch()
    return ok({"status": "human_input_required", "question": question})


async def loop_status(state: LoopState) -> dict:
    max_steps = state.scratchpad.get("_max_steps")
    try:
        max_steps_int = int(max_steps) if max_steps is not None else 0
    except (TypeError, ValueError):
        max_steps_int = 0

    if max_steps_int > 0:
        progress_pct = min(1.0, max(0.0, state.step_count / max_steps_int))
    else:
        progress_pct = 0.0

    verifier_verdict = _normalized_verifier_verdict(state)
    acceptance_checklist = state.acceptance_checklist()
    contract_phase = state.contract_phase()
    write_session_failure = _write_session_schema_failure(state)
    write_session_payload = state.write_session.to_dict() if state.write_session else None
    next_required_tool = _write_session_resume_action(state, write_session_failure)
    if write_session_payload is not None and write_session_failure is not None:
        write_session_payload = dict(write_session_payload)
        write_session_payload["last_schema_failure"] = write_session_failure
    if write_session_payload is not None and next_required_tool is not None:
        write_session_payload = dict(write_session_payload)
        write_session_payload["resume_action"] = next_required_tool

    return ok(
        {
            "phase": state.current_phase,
            "contract_phase": contract_phase,
            "step_count": state.step_count,
            "token_usage": state.token_usage,
            "elapsed_seconds": state.elapsed_seconds,
            "recent_errors": state.recent_errors[-5:],
            "cwd": state.cwd,
            "active_tool_profiles": state.active_tool_profiles,
            "max_steps": max_steps_int or None,
            "progress_pct": round(progress_pct, 4),
            "acceptance_ready": state.acceptance_ready(),
            "acceptance_waived": state.acceptance_waived,
            "acceptance_checklist": acceptance_checklist,
            "pending_acceptance_criteria": [item["criterion"] for item in acceptance_checklist if not item["satisfied"]],
            "last_verifier_verdict": verifier_verdict,
            "last_failure_class": state.last_failure_class,
            "files_changed_this_cycle": state.files_changed_this_cycle,
            "repair_cycle_id": state.repair_cycle_id,
            "stagnation_counters": state.stagnation_counters,
            "next_required_tool": next_required_tool,
            "write_session": write_session_payload,
        }
    )
