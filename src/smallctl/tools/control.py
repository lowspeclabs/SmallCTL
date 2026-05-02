from __future__ import annotations

from typing import Any

from ..state import LoopState, clip_text_value
from ..write_session_fsm import recent_write_session_events, record_write_session_event
from .common import fail, ok
from .fs_loop_guard import build_loop_guard_status


_WRITE_SESSION_SCHEMA_FAILURE_KEY = "_last_write_session_schema_failure"
_REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"


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


def _remote_mutation_verification_requirement(state: LoopState) -> dict[str, Any] | None:
    payload = state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(payload, dict) or not payload:
        return None
    if payload.get("failed_verification_attempts", 0) >= 3:
        return None
    return payload


def _remote_mutation_block_payload(requirement: dict[str, Any]) -> dict[str, Any]:
    guessed_paths = requirement.get("guessed_paths")
    if not isinstance(guessed_paths, list):
        guessed_paths = []
    path_hint = ", ".join(str(path) for path in guessed_paths[:3] if str(path).strip())
    first_path = next((str(path).strip() for path in guessed_paths if str(path).strip()), "")
    host = str(requirement.get("host") or "").strip()
    user = str(requirement.get("user") or "").strip()
    mutation_type = str(requirement.get("mutation_type") or "").strip().lower()
    required_arguments: dict[str, Any] = {}
    if first_path:
        required_arguments["path"] = first_path
    if host:
        required_arguments["host"] = host
    if user:
        required_arguments["user"] = user
    if host and "@" in host:
        required_arguments.pop("host", None)
        required_arguments["target"] = host

    if mutation_type == "deletion":
        error = (
            "Cannot complete the task while a raw `ssh_exec` remote file deletion still needs meaningful verification. "
            "Verify the target is gone with `ssh_file_read`; a `not found` / `no such file` result counts as successful verification."
        )
        next_required_action = {
            "tool_names": ["ssh_file_read"],
            "required_fields": sorted(required_arguments),
            "required_arguments": required_arguments,
            "notes": [
                "Read the deleted path directly.",
                "A missing-file result is valid proof for deletion tasks and will clear the requirement.",
            ],
        }
    else:
        error = (
            "Cannot complete the task while a raw `ssh_exec` remote file mutation still needs meaningful verification. "
            "Read back the changed file with `ssh_file_read`, or redo the edit with `ssh_file_patch` / "
            "`ssh_file_replace_between` so the harness can verify the readback hash."
        )
        next_required_action = {
            "tool_names": ["ssh_file_read", "ssh_file_patch", "ssh_file_replace_between"],
            "required_fields": sorted(required_arguments),
            "required_arguments": required_arguments,
            "notes": [
                "A grep-only positive match is not enough for replacement tasks.",
                "Verify that the replacement exists and the old target is gone.",
            ],
        }
    if path_hint:
        error += f" Suspected path(s): {path_hint}."
    if required_arguments:
        verifier_call = "ssh_file_read(" + ", ".join(
            f"{key}={required_arguments[key]!r}" for key in ("target", "host", "user", "path") if key in required_arguments
        ) + ")"
        error += f" Next required verifier: `{verifier_call}`."
    return {
        "error": error,
        "next_required_action": next_required_action,
    }


def _write_session_resume_action(
    state: LoopState,
    failure: dict[str, Any] | None,
) -> dict[str, Any] | None:
    session = state.write_session
    if session is None or str(session.status or "").strip().lower() == "complete":
        return None

    # If the session already has all sections and no next section, suggest finalizing.
    is_finalizable = (
        session.write_sections_completed
        and not str(session.write_next_section or "").strip()
        and str(session.status or "open").strip().lower() in {"open", "verifying"}
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


def _is_weather_lookup_task(state: LoopState) -> bool:
    task_text = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "").strip().lower()
    if not task_text:
        return False
    return any(marker in task_text for marker in ("weather", "forecast", "temperature"))


def _has_specific_weather_answer(message: str) -> bool:
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False

    explicit_unavailable_markers = (
        "could not verify the exact",
        "couldn't verify the exact",
        "unable to verify the exact",
        "exact weather could not be verified",
        "exact current weather could not be verified",
        "exact temperature could not be verified",
        "i could not verify the exact",
    )
    if any(marker in text for marker in explicit_unavailable_markers):
        return True

    temperature_markers = ("°f", "°c", " fahrenheit", " celsius", " degree", " degrees")
    weather_markers = (
        "temperature",
        "temp",
        "forecast",
        "high",
        "low",
        "today",
        "currently",
        "weather",
        "sunny",
        "cloudy",
        "clear",
        "rain",
        "showers",
        "storm",
        "snow",
        "windy",
        "humid",
        "overcast",
        "drizzle",
        "thunder",
    )
    has_temperature = any(marker in text for marker in temperature_markers)
    has_weather_detail = any(marker in text for marker in weather_markers)
    return has_temperature and has_weather_detail


def _looks_like_weather_search_meta_completion(message: str) -> bool:
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False
    if _has_specific_weather_answer(text):
        return False
    meta_markers = (
        "web search completed",
        "search completed",
        "found ",
        "returned ",
    )
    return any(marker in text for marker in meta_markers) and "result" in text


async def task_complete(message: str, state: LoopState, harness: Any) -> dict:
    if state.plan_execution_mode and state.active_step_id:
        return fail(
            "Cannot call `task_complete` while staged execution is active. Use `step_complete` for the active step.",
            metadata={
                "reason": "task_complete_blocked_in_staged_execution",
                "active_step_id": state.active_step_id,
                "active_step_run_id": state.active_step_run_id,
            },
        )
    verifier_verdict = _normalized_verifier_verdict(state)
    remote_requirement = _remote_mutation_verification_requirement(state)
    if remote_requirement is not None:
        block_payload = _remote_mutation_block_payload(remote_requirement)
        return fail(
            str(block_payload["error"]),
            metadata={
                "reason": "remote_mutation_requires_verification",
                "remote_mutation_requirement": remote_requirement,
                "next_required_action": block_payload["next_required_action"],
                "last_verifier_verdict": verifier_verdict,
            },
        )
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
        # P2: auto-finalize if the session is clearly ready but was never finalized.
        from ..graph.write_session_outcomes import _attempt_write_session_finalize
        from ..tools.fs_sessions import _write_session_can_finalize

        is_finalizable = (
            not str(session.write_next_section or "").strip()
            and session.write_sections_completed
            and str(session.status or "open").strip().lower() in {"open", "verifying"}
            and _write_session_can_finalize(session)
        )
        if is_finalizable:
            finalized, _ = await _attempt_write_session_finalize(harness, session)
            if finalized:
                session = state.write_session

        if str(session.status or "").strip().lower() != "complete":
            record_write_session_event(
                state,
                event="task_complete_blocked",
                session=session,
                details={"reason": "session_incomplete"},
            )
            session_status = str(session.status or "open").strip() or "open"
            error = (
                "Cannot complete the task while Write Session "
                f"`{session.write_session_id}` for `{session.write_target_path}` is still {session_status}."
            )
            next_section = str(session.write_next_section or "").strip()
            if next_section:
                error += f" Next required section: `{next_section}`."
            elif session.write_pending_finalize:
                error += " The staged file still needs verification and finalization."
            else:
                error += " The staged file has not been finalized to the target path yet."
            failure = _write_session_schema_failure(state)
            return fail(
                error,
                metadata={
                    "write_session": session.to_dict(),
                    "next_required_tool": _write_session_resume_action(state, failure),
                    "last_verifier_verdict": verifier_verdict,
                    "acceptance_checklist": state.acceptance_checklist(),
                },
            )
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
    if _is_weather_lookup_task(state) and _looks_like_weather_search_meta_completion(message):
        return fail(
            "Task is not complete yet: the user asked for the weather, but your completion message only reports that a search ran. "
            "Provide the actual weather answer with attribution, or explicitly say that the exact weather could not be verified from the evidence you fetched. "
            "Do not finish with only result counts or source lists.",
            metadata={
                "reason": "lookup_answer_missing",
                "lookup_kind": "weather",
                "next_required_action": {
                    "tool_name": "web_fetch",
                    "notes": [
                        "Fetch a returned search result by `result_id` instead of inventing a forecast URL.",
                        "If fetches still fail, answer explicitly that the exact weather could not be verified from the available evidence.",
                    ],
                },
            },
        )
    state.scratchpad["_task_complete"] = True
    state.scratchpad["_task_complete_message"] = message
    state.touch()
    return ok({"status": "complete", "message": message})


async def step_complete(message: str, state: LoopState, harness: Any) -> dict:
    if not state.plan_execution_mode or not state.active_step_id:
        return fail(
            "`step_complete` is only available while staged execution has an active step.",
            metadata={"reason": "staged_execution_inactive"},
        )
    remote_requirement = _remote_mutation_verification_requirement(state)
    if remote_requirement is not None:
        block_payload = _remote_mutation_block_payload(remote_requirement)
        return fail(
            str(block_payload["error"]).replace("task", "step", 1),
            metadata={
                "reason": "remote_mutation_requires_verification",
                "remote_mutation_requirement": remote_requirement,
                "next_required_action": block_payload["next_required_action"],
            },
        )
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
        return fail(
            "Cannot complete the step while an active write session is incomplete.",
            metadata={
                "reason": "write_session_incomplete",
                "write_session": session.to_dict(),
            },
        )
    state.scratchpad["_step_complete_requested"] = True
    state.scratchpad["_step_complete_message"] = str(message or "")
    state.scratchpad.pop("_step_failed_requested", None)
    state.scratchpad.pop("_step_failed_message", None)
    state.touch()
    log = getattr(harness, "log", None)
    if log is not None and callable(getattr(log, "info", None)):
        log.info(
            "staged_step_completion_requested step_id=%s step_run_id=%s",
            state.active_step_id,
            state.active_step_run_id,
        )
    return ok(
        {
            "status": "step_completion_requested",
            "message": message,
            "step_id": state.active_step_id,
            "step_run_id": state.active_step_run_id,
        }
    )


async def step_fail(message: str, state: LoopState, harness: Any) -> dict:
    if not state.plan_execution_mode or not state.active_step_id:
        return fail(
            "`step_fail` is only available while staged execution has an active step.",
            metadata={"reason": "staged_execution_inactive"},
        )
    state.scratchpad["_step_failed_requested"] = True
    state.scratchpad["_step_failed_message"] = str(message or "")
    state.scratchpad.pop("_step_complete_requested", None)
    state.scratchpad.pop("_step_complete_message", None)
    state.recent_errors.append(str(message or "Step failed."))
    state.touch()
    return ok(
        {
            "status": "step_failed_requested",
            "message": message,
            "step_id": state.active_step_id,
            "step_run_id": state.active_step_run_id,
        }
    )


async def finalize_write_session(state: LoopState, harness: Any) -> dict:
    session = state.write_session
    if not session:
        return fail("No active write session to finalize.")
    if str(session.status or "").strip().lower() == "complete":
        target_path = str(getattr(session, "write_target_path", "") or "").strip()
        message = "Write session is already complete."
        if target_path:
            message += f" Promoted file: `{target_path}`."
        return ok({"status": "already_finalized", "message": message})

    from ..graph.write_session_outcomes import _attempt_write_session_finalize
    success, detail = await _attempt_write_session_finalize(harness, session)
    if success:
        return ok({"status": "finalized", "message": detail})
    return fail(f"Unable to finalize write session: {detail}")


async def task_fail(message: str, state: LoopState) -> dict:
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
        record_write_session_event(
            state,
            event="session_abandoned",
            session=session,
            details={"reason": "task_fail"},
        )
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
    write_session_events = recent_write_session_events(state, limit=10)
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
            "plan_execution_mode": state.plan_execution_mode,
            "active_step_id": state.active_step_id,
            "active_step_run_id": state.active_step_run_id,
            "step_verification_result": state.step_verification_result,
            "max_steps": max_steps_int or None,
            "progress_pct": round(progress_pct, 4),
            "acceptance_ready": state.acceptance_ready(),
            "acceptance_waived": state.acceptance_waived,
            "acceptance_checklist": acceptance_checklist,
            "pending_acceptance_criteria": [item["criterion"] for item in acceptance_checklist if not item["satisfied"]],
            "last_verifier_verdict": verifier_verdict,
            "last_failure_class": state.last_failure_class,
            "files_changed_this_cycle": state.files_changed_this_cycle,
            "system_repair_cycle_id": state.repair_cycle_id,
            "stagnation_counters": state.stagnation_counters,
            "next_required_tool": next_required_tool,
            "write_session": write_session_payload,
            "write_session_events": write_session_events,
            "loop_guard": build_loop_guard_status(state),
        }
    )
