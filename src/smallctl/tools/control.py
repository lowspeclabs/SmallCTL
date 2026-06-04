from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..phase_contracts import phase_contract_status
from ..runtime_error_repair import (
    runtime_error_ask_human_block,
    runtime_error_task_fail_block,
)
from ..state import LoopState
from ..write_session_fsm import record_write_session_event
from .common import fail, ok
from .control_objective_ledger import multi_objective_completion_block as _multi_objective_completion_block
from .control_phase_contracts import (
    normalize_phase_contract_payload as _normalize_phase_contract_payload,
    phase_contract_validation_error as _phase_contract_validation_error,
)
from .control_remote_mutation import (
    remote_mutation_block_payload as _remote_mutation_block_payload,
    remote_mutation_verification_requirement as _remote_mutation_verification_requirement,
)
from .control_verifier_helpers import (
    normalized_verifier_verdict as _normalized_verifier_verdict,
)
from .control_weather import (
    is_weather_lookup_task as _is_weather_lookup_task,
    looks_like_weather_search_meta_completion as _looks_like_weather_search_meta_completion,
)
from .control_write_session_helpers import (
    write_session_resume_action as _write_session_resume_action,
    write_session_schema_failure as _write_session_schema_failure,
    write_session_warning as _write_session_warning,
)
from .control_loop_status_helpers import (
    max_steps_progress as _max_steps_progress,
    subtask_ledger_status as _subtask_ledger_status,
    write_session_status_events as _write_session_status_events,
    write_session_status_payload as _write_session_status_payload,
)
from .control_task_complete_gates import (
    task_complete_gate_acceptance as _task_complete_gate_acceptance,
    task_complete_gate_interactive_program as _task_complete_gate_interactive_program,
    task_complete_gate_missing_input as _task_complete_gate_missing_input,
    task_complete_gate_mutation_expectation as _task_complete_gate_mutation_expectation,
    task_complete_gate_phase_contract as _task_complete_gate_phase_contract,
    task_complete_gate_phase_promotion as _task_complete_gate_phase_promotion,
    task_complete_gate_plan_subtasks as _task_complete_gate_plan_subtasks,
    task_complete_gate_post_change as _task_complete_gate_post_change,
    task_complete_gate_remote_mutation as _task_complete_gate_remote_mutation,
    task_complete_gate_runtime_error as _task_complete_gate_runtime_error,
    task_complete_gate_staged_execution as _task_complete_gate_staged_execution,
    task_complete_gate_verifier_approval as _task_complete_gate_verifier_approval,
    task_complete_gate_verifier_failure as _task_complete_gate_verifier_failure,
)
from .fs_loop_guard import build_loop_guard_status
from .verifier_quality import (
    verifier_quality as _verifier_quality,
)

async def _task_complete_gate_write_session(state: LoopState, harness: Any) -> dict | None:
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
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
            scratchpad = state.scratchpad
            blocker_key = f"write_session:{session.write_session_id}"
            last_blocker = scratchpad.get("_task_complete_last_blocker")
            if last_blocker == blocker_key:
                scratchpad["_task_complete_blocker_count"] = scratchpad.get("_task_complete_blocker_count", 0) + 1
            else:
                scratchpad["_task_complete_last_blocker"] = blocker_key
                scratchpad["_task_complete_blocker_count"] = 1

            if scratchpad.get("_task_complete_blocker_count", 0) >= 2:
                has_no_sections = not session.write_sections_completed
                from ..tools.fs import _resolve
                try:
                    target = _resolve(session.write_target_path, getattr(state, "cwd", None))
                    file_exists = target.exists() and target.is_file() and target.stat().st_size > 0
                except Exception:
                    file_exists = False

                if has_no_sections and file_exists:
                    record_write_session_event(
                        state,
                        event="session_abandoned",
                        session=session,
                        details={
                            "reason": "auto_abandoned_orphan_session",
                            "rejection_count": scratchpad["_task_complete_blocker_count"],
                        },
                    )
                    state.write_session = None
                    scratchpad.pop("_task_complete_last_blocker", None)
                    scratchpad.pop("_task_complete_blocker_count", None)
                    return await task_complete("", state, harness)

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
                    "last_verifier_verdict": _normalized_verifier_verdict(state),
                    "acceptance_checklist": state.acceptance_checklist(),
                },
            )
    return None


async def task_complete(message: str, state: LoopState, harness: Any) -> dict:
    verifier_verdict = _normalized_verifier_verdict(state)
    gates = [
        lambda: _task_complete_gate_staged_execution(state),
        lambda: _task_complete_gate_runtime_error(state),
        lambda: _task_complete_gate_post_change(state),
        lambda: _task_complete_gate_interactive_program(state),
        lambda: _task_complete_gate_remote_mutation(state),
        lambda: _task_complete_gate_missing_input(state),
        lambda: _task_complete_gate_mutation_expectation(state, message),
    ]
    for gate in gates:
        result = gate()
        if result is not None:
            return result

    ledger_service = getattr(harness, "subtask_ledger", None)
    if ledger_service is not None:
        try:
            ledger_service.import_plan_if_needed(replace_synthetic_root=True)
        except Exception:
            pass

    result = await _task_complete_gate_write_session(state, harness)
    if result is not None:
        return result

    gates = [
        lambda: _task_complete_gate_verifier_approval(state),
        lambda: _task_complete_gate_verifier_failure(state, message),
        lambda: _task_complete_gate_phase_contract(state),
        lambda: _task_complete_gate_phase_promotion(state, message),
        lambda: _task_complete_gate_plan_subtasks(state),
        lambda: _task_complete_gate_acceptance(state, message),
    ]
    for gate in gates:
        result = gate()
        if result is not None:
            return result

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
    objective_block = _multi_objective_completion_block(
        state,
        message=message,
        verifier_verdict=verifier_verdict,
    )
    if objective_block is not None:
        remaining = objective_block["remaining_objectives"]
        first_remaining = str(remaining[0].get("title") or "the next open objective")
        completed_now = objective_block["completed_now"]
        if completed_now:
            error = (
                "Marked the current subobjective complete, but the parent task still has open objectives. "
                f"Next open objective: {first_remaining}"
            )
        else:
            error = (
                "Cannot complete the parent task because it still has open objectives and the completion message "
                f"did not match a specific open objective. Next open objective: {first_remaining}"
            )
        return fail(
            error,
            metadata={
                "reason": "multi_objective_incomplete",
                "completed_objectives": completed_now,
                "remaining_objectives": remaining,
                "multi_objective_ledger": objective_block["ledger"],
                "last_verifier_verdict": objective_block["last_verifier_verdict"],
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


async def phase_contract_update(contract: dict[str, Any], state: LoopState, persist: bool = False) -> dict:
    if not isinstance(contract, dict):
        return fail(
            "phase_contract_update requires a JSON object contract.",
            metadata={"reason": "invalid_phase_contract"},
        )
    contract = _normalize_phase_contract_payload(contract)
    validation_error = _phase_contract_validation_error(contract)
    if validation_error:
        return fail(
            validation_error,
            metadata={"reason": "invalid_phase_contract"},
        )
    normalized = dict(contract)
    normalized.setdefault("version", 1)
    state.scratchpad["_phase_contract"] = normalized
    persisted_path = ""
    if persist:
        path = Path(str(getattr(state, "cwd", "") or ".")) / ".smallctl" / "phase_contract.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            persisted_path = str(path)
        except OSError as exc:
            return fail(
                f"Unable to persist phase contract: {exc}",
                metadata={"reason": "phase_contract_persist_failed", "path": str(path)},
            )
    verifier_verdict = _normalized_verifier_verdict(state)
    verifier_command = str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip() if isinstance(verifier_verdict, dict) else ""
    status = phase_contract_status(
        state,
        verifier_verdict=verifier_verdict,
        verifier_quality=_verifier_quality(verifier_command),
    )
    state.touch()
    return ok(
        {
            "status": "updated",
            "persisted_path": persisted_path or None,
            "phase_contract": status,
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


async def task_fail(message: str, state: LoopState, harness: Any | None = None) -> dict:
    verifier_verdict = _normalized_verifier_verdict(state)
    runtime_error_block = runtime_error_task_fail_block(
        state,
        message=message,
        verifier_verdict=verifier_verdict,
    )
    if runtime_error_block is not None:
        return fail(
            "Cannot fail the task with an unsupported explanation while a reported runtime error is open.",
            metadata=runtime_error_block,
        )
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
        # P2: auto-finalize if the session is clearly ready but was never finalized.
        #     This prevents stranded staged files when the model gives up after
        #     writing all content but forgetting to finalize.
        if harness is not None:
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
    runtime_error_block = runtime_error_ask_human_block(state, question=question)
    if runtime_error_block is not None:
        return fail(
            "Cannot ask the user to retry before repairing and verifying the reported runtime error.",
            metadata=runtime_error_block,
        )
    state.scratchpad["_ask_human"] = True
    state.scratchpad["_ask_human_question"] = question
    state.touch()
    return ok({"status": "human_input_required", "question": question})


async def loop_status(state: LoopState) -> dict:
    max_steps_int, progress_pct = _max_steps_progress(state)
    verifier_verdict = _normalized_verifier_verdict(state)
    verifier_command = str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip() if isinstance(verifier_verdict, dict) else ""
    phase_contract_payload = phase_contract_status(
        state,
        verifier_verdict=verifier_verdict,
        verifier_quality=_verifier_quality(verifier_command),
    )
    acceptance_checklist = state.acceptance_checklist()
    contract_phase = state.contract_phase()
    write_session_failure = _write_session_schema_failure(state)
    next_required_tool = _write_session_resume_action(state, write_session_failure)
    write_session_payload = _write_session_status_payload(
        state,
        schema_failure=write_session_failure,
        resume_action=next_required_tool,
    )

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
            "subtask_ledger": _subtask_ledger_status(state),
            "phase_contract": phase_contract_payload,
            "last_verifier_verdict": verifier_verdict,
            "last_failure_class": state.last_failure_class,
            "files_changed_this_cycle": state.files_changed_this_cycle,
            "system_repair_cycle_id": state.repair_cycle_id,
            "stagnation_counters": state.stagnation_counters,
            "next_required_tool": next_required_tool,
            "write_session": write_session_payload,
            "write_session_warning": _write_session_warning(state),
            "write_session_events": _write_session_status_events(state, limit=10),
            "stderr_signature_circuit_breaker": state.scratchpad.get("_stderr_signature_circuit_breaker"),
            "loop_guard": build_loop_guard_status(state),
        }
    )
