from __future__ import annotations

from typing import Any

from ..diagnostic_tasks import diagnostic_failure_completion_allowed
from ..phase_contracts import phase_contract_completion_block
from ..runtime_error_repair import runtime_error_completion_block
from ..state import LoopState
from .common import fail
from .control_objective_ledger import multi_objective_completion_block as _multi_objective_completion_block
from .control_phase_gates import (
    mutation_expectation_block as _mutation_expectation_block,
    phase_promotion_gate_block as _phase_promotion_gate_block,
    task_involves_interactive_program as _task_involves_interactive_program,
)
from .control_plan_subtasks import plan_subtask_completion_block as _plan_subtask_completion_block
from .control_post_change import post_change_verification_block as _post_change_verification_block
from .control_remote_mutation import (
    remote_mutation_block_payload as _remote_mutation_block_payload,
    remote_mutation_verification_requirement as _remote_mutation_verification_requirement,
)
from .control_verifier_helpers import (
    normalized_verifier_verdict as _normalized_verifier_verdict,
    verifier_failure_summary as _verifier_failure_summary,
    verifier_requires_human_approval as _verifier_requires_human_approval,
)
from .verifier_quality import verifier_quality as _verifier_quality


def unresolved_missing_input_file(state: LoopState) -> dict | None:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return None
    blocker = scratchpad.get("_unresolved_missing_input_file")
    if isinstance(blocker, dict) and str(blocker.get("path") or "").strip():
        return blocker
    return None


def task_complete_gate_staged_execution(state: LoopState) -> dict | None:
    if state.plan_execution_mode and state.active_step_id:
        return fail(
            "Cannot call `task_complete` while staged execution is active. Use `step_complete` for the active step.",
            metadata={
                "reason": "task_complete_blocked_in_staged_execution",
                "active_step_id": state.active_step_id,
                "active_step_run_id": state.active_step_run_id,
            },
        )
    return None


def task_complete_gate_runtime_error(state: LoopState) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    runtime_error_block = runtime_error_completion_block(state, verifier_verdict=verifier_verdict)
    if runtime_error_block is not None:
        return fail(
            "Cannot complete the task until the reported runtime error is verified fixed.",
            metadata={
                **runtime_error_block,
                "acceptance_checklist": state.acceptance_checklist(),
            },
        )
    return None


def task_complete_gate_post_change(state: LoopState) -> dict | None:
    post_change_block = _post_change_verification_block(state)
    if post_change_block is not None:
        return fail(
            "Cannot complete the task until the latest file change is verified.",
            metadata={
                **post_change_block,
                "last_verifier_verdict": _normalized_verifier_verdict(state),
                "acceptance_checklist": state.acceptance_checklist(),
            },
        )
    return None


def task_complete_gate_interactive_program(state: LoopState) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    if _task_involves_interactive_program(state) and verifier_verdict:
        verifier_command = str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip()
        quality = _verifier_quality(verifier_command)
        if int(quality.get("score") or 0) < 3:
            return fail(
                "Cannot complete an interactive/GUI program task with only a syntax or import verifier. "
                "Run a behavioral verifier that exercises the game loop, event handling, or rendering.",
                metadata={
                    "reason": "interactive_program_requires_behavioral_verifier",
                    "verifier_quality": quality,
                    "required_quality": {"score": 3, "label": "behavioral"},
                    "last_verifier_verdict": verifier_verdict,
                    "acceptance_checklist": state.acceptance_checklist(),
                    "next_required_action": {
                        "tool_name": "shell_exec",
                        "notes": [
                            "Run a verifier that exercises the interactive/game behavior, not just syntax.",
                            "Examples: run the game with a mock event, test a frame update, or verify output files.",
                        ],
                    },
                },
            )
    return None


def task_complete_gate_remote_mutation(state: LoopState) -> dict | None:
    remote_requirement = _remote_mutation_verification_requirement(state)
    if remote_requirement is not None:
        block_payload = _remote_mutation_block_payload(remote_requirement)
        return fail(
            str(block_payload["error"]),
            metadata={
                "reason": "remote_mutation_requires_verification",
                "remote_mutation_requirement": remote_requirement,
                "next_required_action": block_payload["next_required_action"],
                "last_verifier_verdict": _normalized_verifier_verdict(state),
            },
        )
    return None


def task_complete_gate_missing_input(state: LoopState) -> dict | None:
    missing_input = unresolved_missing_input_file(state)
    if missing_input is not None:
        path = str(missing_input.get("path") or "").strip()
        return fail(
            f"Cannot complete the task because required input file `{path}` was not found.",
            metadata={
                "reason": "missing_required_input_file",
                "missing_input_file": missing_input,
                "next_required_action": {
                    "tool_names": ["file_read", "ask_human", "task_fail"],
                    "notes": [
                        "Read the correct input file if the path was a typo.",
                        "Ask the user for the correct path if no intended file is available.",
                        "Do not infer the missing file contents from memory or directory listings.",
                    ],
                },
                "last_verifier_verdict": _normalized_verifier_verdict(state),
            },
        )
    return None


def task_complete_gate_mutation_expectation(state: LoopState, message: str) -> dict | None:
    mutation_block = _mutation_expectation_block(state, message=message)
    if mutation_block is not None:
        return fail(
            "Cannot complete a phase implementation task with zero code changes.",
            metadata={
                **mutation_block,
                "last_verifier_verdict": _normalized_verifier_verdict(state),
                "acceptance_checklist": state.acceptance_checklist(),
            },
        )
    return None


def task_complete_gate_verifier_approval(state: LoopState) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    if (
        verifier_verdict
        and _verifier_requires_human_approval(verifier_verdict)
        and not state.acceptance_waived
    ):
        error = "Cannot complete the task until the latest verifier check is approved or rerun with approval."
        verifier_summary = _verifier_failure_summary(verifier_verdict)
        if verifier_summary:
            error = f"{error} Latest verifier: {verifier_summary}."
        return fail(
            error,
            metadata={
                "last_verifier_verdict": verifier_verdict,
                "acceptance_checklist": state.acceptance_checklist(),
                "approval_required": True,
            },
        )
    return None


def task_complete_gate_verifier_failure(state: LoopState, message: str) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    verifier_failure_satisfies_diagnostic = (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and diagnostic_failure_completion_allowed(state, message=message, verifier=verifier_verdict)
    )
    if (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and not state.acceptance_waived
        and not verifier_failure_satisfies_diagnostic
    ):
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
    return None


def task_complete_gate_phase_contract(state: LoopState) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    verifier_command = str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip() if isinstance(verifier_verdict, dict) else ""
    phase_contract_block = phase_contract_completion_block(
        state,
        verifier_verdict=verifier_verdict,
        verifier_quality=_verifier_quality(verifier_command),
    )
    if phase_contract_block is not None:
        return fail(
            "Cannot complete this phase until the active phase contract passes.",
            metadata={
                **phase_contract_block,
                "acceptance_checklist": state.acceptance_checklist(),
            },
        )
    return None


def task_complete_gate_phase_promotion(state: LoopState, message: str) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    phase_promotion_block = _phase_promotion_gate_block(
        state,
        message=message,
        verifier_verdict=verifier_verdict,
    )
    if phase_promotion_block is not None:
        return fail(
            "Cannot complete this phase until a behavioral promotion gate passes.",
            metadata={
                **phase_promotion_block,
                "acceptance_checklist": state.acceptance_checklist(),
            },
        )
    return None


def task_complete_gate_plan_subtasks(state: LoopState) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    plan_subtask_block = _plan_subtask_completion_block(state, verifier_verdict=verifier_verdict)
    if plan_subtask_block is not None:
        next_subtask = plan_subtask_block["next_required_subtask"]
        return fail(
            "Cannot complete the task while plan subtasks are still open. "
            f"Next required subtask: {next_subtask.get('subtask_id')} - {next_subtask.get('title')}.",
            metadata={
                "reason": "plan_subtasks_incomplete",
                **plan_subtask_block,
            },
        )
    return None


def task_complete_gate_acceptance(state: LoopState, message: str) -> dict | None:
    verifier_verdict = _normalized_verifier_verdict(state)
    verifier_failure_satisfies_diagnostic = (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and diagnostic_failure_completion_allowed(state, message=message, verifier=verifier_verdict)
    )
    if not state.acceptance_ready() and not verifier_failure_satisfies_diagnostic:
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
    return None
