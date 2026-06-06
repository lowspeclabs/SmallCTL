from __future__ import annotations

import re
from pathlib import Path
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
    has_manual_success = (
        state.scratchpad.get("manual_success")
        or state.scratchpad.get("manual_success_override")
        or state.scratchpad.get("manual_override")
        or any("manual success" in str(f).lower() or "verified manually" in str(f).lower() for f in getattr(state.working_memory, "known_facts", []))
    )
    if has_manual_success:
        return None
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


def _command_backed_file_requirements(state: LoopState) -> list[dict[str, str]]:
    task_text = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "")
    if not task_text.strip():
        return []

    requirements: list[dict[str, str]] = []
    pattern = re.compile(
        r"Create\s+(?:a\s+)?(?P<scope>local|remote)\s+[^\n:]*file\s*:\s*\n+\s*"
        r"(?P<path>\S+)\s*\n+\s*using\s+the\s+(?:local|remote)\s+system's\s+"
        r"(?P<command>[A-Za-z0-9_.+-]+)\s+command\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(task_text):
        scope = match.group("scope").lower()
        requirements.append(
            {
                "scope": scope,
                "path": match.group("path").strip(),
                "command": match.group("command").strip().lower(),
                "tool_name": "shell_exec" if scope == "local" else "ssh_exec",
            }
        )
    lower_task = task_text.lower()
    if "output of listing" in lower_task or "output of list" in lower_task or "*.log" in lower_task or "glob" in lower_task:
        local_paths = re.findall(r"(?m)^\s*(\.?/?[^\s:]+/[^\s:]*result[^\s:]*)\s*$", task_text)
        for path in local_paths:
            scope = "remote" if path.startswith("/tmp/") or path.startswith("/var/") or "remote" in path.lower() else "local"
            requirements.append(
                {
                    "scope": scope,
                    "path": path.strip(),
                    "command": "ls",
                    "tool_name": "ssh_exec" if scope == "remote" else "shell_exec",
                }
            )
    return requirements


def _artifact_text_for_command_match(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    arguments = metadata.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    parts = [
        getattr(artifact, "source", ""),
        getattr(artifact, "summary", ""),
        getattr(artifact, "inline_content", ""),
        getattr(artifact, "preview_text", ""),
        str(arguments.get("command") or ""),
        str(metadata.get("command") or ""),
        str(metadata.get("verifier_command") or ""),
    ]
    return "\n".join(part for part in parts if part).lower()


def _command_requirement_satisfied(state: LoopState, requirement: dict[str, str]) -> bool:
    required_tool = requirement["tool_name"]
    required_path = requirement["path"].lower()
    required_command = requirement["command"].lower()
    for artifact in getattr(state, "artifacts", {}).values():
        tool_name = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if tool_name != required_tool:
            continue
        metadata = getattr(artifact, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("success") is False:
            continue
        text = _artifact_text_for_command_match(artifact)
        if required_path in text and re.search(rf"\b{re.escape(required_command)}\b", text):
            return True
    return False


def task_complete_gate_command_backed_file_creation(state: LoopState) -> dict | None:
    requirements = _command_backed_file_requirements(state)
    pending = [req for req in requirements if not _command_requirement_satisfied(state, req)]
    if not pending:
        return None
    return fail(
        "Cannot complete the task until command-backed file creation requirements are satisfied.",
        metadata={
            "reason": "command_backed_file_creation_required",
            "pending_command_backed_file_requirements": pending,
            "next_required_action": {
                "tool_names": sorted({req["tool_name"] for req in pending}),
                "notes": [
                    "Create the requested file from the required system command in the same shell/SSH command.",
                    "Direct file_write or ssh_file_write does not satisfy a user requirement to create the file using a command.",
                ],
            },
            "last_verifier_verdict": _normalized_verifier_verdict(state),
        },
    )


def _sysadmin_report_task_text(state: LoopState) -> str:
    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)
    return "\n".join(
        str(part or "")
        for part in (
            getattr(run_brief, "original_task", ""),
            getattr(run_brief, "current_phase_objective", ""),
            getattr(working_memory, "current_goal", ""),
        )
    )


def _is_remote_sysadmin_report_task(state: LoopState) -> bool:
    text = _sysadmin_report_task_text(state).lower()
    return (
        str(getattr(state, "task_mode", "") or "").strip().lower() == "remote_execute"
        and "sysadmin challenge" in text
        and ("root-cause" in text or " rca" in f" {text}")
        and "report" in text
        and "/root/" in text
    )


def _artifact_text(artifact: Any) -> str:
    parts = [
        getattr(artifact, "inline_content", None),
        getattr(artifact, "preview_text", None),
        getattr(artifact, "summary", None),
    ]
    content_path = str(getattr(artifact, "content_path", "") or "").strip()
    if content_path:
        try:
            path = Path(content_path)
            if path.is_file() and path.stat().st_size <= 250_000:
                parts.insert(0, path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(str(part) for part in parts if str(part or "").strip())


def _latest_sysadmin_report_artifact(state: LoopState) -> Any | None:
    candidates: list[Any] = []
    for artifact in getattr(state, "artifacts", {}).values():
        tool_name = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if tool_name not in {"ssh_file_write", "file_write"}:
            continue
        metadata = getattr(artifact, "metadata", None)
        arguments = metadata.get("arguments") if isinstance(metadata, dict) else {}
        if not isinstance(arguments, dict):
            arguments = {}
        target = " ".join(
            str(value or "")
            for value in (
                getattr(artifact, "source", ""),
                arguments.get("path"),
                getattr(artifact, "summary", ""),
            )
        ).lower()
        if re.search(r"/root/[^\s]+report[^\s]*\.txt\b", target):
            candidates.append(artifact)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: str(getattr(item, "created_at", "") or ""))[-1]


def _tool_history_has_process_mapped_network_probe(state: LoopState) -> bool:
    weak_probe = re.compile(r"\bss\s+-tu?ln\b|\bss\s+-tuln\b", re.IGNORECASE)
    saw_network_probe = False
    saw_weak_probe = False
    for record in getattr(state, "tool_execution_records", {}).values():
        if not isinstance(record, dict) or str(record.get("tool_name") or "") != "ssh_exec":
            continue
        args = record.get("args")
        if not isinstance(args, dict):
            args = {}
        command = str(args.get("command") or "")
        ss_options = re.findall(r"\bss\s+-([A-Za-z]+)", command, re.IGNORECASE)
        if ss_options:
            saw_network_probe = True
        if any("p" in options.lower() and "n" in options.lower() for options in ss_options):
            return True
        if weak_probe.search(command):
            saw_weak_probe = True
    return saw_network_probe and not saw_weak_probe


def _sysadmin_report_consistency_issues(state: LoopState, message: str) -> list[str]:
    artifact = _latest_sysadmin_report_artifact(state)
    if artifact is None:
        return []
    report = _artifact_text(artifact)
    lowered_report = re.sub(r"\s+", " ", report.lower()).strip()
    lowered_message = re.sub(r"\s+", " ", str(message or "").lower()).strip()
    combined = f"{lowered_report} {lowered_message}"
    issues: list[str] = []

    created_date = str(getattr(artifact, "created_at", "") or "")[:10]
    date_match = re.search(r"\b(?:report generated|date)\s*:\s*(\d{4}-\d{2}-\d{2})", report, re.IGNORECASE)
    if created_date and date_match and date_match.group(1) != created_date:
        issues.append(f"report date {date_match.group(1)} does not match artifact date {created_date}")

    if (
        "no suspicious network exposure" in lowered_report
        and any(marker in combined for marker in ("mysql exposed", "tftp", "ftp listening", "suspicious services found"))
    ):
        issues.append("network exposure summary contradicts detailed findings")

    if re.search(r"listening udp services:.*\bport\s+\d+/tcp\b", lowered_report, re.DOTALL):
        issues.append("UDP listener section labels ports as tcp")

    task_text = _sysadmin_report_task_text(state).lower()
    if "owning process" in task_text and not _tool_history_has_process_mapped_network_probe(state):
        issues.append("network listener evidence lacks owning process mapping; use ss -tulpn or equivalent")

    if "top 10 largest files under /var/log" in lowered_report and "unable to obtain list" in lowered_report:
        issues.append("required /var/log largest-file list is missing")

    return issues


def task_complete_gate_sysadmin_report_consistency(state: LoopState, message: str) -> dict | None:
    if not _is_remote_sysadmin_report_task(state):
        return None
    issues = _sysadmin_report_consistency_issues(state, message)
    if not issues:
        return None
    return fail(
        "Cannot complete the sysadmin report task until the generated report is internally consistent and evidence-backed.",
        metadata={
            "reason": "sysadmin_report_consistency_required",
            "sysadmin_report_issues": issues,
            "next_required_action": {
                "tool_names": ["ssh_exec", "ssh_file_write"],
                "notes": [
                    "Regather any weak evidence with process-mapped commands such as `ss -tulpn`.",
                    "Rewrite the report with matching dates, protocol labels, and non-contradictory exposure findings.",
                    "Preserve required sections: baseline, disk, process, network, SSH/security, logs, and recommendations.",
                ],
            },
            "last_verifier_verdict": _normalized_verifier_verdict(state),
        },
    )


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
    has_manual_success = (
        state.scratchpad.get("manual_success")
        or state.scratchpad.get("manual_success_override")
        or state.scratchpad.get("manual_override")
        or any("manual success" in str(f).lower() or "verified manually" in str(f).lower() for f in getattr(state.working_memory, "known_facts", []))
    )
    if has_manual_success:
        return None
    verifier_verdict = _normalized_verifier_verdict(state)
    verifier_failure_satisfies_diagnostic = (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and diagnostic_failure_completion_allowed(state, message=message, verifier=verifier_verdict)
    )
    # A removal_absence_probe verdict on a non-removal task is a misclassification
    # (e.g. inventory task with "do not remove packages" in its constraints).  It
    # must not gate task completion for tasks that have no actual removal intent.
    # Lazy import: top-level import creates a circular dependency via harness/__init__
    # → core_facade → tools (build_registry) before tools is fully initialized.
    from ..harness.tool_result_verification_removal import _task_has_removal_intent  # noqa: PLC0415
    verifier_is_spurious_absence_probe = (
        verifier_verdict
        and verifier_verdict.get("verifier_kind") == "removal_absence_probe"
        and not _task_has_removal_intent(state)
    )
    if (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and not state.acceptance_waived
        and not verifier_failure_satisfies_diagnostic
        and not verifier_is_spurious_absence_probe
    ):
        vkind = str(verifier_verdict.get("verifier_kind") or "unknown")
        vreason = str(
            verifier_verdict.get("absence_probe_reason")
            or verifier_verdict.get("failure_mode")
            or ""
        )
        error = (
            f"Cannot complete the task while the latest verifier verdict is still failing "
            f"[verifier_kind={vkind}{f', reason={vreason}' if vreason else ''}]."
        )
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
