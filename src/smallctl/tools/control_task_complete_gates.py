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


_ENVIRONMENT_FAILURE_MARKERS = (
    "connection refused",
    "connection timed out",
    "network is unreachable",
    "no route to host",
    "could not resolve hostname",
    "permission denied",
)

_RUN_REPORT_TASK_MARKERS = ("run", "execute", "test", "check", "verify")
_REPORT_DELIVERABLE_MARKERS = (
    "propose",
    "proposal",
    "recommend",
    "report",
    "summarize",
    "inspect",
    "review",
    "read",
)
_MUTATION_TASK_MARKERS = (
    "implement",
    "improve",
    "patch",
    "change",
    "update",
    "add",
    "modify",
)


def unresolved_missing_input_file(state: LoopState) -> dict | None:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return None
    blocker = scratchpad.get("_unresolved_missing_input_file")
    if isinstance(blocker, dict) and str(blocker.get("path") or "").strip():
        return blocker
    return None


def _state_task_text(state: LoopState) -> str:
    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)
    parts = (
        getattr(run_brief, "original_task", ""),
        getattr(working_memory, "current_goal", ""),
        getattr(state, "active_intent", ""),
        " ".join(str(item) for item in (getattr(state, "secondary_intents", []) or [])),
        " ".join(str(item) for item in (getattr(state, "intent_tags", []) or [])),
    )
    return " ".join(
        str(part or "").strip()
        for part in parts
        if str(part or "").strip()
    ).casefold()


def _environment_failure_completion_allowed(
    state: LoopState,
    *,
    message: str,
    verifier: dict[str, Any],
) -> bool:
    verdict = str(verifier.get("verdict") or "").strip().lower()
    if verdict not in {"fail", "failed", "error"}:
        return False
    verifier_text = " ".join(
        str(verifier.get(key) or "").strip().casefold()
        for key in (
            "failure_mode",
            "absence_probe_reason",
            "key_stdout",
            "key_stderr",
            "command",
            "target",
        )
    )
    if "environment" not in verifier_text and not any(
        marker in verifier_text for marker in _ENVIRONMENT_FAILURE_MARKERS
    ):
        return False
    message_text = str(message or "").casefold()
    if "environment" not in message_text and not any(
        marker in message_text for marker in _ENVIRONMENT_FAILURE_MARKERS
    ):
        return False
    task_text = _state_task_text(state)
    if not any(marker in task_text for marker in _RUN_REPORT_TASK_MARKERS):
        return False
    if not any(marker in task_text for marker in _REPORT_DELIVERABLE_MARKERS):
        return False
    if "propose" not in task_text and any(marker in task_text for marker in _MUTATION_TASK_MARKERS):
        return False
    return True


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


def _docker_compose_lifecycle_task_text(state: LoopState) -> str:
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


def _is_docker_compose_lifecycle_report_task(state: LoopState) -> bool:
    text = _docker_compose_lifecycle_task_text(state).lower()
    if "docker compose" not in text and "docker-compose" not in text:
        return False
    if "report" not in text:
        return False
    if not any(marker in text for marker in ("recreate", "re-create", "bring it back", "start it again")):
        return False
    if not any(marker in text for marker in ("tear down", "teardown", "compose down", "docker compose down", "stop and remove")):
        return False
    return any(marker in text for marker in ("create", "start", "bring up", "compose up", "docker compose up"))


def _latest_docker_lifecycle_report_artifact(state: LoopState) -> Any | None:
    candidates: list[Any] = []
    for artifact in getattr(state, "artifacts", {}).values():
        tool_name = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if tool_name not in {"ssh_file_write", "file_write", "ssh_exec", "shell_exec"}:
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
        if "report" in target and re.search(r"(?:^|/|\s)[^\s]*report[^\s]*\.txt\b", target):
            candidates.append(artifact)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: str(getattr(item, "created_at", "") or ""))[-1]


def _docker_lifecycle_report_text(state: LoopState, message: str) -> str:
    artifact = _latest_docker_lifecycle_report_artifact(state)
    report = _artifact_text(artifact) if artifact is not None else ""
    return re.sub(r"\s+", " ", f"{report} {message}".lower()).strip()


def _tool_record_command(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    args = record.get("args")
    if not isinstance(args, dict):
        args = {}
    return str(args.get("command") or "").strip()


def _tool_record_output_text(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    result = record.get("result")
    if not isinstance(result, dict):
        return ""
    parts: list[str] = []
    for key in ("error", "output"):
        value = result.get(key)
        if isinstance(value, dict):
            parts.extend(str(value.get(out_key) or "") for out_key in ("stdout", "stderr", "message"))
        else:
            parts.append(str(value or ""))
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        parts.extend(str(metadata.get(key) or "") for key in ("error", "stdout", "stderr", "message"))
    return "\n".join(part for part in parts if part).lower()


def _tool_record_success(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    result = record.get("result")
    if not isinstance(result, dict):
        return False
    if result.get("success") is False:
        return False
    output = result.get("output")
    if isinstance(output, dict):
        exit_code = output.get("exit_code")
        if exit_code not in {None, 0, "0"}:
            return False
    return True


def _docker_lifecycle_evidence(state: LoopState) -> dict[str, bool]:
    saw_down = False
    saw_down_verifier = False
    saw_recreate_up = False
    saw_recreate_verifier = False
    saw_any_up = False
    for record in getattr(state, "tool_execution_records", {}).values():
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "") not in {"ssh_exec", "shell_exec"}:
            continue
        if not _tool_record_success(record):
            continue
        command = _tool_record_command(record).lower()
        if not command:
            continue
        is_compose = "docker compose" in command or "docker-compose" in command
        is_up = is_compose and re.search(r"\bup\b", command) is not None
        is_down = is_compose and re.search(r"\bdown\b", command) is not None
        is_verify = (
            (is_compose and re.search(r"\b(?:ps|logs)\b", command) is not None)
            or re.search(r"(?:^|[;&|]\s*)curl\b", command) is not None
            or re.search(r"(?:^|[;&|]\s*)wget\b", command) is not None
            or ("docker ps" in command)
        )
        if is_up:
            saw_any_up = True
            if saw_down:
                saw_recreate_up = True
        if is_down:
            saw_down = True
            saw_down_verifier = False
            saw_recreate_up = False
            saw_recreate_verifier = False
            continue
        if saw_down and not saw_recreate_up and is_verify:
            saw_down_verifier = True
        if saw_recreate_up and is_verify:
            saw_recreate_verifier = True
    return {
        "saw_initial_up": saw_any_up,
        "saw_down": saw_down,
        "saw_down_verifier": saw_down_verifier,
        "saw_recreate_up": saw_recreate_up,
        "saw_recreate_verifier": saw_recreate_verifier,
    }


def _remote_service_task_text(state: LoopState) -> str:
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


def _is_remote_service_install_task(state: LoopState) -> bool:
    text = _remote_service_task_text(state).lower()
    if not any(marker in text for marker in ("ssh", "remote", "host", "192.168.", "10.")):
        return False
    if not any(marker in text for marker in ("install", "deploy", "spin up", "run", "launch", "start")):
        return False
    return any(marker in text for marker in ("service", "container", "docker", "app", "application", "netbox"))


def _remote_service_readiness_evidence(state: LoopState) -> dict[str, Any]:
    saw_detached_start = False
    saw_readiness_probe = False
    saw_weak_container_probe = False
    saw_unhealthy_logs = False
    latest_start_command = ""
    negative_log_markers = (
        "waiting on db",
        "waited 30s or more for the db",
        "database is not ready",
        "connection refused",
        "traceback",
        "fatal:",
        "error:",
        "exception",
    )
    records = [record for record in getattr(state, "tool_execution_records", {}).values() if isinstance(record, dict)]
    records.sort(key=lambda record: (int(record.get("step_count") or 0), str(record.get("operation_id") or "")))
    for record in records:
        if str(record.get("tool_name") or "") not in {"ssh_exec", "shell_exec"}:
            continue
        if not _tool_record_success(record):
            continue
        command = _tool_record_command(record).lower()
        if not command:
            continue
        starts_service = bool(
            re.search(r"\bdocker\s+run\b.*\s-d\b", command)
            or re.search(r"\bdocker\s+compose\b.*\bup\b.*\s-d\b", command)
            or re.search(r"\bdocker-compose\b.*\bup\b.*\s-d\b", command)
            or re.search(r"\bdocker\s+start\b", command)
            or re.search(r"\b(?:systemctl|service)\s+(?:start|restart)\b", command)
        )
        if starts_service:
            saw_detached_start = True
            saw_readiness_probe = False
            saw_weak_container_probe = False
            saw_unhealthy_logs = False
            latest_start_command = command
            continue
        if not saw_detached_start:
            continue
        output_text = _tool_record_output_text(record)
        reads_logs = bool(re.search(r"\bdocker(?:\s+compose)?\s+logs\b|\bdocker-compose\s+logs\b|\bjournalctl\b", command))
        if reads_logs and any(marker in output_text for marker in negative_log_markers):
            saw_unhealthy_logs = True
        weak_probe = bool(re.search(r"\bdocker\s+ps\b|\bdocker(?:\s+compose)?\s+ps\b|\bdocker-compose\s+ps\b", command))
        if weak_probe:
            saw_weak_container_probe = True
        strong_probe = bool(
            re.search(r"(?:^|[;&|]\s*)(?:curl|wget|nc|nmap)\b", command)
            or re.search(r"\bdocker\s+inspect\b.*\b(?:health|status)\b", command)
            or (reads_logs and not saw_unhealthy_logs and bool(output_text.strip()))
            or re.search(r"\b(?:systemctl|service)\s+(?:status|is-active)\b", command)
        )
        if strong_probe and not saw_unhealthy_logs:
            saw_readiness_probe = True
    return {
        "saw_detached_start": saw_detached_start,
        "saw_readiness_probe": saw_readiness_probe,
        "saw_weak_container_probe": saw_weak_container_probe,
        "saw_unhealthy_logs": saw_unhealthy_logs,
        "latest_start_command": latest_start_command,
    }


def task_complete_gate_remote_service_readiness(state: LoopState) -> dict | None:
    if not _is_remote_service_install_task(state):
        return None
    evidence = _remote_service_readiness_evidence(state)
    if not evidence["saw_detached_start"]:
        return None
    if evidence["saw_readiness_probe"] and not evidence["saw_unhealthy_logs"]:
        return None
    issues: list[str] = []
    if evidence["saw_unhealthy_logs"]:
        issues.append("post-start logs show the service is not ready or is failing")
    if not evidence["saw_readiness_probe"]:
        issues.append("detached service start lacks a post-start readiness probe")
    if evidence["saw_weak_container_probe"] and not evidence["saw_readiness_probe"]:
        issues.append("docker ps/container listing alone is not service readiness proof")
    return fail(
        "Cannot complete the remote service install until the service has post-start readiness evidence.",
        metadata={
            "reason": "remote_service_readiness_required",
            "remote_service_readiness_issues": issues,
            "remote_service_readiness_evidence": evidence,
            "next_required_action": {
                "tool_names": ["ssh_exec", "task_fail"],
                "notes": [
                    "After detached service start, run an HTTP/readiness probe or inspect stable service logs.",
                    "Do not treat `docker ps` alone as proof that the application is usable.",
                    "If logs show a missing dependency such as DB/Redis, repair the deployment or call task_fail with the blocker.",
                ],
            },
            "last_verifier_verdict": _normalized_verifier_verdict(state),
        },
    )


def _docker_lifecycle_report_issues(state: LoopState, message: str) -> list[str]:
    evidence = _docker_lifecycle_evidence(state)
    report_text = _docker_lifecycle_report_text(state, message)
    issues: list[str] = []
    if not evidence["saw_down"]:
        issues.append("docker compose teardown was not directly evidenced by a successful down command")
    claims_absence = any(
        marker in report_text
        for marker in (
            "offline",
            "removed",
            "stopped and removed",
            "torn down",
            "taken down",
            "no longer running",
        )
    )
    if claims_absence and not evidence["saw_down_verifier"]:
        issues.append("teardown/offline report claim lacks a direct post-down Docker or HTTP verification probe")
    if not evidence["saw_recreate_up"]:
        issues.append("docker compose recreation was not directly evidenced by a successful up command after down")
    if not evidence["saw_recreate_verifier"]:
        issues.append("recreated service was not directly verified after the post-down up command")
    return issues


def task_complete_gate_docker_compose_lifecycle_report(state: LoopState, message: str) -> dict | None:
    if not _is_docker_compose_lifecycle_report_task(state):
        return None
    issues = _docker_lifecycle_report_issues(state, message)
    if not issues:
        return None
    return fail(
        "Cannot complete the Docker Compose lifecycle report until teardown and recreate claims are directly evidence-backed.",
        metadata={
            "reason": "docker_compose_lifecycle_report_grounding_required",
            "docker_compose_lifecycle_issues": issues,
            "docker_compose_lifecycle_evidence": _docker_lifecycle_evidence(state),
            "next_required_action": {
                "tool_names": ["ssh_exec"],
                "notes": [
                    "Run or show the successful `docker compose down` command for the stack.",
                    "Probe after teardown before claiming the service is offline or removed.",
                    "Run `docker compose up -d` after teardown and verify the recreated service with `docker compose ps`, `docker ps`, or curl/wget.",
                    "Update the report only with lifecycle claims supported by those command outputs.",
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
    verifier_failure_is_reported_environment_blocker = (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and _environment_failure_completion_allowed(state, message=message, verifier=verifier_verdict)
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
        and not verifier_failure_is_reported_environment_blocker
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
    verifier_failure_is_reported_environment_blocker = (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and _environment_failure_completion_allowed(state, message=message, verifier=verifier_verdict)
    )
    if (
        not state.acceptance_ready()
        and not verifier_failure_satisfies_diagnostic
        and not verifier_failure_is_reported_environment_blocker
    ):
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


def task_complete_gate_shell_table_coverage(state: LoopState, message: str) -> dict | None:
    """Block incomplete summaries of numbered shell-output tables."""
    table = _latest_numbered_shell_table(state)
    if table is None:
        return None
    row_ids = table["row_ids"]
    if len(row_ids) < 12:
        return None
    message_ids = set(re.findall(r"(?<!\d)(\d{1,6})(?!\d)", str(message or "")))
    mentioned = [row_id for row_id in row_ids if row_id in message_ids]
    if len(mentioned) >= max(8, len(row_ids) // 2):
        return None
    if any(marker in str(message or "").lower() for marker in ("first ", "partial", "sample", "truncated", "not all")):
        return None
    return fail(
        "Cannot complete the task: the latest command output contained "
        f"{len(row_ids)} numbered rows, but the completion message only mentions "
        f"{len(mentioned)} of them. Summarize all rows, state that you are intentionally "
        "showing a partial sample, or read the artifact before completing.",
        metadata={
            "reason": "shell_table_answer_incomplete",
            "artifact_id": table["artifact_id"],
            "row_count": len(row_ids),
            "mentioned_row_count": len(mentioned),
            "row_ids_preview": row_ids[:80],
        },
    )


def _latest_numbered_shell_table(state: LoopState) -> dict[str, Any] | None:
    artifacts = getattr(state, "artifacts", {}) or {}
    for artifact_id, artifact in reversed(list(artifacts.items())):
        tool_name = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if tool_name not in {"shell_exec", "ssh_exec"}:
            continue
        text = _artifact_text(artifact)
        rows = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if parts and parts[0].isdigit():
                rows.append((parts[0], stripped))
        if len(rows) >= 12:
            return {
                "artifact_id": str(artifact_id),
                "row_ids": [row_id for row_id, _ in rows],
                "rows": [row for _, row in rows],
            }
    return None


def _artifact_text(artifact: Any) -> str:
    text = str(getattr(artifact, "inline_content", "") or getattr(artifact, "preview_text", "") or "")
    if text:
        return text
    content_path = str(getattr(artifact, "content_path", "") or "").strip()
    if not content_path:
        return ""
    try:
        return Path(content_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
