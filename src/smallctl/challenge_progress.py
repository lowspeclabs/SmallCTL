from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from .models.tool_result import ToolEnvelope

_CODING_PATH_RE = re.compile(r"(?:^|\s|`)(?P<path>\./temp/[A-Za-z0-9_.\-/]+\.(?:py|html?|js|css))(?:`|\s|$)")
_CODING_MARKERS = (
    "build a self-contained python script",
    "includes built-in unittest",
    "unittest cases",
    "./temp/",
)
_SYSADMIN_MARKERS = (
    "ssh to the host",
    "remote credentials",
    "sysadmin task",
    "ssh_exec",
    "ssh_file_read",
    "ssh_file_write",
)
_FILE_MUTATION_TOOLS = {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"}
_VERIFIER_SCAFFOLDING_RE = re.compile(
    r"(?:^|/)temp/(?:verify_[A-Za-z0-9_.-]*|run_verification)\.py$|(?:^|/)run_verification\.py$"
)


def initialize_challenge_progress_from_task(state: Any, task: str) -> None:
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return
    task_text = str(task or "")
    lower_task = task_text.lower()
    if not progress.task_category:
        if any(marker in lower_task for marker in _SYSADMIN_MARKERS):
            progress.task_category = "sysadmin"
        elif any(marker in lower_task for marker in _CODING_MARKERS):
            progress.task_category = "coding"
    if not progress.required_output_paths:
        paths = []
        for match in _CODING_PATH_RE.finditer(task_text):
            path = match.group("path").strip()
            if path and path not in paths:
                paths.append(path)
        progress.required_output_paths = paths
    if not progress.phase:
        if progress.task_category == "coding":
            progress.phase = "implement"
        elif progress.task_category == "sysadmin":
            progress.phase = "explore"
        progress.phase_started_at_step = int(getattr(state, "step_count", 0) or 0)


def record_code_change(
    state: Any,
    *,
    tool_name: str,
    path: str = "",
    changed: bool = True,
) -> None:
    if tool_name not in _FILE_MUTATION_TOOLS or not changed:
        return
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return
    _initialize_from_state_task(state)
    if not progress.task_category and _path_is_coding_output(path):
        progress.task_category = "coding"
    if progress.task_category != "coding":
        return
    if _path_is_verifier_scaffolding(path) and not _path_is_required_output(progress, path):
        _record_verifier_artifact_path(progress, path)
        return
    step = int(getattr(state, "step_count", 0) or 0)
    progress.last_code_change_step = step
    progress.code_change_count += 1
    if path and path not in progress.last_code_change_paths:
        progress.last_code_change_paths.append(path)
        progress.last_code_change_paths = progress.last_code_change_paths[-8:]
    progress.verified_after_last_change = False
    progress.redundant_verifier_count = 0
    progress.post_pass_nonterminal_steps = 0
    progress.no_change_steps_after_write = 0
    progress.nonterminal_steps_after_verified_write = 0
    if path and any(p == path for p in getattr(progress, "required_output_paths", []) or []):
        progress.successful_artifact_write_step = step
        progress.first_post_write_verification_step = None
    _set_phase(progress, "debug" if progress.last_verifier_verdict == "fail" else "final_verify", step=step)


def record_verifier_result(
    state: Any,
    *,
    tool_name: str,
    command: str,
    verifier_kind: str,
    verdict: str,
    exit_code: Any,
) -> None:
    if tool_name != "shell_exec":
        return
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return
    _initialize_from_state_task(state)
    if not progress.task_category and _command_mentions_required_output(progress, command):
        progress.task_category = "coding"
    if progress.task_category != "coding":
        return
    step = int(getattr(state, "step_count", 0) or 0)
    progress.last_verifier_step = step
    progress.last_verifier_command = str(command or "").strip()
    progress.last_verifier_kind = str(verifier_kind or "").strip()
    progress.last_verifier_verdict = str(verdict or "").strip().lower()
    try:
        progress.last_verifier_exit_code = None if exit_code is None else int(exit_code)
    except (TypeError, ValueError):
        progress.last_verifier_exit_code = None
    if progress.last_verifier_verdict == "pass":
        progress.verified_after_last_change = step >= progress.last_code_change_step
        progress.post_pass_nonterminal_steps = 0
        progress.no_change_steps_after_write = 0
        if getattr(progress, "successful_artifact_write_step", None) is not None and step >= progress.successful_artifact_write_step:
            if getattr(progress, "first_post_write_verification_step", None) is None:
                progress.first_post_write_verification_step = step
        _set_phase(progress, "finalize", step=step)
    else:
        progress.verified_after_last_change = False
        _set_phase(progress, "debug", step=step)


def _write_session_deadline_block(state: Any, *, tool_name: str) -> ToolEnvelope | None:
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return None
    if tool_name in {"task_complete", "task_fail"} or tool_name in _FILE_MUTATION_TOOLS:
        return None
    if not getattr(progress, "verified_after_last_change", False):
        return None
    write_step = getattr(progress, "successful_artifact_write_step", None)
    verify_step = getattr(progress, "first_post_write_verification_step", None)
    if write_step is None or verify_step is None:
        return None
    progress.nonterminal_steps_after_verified_write = int(getattr(progress, "nonterminal_steps_after_verified_write", 0)) + 1
    if progress.nonterminal_steps_after_verified_write <= 3:
        return None
    message = (
        "Write-session deadline reached. The artifact was written and verified. "
        "Either call task_complete or make one focused repair if the verifier changed."
    )
    return ToolEnvelope(
        success=False,
        status="blocked",
        error=message,
        metadata={
            "tool_name": tool_name,
            "reason": "write_deadline_terminal_guard",
            "active_mitigation": "write_session_deadline",
            "nonterminal_steps_after_verified_write": progress.nonterminal_steps_after_verified_write,
        },
    )


def redundant_verifier_block(state: Any, *, tool_name: str, arguments: dict[str, Any] | None) -> ToolEnvelope | None:
    _initialize_from_state_task(state)
    early_stop = _no_change_after_write_block(state, tool_name=tool_name)
    if early_stop is not None:
        return early_stop
    deadline_stop = _write_session_deadline_block(state, tool_name=tool_name)
    if deadline_stop is not None:
        return deadline_stop
    if tool_name != "shell_exec":
        _count_post_pass_nonterminal(state, tool_name=tool_name)
        return None
    progress = getattr(state, "challenge_progress", None)
    if progress is None or progress.task_category != "coding" or not progress.verified_after_last_change:
        return None
    command = str((arguments or {}).get("command") or "").strip()
    if not command:
        return None
    if _command_fingerprint(command) != _command_fingerprint(progress.last_verifier_command):
        return None
    progress.redundant_verifier_count += 1
    progress.post_pass_nonterminal_steps += 1
    message = (
        "Verification already passed after the latest code change. Do not run tests again "
        "unless you changed code. Call task_complete or task_fail."
    )
    return ToolEnvelope(
        success=False,
        status="blocked",
        error=message,
        metadata={
            "tool_name": tool_name,
            "reason": "redundant_verifier_after_pass",
            "active_mitigation": "coding_verify_once_per_change",
            "last_verifier_command": progress.last_verifier_command,
            "redundant_verifier_count": progress.redundant_verifier_count,
        },
    )


def challenge_progress_report(state: Any) -> dict[str, Any]:
    _initialize_from_state_task(state)
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return {}
    if progress.task_category != "coding" and progress.code_change_count <= 0 and not progress.last_verifier_command:
        return {}
    return {
        "task_category": progress.task_category,
        "challenge_id": progress.challenge_id,
        "required_output_paths": list(progress.required_output_paths),
        "phase": progress.phase,
        "code_change_count": progress.code_change_count,
        "last_code_change_step": progress.last_code_change_step,
        "last_code_change_paths": list(progress.last_code_change_paths),
        "last_verifier_artifact_paths": list(getattr(progress, "last_verifier_artifact_paths", []) or []),
        "last_verifier_step": progress.last_verifier_step,
        "last_verifier_command": progress.last_verifier_command,
        "last_verifier_kind": progress.last_verifier_kind,
        "last_verifier_verdict": progress.last_verifier_verdict,
        "verified_after_last_change": progress.verified_after_last_change,
        "redundant_verifier_count": progress.redundant_verifier_count,
        "post_pass_nonterminal_steps": progress.post_pass_nonterminal_steps,
        "no_change_steps_after_write": progress.no_change_steps_after_write,
        "successful_artifact_write_step": getattr(progress, "successful_artifact_write_step", None),
        "first_post_write_verification_step": getattr(progress, "first_post_write_verification_step", None),
        "nonterminal_steps_after_verified_write": getattr(progress, "nonterminal_steps_after_verified_write", 0),
    }


def _no_change_after_write_block(state: Any, *, tool_name: str) -> ToolEnvelope | None:
    progress = getattr(state, "challenge_progress", None)
    if progress is None or progress.task_category != "coding":
        return None
    if progress.code_change_count <= 0 or progress.verified_after_last_change:
        return None
    if tool_name in {"task_complete", "task_fail"} or tool_name in _FILE_MUTATION_TOOLS:
        return None
    progress.no_change_steps_after_write += 1
    if progress.no_change_steps_after_write <= 20:
        return None
    message = (
        "No code changes have been made for more than 20 tool steps after the initial write. "
        "Stop the verification loop: patch the code, call task_complete with current evidence, or call task_fail."
    )
    return ToolEnvelope(
        success=False,
        status="blocked",
        error=message,
        metadata={
            "tool_name": tool_name,
            "reason": "coding_no_change_after_write_early_stop",
            "active_mitigation": "coding_progress_early_stop",
            "no_change_steps_after_write": progress.no_change_steps_after_write,
            "last_code_change_paths": list(progress.last_code_change_paths),
        },
    )


def terminal_readiness_state(state: Any) -> dict[str, Any] | None:
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return None
    # Condition A: required output path exists
    required_paths = list(getattr(progress, "required_output_paths", []) or [])
    existing_paths = [p for p in required_paths if Path(p).exists()]
    if not existing_paths:
        return None
    # Condition B: successful write was followed by at least one read/verification
    verified = bool(getattr(progress, "verified_after_last_change", False))
    post_write_read = bool(getattr(state, "scratchpad", {}).get("_post_write_verification_done"))
    if not verified and not post_write_read:
        return None
    # Condition C: no failing verifier
    if getattr(progress, "last_verifier_verdict", "") == "fail":
        return None
    # Condition D: no open write session
    session = getattr(state, "write_session", None)
    if session is not None and str(getattr(session, "status", "")).strip().lower() != "complete":
        return None
    return {
        "ready": True,
        "existing_paths": existing_paths,
        "verified": verified,
        "phase": getattr(progress, "phase", ""),
    }


def _initialize_from_state_task(state: Any) -> None:
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return
    if progress.task_category and progress.required_output_paths:
        return
    run_brief = getattr(state, "run_brief", None)
    task = str(getattr(run_brief, "original_task", "") or getattr(run_brief, "task_contract", "") or "")
    if not task:
        working_memory = getattr(state, "working_memory", None)
        task = str(getattr(working_memory, "current_goal", "") or "")
    if task:
        initialize_challenge_progress_from_task(state, task)


def _count_post_pass_nonterminal(state: Any, *, tool_name: str) -> None:
    progress = getattr(state, "challenge_progress", None)
    if progress is None or progress.task_category != "coding" or not progress.verified_after_last_change:
        return
    if tool_name in {"task_complete", "task_fail"} or tool_name in _FILE_MUTATION_TOOLS:
        return
    progress.post_pass_nonterminal_steps += 1


def _path_is_coding_output(path: str) -> bool:
    normalized = str(path or "").strip()
    return normalized.endswith((".py", ".html", ".htm", ".js", ".css")) and (
        "/temp/" in normalized or normalized.startswith("./temp/")
    )


def _path_is_verifier_scaffolding(path: str) -> bool:
    normalized = str(path or "").strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return bool(_VERIFIER_SCAFFOLDING_RE.search(normalized))


def _path_is_required_output(progress: Any, path: str) -> bool:
    normalized = str(path or "").strip().lstrip("./")
    return any(
        normalized == str(required or "").strip().lstrip("./")
        for required in getattr(progress, "required_output_paths", []) or []
    )


def _record_verifier_artifact_path(progress: Any, path: str) -> None:
    normalized = str(path or "").strip()
    if not normalized:
        return
    paths = list(getattr(progress, "last_verifier_artifact_paths", []) or [])
    if normalized not in paths:
        paths.append(normalized)
    progress.last_verifier_artifact_paths = paths[-8:]


def _command_mentions_required_output(progress: Any, command: str) -> bool:
    normalized = str(command or "")
    return any(path and path.lstrip("./") in normalized for path in getattr(progress, "required_output_paths", []) or [])


def _set_phase(progress: Any, phase: str, *, step: int) -> None:
    if progress.phase != phase:
        progress.phase = phase
        progress.phase_started_at_step = step


def _command_fingerprint(command: str) -> str:
    normalized = re.sub(r"\s+", " ", str(command or "").strip())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
