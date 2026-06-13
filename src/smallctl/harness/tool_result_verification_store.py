from __future__ import annotations

import re
from typing import Any

from ..challenge_progress import record_verifier_result
from ..models.tool_result import ToolEnvelope
from ..shell_utils import strip_benign_shell_redirections as _strip_benign_shell_redirections
from .tool_result_verification_audit import _is_audit_task
from .tool_result_verification_blocker import (
    _extract_latest_execution_blocker,
    _store_latest_execution_blocker,
)
from .tool_result_verification_constants import _INTERACTIVE_PROMPT_RE
from .tool_result_verification_helpers import (
    classify_execution_failure,
    command_is_binary_probe,
    command_is_install_absence_probe,
    exit_code_matches,
    install_absence_probe_confirmed,
    looks_like_infinite_loop,
    output_confirms_not_found,
    snip_text,
    verifier_kind_for_command,
)
from .tool_result_verification_removal import _classify_removal_absence_probe, _task_has_removal_intent
from .tool_result_verification_repair import (
    _record_docker_retry_state,
    _update_acceptance_ledger,
    _update_repair_cycle_state,
)
from .tool_result_verification_semantic import (
    _insufficient_verifier_message,
    _install_task_requires_strong_verifier,
    _passing_verifier_is_weaker_than_prior_failure,
    _semantic_verifier_failure,
)
from .tool_result_verification_ssh_recovery import _update_ssh_auth_recovery_state
from .tool_result_verification_timeout import _is_long_running_remote_command_timeout


def _store_verifier_verdict(
    state: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if tool_name not in {"shell_exec", "ssh_exec"}:
        return None
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("reason") == "tool_not_exposed_this_turn":
        return None
    output = result.output if isinstance(result.output, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    args = arguments if isinstance(arguments, dict) else {}
    command = str(
        args.get("command")
        or metadata.get("command")
        or metadata.get("target")
        or ""
    ).strip()
    target = command
    if tool_name == "ssh_exec":
        host = str(args.get("host") or metadata.get("host") or "").strip()
        remote_command = str(args.get("command") or metadata.get("command") or "").strip()
        if host and remote_command:
            target = f"{host} :: {remote_command}"
        elif host:
            target = host
        elif remote_command:
            target = f"(missing host) :: {remote_command}"

    exit_code = output.get("exit_code") if isinstance(output, dict) else None
    raw_stdout = str(output.get("stdout") if isinstance(output, dict) else "")
    raw_stderr = str(output.get("stderr") if isinstance(output, dict) else "")
    stdout = snip_text(raw_stdout, limit=400)
    stderr = snip_text(raw_stderr, limit=400)
    semantic_failure = _semantic_verifier_failure(command=command, stdout=raw_stdout, stderr=raw_stderr)
    absence_probe = _classify_removal_absence_probe(
        state,
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        tool_name=tool_name,
    )
    status = str(result.status or metadata.get("status") or "").strip()
    approval_denied = bool(metadata.get("approval_denied"))
    current_verifier_kind = verifier_kind_for_command(command)
    if status == "needs_human" or approval_denied:
        verdict = "needs_human"
    elif semantic_failure:
        verdict = "fail"
    elif absence_probe["is_absence_probe"] and absence_probe["found_resources"]:
        if _is_audit_task(state):
            # For audit tasks, finding resources is expected (we're documenting state)
            verdict = "pass"
        else:
            verdict = "fail"
            semantic_failure = str(absence_probe["reason"] or "absence probe found matching resources")
    elif absence_probe["is_absence_probe"] and absence_probe["absence_confirmed"]:
        verdict = "pass"
    elif result.success and (exit_code in (0, None)):
        verdict = "pass"
    elif (
        exit_code == 127
        and command_is_binary_probe(command)
        and output_confirms_not_found(stdout, stderr)
        and _task_has_removal_intent(state)
    ):
        # Exit 127 = "command not found" which is the expected outcome of a
        # successful removal task. Treat this as a pass rather than a verifier
        # failure so the session can exit the repair phase normally.
        verdict = "pass"
    elif (
        exit_code == 1
        and command_is_binary_probe(command)
        and output_confirms_not_found(stdout, stderr)
    ):
        # Exit 1 from a diagnostic probe (systemctl status, dpkg -l, apt list,
        # which, whereis, etc.) that confirms the resource is absent is
        # informational negative intelligence, not a failure.
        verdict = "pass"
    elif (
        exit_code == 1
        and command_is_install_absence_probe(command)
        and install_absence_probe_confirmed(command=command, exit_code=exit_code, stdout=raw_stdout, stderr=raw_stderr)
    ):
        # Exit 1 from a piped absence probe such as `dpkg -l | grep -i fog`
        # with no matches is valid negative intelligence for install tasks.
        verdict = "pass"
        semantic_failure = "install absence probe returned no matches"
    else:
        verdict = "fail"

    if verdict == "pass" and tool_name == "ssh_exec":
        combined = "\n".join(part for part in (raw_stdout, raw_stderr) if str(part or "").strip())
        if _INTERACTIVE_PROMPT_RE.search(combined) or re.search(r"\([yYnN]/[yYnN]\)|continue\?.*\([yYnN]/[yYnN]\)|password:\s*$", combined, re.IGNORECASE | re.DOTALL):
            verdict = "fail"
            semantic_failure = "interactive prompt detected in ssh_exec output"

    # Fix 2: detect false-negative path verifiers for SSH exec
    false_negative_path = False
    if verdict == "fail" and tool_name == "ssh_exec":
        if _verifier_path_failure_is_false_negative(
            state, host=host, command=command, stdout=raw_stdout, stderr=raw_stderr
        ):
            verdict = "pass"
            semantic_failure = ""
            false_negative_path = True

    insufficient_verifier = False
    if verdict == "pass":
        insufficient_verifier = _passing_verifier_is_weaker_than_prior_failure(
            state,
            current_command=command,
            current_kind=current_verifier_kind,
        )
        if insufficient_verifier:
            verdict = "fail"
            semantic_failure = _insufficient_verifier_message(state, command=command)

    # Reject weak verifiers for install tasks
    install_verifier_weak = False
    if verdict == "pass" and not insufficient_verifier:
        install_verifier_weak, install_weak_reason = _install_task_requires_strong_verifier(
            state, command=command
        )
        if install_verifier_weak:
            verdict = "fail"
            semantic_failure = install_weak_reason

    failure_class = "approval_denied" if approval_denied else classify_execution_failure(result.error or stderr or stdout)
    if not approval_denied and tool_name == "ssh_exec" and exit_code_matches(exit_code, {255}):
        failure_class = "environment"
    if insufficient_verifier:
        failure_class = "insufficient_verifier"
    if install_verifier_weak:
        failure_class = "insufficient_verifier"
    if false_negative_path:
        failure_class = "false_negative_path_verifier"
    if not approval_denied and failure_class == "environment":
        if looks_like_infinite_loop(command, str(result.error or ""), stdout, stderr):
            failure_class = "infinite_loop_suspected"
    if absence_probe["is_absence_probe"]:
        failure_class = "" if verdict == "pass" else "removal_residue"
    if _is_long_running_remote_command_timeout(
        tool_name=tool_name,
        command=command,
        result=result,
        stdout=stdout,
        stderr=stderr,
    ):
        failure_class = "long_running_remote_command"
    if semantic_failure == "interactive prompt detected in ssh_exec output":
        failure_class = "interactive_prompt"
    docker_retry = _record_docker_retry_state(
        state,
        command=command,
        failure_class=failure_class,
        verdict=verdict,
    )
    if verdict == "pass":
        acceptance_delta = {
            "status": "satisfied",
            "notes": ["execution succeeded"],
        }
    elif verdict == "needs_human":
        acceptance_delta = {
            "status": "pending",
            "notes": [str(result.error or status or "human approval required")],
        }
    else:
        acceptance_delta = {
            "status": "blocked",
            "notes": [str(semantic_failure or result.error or stderr or stdout or status or "execution failed")],
        }
    normalized = {
        "tool": tool_name,
        "target": target,
        "command": command,
        "exit_code": exit_code,
        "key_stdout": stdout,
        "key_stderr": stderr,
        "verdict": verdict,
        "failure_mode": failure_class,
        "acceptance_delta": acceptance_delta,
    }
    if insufficient_verifier or install_verifier_weak:
        normalized["insufficient_verifier"] = True
        normalized["verifier_kind"] = current_verifier_kind
    if approval_denied:
        normalized["approval_denied"] = True
    if absence_probe["is_absence_probe"]:
        normalized["verifier_kind"] = "removal_absence_probe"
        normalized["absence_probe_reason"] = absence_probe["reason"]
    if docker_retry:
        normalized.update(docker_retry)
    blocker = _extract_latest_execution_blocker(
        tool_name=tool_name,
        command=command,
        target=target,
        exit_code=exit_code,
        failure_class=failure_class,
        stdout=raw_stdout,
        stderr=raw_stderr,
        error=str(result.error or ""),
        verdict=verdict,
    )
    if blocker:
        normalized["latest_blocker"] = blocker
    state.last_verifier_verdict = normalized
    state.scratchpad["_last_verifier_verdict"] = normalized
    record_verifier_result(
        state,
        tool_name=tool_name,
        command=command,
        verifier_kind=str(normalized.get("verifier_kind") or current_verifier_kind),
        verdict=verdict,
        exit_code=exit_code,
    )
    if blocker:
        _store_latest_execution_blocker(state, blocker)
    state.scratchpad.pop("_last_verifier_stale_after_mutation", None)
    state.last_failure_class = failure_class
    state.scratchpad["_last_failure_class"] = failure_class
    if failure_class == "long_running_remote_command":
        state.scratchpad["_last_long_running_remote_command_timeout"] = {
            "tool": tool_name,
            "target": target,
            "command": command,
            "host": str((arguments or {}).get("host") or "").strip(),
            "stdout": stdout,
            "stderr": stderr,
            "error": str(result.error or ""),
        }
    _update_acceptance_ledger(state, verdict=verdict)
    _update_ssh_auth_recovery_state(
        state,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
    )
    _update_repair_cycle_state(
        state,
        tool_name=tool_name,
        result=result,
        command=command,
        target=target,
        verdict=verdict,
        failure_class=failure_class,
        docker_retry=docker_retry,
    )
    return normalized


def _verifier_path_failure_is_false_negative(
    state: Any,
    *,
    host: str,
    command: str,
    stdout: str,
    stderr: str,
) -> bool:
    """Return True if a path verifier failure contradicts a recent ssh_file_write verification."""
    combined_output = str(stdout or "") + "\n" + str(stderr or "")
    if "no such file or directory" not in combined_output.lower():
        return False
    if not host:
        return False
    # Extract path from command
    words = str(command or "").strip().split()
    path = ""
    for word in words:
        if word.startswith("/"):
            path = word
            break
    if not path:
        return False
    recent_messages = getattr(state, "recent_messages", None) or []
    if not isinstance(recent_messages, list):
        return False
    for msg in reversed(recent_messages):
        if msg is None:
            continue
        # Support both dict and object-style messages (e.g. SimpleNamespace)
        if isinstance(msg, dict):
            role = msg.get("role")
            name = msg.get("name")
            metadata = msg.get("metadata") or {}
        else:
            role = getattr(msg, "role", None)
            name = getattr(msg, "name", None)
            metadata = getattr(msg, "metadata", None) or {}
        if role != "tool" or name != "ssh_file_write":
            continue
        if not isinstance(metadata, dict):
            continue
        if not metadata.get("success"):
            continue
        msg_host = str(metadata.get("host") or "").strip().lower()
        msg_path = str(metadata.get("path") or "").strip()
        if msg_host == host.strip().lower() and msg_path == path:
            return True
    return False
