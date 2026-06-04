from __future__ import annotations

import hashlib
import re
import shlex
from typing import Any

from ..docker_retry_normalization import (
    classify_docker_failure,
    docker_failure_is_registry_resolution,
    docker_retry_family,
    docker_retry_key,
    extract_docker_command_target,
)
from .directory_empty_checks import (
    match_directory_empty_check,
    parse_directory_empty_checks,
)
from ..models.tool_result import ToolEnvelope
from ..challenge_progress import record_verifier_result
from ..shell_utils import strip_benign_shell_redirections as _strip_benign_shell_redirections
from .tool_result_verification_constants import (
    _BINARY_PROBE_RE,
    _CURL_VERIFIER_FAILURE_RE,
    _FOG_RESOURCE_RE,
    _INTERACTIVE_PROMPT_RE,
    _LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE,
    _LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE,
    _LS_NO_SUCH_FILE_RE,
    _NGINX_VERIFIER_COMMAND_RE,
    _NGINX_VERIFIER_FAILURE_RE,
    _NOT_FOUND_MARKERS,
    _RAW_SSH_COMMAND_RE,
    _REMOTE_APPLICATION_BLOCKERS,
    _REMOTE_FILE_PRESENCE_PROBE_RE,
    _REMOTE_MUTATING_COMMAND_RE,
    _REMOTE_READBACK_COMMANDS,
    _REMOVAL_ABSENCE_PIPE_RE,
    _REMOVAL_ABSENCE_PROBE_RE,
    _REMOVAL_TASK_KEYWORDS,
    _SSH_AUTH_RECOVERY_KEY,
    _TEST_FAILURE_COUNT_RE,
    _TEST_FAILURE_OUTPUT_RE,
    _TEST_FAILURE_SUMMARY_RE,
    _ZERO_TESTS_RAN_RE,
)
from .tool_result_verification_helpers import (
    _VERIFIER_KIND_STRENGTH,
    classify_execution_failure,
    command_has_write_or_heredoc_shape,
    command_is_binary_probe,
    exit_code_matches,
    looks_like_infinite_loop,
    output_confirms_not_found,
    snip_text,
    verifier_kind_for_command,
    verifier_strength,
)


def _ssh_auth_recovery_entry_key(host: str, user: str) -> str:
    normalized_host = str(host or "").strip().lower()
    normalized_user = str(user or "").strip().lower()
    return f"{normalized_user}@{normalized_host}" if normalized_user else normalized_host


def _ssh_password_fingerprint(password: str) -> str:
    value = str(password or "").strip()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _update_ssh_auth_recovery_state(
    state: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
) -> None:
    if tool_name != "ssh_exec":
        return
    args = arguments if isinstance(arguments, dict) else {}
    command = str(args.get("command") or "").strip()
    if _RAW_SSH_COMMAND_RE.match(command):
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    recovery_state = scratchpad.setdefault(_SSH_AUTH_RECOVERY_KEY, {})
    if not isinstance(recovery_state, dict):
        recovery_state = {}
        scratchpad[_SSH_AUTH_RECOVERY_KEY] = recovery_state

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("reason") == "tool_not_exposed_this_turn":
        return None
    output = result.output if isinstance(result.output, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    host = str(args.get("host") or metadata.get("host") or "").strip().lower()
    user = str(args.get("user") or "").strip()
    if not host:
        return
    entry_key = _ssh_auth_recovery_entry_key(host, user)
    reached_remote_host = (
        result.success
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    )
    if reached_remote_host:
        recovery_state.pop(entry_key, None)
        return

    error_text = " ".join(
        part
        for part in [
            str(result.error or "").strip(),
            str(output.get("stderr") or "").strip(),
            str(metadata.get("error") or "").strip(),
        ]
        if part
    ).strip()
    if "permission denied" not in error_text.lower():
        return

    prior = recovery_state.get(entry_key)
    prior_failures = 0
    if isinstance(prior, dict):
        try:
            prior_failures = max(0, int(prior.get("failure_count") or 0))
        except (TypeError, ValueError):
            prior_failures = 0
    password = str(args.get("password") or "").strip()
    recovery_state[entry_key] = {
        "host": host,
        "user": user,
        "failure_count": prior_failures + 1,
        "password_provided": bool(password),
        "password_fingerprint": _ssh_password_fingerprint(password),
        "ssh_auth_mode": str(metadata.get("ssh_auth_mode") or "").strip(),
        "ssh_auth_transport": str(metadata.get("ssh_auth_transport") or "").strip(),
        "last_command": str(args.get("command") or "").strip(),
        "last_error": _snip_text(error_text, limit=240),
    }


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
    stdout = _snip_text(raw_stdout, limit=400)
    stderr = _snip_text(raw_stderr, limit=400)
    semantic_failure = _semantic_verifier_failure(command=command, stdout=raw_stdout, stderr=raw_stderr)
    absence_probe = _classify_removal_absence_probe(
        state,
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )
    status = str(result.status or metadata.get("status") or "").strip()
    approval_denied = bool(metadata.get("approval_denied"))
    current_verifier_kind = _verifier_kind_for_command(command)
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
        and _command_is_binary_probe(command)
        and _output_confirms_not_found(stdout, stderr)
        and _task_has_removal_intent(state)
    ):
        # Exit 127 = "command not found" which is the expected outcome of a
        # successful removal task. Treat this as a pass rather than a verifier
        # failure so the session can exit the repair phase normally.
        verdict = "pass"
    elif (
        exit_code == 1
        and _command_is_binary_probe(command)
        and _output_confirms_not_found(stdout, stderr)
    ):
        # Exit 1 from a diagnostic probe (systemctl status, dpkg -l, apt list,
        # which, whereis, etc.) that confirms the resource is absent is
        # informational negative intelligence, not a failure.
        verdict = "pass"
    else:
        verdict = "fail"
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

    failure_class = "approval_denied" if approval_denied else _classify_execution_failure(result.error or stderr or stdout)
    if insufficient_verifier:
        failure_class = "insufficient_verifier"
    if not approval_denied and failure_class == "environment":
        if _looks_like_infinite_loop(command, str(result.error or ""), stdout, stderr):
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
    if insufficient_verifier:
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


def _extract_latest_execution_blocker(
    *,
    tool_name: str,
    command: str,
    target: str,
    exit_code: Any,
    failure_class: str,
    stdout: str,
    stderr: str,
    error: str,
    verdict: str,
) -> dict[str, Any] | None:
    if tool_name not in {"shell_exec", "ssh_exec"} or verdict == "pass":
        return None
    combined = "\n".join(str(part or "").strip() for part in (error, stderr, stdout) if str(part or "").strip())
    if not combined:
        return None
    blocker_class = ""
    salient = ""
    for candidate_class, pattern in _REMOTE_APPLICATION_BLOCKERS:
        match = pattern.search(combined)
        if match:
            blocker_class = candidate_class
            salient = _snip_text(match.group(0), limit=280)
            break
    if not blocker_class and _INTERACTIVE_PROMPT_RE.search(combined):
        blocker_class = "interactive_prompt"
        match = _INTERACTIVE_PROMPT_RE.search(combined)
        salient = _snip_text(match.group(0) if match else combined, limit=280)
    if not blocker_class:
        return None
    signature_seed = "|".join([tool_name, command, blocker_class, salient.lower()])
    return {
        "tool": tool_name,
        "command": command,
        "target": target,
        "exit_code": exit_code,
        "blocker_class": blocker_class,
        "failure_class": failure_class,
        "salient_error": salient,
        "is_interactive_prompt": blocker_class == "interactive_prompt",
        "signature": hashlib.sha1(signature_seed.encode("utf-8")).hexdigest()[:16],
    }


def _store_latest_execution_blocker(state: Any, blocker: dict[str, Any]) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    prior = scratchpad.get("_latest_execution_blocker")
    prior_signature = str(prior.get("signature") or "") if isinstance(prior, dict) else ""
    new_signature = str(blocker.get("signature") or "")
    scratchpad["_latest_execution_blocker"] = blocker
    if prior_signature and new_signature and prior_signature != new_signature:
        counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
        counters["no_progress"] = 0
        counters["repeat_command"] = 0
        state.stagnation_counters = counters
        scratchpad["_repair_last_failure_signature"] = new_signature


def _semantic_verifier_failure(*, command: str, stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in (stdout, stderr) if str(part or "").strip())
    if not combined:
        return ""
    normalized_command = re.sub(r"\s+", " ", str(command or "").strip().lower())
    normalized_output = re.sub(r"\s+", " ", combined.strip().lower())
    if (
        "test -f" in normalized_command
        and "echo" in normalized_command
        and re.search(r"\bmissing\b", normalized_output)
        and not re.search(r"\bexists\b", normalized_output)
    ):
        return "file existence verifier reported MISSING"
    if _NGINX_VERIFIER_COMMAND_RE.search(command) or "nginx:" in combined.lower():
        match = _NGINX_VERIFIER_FAILURE_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    if "curl" in str(command or "").lower() or "curl:" in combined.lower():
        match = _CURL_VERIFIER_FAILURE_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    match = _ZERO_TESTS_RAN_RE.search(combined)
    if match:
        return _snip_text(match.group(0), limit=240)
    if _verifier_kind_for_command(command) in {"test_suite", "run_target"}:
        match = _TEST_FAILURE_SUMMARY_RE.search(combined)
        if match:
            summary = _snip_text(match.group(0), limit=240)
            count_match = _TEST_FAILURE_COUNT_RE.search(combined)
            if count_match:
                failures = next(g for g in count_match.groups() if g is not None)
                summary = f"{summary} ({failures} test failure(s) detected)"
            return summary
        match = _TEST_FAILURE_OUTPUT_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    return ""


_VERIFIER_KIND_STRENGTH = {
    "syntax_only": 1,
    "lint_typecheck": 2,
    "diagnostic": 2,
    "run_target": 3,
    "test_suite": 4,
}


def _verifier_kind_for_command(command: str) -> str:
    return verifier_kind_for_command(command)


def _verifier_strength(kind: str) -> int:
    return verifier_strength(kind)


def _task_or_history_requires_runtime_verifier(state: Any) -> bool:
    prior_command = _prior_failed_verifier_command(state)
    if verifier_strength(verifier_kind_for_command(prior_command)) > verifier_strength("syntax_only"):
        return True
    texts: list[str] = []
    run_brief = getattr(state, "run_brief", None)
    texts.append(str(getattr(run_brief, "original_task", "") or ""))
    working_memory = getattr(state, "working_memory", None)
    texts.append(str(getattr(working_memory, "current_goal", "") or ""))
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        handoff = scratchpad.get("_last_task_handoff")
        if isinstance(handoff, dict):
            texts.append(str(handoff.get("effective_task") or ""))
            texts.append(str(handoff.get("current_goal") or ""))
    combined = " ".join(texts).lower()
    return any(
        marker in combined
        for marker in (
            "run script",
            "run the script",
            "run it",
            "test",
            "tests",
            "unittest",
            "pytest",
            "verify functionality",
            "fix until complete",
        )
    )


def _prior_failed_verifier_command(state: Any) -> str:
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        failed = scratchpad.get("_last_failed_verifier")
        if isinstance(failed, dict):
            command = str(failed.get("command") or "").strip()
            if command:
                return command
    prior = getattr(state, "last_verifier_verdict", None)
    if isinstance(prior, dict) and str(prior.get("verdict") or "").strip().lower() == "fail":
        return str(prior.get("command") or "").strip()
    return ""


def _passing_verifier_is_weaker_than_prior_failure(
    state: Any,
    *,
    current_command: str,
    current_kind: str,
) -> bool:
    if verifier_strength(current_kind) > verifier_strength("syntax_only"):
        return False
    if not _task_or_history_requires_runtime_verifier(state):
        return False
    prior_command = _prior_failed_verifier_command(state)
    if not prior_command:
        return False
    prior_kind = verifier_kind_for_command(prior_command)
    if verifier_strength(prior_kind) <= verifier_strength(current_kind):
        return False
    normalized_current = re.sub(r"\s+", " ", str(current_command or "").strip().lower())
    normalized_prior = re.sub(r"\s+", " ", prior_command.strip().lower())
    return normalized_current != normalized_prior


def _insufficient_verifier_message(state: Any, *, command: str) -> str:
    prior_command = _prior_failed_verifier_command(state)
    if prior_command:
        return (
            f"Verifier `{command}` only checks syntax and is weaker than the prior failed verifier "
            f"`{prior_command}`; rerun the script/tests that failed."
        )
    return f"Verifier `{command}` only checks syntax; rerun the script/tests required by the task."


def assess_remote_mutation_verification(
    *,
    requirement: dict[str, Any] | None,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tool_name != "ssh_exec" or not isinstance(requirement, dict) or not isinstance(arguments, dict):
        return {"is_verifier_attempt": False}
    command = str(arguments.get("command") or "").strip()
    mutation_command = _strip_benign_shell_redirections(command, preserve_newlines=True)
    if not mutation_command or _REMOTE_MUTATING_COMMAND_RE.search(mutation_command):
        return {"is_verifier_attempt": False}

    output = result.output if isinstance(result.output, dict) else {}
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    stdout = str(output.get("stdout") or "")
    stderr = str(output.get("stderr") or "")
    exit_code = output.get("exit_code")

    host = str(arguments.get("host") or metadata.get("host") or "").strip().lower()
    requirement_host = str(requirement.get("host") or "").strip().lower()
    if requirement_host and host and requirement_host != host:
        return {"is_verifier_attempt": False}

    guessed_paths = [str(item).strip() for item in requirement.get("guessed_paths", []) if str(item).strip()]
    path_match = any(path in command for path in guessed_paths) if guessed_paths else False
    directory_checks = parse_directory_empty_checks(requirement)
    directory_check = match_directory_empty_check(command, directory_checks)
    profile = str(requirement.get("verification_profile") or "").strip().lower()
    patterns = requirement.get("verification_patterns")
    new_patterns = patterns.get("new_present", []) if isinstance(patterns, dict) else []
    old_patterns = patterns.get("old_absent", []) if isinstance(patterns, dict) else []

    mentions_new = any(str(pattern) and str(pattern) in command for pattern in new_patterns)
    mentions_old = any(str(pattern) and str(pattern) in command for pattern in old_patterns)
    is_verifier_attempt = path_match or directory_check is not None or mentions_new or mentions_old
    if not is_verifier_attempt:
        return {"is_verifier_attempt": False}

    assessment: dict[str, Any] = {
        "is_verifier_attempt": True,
        "host": host,
        "command": command,
        "exit_code": exit_code,
        "path_match": path_match,
        "verification_profile": profile,
    }

    if directory_check is not None:
        check_path = str(directory_check.get("path") or "").strip().rstrip("/")
        if result.success and exit_code in (0, None) and not stdout.strip():
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": True,
                    "reason": "directory_empty_check_passed",
                    "verified_directory_empty_check": check_path,
                }
            )
            return assessment
        assessment.update(
            {
                "verification_strength": "strong",
                "clears_requirement": False,
                "reason": "directory_empty_check_failed",
                "verified_directory_empty_check": check_path,
                "message": _snip_text(stderr or stdout or "Directory-empty verifier did not return empty output.", limit=240),
            }
        )
        return assessment

    if not isinstance(patterns, dict):
        readback_path = _simple_remote_readback_path(command)
        if readback_path and readback_path in guessed_paths:
            if result.success and exit_code in (0, None) and stdout.strip():
                assessment.update(
                    {
                        "verification_strength": "strong",
                        "clears_requirement": True,
                        "reason": "simple_readback_path_verified",
                        "verified_path": readback_path,
                    }
                )
                return assessment
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": False,
                    "reason": "simple_readback_path_empty_or_failed",
                    "verified_path": readback_path,
                    "message": _snip_text(stderr or stdout or "Readback command did not return file content.", limit=240),
                }
            )
            return assessment
        if path_match and _REMOTE_FILE_PRESENCE_PROBE_RE.search(command):
            if result.success and exit_code in (0, None):
                assessment.update(
                    {
                        "verification_strength": "strong",
                        "clears_requirement": True,
                        "reason": "file_presence_probe_passed",
                    }
                )
                return assessment
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": False,
                    "reason": "file_presence_probe_failed",
                    "message": _snip_text(stderr or stdout or "Presence/hash verifier did not confirm the file.", limit=240),
                }
            )
            return assessment

    if profile == "html_stylesheet_swap":
        command_lower = command.lower()
        stdout_lower = stdout.lower()
        has_positive_check = "theme" in command_lower or "stylesheet" in command_lower or any(
            str(pattern).lower() in command_lower for pattern in new_patterns
        )
        has_negative_check = "<style>" in command_lower and (
            "no_style" in command_lower
            or "! grep" in command_lower
            or "grep -l" not in command_lower
            or "grep -l" in command_lower and "has_style" in command_lower
        )
        weak_positive_only = "grep -l" in command_lower and has_positive_check and "<style>" not in command_lower
        if weak_positive_only:
            assessment.update(
                {
                    "verification_strength": "weak_positive_only",
                    "clears_requirement": False,
                    "reason": "replacement_positive_only",
                    "message": "Positive-only grep verifier does not prove the old target is gone.",
                }
            )
            return assessment

        if has_positive_check and has_negative_check:
            style_present = "has_style" in stdout_lower
            link_missing = "no_link" in stdout_lower
            link_present = "has_link" in stdout_lower or any(
                str(pattern).lower() in stdout_lower for pattern in new_patterns
            )
            if result.success and exit_code in (0, None) and not style_present and not link_missing and link_present:
                assessment.update(
                    {
                        "verification_strength": "strong",
                        "clears_requirement": True,
                        "reason": "strong_remote_replacement_verifier",
                    }
                )
                return assessment
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": False,
                    "reason": "strong_verifier_failed",
                    "message": _snip_text(stderr or stdout or "Strong replacement verifier did not show the expected NO_STYLE/HAS_LINK outcome.", limit=240),
                }
            )
            return assessment

    content_based_positive = [str(pattern) for pattern in new_patterns if str(pattern)]
    content_based_negative = [str(pattern) for pattern in old_patterns if str(pattern)]
    if result.success and exit_code in (0, None) and content_based_positive:
        positive_seen = all(pattern.lower() in stdout.lower() for pattern in content_based_positive)
        negative_absent = all(pattern.lower() not in stdout.lower() for pattern in content_based_negative)
        if positive_seen and negative_absent:
            assessment.update(
                {
                    "verification_strength": "strong",
                    "clears_requirement": True,
                    "reason": "stdout_verifier_patterns_matched",
                }
            )
            return assessment

    assessment.update(
        {
            "verification_strength": "unknown",
            "clears_requirement": False,
        }
    )
    return assessment


def _simple_remote_readback_path(command: str) -> str:
    try:
        tokens = shlex.split(str(command or "").strip())
    except ValueError:
        return ""
    if not tokens:
        return ""
    if tokens[0] == "sudo":
        tokens = tokens[1:]
    if not tokens:
        return ""

    command_name = tokens[0]
    if command_name in _REMOTE_READBACK_COMMANDS:
        path_operands = [token for token in tokens[1:] if token.startswith("/")]
        return path_operands[0] if len(path_operands) == 1 else ""

    if command_name == "sed" and "-n" in tokens:
        path_operands = [token for token in tokens[1:] if token.startswith("/")]
        if len(path_operands) == 1:
            return path_operands[0]
    return ""



def _annotate_verifier_artifact(
    artifact: Any,
    *,
    verifier_verdict: dict[str, Any],
) -> None:
    if not hasattr(artifact, "metadata") or not isinstance(artifact.metadata, dict):
        artifact.metadata = {}
    metadata = artifact.metadata
    verdict = str(verifier_verdict.get("verdict") or "").strip()
    target = str(verifier_verdict.get("target") or verifier_verdict.get("command") or "").strip()
    command = str(verifier_verdict.get("command") or "").strip()
    metadata["verifier_verdict"] = verdict
    metadata["verifier_target"] = target
    metadata["verifier_command"] = command
    metadata["verifier_exit_code"] = verifier_verdict.get("exit_code")
    metadata["verifier_stdout"] = verifier_verdict.get("key_stdout")
    metadata["verifier_stderr"] = verifier_verdict.get("key_stderr")
    if target and (not getattr(artifact, "source", "")):
        artifact.source = target
    status_label = "SUCCESS" if verdict == "pass" else "FAILURE"
    summary = f"{artifact.tool_name or artifact.kind} {status_label}"
    if command:
        summary = f"{summary}: {command}"
    elif target:
        summary = f"{summary}: {target}"
    artifact.summary = summary[:160]


def _snip_text(value: Any, *, limit: int = 400) -> str:
    return snip_text(value, limit=limit)


def _classify_execution_failure(text: str) -> str:
    return classify_execution_failure(text)


def _looks_like_infinite_loop(command: str, error: str, stdout: str, stderr: str) -> bool:
    return looks_like_infinite_loop(command, error, stdout, stderr)


def _is_long_running_remote_command_timeout(
    *,
    tool_name: str,
    command: str,
    result: ToolEnvelope,
    stdout: str,
    stderr: str,
) -> bool:
    if tool_name != "ssh_exec":
        return False
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    failure_kind = str(metadata.get("failure_kind") or "").strip()
    ssh_error_class = str(metadata.get("ssh_error_class") or "").strip()
    error_text = str(result.error or "").lower()
    is_timeout = (
        failure_kind == "timeout"
        or ssh_error_class == "command_timeout"
        or "timed out" in error_text
        or "timeout" in error_text
    )
    if not is_timeout:
        return False
    command_text = str(command or "")
    output_text = "\n".join(part for part in (stdout, stderr) if str(part or "").strip())
    return bool(
        _LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE.search(command_text)
        or _LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE.search(output_text)
    )


def _command_is_binary_probe(command: str) -> bool:
    return command_is_binary_probe(command)


def _output_confirms_not_found(stdout: str, stderr: str) -> bool:
    return output_confirms_not_found(stdout, stderr)


def _classify_removal_absence_probe(
    state: Any,
    *,
    command: str,
    exit_code: Any,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    default = {
        "is_absence_probe": False,
        "absence_confirmed": False,
        "found_resources": False,
        "reason": "",
    }
    if not _task_has_removal_intent(state):
        return default
    cmd = str(command or "").strip()
    if not cmd or not _command_is_removal_absence_probe(cmd, state):
        return default

    out = str(stdout or "").strip()
    err = str(stderr or "").strip()
    combined = "\n".join(part for part in (out, err) if part).strip()
    found_resources = _absence_probe_found_resources(command=cmd, stdout=out, stderr=err, exit_code=exit_code)
    if found_resources:
        return {
            "is_absence_probe": True,
            "absence_confirmed": False,
            "found_resources": True,
            "reason": "absence probe found matching cleanup/removal resources",
        }

    if _LS_NO_SUCH_FILE_RE.search(err) or _LS_NO_SUCH_FILE_RE.search(out):
        return {
            "is_absence_probe": True,
            "absence_confirmed": True,
            "found_resources": False,
            "reason": "ls reported deleted resource is absent",
        }

    lowered_cmd = cmd.lower()
    empty_output = not combined
    if empty_output and _exit_code_matches(exit_code, {0, 1}):
        if "grep" in lowered_cmd:
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "grep absence probe returned no matches",
            }
        if "find" in lowered_cmd:
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "find absence probe returned no matches",
            }
        if "systemctl" in lowered_cmd and _REMOVAL_ABSENCE_PIPE_RE.search(cmd):
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "systemctl absence probe returned no matching units",
            }
        if "pgrep" in lowered_cmd or re.search(r"(?:^|\s)ps(?:\s|$)", lowered_cmd):
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "process absence probe returned no matches",
            }

    return {
        "is_absence_probe": True,
        "absence_confirmed": False,
        "found_resources": False,
        "reason": "absence probe did not prove resource absence",
    }


def _command_is_removal_absence_probe(command: str, state: Any) -> bool:
    cmd = str(command or "").strip()
    if not _REMOVAL_ABSENCE_PROBE_RE.search(cmd):
        return False
    if _command_has_write_or_heredoc_shape(cmd):
        return False
    lowered = cmd.lower()
    has_absence_tool_shape = (
        re.search(r"(?:^|[;&|]\s*)find\s+", lowered) is not None
        or re.search(r"(?:^|[;&|]\s*)ls\s+", lowered) is not None
        or "grep" in lowered
        or "pgrep" in lowered
        or re.search(r"(?:^|[;&|]\s*)ps(?:\s|$)", lowered) is not None
        or ("systemctl" in lowered and _REMOVAL_ABSENCE_PIPE_RE.search(cmd) is not None)
    )
    if not has_absence_tool_shape:
        return False
    return _command_mentions_removal_subject(cmd, state)


def _command_has_write_or_heredoc_shape(command: str) -> bool:
    return command_has_write_or_heredoc_shape(command)


def _command_mentions_removal_subject(command: str, state: Any) -> bool:
    lowered = str(command or "").lower()
    if _FOG_RESOURCE_RE.search(lowered):
        return True
    task_terms = _removal_task_subject_terms(state)
    return any(term in lowered for term in task_terms)


def _removal_task_subject_terms(state: Any) -> set[str]:
    task_text = _removal_task_text(state)
    if not task_text:
        return set()
    candidates = re.findall(r"[a-z0-9][a-z0-9_.@/-]{2,}", task_text.lower())
    stop_words = {
        "cleanup", "clean", "remove", "removed", "removal", "delete", "deleted",
        "purge", "stop", "disable", "disabled", "mask", "masked", "verify",
        "remote", "server", "host", "file", "files", "service", "services",
        "systemd", "process", "processes", "user", "users", "database",
    }
    terms = set()
    for candidate in candidates:
        stripped = candidate.strip(".,:;()[]{}'\"")
        if stripped in stop_words:
            continue
        if stripped.startswith(("http", "/tmp/")):
            continue
        terms.add(stripped)
    return terms


def _removal_task_text(state: Any) -> str:
    run_brief = getattr(state, "run_brief", None)
    original_task = str(getattr(run_brief, "original_task", "") or "")
    wm = getattr(state, "working_memory", None)
    current_goal = str(getattr(wm, "current_goal", "") or "")
    return " ".join(part for part in (original_task, current_goal) if part).strip()


def _absence_probe_found_resources(
    *,
    command: str,
    stdout: str,
    stderr: str,
    exit_code: Any,
) -> bool:
    out = str(stdout or "").strip()
    err = str(stderr or "").strip()
    if _LS_NO_SUCH_FILE_RE.search(err) or _LS_NO_SUCH_FILE_RE.search(out):
        return False
    lowered_cmd = str(command or "").lower()
    if ("grep -q" in lowered_cmd or "grep --quiet" in lowered_cmd) and _exit_code_matches(exit_code, {0}):
        return True
    if out:
        return True
    if err and not _output_confirms_not_found("", err):
        return True
    return False


def _exit_code_matches(exit_code: Any, values: set[int]) -> bool:
    return exit_code_matches(exit_code, values)


def _task_has_removal_intent(state: Any) -> bool:
    """Return True when the original task description signals a removal intent."""
    original_task = _removal_task_text(state).lower()
    if not original_task:
        return False
    return any(kw in original_task for kw in _REMOVAL_TASK_KEYWORDS)


# Keywords that indicate an audit/investigation task where documenting
# existence of resources is expected (not a failure).
_AUDIT_TASK_KEYWORDS = frozenset([
    "audit", "investigate", "review", "assess", "check", "report on",
    "inspect", "examine", "analyze", "verify compliance", "document",
])


def _is_audit_task(state: Any) -> bool:
    """Return True when the original task is an audit/investigation."""
    task_text = _removal_task_text(state).lower()
    if not task_text:
        return False
    return any(kw in task_text for kw in _AUDIT_TASK_KEYWORDS)


def _update_acceptance_ledger(state: Any, *, verdict: str) -> None:
    criteria = []
    if hasattr(state, "active_acceptance_criteria"):
        try:
            criteria = list(state.active_acceptance_criteria())
        except Exception:
            criteria = []
    if not criteria:
        return
    ledger = state.acceptance_ledger if isinstance(getattr(state, "acceptance_ledger", None), dict) else {}
    if verdict == "pass":
        for criterion in criteria:
            ledger[criterion] = "passed"
    elif verdict == "needs_human":
        for criterion in criteria:
            ledger.setdefault(criterion, "pending")
    state.acceptance_ledger = ledger


def _record_docker_retry_state(
    state: Any,
    *,
    command: str,
    failure_class: str,
    verdict: str,
) -> dict[str, Any]:
    if verdict != "fail" or not docker_failure_is_registry_resolution(failure_class):
        return {}

    parsed = extract_docker_command_target(command)
    if parsed is None:
        return {}
    command_kind, image_ref = parsed
    retry_key = docker_retry_key(command, failure_class)
    retry_family = docker_retry_family(command)
    if not retry_key or not retry_family:
        return {}

    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    state.scratchpad = scratchpad

    retry_counts = scratchpad.setdefault("_docker_registry_retry_counts", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}
        scratchpad["_docker_registry_retry_counts"] = retry_counts
    retry_count = int(retry_counts.get(retry_key, 0) or 0) + 1
    retry_counts[retry_key] = retry_count

    family_counts = scratchpad.setdefault("_docker_registry_family_counts", {})
    if not isinstance(family_counts, dict):
        family_counts = {}
        scratchpad["_docker_registry_family_counts"] = family_counts
    family_count = int(family_counts.get(retry_family, 0) or 0) + 1
    family_counts[retry_family] = family_count

    exhausted_families = scratchpad.setdefault("_docker_registry_exhausted_families", [])
    if not isinstance(exhausted_families, list):
        exhausted_families = []
        scratchpad["_docker_registry_exhausted_families"] = exhausted_families
    if family_count >= 4 and retry_family not in exhausted_families:
        exhausted_families.append(retry_family)

    scratchpad["_last_docker_registry_retry_key"] = retry_key
    scratchpad["_last_docker_registry_retry_family"] = retry_family

    return {
        "docker_command_kind": command_kind,
        "docker_image_ref": image_ref,
        "docker_retry_family": retry_family,
        "docker_retry_key": retry_key,
        "docker_retry_count": retry_count,
        "docker_retry_family_count": family_count,
    }


def _update_repair_cycle_state(
    state: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    command: str,
    target: str,
    verdict: str,
    failure_class: str,
    docker_retry: dict[str, Any] | None = None,
) -> None:
    if verdict == "pass":
        if getattr(state, "acceptance_ready", None) and state.acceptance_ready():
            state.scratchpad["_contract_phase"] = "execute"
        elif getattr(state, "repair_cycle_id", ""):
            state.scratchpad["_contract_phase"] = "verify"
        return

    if verdict == "needs_human":
        return

    docker_retry = docker_retry if isinstance(docker_retry, dict) else {}
    semantic_target = str(docker_retry.get("docker_retry_family") or target)
    semantic_failure = str(docker_retry.get("docker_retry_key") or "")
    semantic_command = str(docker_retry.get("docker_retry_family") or command)

    signature_seed = "|".join(
        [
            str(getattr(state, "thread_id", "") or ""),
            tool_name,
            semantic_command or command,
            semantic_target or target,
            semantic_failure or failure_class,
            str(result.error or ""),
        ]
    )
    repair_cycle_id = f"repair-{hashlib.sha1(signature_seed.encode('utf-8')).hexdigest()[:8]}"
    if getattr(state, "repair_cycle_id", "") != repair_cycle_id:
        state.repair_cycle_id = repair_cycle_id
        state.scratchpad["_repair_cycle_reads"] = []
        state.files_changed_this_cycle = []
    # Increment repair step count every time we enter repair
    repair_steps = int(state.scratchpad.get("_repair_step_count", 0) or 0) + 1
    state.scratchpad["_repair_step_count"] = repair_steps
    max_repair = int(state.scratchpad.get("_max_repair_steps", 3) or 3)
    if repair_steps >= max_repair:
        state.acceptance_waived = True
    # Flag for auto-escalation after 2 repair cycles on same target
    if repair_steps >= 2:
        state.scratchpad["_repair_cycle_escalation_ready"] = True
    state.scratchpad["_contract_phase"] = "repair"

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    failure_signature = semantic_failure or f"{tool_name}|{command}|{target}|{failure_class}|{result.error or ''}"
    last_failure_signature = str(state.scratchpad.get("_repair_last_failure_signature", "") or "")
    if last_failure_signature == failure_signature:
        counters["no_progress"] = int(counters.get("no_progress", 0)) + 1
    state.scratchpad["_repair_last_failure_signature"] = failure_signature

    command_fingerprint = hashlib.sha1(f"{tool_name}|{semantic_command or command}|{semantic_target or target}".encode("utf-8")).hexdigest()
    last_command_fingerprint = str(state.scratchpad.get("_repair_last_command_fingerprint", "") or "")
    if last_command_fingerprint == command_fingerprint:
        counters["repeat_command"] = int(counters.get("repeat_command", 0)) + 1
    state.scratchpad["_repair_last_command_fingerprint"] = command_fingerprint
    state.stagnation_counters = counters
