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
from ..models.tool_result import ToolEnvelope
from ..shell_utils import strip_benign_shell_redirections as _strip_benign_shell_redirections

# Patterns that indicate the command is probing for a binary's presence.
# These are the canonical "is X installed?" commands.
# The lookahead after the subcommand keyword accepts whitespace, shell
# redirects (2>&1), pipes, or end-of-string so that commands like
# "docker info 2>&1 | head -5" are still recognised as probes.
_BINARY_PROBE_RE = re.compile(
    r"^(?:"
    r"which\s+\S+"
    r"|whereis\s+\S+"
    r"|\S+\s+(?:--version|version|info|status|--help)(?:\s|\||\>|\&|$)"
    r"|type\s+\S+"
    r"|command\s+-v\s+\S+"
    r"|dpkg\s+-l\s+\S+"
    r"|apt\s+(?:list|show|search)\s+\S+"
    r"|rpm\s+-q\s+\S+"
    r"|apk\s+info\s+\S+"
    r")"
    r"(?:.*)?$",
    re.IGNORECASE,
)
_REMOVAL_ABSENCE_PROBE_RE = re.compile(
    r"\b(?:grep|egrep|fgrep|find|ls|systemctl|pgrep|ps)\b",
    re.IGNORECASE,
)
_REMOVAL_ABSENCE_PIPE_RE = re.compile(
    r"\|\s*(?:grep|egrep|fgrep)\b",
    re.IGNORECASE,
)
_LS_NO_SUCH_FILE_RE = re.compile(
    r"\bls:\s+cannot\s+access\b.*\b(?:no such file|not found)\b",
    re.IGNORECASE | re.DOTALL,
)
_FOG_RESOURCE_RE = re.compile(
    r"\bfog(?:project|server|\.service|[_-]?(?:nfs|scheduler|multicast|snapin|replicator|image|web|php|worker))?\b",
    re.IGNORECASE,
)

# Keywords that indicate a removal/uninstall task.
_REMOVAL_TASK_KEYWORDS = frozenset([
    "uninstall", "remove", "delete", "purge", "rm ", "rm -f",
    "stop and remove", "stop all", "clean up", "clean-up",
    "get rid of", "wipe", "tear down", "teardown", "disable",
])

# Strings in stderr/stdout that confirm exit-127 means "not found".
_NOT_FOUND_MARKERS = ("command not found", "not found", "no such file", "permission denied", "could not be found")
_SSH_AUTH_RECOVERY_KEY = "_ssh_auth_recovery_state"
_REMOTE_MUTATING_COMMAND_RE = re.compile(
    r"\bsed\s+-i\b"
    r"|"
    r"\bperl\s+-p(?:i|[^A-Za-z0-9_]*-i)\b"
    r"|"
    r"\bpython3?\s+-c\b.*\bopen\s*\([^)]*['\"]w"
    r"|"
    r"(?:^|\s)(?:\d?>|\d?>>|>>|>)\s*\S+"
    r"|"
    r"\btee(?:\s+-a)?\s+/\S+"
    r"|"
    r"\bcat\s*>\s*/\S+"
    r"|"
    r"\b(?:rm|truncate|install\s+-m)\b"
    r"|"
    r"\b(?:mv|cp)\b.+\s+/(?:etc|var|usr|opt|srv|root)/\S+",
    re.IGNORECASE | re.DOTALL,
)
_REMOTE_READBACK_COMMANDS = {"cat", "head", "tail"}
_NGINX_VERIFIER_COMMAND_RE = re.compile(r"\bnginx\s+-t\b", re.IGNORECASE)
_NGINX_VERIFIER_FAILURE_RE = re.compile(
    r"nginx:\s*configuration\s*file\b.*\btest\s*failed\b"
    r"|"
    r"\[\s*emerg\s*\].*?\bin\s+/etc/nginx/"
    r"|"
    r"unexpected\s+end\s+of\s*file,\s*expecting\s+[\"']?[;}]",
    re.IGNORECASE | re.DOTALL,
)
_CURL_VERIFIER_FAILURE_RE = re.compile(
    r"\bcurl:\s*\(\d+\)"
    r"|"
    r"\bfailed to connect\b"
    r"|"
    r"\bconnection refused\b",
    re.IGNORECASE | re.DOTALL,
)
_LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE = re.compile(
    r"\binstallfog\.sh\b"
    r"|"
    r"\b(?:apt-get|apt|dnf|yum|zypper|pacman)\b.*\b(?:install|upgrade|dist-upgrade|full-upgrade)\b"
    r"|"
    r"\b(?:docker\s+compose|docker-compose)\s+up\b"
    r"|"
    r"\b(?:make|ninja)\s+(?:install|build)\b",
    re.IGNORECASE | re.DOTALL,
)
_LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE = re.compile(
    r"\binstallation started\b"
    r"|"
    r"\btesting internet connection\b"
    r"|"
    r"\binstalling\b.+\bas needed\b"
    r"|"
    r"\bbuilding dependency tree\b"
    r"|"
    r"\bsetting up\b\s+\S+"
    r"|"
    r"\bpulling\b.+\bimage\b",
    re.IGNORECASE | re.DOTALL,
)
_REMOTE_APPLICATION_BLOCKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "fogproject_account_exists",
        re.compile(
            r'The account\s+"fogproject"\s+already exists.*?'
            r"(?:Please remove the account\s+\"fogproject\".*?|"
            r"username=<usernameForSystem>\s+\./installfog\.sh\s+-y)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "account_exists",
        re.compile(
            r'The account\s+"[^"]+"\s+already exists.*?'
            r"(?:Please remove the account|set a new service username|userdel\s+\S+)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "command_not_found",
        re.compile(r"\b(?:bash|sh):\s+line\s+\d+:\s+\S+:\s+command not found\b", re.IGNORECASE),
    ),
    (
        "file_exists",
        re.compile(r"\bfailed to create symbolic link\b.*?\bFile exists\b", re.IGNORECASE | re.DOTALL),
    ),
    (
        "permission_denied",
        re.compile(r"\bpermission denied\b", re.IGNORECASE),
    ),
)
_INTERACTIVE_PROMPT_RE = re.compile(
    r"\b(?:Choice:\s*\[\d+\]|Are you sure you wish to continue|Should .*?\?\s*\([yYnN]/|"
    r"Sorry,\s+answer not recognized|Hit \[?Enter\]?)\b",
    re.IGNORECASE | re.DOTALL,
)
_RAW_SSH_COMMAND_RE = re.compile(r"^\s*(?:ssh\b|scp\b|sftp\b|sshpass\b)", re.IGNORECASE)


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
    semantic_failure = _semantic_verifier_failure(command=command, stdout=stdout, stderr=stderr)
    absence_probe = _classify_removal_absence_probe(
        state,
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )
    status = str(result.status or metadata.get("status") or "").strip()
    if status == "needs_human":
        verdict = "needs_human"
    elif semantic_failure:
        verdict = "fail"
    elif absence_probe["is_absence_probe"] and absence_probe["found_resources"]:
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
    failure_class = _classify_execution_failure(result.error or stderr or stdout)
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
    acceptance_delta = {
        "status": "satisfied" if verdict == "pass" else "blocked",
        "notes": ["execution succeeded"] if verdict == "pass" else [
            str(semantic_failure or result.error or stderr or stdout or status or "execution failed")
        ],
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
    if _NGINX_VERIFIER_COMMAND_RE.search(command) or "nginx:" in combined.lower():
        match = _NGINX_VERIFIER_FAILURE_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    if "curl" in str(command or "").lower() or "curl:" in combined.lower():
        match = _CURL_VERIFIER_FAILURE_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    return ""


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
    directory_checks = _remote_mutation_directory_empty_checks(requirement)
    directory_check = _matched_directory_empty_check(command, directory_checks)
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

    if profile == "html_stylesheet_swap":
        command_lower = command.lower()
        stdout_lower = stdout.lower()
        stderr_lower = stderr.lower()
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


def _remote_mutation_directory_empty_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
    raw_checks = requirement.get("directory_empty_checks")
    if not isinstance(raw_checks, list):
        return []
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_checks:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().rstrip("/")
        if not path or path in seen:
            continue
        seen.add(path)
        glob = str(item.get("glob") or "").strip()
        checks.append({"path": path, "glob": glob})
    return checks


def _matched_directory_empty_check(command: str, checks: list[dict[str, str]]) -> dict[str, str] | None:
    command_text = str(command or "")
    if not command_text or not checks:
        return None
    lowered = command_text.lower()
    if not any(marker in lowered for marker in ("find ", "ls ", "test ", "compgen ")):
        return None
    if not any(marker in lowered for marker in ("-mindepth", "-maxdepth", "-print", "-quit", "-a", "-z", "compgen")):
        return None
    for check in checks:
        path = str(check.get("path") or "").strip().rstrip("/")
        glob = str(check.get("glob") or "").strip()
        if path and (path in command_text or glob and glob in command_text):
            return check
    return None


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
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _classify_execution_failure(text: str) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    docker_class = classify_docker_failure(lowered)
    if docker_class:
        return docker_class
    if "syntaxerror" in lowered or "parseerror" in lowered:
        return "syntax"
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "import"
    if "no such file" in lowered or "not found" in lowered:
        return "path"
    if "timed out" in lowered or "timeout" in lowered or "connection timed out" in lowered:
        return "environment"
    if "permission denied" in lowered or "password" in lowered or "sudo" in lowered:
        return "environment"
    if "assert" in lowered or "failed" in lowered or "traceback" in lowered:
        return "test"
    return "logic"


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
    """Return True when the command looks like a binary-presence probe."""
    cmd = str(command or "").strip()
    if not cmd:
        return False
    return bool(_BINARY_PROBE_RE.match(cmd))


def _output_confirms_not_found(stdout: str, stderr: str) -> bool:
    """Return True when stdout/stderr text confirms the binary is absent."""
    combined = (str(stdout or "") + " " + str(stderr or "")).lower()
    # A completely empty combined output with exit 127 also qualifies — the
    # shell itself swallowed the "not found" message (happens on some remotes).
    if not combined.strip():
        return True
    return any(marker in combined for marker in _NOT_FOUND_MARKERS)


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
    try:
        return int(exit_code) in values
    except (TypeError, ValueError):
        return exit_code is None and 0 in values


def _task_has_removal_intent(state: Any) -> bool:
    """Return True when the original task description signals a removal intent."""
    original_task = _removal_task_text(state).lower()
    if not original_task:
        return False
    return any(kw in original_task for kw in _REMOVAL_TASK_KEYWORDS)


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
