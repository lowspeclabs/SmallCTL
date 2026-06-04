from __future__ import annotations

import asyncio
import asyncio.subprocess
import re
import shlex
import shutil
import time
import uuid
from typing import Any, TYPE_CHECKING

from ..models.events import UIEventType
from .common import fail, ok
from ..risk_policy import evaluate_risk_policy
from ..state import LoopState
from .shell import create_process
from .installer_preflight import run_installer_preflight_probes
from .process_lifecycle import stop_process, truncate_output, unregister_process
from .process_streams import read_stream_chunks
from .ssh_parsing import (
    is_shell_redirection_token as _is_shell_redirection_token,
    join_remote_shell_tokens as _join_remote_shell_tokens,
    normalize_ssh_arguments as _normalize_ssh_arguments,
    normalize_ssh_target as _normalize_ssh_target,
    normalize_optional_ssh_string as _normalize_optional_ssh_string,
    parse_int_option as _parse_int_option,
    parse_ssh_exec_args_from_shell_command as _parse_ssh_exec_args_from_shell_command,
    shell_join as _shell_join,
    shell_tokens_with_spans as _shell_tokens_with_spans,
    split_ssh_option_value as _split_ssh_option_value,
    strip_redundant_root_sudo as _strip_redundant_root_sudo,
)

# Public re-exports for downstream callers
normalize_ssh_arguments = _normalize_ssh_arguments
normalize_ssh_target = _normalize_ssh_target
parse_ssh_exec_args_from_shell_command = _parse_ssh_exec_args_from_shell_command
shell_join = _shell_join

from .shell_support import (
    InvalidInputLoopDetector,
    _apt_deb822_preflight_guard,
    _expose_interactive_session_tools,
    _foreground_command_guard,
    _interactive_installer_yes_pipe_guard,
    _installer_command_suggested_timeout,
    _mark_remote_installer_preflight_clean,
    _remote_installer_cwd_and_script,
    _remote_installer_preflight_guard,
)
from .ui_streaming import BufferedUIEventEmitter

if TYPE_CHECKING:
    from ..state import LoopState


_SSH_DIAGNOSTIC_NOT_FOUND_MARKERS = (
    "not found",
    "could not be found",
    "no such file",
    "command not found",
)
_SSH_TRANSPORT_FAILURE_MARKERS = (
    "permission denied",
    "connection timed out",
    "connection refused",
    "connection closed by remote host",
    "could not resolve hostname",
    "no route to host",
    "host key verification failed",
    "kex_exchange_identification",
    "connection reset by peer",
    "operation timed out",
    "network is unreachable",
)
_INTERACTIVE_PROMPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("choice", re.compile(r"Choice:\s*\[(?P<choices>[^\]]+)\]", re.IGNORECASE)),
    ("yes_no", re.compile(r"(?P<question>[^\n\r]{0,120}\?)\s*\(?(?P<default>[yYnN])/?[yYnN]\)?", re.IGNORECASE)),
    ("free_text", re.compile(r"(?P<question>[^\n\r]{0,240}:\s*)$", re.IGNORECASE)),
)
_SSH_INTERACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}
_SSH_ACCEPT_NEW_INCOMPATIBLE_MARKERS = (
    "keyword stricthostkeychecking extra arguments at end of line",
    "bad configuration option",
)

def _build_ssh_command(
    *,
    host: str,
    command: str,
    user: str | None,
    port: int,
    identity_file: str | None,
    password: str | None,
    strict_host_key_checking: str = "accept-new",
    force_tty: bool = False,
) -> tuple[str, dict[str, str] | None]:
    host, user = normalize_ssh_target(host=host, user=user)
    ssh_args = [
        "-p", str(port),
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", f"StrictHostKeyChecking={strict_host_key_checking}",
    ]
    env_overrides: dict[str, str] | None = None

    if password:
        if shutil.which("sshpass") is None:
            raise FileNotFoundError("sshpass")
        ssh_args.extend([
            "-o", "BatchMode=no",
            "-o", "PasswordAuthentication=yes",
            "-o", "PreferredAuthentications=password,keyboard-interactive",
            "-o", "PubkeyAuthentication=no",
            "-o", "NumberOfPasswordPrompts=1",
        ])
        command_args = ["sshpass", "-e", "ssh"]
        env_overrides = {"SSHPASS": password}
    else:
        ssh_args.extend(["-o", "BatchMode=yes"])
        command_args = ["ssh"]

    if identity_file:
        ssh_args.extend(["-i", identity_file])
    if force_tty:
        ssh_args.append("-tt")

    target = f"{user}@{host}" if user else host
    ssh_args.extend([target, command])
    return _shell_join([*command_args, *ssh_args]), env_overrides


def _detect_interactive_prompt(text: str) -> dict[str, Any] | None:
    tail = str(text or "")[-4096:]
    if not tail.strip():
        return None
    normalized_tail = tail.rstrip()
    for prompt_type, pattern in _INTERACTIVE_PROMPT_PATTERNS:
        match = pattern.search(normalized_tail)
        if not match:
            continue
        question = " ".join(str(match.group("question") or "").split())
        detected: dict[str, Any] = {
            "type": prompt_type,
            "question": question,
        }
        if prompt_type == "yes_no":
            marker = question[-5:].lower()
            if "/n" in marker:
                detected["default"] = "N"
            elif "/y" in marker:
                detected["default"] = "Y"
        return detected
    return None


def _interactive_session_snapshot(session_id: str, session: dict[str, Any], *, max_chars: int = 6000) -> dict[str, Any]:
    proc = session.get("proc")
    stdout = "".join(session.get("stdout", []))
    stderr = "".join(session.get("stderr", []))
    combined = stdout + stderr
    detected_prompt = _detect_interactive_prompt(combined)
    returncode = getattr(proc, "returncode", None)
    if returncode is not None:
        status = "exited"
    elif detected_prompt is not None:
        status = "waiting_for_input"
    else:
        status = "running"
    return {
        "session_id": session_id,
        "status": status,
        "detected_prompt": detected_prompt,
        "stdout_tail": stdout[-max_chars:],
        "stderr_tail": stderr[-max_chars:],
        "output_tail": combined[-max_chars:],
        "exit_code": returncode,
        "host": session.get("host"),
        "user": session.get("user"),
        "command": session.get("command"),
    }


async def _cleanup_interactive_session(
    session_id: str,
    session: dict[str, Any],
    *,
    harness: Any = None,
    terminate: bool = False,
) -> None:
    _SSH_INTERACTIVE_SESSIONS.pop(str(session_id or "").strip(), None)
    proc = session.get("proc")
    if terminate:
        await stop_process(proc, harness=harness, timeout=2.0)
    else:
        unregister_process(harness, proc)
    tasks = [task for task in session.get("tasks", []) if hasattr(task, "cancel") and hasattr(task, "done")]
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _ssh_accept_new_is_incompatible(stderr: str) -> bool:
    lowered = str(stderr or "").strip().lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in _SSH_ACCEPT_NEW_INCOMPATIBLE_MARKERS):
        return "stricthostkeychecking" in lowered or "accept-new" in lowered
    return "accept-new" in lowered and "unsupported option" in lowered


def _ssh_failure_kind(*, exit_code: int, stderr: str) -> str:
    lowered = str(stderr or "").strip().lower()
    if any(marker in lowered for marker in _SSH_TRANSPORT_FAILURE_MARKERS):
        return "transport"
    if exit_code == 255 and not lowered:
        return "transport"
    return "remote_command"


def _ssh_error_class(*, exit_code: int, stderr: str) -> str:
    lowered = str(stderr or "").strip().lower()
    if "permission denied" in lowered:
        return "auth_permission_denied"
    if "host key verification failed" in lowered or "remote host identification has changed" in lowered:
        return "host_key_verification"
    if "could not resolve hostname" in lowered or "name or service not known" in lowered or "temporary failure in name resolution" in lowered:
        return "dns_resolution"
    if "connection refused" in lowered:
        return "connection_refused"
    if "connection timed out" in lowered or "operation timed out" in lowered or "timed out" in lowered:
        return "connection_timeout"
    return "transport_failure" if _ssh_failure_kind(exit_code=exit_code, stderr=stderr) == "transport" else "remote_exit_nonzero"


def _ssh_diagnostic_not_found(command: str, output: dict[str, Any]) -> bool:
    """Return True when an exit-1 SSH result is a diagnostic 'not found' probe."""
    stderr = str(output.get("stderr") or "").lower()
    stdout = str(output.get("stdout") or "").lower()
    combined = stdout + " " + stderr
    return any(marker in combined for marker in _SSH_DIAGNOSTIC_NOT_FOUND_MARKERS)


def _ssh_execution_debug_metadata(
    *,
    password: str | None,
    identity_file: str | None,
    strict_host_key_checking: str,
) -> dict[str, Any]:
    password_text = str(password or "").strip()
    identity_file_text = str(identity_file or "").strip()
    return {
        "ssh_auth_mode": "password" if password_text else "key",
        "ssh_auth_transport": "sshpass_env" if password_text else "ssh",
        "ssh_password_provided": bool(password_text),
        "ssh_identity_file_supplied": bool(identity_file_text),
        "ssh_strict_host_key_checking": strict_host_key_checking,
    }


async def ssh_exec(
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    timeout_sec: int = 60,
    stdin_data: str | None = None,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """
    Execute a command on a remote host via SSH with live streaming support.
    """
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    if plan is not None and not getattr(plan, "approved", False):
        return fail(
            "SSH execution is blocked until the spec contract is approved.",
            metadata={
                "host": host,
                "command": command,
                "reason": "spec_not_approved",
                "plan_id": getattr(plan, "plan_id", ""),
            },
        )
    if state is not None and state.contract_phase() == "author":
        if not state.files_changed_this_cycle:
            return fail(
                "SSH execution is blocked until the authoring contract has produced a target artifact.",
                metadata={
                    "host": host,
                    "command": command,
                    "reason": "authoring_target_missing",
                    "contract_phase": state.contract_phase(),
                    "files_changed_this_cycle": state.files_changed_this_cycle,
                },
            )
    policy_state = state if state is not None else LoopState()
    approval_fn = getattr(harness, "request_shell_approval", None)
    approval_available = callable(approval_fn) and getattr(harness, "event_handler", None) is not None
    risk_decision = evaluate_risk_policy(
        policy_state,
        tool_name="ssh_exec",
        tool_risk="high",
        phase=str(policy_state.current_phase or ""),
        action=command,
        expected_effect="Run the requested SSH command on the remote host.",
        rollback="Stop the command and revert any in-progress remote changes if needed.",
        verification="Inspect the remote command output and any follow-up verifier result.",
        approval_available=approval_available,
    )
    if not risk_decision.allowed:
        return fail(
            risk_decision.reason,
            metadata={
                "host": host,
                "command": command,
                "reason": "missing_supported_claim",
                "proof_bundle": risk_decision.proof_bundle,
            },
        )
    approval_wait_sec = 0.0
    if risk_decision.requires_approval and callable(approval_fn) and approval_available:
        approval_start = time.monotonic()
        approved = await approval_fn(
            command=command,
            cwd=str(getattr(policy_state, "cwd", ".") or "."),
            timeout_sec=timeout_sec,
            proof_bundle=risk_decision.proof_bundle,
        )
        approval_wait_sec = time.monotonic() - approval_start
        if not approved:
            denied = fail(
                "SSH execution denied by user.",
                metadata={
                    "approval_denied": True,
                    "command": command,
                    "cwd": str(getattr(policy_state, "cwd", ".") or "."),
                    "timeout_sec": timeout_sec,
                    "host": host,
                },
            )
            denied["status"] = "denied"
            return denied

    result = await run_ssh_command(
        host=host,
        command=command,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        stdin_data=stdin_data,
        state=state,
        harness=harness,
    )
    if isinstance(result, dict) and isinstance(result.get("metadata"), dict):
        if approval_wait_sec > 0:
            result["metadata"]["approval_wait_sec"] = round(approval_wait_sec, 3)
    return result


async def ssh_session_start(
    *,
    host: str = "",
    target: str | None = None,
    command: str = "",
    user: str | None = None,
    username: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    timeout_sec: int = 900,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Start an interactive SSH command and keep stdin/stdout open."""
    if password and str(password).startswith("[REDACTED"):
        return fail(
            "The SSH password provided was literally redacted. Ask the human user to provide the actual password.",
            metadata={"host": host, "command": command, "reason": "redacted_password_provided"},
        )
    try:
        normalized = normalize_ssh_arguments(
            {
                "host": host,
                "target": target,
                "user": user,
                "username": username,
            }
        )
        host = str(normalized.get("host") or "")
        user = str(normalized.get("user") or "") or None
        host, user = normalize_ssh_target(host=host, user=user)
    except ValueError as exc:
        return fail(str(exc), metadata={"host": host, "user": user, "command": command})
    command, stripped_root_sudo = _strip_redundant_root_sudo(command, user)
    # ssh_session_start is the designated tool for interactive installers;
    # do not block it with the remote-installer preflight guard.

    proc = None
    try:
        full_cmd, env_overrides = _build_ssh_command(
            host=host,
            command=command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
            force_tty=True,
        )
    except FileNotFoundError as exc:
        if str(exc) == "sshpass":
            return fail(
                "sshpass is required for password-based SSH but was not found.",
                metadata={"host": host, "user": user, "command": command, "reason": "sshpass_missing"},
            )
        raise

    proc = await create_process(
        command=full_cmd,
        cwd=".",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        env_overrides=env_overrides,
        harness=harness,
    )
    session_id = f"sshint-{uuid.uuid4().hex[:12]}"
    session: dict[str, Any] = {
        "proc": proc,
        "stdout": [],
        "stderr": [],
        "host": host,
        "user": user,
        "command": command,
        "started_at": time.time(),
        "timeout_sec": max(1, int(timeout_sec or 900)),
        "harness": harness,
    }
    _SSH_INTERACTIVE_SESSIONS[session_id] = session

    async def collect(stream: Any, key: str) -> None:
        await read_stream_chunks(stream, session[key], chunk_size=4096, idle_timeout_sec=None)

    session["tasks"] = [
        asyncio.create_task(collect(proc.stdout, "stdout")),
        asyncio.create_task(collect(proc.stderr, "stderr")),
    ]
    await asyncio.sleep(0.2)
    _expose_interactive_session_tools(state)
    return ok(
        _interactive_session_snapshot(session_id, session),
        metadata={
            "interactive_session": True,
            "pty_requested": True,
            "next_tools": ["ssh_session_read", "ssh_session_send", "ssh_session_close"],
            **_ssh_execution_debug_metadata(
                password=password,
                identity_file=identity_file,
                strict_host_key_checking="accept-new",
            ),
        },
    )


async def ssh_session_read(
    *,
    session_id: str,
    wait_sec: float = 1.0,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    if snapshot.get("status") == "exited":
        await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
    return ok(snapshot, metadata={"interactive_session": True})


async def ssh_session_send(
    *,
    session_id: str,
    input: str,
    wait_sec: float = 0.5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    proc = session.get("proc")
    if getattr(proc, "returncode", None) is not None:
        snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
        await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
        return fail(
            "SSH interactive session has already exited.",
            metadata=snapshot,
        )
    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return fail("SSH interactive session stdin is unavailable.", metadata={"session_id": session_id})
    stdin.write(str(input or "").encode("utf-8"))
    await stdin.drain()
    await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
    return ok(_interactive_session_snapshot(session_id, session, max_chars=max_chars), metadata={"interactive_session": True})


async def ssh_session_close(
    *,
    session_id: str,
    terminate: bool = True,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=terminate)
    return ok(snapshot, metadata={"interactive_session": True})


async def _run_remote_installer_preflight_probes(
    *,
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Run automated SSH probes to discover installer environment state."""

    def _build_probe_command(probe_script: str) -> str:
        full_cmd, _env_overrides = _build_ssh_command(
            host=host,
            command=probe_script,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
        )
        return full_cmd

    probes = await run_installer_preflight_probes(
        command=command,
        state=state,
        create_process=create_process,
        harness=harness,
        host=host,
        user=user or "",
        build_probe_command=_build_probe_command,
    )
    # SSH-specific wording
    if probes.get("noninteractive_flags"):
        probes["recommended_approach"] = (
            f"This installer supports non-interactive mode. "
            f"Retry with `ssh_exec` and the flag `{probes['noninteractive_flags'][0]}`."
        )
    elif not probes.get("probe_error"):
        probes["recommended_approach"] = (
            "This appears to be an interactive installer. "
            "Use `ssh_session_start` to run it with a pseudo-terminal, "
            "then answer prompts with `ssh_session_send`."
        )
    return probes


async def run_ssh_command(
    *,
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
    stdin_data: str | None = None,
) -> dict[str, Any]:
    """Run a generated SSH command and return the same result shape as ssh_exec."""
    if password and str(password).startswith("[REDACTED"):
        return fail(
            "The SSH password provided was literally redacted. This means you have lost access to the real password due to security scrubbing. "
            "You MUST ask the human user to provide the actual password in plain text. Do NOT retry this command blindly.",
            metadata={
                "host": host,
                "command": command,
                "reason": "redacted_password_provided",
            },
        )
    strict_host_key_mode = "accept-new"
    try:
        host, user = normalize_ssh_target(host=host, user=user)
    except ValueError as exc:
        return fail(
            str(exc),
            metadata={
                "host": host,
                "command": command,
                "user": user,
                "reason": "invalid_ssh_target",
            },
        )
    command, stripped_root_sudo = _strip_redundant_root_sudo(command, user)
    requested_timeout_sec = timeout_sec
    timeout_sec = _installer_command_suggested_timeout(command, timeout_sec)

    yes_pipe_guard = _interactive_installer_yes_pipe_guard(command, tool_name="ssh_exec")
    if yes_pipe_guard is not None:
        metadata = dict(yes_pipe_guard.get("metadata") or {})
        metadata.update(
            {
                "host": host,
                "user": user,
                **_ssh_execution_debug_metadata(
                    password=password,
                    identity_file=identity_file,
                    strict_host_key_checking=strict_host_key_mode,
                ),
            }
        )
        yes_pipe_guard["metadata"] = metadata
        return yes_pipe_guard

    foreground_guard = _foreground_command_guard(command, tool_name="ssh_exec")
    if foreground_guard is not None:
        metadata = dict(foreground_guard.get("metadata") or {})
        if stripped_root_sudo:
            metadata["stripped_redundant_root_sudo"] = True
        metadata.update(
            {
                "host": host,
                "user": user,
                **_ssh_execution_debug_metadata(
                    password=password,
                    identity_file=identity_file,
                    strict_host_key_checking=strict_host_key_mode,
                ),
            }
        )
        foreground_guard["metadata"] = metadata
        return foreground_guard

    apt_guard = _apt_deb822_preflight_guard(command, tool_name="ssh_exec")
    if apt_guard is not None:
        metadata = dict(apt_guard.get("metadata") or {})
        metadata.update({"host": host, "user": user})
        apt_guard["metadata"] = metadata
        return apt_guard

    preflight_guard = None
    if stdin_data is None:
        preflight_guard = _remote_installer_preflight_guard(
            command,
            host=host,
            user=user,
            state=state,
        )
    if preflight_guard is not None:
        metadata = dict(preflight_guard.get("metadata") or {})
        # If the guard already detected a hard failure (missing/corrupt files),
        # preserve that actionable error instead of running probes.
        if metadata.get("reason") == "remote_installer_preflight_failed":
            if stripped_root_sudo:
                metadata["stripped_redundant_root_sudo"] = True
            metadata.update(
                {
                    "host": host,
                    "user": user,
                    **_ssh_execution_debug_metadata(
                        password=password,
                        identity_file=identity_file,
                        strict_host_key_checking=strict_host_key_mode,
                    ),
                }
            )
            preflight_guard["metadata"] = metadata
            return preflight_guard

        # Run automatic environment probes before returning the block
        probes = await _run_remote_installer_preflight_probes(
            host=host,
            command=command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
            state=state,
            harness=harness,
        )
        if stripped_root_sudo:
            metadata["stripped_redundant_root_sudo"] = True
        metadata.update(
            {
                "host": host,
                "user": user,
                "preflight_probes": probes,
                "suggested_tool_after_preflight": (
                    "ssh_session_start" if probes.get("is_interactive") else "ssh_exec"
                ),
                **_ssh_execution_debug_metadata(
                    password=password,
                    identity_file=identity_file,
                    strict_host_key_checking=strict_host_key_mode,
                ),
            }
        )

        # Auto-clear preflight if basic checks pass so the model can retry immediately
        if probes.get("script_exists") and probes.get("script_executable"):
            _mark_remote_installer_preflight_clean(
                state, host=host, user=user, cwd=probes.get("cwd", "")
            )

        # Expose interactive session tools when the installer requires interactivity
        if probes.get("is_interactive"):
            _expose_interactive_session_tools(state)

        # Build enriched, actionable error text
        parts: list[str] = []
        parts.append("Remote installer environment scan completed.")
        if probes.get("script_exists"):
            if probes.get("script_executable"):
                parts.append(
                    f"- Script: {probes['script_path']} (exists and is executable)"
                )
            else:
                parts.append(
                    f"- Script: {probes['script_path']} (exists but NOT executable)"
                )
        else:
            parts.append(f"- Script: {probes['script_path']} (NOT FOUND)")
            if probes.get("cwd"):
                parts.append(f"  Check the correct path in `{probes['cwd']}/` and retry.")

        if probes.get("repo_clean"):
            parts.append("- Git repo: clean")
        elif probes.get("cwd"):
            parts.append("- Git repo: dirty or not a git repository")

        if probes.get("noninteractive_flags"):
            parts.append(
                f"- Non-interactive flags detected: {', '.join(probes['noninteractive_flags'])}"
            )
        if probes.get("preseed_files"):
            parts.append(
                f"- Preseed/config files found: {', '.join(probes['preseed_files'])}"
            )

        parts.append(f"\n{probes['recommended_approach']}")

        if not probes.get("script_exists"):
            parts.append(
                "\nAction required: Verify the installer path before retrying."
            )

        preflight_guard["error"] = "\n".join(parts)
        preflight_guard["metadata"] = metadata
        return preflight_guard

    try:
        full_cmd, env_overrides = _build_ssh_command(
            host=host,
            command=command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
        )
    except FileNotFoundError as exc:
        if str(exc) == "sshpass":
            return fail(
                "Password authentication requires `sshpass`, but it is not installed.",
                metadata={
                    "host": host,
                    "command": command,
                    "user": user,
                    "reason": "sshpass_missing",
                    **_ssh_execution_debug_metadata(
                        password=password,
                        identity_file=identity_file,
                        strict_host_key_checking="accept-new",
                    ),
                },
            )
        raise

    execution_debug_metadata = _ssh_execution_debug_metadata(
        password=password,
        identity_file=identity_file,
        strict_host_key_checking=strict_host_key_mode,
    )
    if timeout_sec != requested_timeout_sec:
        execution_debug_metadata["timeout_sec_auto_extended"] = {
            "from": requested_timeout_sec,
            "to": timeout_sec,
            "reason": "installer_like_command",
        }
    if stripped_root_sudo:
        execution_debug_metadata["stripped_redundant_root_sudo"] = True

    last_process_output: dict[str, Any] | None = None
    invalid_input_loop: dict[str, Any] | None = None

    def _build_process_output(
        *,
        stdout: str,
        stderr: str,
        exit_code: int | None,
        elapsed: float,
    ) -> dict[str, Any]:
        final_stdout = truncate_output(stdout)
        final_stderr = truncate_output(stderr)
        return {
            "stdout": final_stdout,
            "stderr": final_stderr,
            "exit_code": exit_code,
            "metrics": {
                "duration_sec": round(elapsed, 3) if isinstance(elapsed, (int, float)) else 0.0,
                "host": host,
                "user": user,
            },
        }

    async def _run_ssh_process(command_text: str, stdin_payload: str | None = None) -> tuple[dict[str, Any], asyncio.subprocess.Process | None]:
        nonlocal last_process_output
        start_time = time.time()
        proc = await create_process(
            command=command_text,
            cwd=state.cwd if state else ".",
            env_overrides=env_overrides,
            harness=harness,
            stdin=asyncio.subprocess.PIPE if stdin_payload is not None else asyncio.subprocess.DEVNULL,
        )
        if stdin_payload is not None and proc.stdin is not None:
            proc.stdin.write(stdin_payload.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

        stdout_data: list[str] = []
        stderr_data: list[str] = []
        invalid_input_detector = InvalidInputLoopDetector()
        stream_emitter = BufferedUIEventEmitter(
            harness=harness,
            event_type=UIEventType.SHELL_STREAM,
        )

        async def read_stream(stream: Any, out_list: list[str]) -> None:
            async def handle_chunk(chunk_str: str) -> None:
                nonlocal invalid_input_loop
                if invalid_input_loop is None:
                    loop_metadata = invalid_input_detector.observe(chunk_str)
                    if loop_metadata is not None:
                        invalid_input_loop = {
                            **loop_metadata,
                            "command": command,
                            "tool_name": "ssh_exec",
                            "host": host,
                            "user": user,
                        }
                        if proc.returncode is None:
                            try:
                                proc.terminate()
                            except Exception:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                await stream_emitter.emit_text(chunk_str)

            await read_stream_chunks(stream, out_list, chunk_size=4096, on_chunk=handle_chunk, idle_timeout_sec=30)

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(proc.stdout, stdout_data),
                    read_stream(proc.stderr, stderr_data),
                    proc.wait(),
                ),
                timeout=timeout_sec,
            )
        finally:
            elapsed = time.time() - start_time
            last_process_output = _build_process_output(
                stdout="".join(stdout_data),
                stderr="".join(stderr_data),
                exit_code=proc.returncode,
                elapsed=elapsed,
            )
            await stream_emitter.flush()
            unregister_process(harness, proc)

        return last_process_output, proc

    proc = None
    retry_metadata: dict[str, Any] = {}
    max_auth_retries = 3
    auth_retry_count = 0

    try:
        for auth_attempt in range(max_auth_retries):
            output, proc = await _run_ssh_process(full_cmd, stdin_data)
            stderr_text = str(output.get("stderr") or "")
            # Retry on auth failures with exponential backoff
            if (
                int(output.get("exit_code") or 0) != 0
                and "permission denied" in stderr_text.lower()
                and auth_attempt < max_auth_retries - 1
            ):
                auth_retry_count += 1
                wait_sec = 2 ** auth_attempt
                if harness and hasattr(harness, "emit"):
                    await harness.emit(
                        "ssh_auth_retry",
                        f"SSH auth failed (attempt {auth_attempt + 1}/{max_auth_retries}), retrying in {wait_sec}s...",
                        data={"attempt": auth_attempt + 1, "wait_sec": wait_sec},
                    )
                await asyncio.sleep(wait_sec)
                continue
            break

        if int(output.get("exit_code") or 0) != 0 and _ssh_accept_new_is_incompatible(str(output.get("stderr") or "")):
            strict_host_key_mode = "no"
            full_cmd, env_overrides = _build_ssh_command(
                host=host,
                command=command,
                user=user,
                port=port,
                identity_file=identity_file,
                password=password,
                strict_host_key_checking=strict_host_key_mode,
            )
            execution_debug_metadata = _ssh_execution_debug_metadata(
                password=password,
                identity_file=identity_file,
                strict_host_key_checking=strict_host_key_mode,
            )
            if stripped_root_sudo:
                execution_debug_metadata["stripped_redundant_root_sudo"] = True
            output, proc = await _run_ssh_process(full_cmd, stdin_data)
            retry_metadata = {
                "ssh_option_retry": "strict_host_key_checking_no",
                "ssh_option_retry_reason": "accept_new_incompatible",
            }

        if auth_retry_count > 0:
            retry_metadata["ssh_auth_retries"] = auth_retry_count

        if invalid_input_loop is not None:
            return fail(
                "SSH command stopped after repeated invalid interactive input. "
                "Use documented non-interactive flags, a config/preseed file, or an explicit prompt answer script.",
                metadata={
                    "output": output,
                    "output_received": bool(
                        str(output.get("stdout") or "").strip()
                        or str(output.get("stderr") or "").strip()
                    ),
                    "failure_kind": "interactive_input_loop",
                    "ssh_error_class": "interactive_invalid_input_loop",
                    "ssh_transport_succeeded": True,
                    **execution_debug_metadata,
                    **retry_metadata,
                    **invalid_input_loop,
                },
            )

        if proc.returncode != 0:
            err_output = output.get("stderr", "")
            if not isinstance(err_output, str):
                err_output = str(err_output or "")
            # Diagnostic probes that explicitly report "not found" are informational
            # successes, not execution failures. Treating them as ok prevents the
            # harness from entering a repair loop over a negative result.
            if (
                int(proc.returncode) == 1
                and _ssh_diagnostic_not_found(command, output)
            ):
                return ok(output, metadata={**execution_debug_metadata, **retry_metadata})
            failure_kind = _ssh_failure_kind(
                exit_code=int(proc.returncode),
                stderr=err_output,
            )
            ssh_error_class = _ssh_error_class(
                exit_code=int(proc.returncode),
                stderr=err_output,
            )
            hints = []
            if failure_kind == "transport":
                error_msg = err_output.strip() or f"SSH transport failed with exit code {proc.returncode}"
            else:
                error_msg = err_output.strip() or f"Remote SSH command exited with code {proc.returncode}"
                hints.append(
                    "SSH transport appears to have succeeded; inspect the remote command, stdout, and exit code to decide whether the probe simply returned a non-zero status."
                )
            if "Permission denied" in error_msg:
                if password:
                    hints.append("Check the SSH username/password and verify that password authentication is enabled on the remote host.")
                else:
                    hints.append("Check if SSH keys are correctly configured on the remote host.")
            if "Connection timed out" in error_msg:
                hints.append("Verify the host is reachable and the port is open.")

            return fail(
                error_msg,
                metadata={
                    "output": output,
                    "output_received": bool(
                        str(output.get("stdout") or "").strip()
                        or str(output.get("stderr") or "").strip()
                    ),
                    "hints": hints,
                    "failure_kind": failure_kind,
                    "failure_mode": ssh_error_class,
                    "ssh_error_class": ssh_error_class,
                    "ssh_transport_succeeded": failure_kind == "remote_command",
                    **execution_debug_metadata,
                    **retry_metadata,
                },
            )

        return ok(output, metadata={**execution_debug_metadata, **retry_metadata})

    except asyncio.TimeoutError:
        await stop_process(proc, harness=harness, timeout=2.0)
        output = last_process_output if isinstance(last_process_output, dict) else {}
        combined_output = f"{output.get('stdout', '')}{output.get('stderr', '')}"
        detected_prompt = _detect_interactive_prompt(combined_output)
        if detected_prompt is not None:
            return fail(
                "SSH command appears to be waiting for interactive input. "
                "Use documented non-interactive flags/config when available, or retry with `ssh_session_start` and answer prompts with `ssh_session_send`.",
                metadata={
                    "output": output,
                    "output_received": True,
                    "failure_kind": "interactive_prompt_wait",
                    "ssh_error_class": "interactive_prompt_wait",
                    "ssh_transport_succeeded": True,
                    "detected_prompt": detected_prompt,
                    "suggested_tools": ["ssh_session_start", "ssh_session_read", "ssh_session_send", "ssh_session_close"],
                    **execution_debug_metadata,
                },
            )
        return fail(
            f"SSH command timed out after {timeout_sec}s",
            metadata={
                "output": output,
                "output_received": bool(
                    str(output.get("stdout") or "").strip()
                    or str(output.get("stderr") or "").strip()
                ),
                "failure_kind": "timeout",
                "ssh_error_class": "command_timeout",
                "ssh_transport_succeeded": bool(
                    str(output.get("stdout") or "").strip()
                    or str(output.get("stderr") or "").strip()
                ),
                **execution_debug_metadata,
            },
        )
    except Exception as exc:
        return fail(
            f"SSH execution error: {str(exc)}",
            metadata=execution_debug_metadata,
        )
    finally:
        unregister_process(harness, proc)
