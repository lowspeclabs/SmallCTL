from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

log = logging.getLogger("smallctl.tools.shell_foreground")

from ..models.events import UIEvent, UIEventType
from ..risk_policy import evaluate_risk_policy
from ..state import LoopState
from .common import fail, needs_human, ok
from .process_lifecycle import build_process_output, cancel_tasks, stop_process, unregister_process
from .process_streams import read_stream_chunks
from .shell_sudo import SUDO_PROMPT_PATTERNS, ensure_sudo_credentials
from .shell_support import (
    InvalidInputLoopDetector,
    _build_argparse_missing_args_question,
    _build_argparse_unrecognized_args_hint,
    _build_shell_status_update,
    _command_requires_shell,
    _detect_unsupported_shell_syntax,
    _extract_missing_argparse_arguments,
    _extract_unrecognized_argparse_arguments,
    _interactive_installer_yes_pipe_guard,
    _shell_execution_authoring_guard,
    _shell_status_update_interval,
    _shell_workspace_relative_hint,
)
from .ui_streaming import BufferedUIEventEmitter


_SAFE_COMPILE_LINT_COMMANDS = {
    "python3 -m py_compile",
    "flake8",
    "mypy",
    "ruff check",
    "shellcheck",
}


def _is_safe_compile_lint_command(command: str) -> bool:
    """Return True if the command is a read-only compile/lint verifier that can be auto-approved."""
    segments = [s.strip() for s in re.split(r"&&|\|\||[;&|]", str(command or ""))]
    for segment in segments:
        if segment.startswith("cd "):
            continue
        for prefix in _SAFE_COMPILE_LINT_COMMANDS:
            if segment.startswith(prefix):
                return True
    return False


def _leading_command_tokens(command: str, *, max_depth: int = 3) -> list[str]:
    from .. import shell_utils as _shell_attempts

    return _shell_attempts.leading_command_tokens(command, max_depth=max_depth)


def _command_uses_leading_sudo(command: str) -> bool:
    tokens = _leading_command_tokens(command)
    return bool(tokens) and tokens[0].lower() == "sudo"


def _command_may_need_sudo(command: str) -> bool:
    """Return True if the command is likely to prompt for a sudo password."""
    if _command_uses_leading_sudo(command):
        return True
    lowered = str(command or "").lower()
    # Match common sudo invocations inside a shell pipeline, but avoid false
    # positives from words that merely contain "sudo".
    return bool(re.search(r"(^|[\s;&|({\[])sudo([\s-]|$)", lowered))


def _process_runs_under_pty(proc: Any) -> bool:
    """Best-effort check whether a process is attached to a PTY."""
    explicit = getattr(proc, "is_running_under_pty", None)
    if explicit is not None:
        return bool(explicit)
    stdout = getattr(proc, "stdout", None)
    if stdout is not None:
        isatty = getattr(stdout, "isatty", None)
        if callable(isatty):
            try:
                return bool(isatty())
            except Exception:
                pass
        fileno = getattr(stdout, "fileno", None)
        if callable(fileno):
            try:
                import os as _os

                return _os.isatty(fileno())
            except Exception:
                pass
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        get_extra = getattr(transport, "get_extra_info", None)
        if callable(get_extra):
            try:
                return bool(get_extra("pty"))
            except Exception:
                pass
    return False


def _process_child_command_basename(proc: Any) -> str | None:
    """Best-effort basename of the actual child process executable."""
    explicit = getattr(proc, "child_command_basename", None)
    if explicit:
        return str(explicit).lower()
    argv = getattr(proc, "argv", None)
    if isinstance(argv, (list, tuple)) and argv:
        return Path(str(argv[0])).name.lower()
    command = getattr(proc, "command", None)
    if isinstance(command, str) and command:
        parts = command.strip().split()
        if parts:
            return Path(parts[0]).name.lower()
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            import os as _os

            cmdline_path = f"/proc/{pid}/cmdline"
            if _os.path.exists(cmdline_path):
                with open(cmdline_path, "rb") as fh:
                    data = fh.read()
                if data:
                    first = data.split(b"\x00")[0].decode("utf-8", errors="replace")
                    return Path(first).name.lower()
        except Exception:
            pass
    return None


def _verify_sudo_process(proc: Any, command: str) -> bool:
    """Return True if the receiving process is verified to be sudo.

    Feeding a configured password is allowed when the process is running under
    a PTY (so the user can see the interactive prompt) or when the actual child
    executable is sudo.  If process metadata cannot be inspected (e.g. tests or
    restricted environments), fall back to the leading command token already
    vetted by :func:`_command_may_need_sudo`.
    """
    if _process_runs_under_pty(proc):
        return True
    basename = _process_child_command_basename(proc)
    if basename == "sudo":
        return True
    if basename is not None:
        return False
    return _command_uses_leading_sudo(command)


def _classify_shell_failure(command: str, error: str, output: dict[str, Any]) -> dict[str, Any]:
    lowered_error = str(error or "").lower()
    lowered_command = str(command or "").lower()
    stdout = str(output.get("stdout") or "") if isinstance(output, dict) else ""
    stderr = str(output.get("stderr") or "") if isinstance(output, dict) else ""
    combined = "\n".join(part for part in (stdout, stderr, str(error or "")) if part).lower()

    if "connection refused" in combined or "errno 111" in combined:
        return {
            "failure_class": "environment_unavailable",
            "reason": "connection_refused",
            "suggested_next_action": "Confirm the target service is running and listening before retrying the command.",
        }
    if "not found" in lowered_error and re.search(r"(^|[/\s])(?:netstat|ss|lsof)(:|\s|$)", lowered_error):
        return {
            "failure_class": "diagnostic_tool_unavailable",
            "reason": "port_probe_tool_missing",
            "suggested_next_action": "Use an installed port probe such as `ss`, `lsof`, or a Python socket check.",
        }
    if re.search(r"\b(?:ss|lsof|netstat)\b", lowered_command) and re.search(r"\|\s*grep\s+[:]?\d+", lowered_command):
        exit_code = output.get("exit_code") if isinstance(output, dict) else None
        if exit_code == 1 and not stdout.strip() and not stderr.strip():
            return {
                "failure_class": "environment_unavailable",
                "reason": "service_not_listening",
                "suggested_next_action": "The port probe found no listener; start the service or ask the user for the correct endpoint.",
            }
    return {}


async def _feed_sudo_password_to_process(
    proc: Any,
    command: str,
    harness: Any,
) -> dict[str, Any] | None:
    """Feed a configured sudo password to a running process.

    Returns None when a password was written, or a needs_human/fail envelope
    when no password is available.  The caller is responsible for stopping the
    process if this returns a guard envelope.
    """
    if not _command_may_need_sudo(command):
        return fail(
            "Detected a password prompt but the command is not known to use sudo. "
            "Refusing to feed a password to an unverified process.",
            metadata={"command": command, "reason": "sudo_prompt_unexpected"},
        )

    password_fn = getattr(harness, "get_sudo_password", None)
    password: str | None = None
    if callable(password_fn):
        password = password_fn(command=command)
    else:
        store = getattr(harness, "credential_store", None)
        if store is not None:
            password = store.get_sudo_password()
    if not isinstance(password, str) or not password.strip():
        return needs_human(
            f"Command requires sudo/password input: '{command}'. "
            "Set a sudo password in config or configure passwordless sudo.",
            metadata={"command": command, "reason": "sudo_password_required"},
        )

    if not _verify_sudo_process(proc, command):
        return fail(
            "Detected a password prompt but the receiving process is not verified to be sudo. "
            "Enter the password interactively.",
            metadata={"command": command, "reason": "sudo_prompt_unexpected"},
        )

    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return fail(
            "Process has no stdin pipe; cannot feed sudo password.",
            metadata={"command": command, "reason": "sudo_stdin_unavailable"},
        )
    try:
        stdin.write(f"{password}\n".encode("utf-8"))
        await stdin.drain()
        stdin.close()
    except Exception as exc:
        return fail(
            f"Failed to write sudo password to process: {exc}",
            metadata={"command": command, "reason": "sudo_password_write_failed"},
        )
    return None


async def shell_exec_foreground(
    command: str,
    *,
    state: LoopState,
    timeout_sec: int,
    harness: Any = None,
    create_process,
) -> dict[str, Any]:
    approval_wait_sec = 0.0
    execution_sec = 0.0

    async def _run():
        nonlocal approval_wait_sec, execution_sec
        proc = None
        password_prompt_detected = False
        progress_updates: list[str] = []
        start_time = 0.0
        try:
            sudo_human_message = (
                "Sudo execution requires a password. If interactive prompts are unavailable, "
                "configure passwordless sudo or ask the user for help."
            )
            approval_fn = getattr(harness, "request_shell_approval", None)
            approval_available = callable(approval_fn) and getattr(harness, "event_handler", None) is not None
            risk_decision = evaluate_risk_policy(
                state,
                tool_name="shell_exec",
                tool_risk="high",
                phase=str(state.current_phase or ""),
                action=command,
                expected_effect="Run the requested shell command.",
                rollback="Stop the command and undo any manual changes if needed.",
                verification="Inspect the command output and follow-up verifier result.",
                approval_available=approval_available,
            )
            if not risk_decision.allowed:
                return fail(
                    risk_decision.reason,
                    metadata={
                        "command": command,
                        "reason": "missing_supported_claim",
                        "proof_bundle": risk_decision.proof_bundle,
                    },
                )
            authoring_guard = _shell_execution_authoring_guard(state, command)
            if authoring_guard is not None:
                return authoring_guard
            yes_pipe_guard = _interactive_installer_yes_pipe_guard(command, tool_name="shell_exec")
            if yes_pipe_guard is not None:
                return yes_pipe_guard
            unsupported_shell_message = _detect_unsupported_shell_syntax(command)
            if unsupported_shell_message:
                return needs_human(
                    unsupported_shell_message,
                    metadata={"command": command, "reason": "unsupported_shell_syntax"},
                )
            if risk_decision.requires_approval and callable(approval_fn) and approval_available:
                approval_start = time.monotonic()
                if _is_safe_compile_lint_command(command):
                    approved = True
                else:
                    approved = await approval_fn(
                        command=command,
                        cwd=state.cwd,
                        timeout_sec=timeout_sec,
                        proof_bundle=risk_decision.proof_bundle,
                    )
                if not _is_safe_compile_lint_command(command):
                    approval_wait_sec = time.monotonic() - approval_start
                if not approved:
                    denied = fail(
                        "Shell execution denied by user.",
                        metadata={
                            "approval_denied": True,
                            "command": command,
                            "cwd": state.cwd,
                            "timeout_sec": timeout_sec,
                        },
                    )
                    denied["status"] = "denied"
                    return denied

            sudo_password_configured = False
            get_sudo_password_fn = getattr(harness, "get_sudo_password", None)
            if callable(get_sudo_password_fn):
                sudo_password_configured = bool(get_sudo_password_fn(command=command))

            # Only run the pre-flight sudo validation prompt when no configured
            # password is available.  When a password is configured we feed it
            # inline as the process runs, avoiding the TUI password prompt.
            if not sudo_password_configured or not _command_may_need_sudo(command):
                sudo_guard = await ensure_sudo_credentials(
                    command=command,
                    cwd=state.cwd,
                    is_leading_sudo=_command_uses_leading_sudo(command),
                    create_process=create_process,
                    harness=harness,
                    timeout_sec=timeout_sec,
                    sudo_human_message=sudo_human_message,
                )
                if sudo_guard is not None:
                    return sudo_guard

            stdin_setting = asyncio.subprocess.PIPE if _command_may_need_sudo(command) else asyncio.subprocess.DEVNULL
            proc = await create_process(
                command=command, cwd=state.cwd, harness=harness, stdin=stdin_setting, shell=_command_requires_shell(command)
            )

            stdout_data = []
            stderr_data = []
            detection_buffer = ""
            invalid_input_loop: dict[str, Any] | None = None
            invalid_input_detector = InvalidInputLoopDetector()
            heartbeat_interval = _shell_status_update_interval(timeout_sec)
            start_time = time.monotonic()
            sudo_password_fed = False
            sudo_password_unavailable: dict[str, Any] | None = None
            sudo_feed_lock = asyncio.Lock()
            stream_emitter = BufferedUIEventEmitter(
                harness=harness,
                event_type=UIEventType.SHELL_STREAM,
                event_data=_active_tool_event_data(harness, fallback_tool_name="shell_exec"),
            )

            async def read_stream(stream, out_list, is_stderr: bool = False):
                nonlocal password_prompt_detected, detection_buffer, invalid_input_loop, sudo_password_fed, sudo_password_unavailable

                async def handle_chunk(chunk_str: str) -> None:
                    nonlocal password_prompt_detected, detection_buffer, invalid_input_loop, sudo_password_fed, sudo_password_unavailable

                    if not password_prompt_detected:
                        detection_buffer += chunk_str
                        if len(detection_buffer) > 4096:
                            detection_buffer = detection_buffer[-2048:]

                        for pattern in SUDO_PROMPT_PATTERNS:
                            if pattern.search(detection_buffer):
                                password_prompt_detected = True
                                async with sudo_feed_lock:
                                    if not sudo_password_fed and sudo_password_unavailable is None:
                                        sudo_password_unavailable = await _feed_sudo_password_to_process(
                                            proc, command, harness
                                        )
                                        if sudo_password_unavailable is None:
                                            sudo_password_fed = True
                                break
                    if invalid_input_loop is None:
                        loop_metadata = invalid_input_detector.observe(chunk_str)
                        if loop_metadata is not None:
                            invalid_input_loop = {
                                **loop_metadata,
                                "command": command,
                                "tool_name": "shell_exec",
                            }
                            if proc and proc.returncode is None:
                                try:
                                    proc.kill()
                                except (OSError, ProcessLookupError) as exc:
                                    log.debug("failed to kill process for invalid-input loop: %s", exc)
                                except Exception as exc:
                                    log.warning("unexpected error killing process: %s", exc)

                    await stream_emitter.emit_text(chunk_str)

                await read_stream_chunks(stream, out_list, chunk_size=2048, on_chunk=handle_chunk)

            async def emit_status_update() -> None:
                elapsed_sec = time.monotonic() - start_time
                status_text = _build_shell_status_update(
                    command,
                    elapsed_sec=elapsed_sec,
                    timeout_sec=timeout_sec,
                )
                progress_updates.append(status_text)
                await stream_emitter.emit_event(
                    UIEvent(event_type=UIEventType.SHELL_STREAM, content=status_text)
                )

            if hasattr(proc, "stdout") and hasattr(proc, "stderr") and hasattr(proc.stdout, "read"):
                stdout_task = asyncio.create_task(read_stream(proc.stdout, stdout_data, is_stderr=False))
                stderr_task = asyncio.create_task(read_stream(proc.stderr, stderr_data, is_stderr=True))
                wait_task = asyncio.create_task(proc.wait())
                deadline = start_time + max(1, timeout_sec)
                timed_out = False

                try:
                    while True:
                        remaining = max(0.0, deadline - time.monotonic())
                        if remaining <= 0:
                            timed_out = True
                            break
                        wait_window = min(heartbeat_interval, remaining)
                        try:
                            await asyncio.wait_for(asyncio.shield(wait_task), timeout=wait_window)
                            break
                        except asyncio.TimeoutError:
                            await emit_status_update()
                            continue
                finally:
                    if timed_out and proc and proc.returncode is None:
                        try:
                            proc.kill()
                        except (OSError, ProcessLookupError) as exc:
                            log.debug("failed to kill timed-out process: %s", exc)
                        except Exception as exc:
                            log.warning("unexpected error killing timed-out process: %s", exc)
                    await stream_emitter.flush()
                    try:
                        await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task), timeout=1.0)
                    except Exception:
                        await cancel_tasks([stdout_task, stderr_task])
                    if not wait_task.done():
                        try:
                            await asyncio.wait_for(wait_task, timeout=1.0)
                        except Exception:
                            await cancel_tasks([wait_task])
                if timed_out:
                    raise asyncio.TimeoutError
                output = build_process_output(
                    stdout="".join(stdout_data),
                    stderr="".join(stderr_data),
                    exit_code=proc.returncode,
                )
                if progress_updates:
                    output["progress_updates"] = progress_updates
            else:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
                output = build_process_output(
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    exit_code=proc.returncode,
                )
                stdout_txt = output.get("stdout", "") if isinstance(output.get("stdout"), str) else ""
                stderr_txt = output.get("stderr", "") if isinstance(output.get("stderr"), str) else ""
                msg = stdout_txt + stderr_txt
                if msg:
                    await stream_emitter.emit_event(
                        UIEvent(event_type=UIEventType.SHELL_STREAM, content=msg)
                    )
                if progress_updates:
                    output["progress_updates"] = progress_updates

            execution_sec = time.monotonic() - start_time

            if invalid_input_loop is not None:
                return fail(
                    "Command stopped after repeated invalid interactive input. "
                    "Use documented non-interactive flags, a config/preseed file, or an explicit prompt answer script.",
                    metadata={"output": output, **invalid_input_loop},
                )

            if password_prompt_detected:
                if sudo_password_unavailable is not None:
                    if proc and proc.returncode is None:
                        try:
                            proc.kill()
                            await asyncio.wait_for(proc.wait(), timeout=1.0)
                        except Exception:
                            pass
                    return sudo_password_unavailable
                if not sudo_password_fed:
                    if proc and proc.returncode is None:
                        try:
                            proc.kill()
                            await asyncio.wait_for(proc.wait(), timeout=1.0)
                        except Exception:
                            pass
                    return needs_human(
                        f"Command requires sudo/password input: '{command}'. {sudo_human_message}",
                        metadata={"output": output, "command": command, "reason": "password_prompt_detected"},
                    )

            if proc.returncode not in (0, None):
                err_output = output.get("stderr", "")
                if not isinstance(err_output, str):
                    err_output = str(err_output or "")

                if not err_output.strip():
                    std_output = output.get("stdout", "")
                    if isinstance(std_output, str) and std_output.strip():
                        err_output = std_output

                error = err_output.strip() or f"Command exited with code {proc.returncode}"

                sudo_prompts = (
                    "sudo: a password is required",
                    "[sudo] password for",
                    "password:",
                    "Password:",
                    "sudo: no tty present",
                    "sudo: a terminal is required",
                    "sudo: must be run from a terminal",
                )
                if any(p.lower() in error.lower() for p in sudo_prompts):
                    if sudo_password_fed:
                        return fail(
                            "Sudo password was provided but the command still failed. "
                            "The password may be incorrect or the user is not in the sudoers file.",
                            metadata={"output": output, "command": command, "reason": "sudo_password_rejected"},
                        )
                    return needs_human(
                        sudo_human_message,
                        metadata={"output": output, "command": command},
                    )

                missing_args = _extract_missing_argparse_arguments(error)
                if missing_args:
                    question = _build_argparse_missing_args_question(command, missing_args)
                    return needs_human(
                        question,
                        metadata={
                            "output": output,
                            "command": command,
                            "reason": "missing_required_arguments",
                            "missing_arguments": missing_args,
                        },
                    )

                unrecognized_args = _extract_unrecognized_argparse_arguments(error)
                if unrecognized_args:
                    hint = _build_argparse_unrecognized_args_hint(command, unrecognized_args)
                    if hint:
                        return needs_human(
                            hint,
                            metadata={
                                "output": output,
                                "command": command,
                                "reason": "unrecognized_arguments",
                                "unrecognized_arguments": unrecognized_args,
                            },
                        )

                path_hint = _shell_workspace_relative_hint(command, cwd=state.cwd)
                if path_hint and any(
                    token in error.lower()
                    for token in (
                        "no such file",
                        "can't open file",
                        "cannot open file",
                        "file not found",
                    )
                ):
                    error = f"{error}\n\n{path_hint}"

                failure_metadata = {"output": output, "command": command}
                failure_metadata.update(_classify_shell_failure(command, error, output))
                return fail(error, metadata=failure_metadata)
            return ok(output)
        except asyncio.TimeoutError:
            if start_time:
                execution_sec = time.monotonic() - start_time
            await stop_process(proc, harness=harness, timeout=1.0)
            if password_prompt_detected:
                if sudo_password_unavailable is not None:
                    return sudo_password_unavailable
                if sudo_password_fed:
                    return fail(
                        f"Command timed out after {timeout_sec}s. "
                        "A sudo password was provided but the command did not complete.",
                        metadata={"command": command, "reason": "sudo_password_timeout"},
                    )
                return needs_human(
                    f"Command timed out waiting for sudo/password input: '{command}'. {sudo_human_message}",
                    metadata={"command": command, "reason": "password_prompt_timeout"},
                )
            metadata: dict[str, Any] = {"command": command}
            if progress_updates:
                metadata["progress_updates"] = progress_updates
            return fail(f"Command timed out after {timeout_sec}s", metadata=metadata)
        except asyncio.CancelledError:
            if start_time:
                execution_sec = time.monotonic() - start_time
            await stop_process(proc, harness=harness, timeout=1.0)
            raise
        except Exception as exc:
            return fail(str(exc))
        finally:
            unregister_process(harness, proc)

        return fail("Unknown shell execution error")
    result = await _run()
    if isinstance(result, dict) and isinstance(result.get("metadata"), dict):
        if approval_wait_sec > 0:
            result["metadata"]["approval_wait_sec"] = round(approval_wait_sec, 3)
        if execution_sec > 0:
            result["metadata"]["execution_sec"] = round(execution_sec, 3)
    return result


def _active_tool_event_data(harness: Any, *, fallback_tool_name: str) -> dict[str, Any]:
    context = getattr(harness, "_active_ui_tool_context", None)
    if not isinstance(context, dict):
        return {"tool_name": fallback_tool_name}
    data = dict(context)
    data.setdefault("tool_name", fallback_tool_name)
    return data
