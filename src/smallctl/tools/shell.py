from __future__ import annotations

import asyncio
import asyncio.subprocess
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .. import shell_utils as _shell_attempts
from ..models.events import UIEvent, UIEventType
from ..state import LoopState
from ..risk_policy import evaluate_risk_policy
from .common import fail, needs_human, ok
from .process_streams import read_stream_chunks
from .shell_processes import cwd_get, cwd_set, env_get, env_set, process_kill, shell_job_launch, shell_job_status
from .shell_foreground import shell_exec_foreground, _command_uses_leading_sudo
from .shell_sudo import SUDO_PROMPT_PATTERNS, ensure_sudo_credentials
from .shell_support import (
    _build_argparse_missing_args_question,
    _build_shell_status_update,
    _detect_unsupported_shell_syntax,
    _expose_interactive_session_tools,
    _extract_missing_argparse_arguments,
    _foreground_command_guard,
    _interactive_installer_yes_pipe_guard,
    _mark_remote_installer_preflight_clean,
    _remote_installer_cwd_and_script,
    _remote_installer_preflight_guard,
    _shell_execution_authoring_guard,
    _shell_status_update_interval,
    _shell_write_session_artifact_delete_guard,
    _shell_write_session_target_path_guard,
    _shell_workspace_relative_hint,
)
from .ui_streaming import BufferedUIEventEmitter


async def _shell_exec_foreground(
    command: str,
    *,
    state: LoopState,
    timeout_sec: int,
    harness: Any = None,
) -> dict[str, Any]:
    proc = None
    password_prompt_detected = False
    progress_updates: list[str] = []
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
        unsupported_shell_message = _detect_unsupported_shell_syntax(command)
        if unsupported_shell_message:
            return needs_human(
                unsupported_shell_message,
                metadata={"command": command, "reason": "unsupported_shell_syntax"},
            )
        if risk_decision.requires_approval and callable(approval_fn) and approval_available:
            approved = await approval_fn(
                command=command,
                cwd=state.cwd,
                timeout_sec=timeout_sec,
                proof_bundle=risk_decision.proof_bundle,
            )
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

        sudo_guard = await ensure_sudo_credentials(
            command=command,
            cwd=state.cwd,
            is_leading_sudo=_command_uses_leading_sudo(command),
            create_process=_create_process,
            harness=harness,
            timeout_sec=timeout_sec,
            sudo_human_message=sudo_human_message,
        )
        if sudo_guard is not None:
            return sudo_guard

        proc = await _create_process(command=command, cwd=state.cwd, harness=harness)

        stdout_data = []
        stderr_data = []
        detection_buffer = ""  # Buffer for pattern detection
        heartbeat_interval = _shell_status_update_interval(timeout_sec)
        start_time = time.monotonic()
        stream_emitter = BufferedUIEventEmitter(
            harness=harness,
            event_type=UIEventType.SHELL_STREAM,
        )

        async def read_stream(stream, out_list, is_stderr: bool = False):
            nonlocal password_prompt_detected, detection_buffer

            async def handle_chunk(chunk_str: str) -> None:
                nonlocal password_prompt_detected, detection_buffer

                # Check for password prompt in output (check both this chunk and buffer)
                if not password_prompt_detected:
                    detection_buffer += chunk_str
                    # Keep buffer size manageable
                    if len(detection_buffer) > 4096:
                        detection_buffer = detection_buffer[-2048:]

                    for pattern in SUDO_PROMPT_PATTERNS:
                        if pattern.search(detection_buffer):
                            password_prompt_detected = True
                            break

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
                    except Exception:
                        pass
                await stream_emitter.flush()
                try:
                    await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task), timeout=1.0)
                except Exception:
                    for task in (stdout_task, stderr_task):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                if not wait_task.done():
                    try:
                        await asyncio.wait_for(wait_task, timeout=1.0)
                    except Exception:
                        wait_task.cancel()
                        await asyncio.gather(wait_task, return_exceptions=True)
            if timed_out:
                raise asyncio.TimeoutError
            final_stdout = "".join(stdout_data)
            final_stderr = "".join(stderr_data)
            
            # Final safety cap for the tool result itself
            MAX_FINAL_RESULT = 256 * 1024
            if len(final_stdout) > MAX_FINAL_RESULT:
                final_stdout = final_stdout[:MAX_FINAL_RESULT] + "\n[OUTPUT TRUNCATED - TOO LARGE]"
            if len(final_stderr) > MAX_FINAL_RESULT:
                final_stderr = final_stderr[:MAX_FINAL_RESULT] + "\n[OUTPUT TRUNCATED - TOO LARGE]"
                
            output = {
                "stdout": final_stdout,
                "stderr": final_stderr,
                "exit_code": proc.returncode,
            }
            if progress_updates:
                output["progress_updates"] = progress_updates
        else:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            output = {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode,
            }
            stdout_txt = output.get("stdout", "") if isinstance(output.get("stdout"), str) else ""
            stderr_txt = output.get("stderr", "") if isinstance(output.get("stderr"), str) else ""
            msg = stdout_txt + stderr_txt
            if msg:
                await stream_emitter.emit_event(
                    UIEvent(event_type=UIEventType.SHELL_STREAM, content=msg)
                )
            if progress_updates:
                output["progress_updates"] = progress_updates

        if password_prompt_detected:
            # Process was waiting for password input - kill it and ask user
            if proc and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    pass
            return needs_human(
                f"Command requires sudo/password input: '{command}'. {sudo_human_message}",
                metadata={"output": output, "command": command, "reason": "password_prompt_detected"}
            )

        if proc.returncode not in (0, None):
            err_output = output.get("stderr", "")
            if not isinstance(err_output, str):
                err_output = str(err_output or "")
            
            # If stderr is empty, fallback to stdout which might contain the error (e.g. if redirected 2>&1)
            if not err_output.strip():
                std_output = output.get("stdout", "")
                if isinstance(std_output, str) and std_output.strip():
                     err_output = std_output
            
            error = err_output.strip() or f"Command exited with code {proc.returncode}"
            
            # Sudo password detection (post-execution check for non-streaming case)
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
                return needs_human(
                    sudo_human_message,
                    metadata={"output": output, "command": command}
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

            return fail(error, metadata={"output": output})
        return ok(output)
    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except Exception:
                pass
        # Check if password prompt was detected before timeout
        if password_prompt_detected:
            return needs_human(
                f"Command timed out waiting for sudo/password input: '{command}'. {sudo_human_message}",
                metadata={"command": command, "reason": "password_prompt_timeout"}
            )
        metadata: dict[str, Any] = {"command": command}
        if progress_updates:
            metadata["progress_updates"] = progress_updates
        return fail(f"Command timed out after {timeout_sec}s", metadata=metadata)
    except asyncio.CancelledError:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except Exception:
                pass
        raise
    except Exception as exc:
        return fail(str(exc))
    finally:
        if harness and proc and hasattr(harness, "_active_processes"):
            try:
                harness._active_processes.discard(proc)
            except Exception:
                pass

    return fail("Unknown shell execution error")


def _local_user() -> str:
    return os.environ.get("USER", "")


async def _run_local_installer_preflight_probes(
    command: str,
    *,
    state: LoopState | None = None,
) -> dict[str, Any]:
    """Run automated local probes to discover installer environment state."""
    import shlex

    cwd, script_path = _remote_installer_cwd_and_script(command)
    probes: dict[str, Any] = {
        "host": "localhost",
        "user": _local_user(),
        "cwd": cwd,
        "script_path": script_path,
        "script_exists": False,
        "script_executable": False,
        "repo_clean": False,
        "noninteractive_flags": [],
        "preseed_files": [],
        "help_output": "",
        "is_interactive": True,
        "recommended_approach": "",
    }

    if not script_path or script_path == "make install":
        probes["recommended_approach"] = (
            "Unable to identify a specific install script for probing. "
            "Verify the correct installer path and retry."
        )
        return probes

    probe_script_parts = [
        'echo "__PREFLIGHT_PWD__"',
        "pwd",
        'echo "__PREFLIGHT_GIT_TOPLEVEL__"',
        f"cd {shlex.quote(cwd or '.')} && git rev-parse --show-toplevel 2>/dev/null || echo 'NO_GIT'",
        'echo "__PREFLIGHT_GIT_STATUS__"',
        f"cd {shlex.quote(cwd or '.')} && git status --short 2>/dev/null || echo 'NO_GIT'",
        'echo "__PREFLIGHT_SCRIPT__"',
        f"test -x {shlex.quote(script_path)} && echo 'EXECUTABLE' || (test -f {shlex.quote(script_path)} && echo 'EXISTS') || echo 'MISSING'",
        'echo "__PREFLIGHT_HELP__"',
        f"{shlex.quote(script_path)} --help 2>&1 || echo 'NO_HELP'",
    ]

    if cwd:
        probe_script_parts.append('echo "__PREFLIGHT_PRESEED__"')
        probe_script_parts.append(
            f"test -f {shlex.quote(cwd.rstrip('/') + '/.fogsettings')} && echo 'FOG_PRESEED' || true"
        )

    probe_script_parts.append('echo "__PREFLIGHT_DONE__"')
    probe_command = "bash -c " + shlex.quote("; ".join(probe_script_parts))

    try:
        proc = await _create_process(
            command=probe_command,
            cwd=state.cwd if state else ".",
        )
        stdout_data: list[str] = []
        stderr_data: list[str] = []
        await asyncio.wait_for(
            asyncio.gather(
                read_stream_chunks(proc.stdout, stdout_data, chunk_size=4096),
                read_stream_chunks(proc.stderr, stderr_data, chunk_size=4096),
                proc.wait(),
            ),
            timeout=30,
        )
        combined = "".join(stdout_data) + "".join(stderr_data)

        if "__PREFLIGHT_SCRIPT__" in combined:
            script_section = combined.split("__PREFLIGHT_SCRIPT__")[1].split("__PREFLIGHT_")[0]
            if "EXECUTABLE" in script_section:
                probes["script_exists"] = True
                probes["script_executable"] = True
            elif "EXISTS" in script_section:
                probes["script_exists"] = True

        if "__PREFLIGHT_GIT_STATUS__" in combined:
            git_section = combined.split("__PREFLIGHT_GIT_STATUS__")[1].split("__PREFLIGHT_")[0]
            probes["repo_clean"] = (
                "NO_GIT" in git_section
                or not git_section.strip()
                or "nothing to commit" in git_section
            )

        if "__PREFLIGHT_HELP__" in combined:
            help_section = combined.split("__PREFLIGHT_HELP__")[1].split("__PREFLIGHT_")[0]
            probes["help_output"] = help_section[:2000]
            help_lower = help_section.lower()
            known_flags = [
                "--autoaccept", "-y", "--yes", "--quiet",
                "--non-interactive", "--unattended", "--batch", "-n",
            ]
            probes["noninteractive_flags"] = [f for f in known_flags if f in help_lower]

        if "__PREFLIGHT_PRESEED__" in combined:
            preseed_section = combined.split("__PREFLIGHT_PRESEED__")[1].split("__PREFLIGHT_")[0]
            if "FOG_PRESEED" in preseed_section:
                probes["preseed_files"].append(".fogsettings")

        probes["is_interactive"] = len(probes["noninteractive_flags"]) == 0

        if probes["noninteractive_flags"]:
            probes["recommended_approach"] = (
                f"This installer supports non-interactive mode. "
                f"Retry with `shell_exec` and the flag `{probes['noninteractive_flags'][0]}`."
            )
        else:
            probes["recommended_approach"] = (
                "This appears to be an interactive installer with no documented non-interactive flags. "
                "If the installer must run interactively, use an explicit `printf` script with known answers, "
                "or configure a preseed/config file when available."
            )
    except asyncio.TimeoutError:
        probes["probe_error"] = "Preflight probes timed out after 30s"
    except Exception as exc:
        probes["probe_error"] = str(exc)

    return probes


async def shell_exec(
    command: str = "",
    state: LoopState | None = None,
    timeout_sec: int = 30,
    job_id: str = "",
    background: bool = False,
    harness: Any = None,
) -> dict[str, Any]:
    poll_job_id = str(job_id or "").strip()
    command = str(command or "").strip()
    if state is None:
        return fail("Shell execution requires state context.", metadata={"reason": "missing_state"})
    if poll_job_id and not command:
        return await shell_job_status(job_id=poll_job_id, state=state, harness=harness)
    if poll_job_id and command and not background:
        return fail(
            "`job_id` is only for polling background shell jobs. For foreground execution, omit `job_id`; "
            "to launch a background job, set `background=true` and provide `command`.",
            metadata={
                "reason": "invalid_shell_exec_job_id_with_foreground_command",
                "job_id": poll_job_id,
                "command": command,
            },
        )
    if not command:
        return fail("Shell execution requires a command or a job_id to poll.", metadata={"reason": "missing_command"})
    artifact_delete_guard = _shell_write_session_artifact_delete_guard(state, command)
    if artifact_delete_guard is not None:
        return artifact_delete_guard
    write_session_guard = _shell_write_session_target_path_guard(state, command)
    if write_session_guard is not None:
        return write_session_guard
    if background:
        return await shell_job_launch(command=command, state=state, create_process=_create_process, harness=harness)
    foreground_guard = _foreground_command_guard(
        command,
        tool_name="shell_exec",
        allow_background_parameter=True,
    )
    if foreground_guard is not None:
        return foreground_guard
    yes_pipe_guard = _interactive_installer_yes_pipe_guard(command, tool_name="shell_exec")
    if yes_pipe_guard is not None:
        return yes_pipe_guard
    preflight_guard = _remote_installer_preflight_guard(
        command,
        host="localhost",
        user=_local_user(),
        state=state,
    )
    if preflight_guard is not None:
        metadata = dict(preflight_guard.get("metadata") or {})
        if metadata.get("reason") == "remote_installer_preflight_failed":
            preflight_guard["metadata"] = metadata
            return preflight_guard

        probes = await _run_local_installer_preflight_probes(command, state=state)
        metadata["preflight_probes"] = probes
        metadata["suggested_tool_after_preflight"] = (
            "shell_exec" if not probes.get("is_interactive") else "interactive_session_unavailable"
        )

        if probes.get("script_exists") and probes.get("script_executable"):
            _mark_remote_installer_preflight_clean(
                state, host="localhost", user=_local_user(), cwd=probes.get("cwd", "")
            )

        if probes.get("is_interactive"):
            _expose_interactive_session_tools(state)

        parts: list[str] = []
        parts.append("Local installer environment scan completed.")
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
            parts.append(
                f"- Script: {probes['script_path']} (NOT FOUND)"
            )
        if probes.get("repo_clean"):
            parts.append("- Git repo: clean (or not a git repo)")
        else:
            parts.append("- Git repo: has uncommitted changes")
        if probes["noninteractive_flags"]:
            parts.append(
                f"- Non-interactive flags detected: {', '.join(probes['noninteractive_flags'])}"
            )
        if probes["preseed_files"]:
            parts.append(
                f"- Preseed/config files detected: {', '.join(probes['preseed_files'])}"
            )
        if probes.get("probe_error"):
            parts.append(f"- Probe error: {probes['probe_error']}")
        parts.append("")
        parts.append(probes.get("recommended_approach", ""))

        error_text = "\n".join(parts)
        return fail(error_text, metadata=metadata)

    return await shell_exec_foreground(
        command,
        state=state,
        timeout_sec=timeout_sec,
        harness=harness,
        create_process=_create_process,
    )


async def create_process(
    *,
    command: str,
    cwd: str,
    stdout: Any = asyncio.subprocess.PIPE,
    stderr: Any = asyncio.subprocess.PIPE,
    stdin: Any = asyncio.subprocess.DEVNULL,
    env_overrides: dict[str, str] | None = None,
    harness: Any = None,
) -> asyncio.subprocess.Process:
    env = _shell_env()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    proc = None
    if os.name == "nt":
        windows_commands = [
            ("powershell", "-NoProfile", "-Command", command),
            (_windows_cmd_exe(), "/c", command),
        ]
        last_error: Exception | None = None
        for candidate in windows_commands:
            try:
                return await asyncio.create_subprocess_exec(
                    *candidate,
                    cwd=cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=stdin,
                )
            except (FileNotFoundError, PermissionError) as exc:
                last_error = exc
        if proc is None: # If no async process was created, try Popen
            try:
                proc = _PopenProcessAdapter(
                    subprocess.Popen(
                        command,
                        cwd=cwd,
                        shell=True,
                        env=env,
                        stdout=_subprocess_stdio(stdout),
                        stderr=_subprocess_stdio(stderr),
                        stdin=_subprocess_stdio(stdin),
                    )
                )
            except Exception as exc:
                last_error = exc
        if proc is None and last_error is not None:
            raise last_error
    else:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            stdin=stdin,
        )

    if harness and hasattr(harness, "_active_processes"):
        harness._active_processes.add(proc)
        # We don't want to leak memory, but we need to remove it when done.
        # However, create_process is a low-level helper.
        # The caller (shell_exec in foreground/background mode) should handle the removal or teardown.

    return proc


async def _create_process(
    *,
    command: str,
    cwd: str,
    stdout: Any = asyncio.subprocess.PIPE,
    stderr: Any = asyncio.subprocess.PIPE,
    stdin: Any = asyncio.subprocess.DEVNULL,
    env_overrides: dict[str, str] | None = None,
    harness: Any = None,
) -> asyncio.subprocess.Process:
    return await create_process(
        command=command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        env_overrides=env_overrides,
        harness=harness,
    )


def _shell_env() -> dict[str, str]:
    env = dict(os.environ)
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    extra_entries: list[str] = []
    for candidate in (
        Path(sys.executable).absolute().parent,
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/bin"),
    ):
        candidate_str = str(candidate)
        if candidate_str not in path_entries and candidate_str not in extra_entries:
            extra_entries.append(candidate_str)
    if extra_entries:
        env["PATH"] = os.pathsep.join([*extra_entries, *path_entries])
    return env


def _windows_cmd_exe() -> str:
    candidate = os.environ.get("COMSPEC")
    if candidate:
        return candidate
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    return str(Path(system_root) / "System32" / "cmd.exe")


def _subprocess_stdio(stream: Any) -> Any:
    if stream is asyncio.subprocess.PIPE:
        return subprocess.PIPE
    if stream is asyncio.subprocess.DEVNULL:
        return subprocess.DEVNULL
    return stream


class _PopenProcessAdapter:
    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        stdout, stderr = await asyncio.to_thread(self._proc.communicate, input)
        return stdout or b"", stderr or b""

    def kill(self) -> None:
        self._proc.kill()

    def terminate(self) -> None:
        self._proc.terminate()

    async def wait(self) -> int:
        return await asyncio.to_thread(self._proc.wait)
