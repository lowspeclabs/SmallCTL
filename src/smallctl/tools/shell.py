from __future__ import annotations

import asyncio
import asyncio.subprocess
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..state import LoopState, _coerce_background_process_record
from .common import fail, needs_human, ok


# Patterns that indicate a sudo/password prompt is waiting for input
SUDO_PROMPT_PATTERNS = [
    re.compile(r'\[sudo\] password for', re.IGNORECASE),
    re.compile(r'^password:', re.IGNORECASE),
    re.compile(r'password for .*:', re.IGNORECASE),
    re.compile(r'sudo:.*no password was provided', re.IGNORECASE),
    re.compile(r'sudo:.*password is required', re.IGNORECASE),
    re.compile(r'sudo:.*a password is required', re.IGNORECASE),
    re.compile(r'sudo:.*no tty present', re.IGNORECASE),
    re.compile(r'sudo:.*a terminal is required', re.IGNORECASE),
    re.compile(r'sudo:.*must be run from a terminal', re.IGNORECASE),
]

SUDO_INVALID_PASSWORD_PATTERNS = [
    re.compile(r"sorry,\s*try again", re.IGNORECASE),
    re.compile(r"incorrect password", re.IGNORECASE),
    re.compile(r"authentication failure", re.IGNORECASE),
    re.compile(r"\bincorrect password attempts?\b", re.IGNORECASE),
]

SUDO_PERMISSION_DENIED_PATTERNS = [
    re.compile(r"not in the sudoers file", re.IGNORECASE),
    re.compile(r"may not run sudo", re.IGNORECASE),
]

_SHELL_WRAPPER_TOKENS = {
    "bash",
    "sh",
    "zsh",
    "dash",
    "ksh",
    "pwsh",
    "powershell",
    "cmd",
    "cmd.exe",
}
_SHELL_WRAPPER_COMMAND_FLAGS = {"-c", "-lc", "/c", "-Command", "-command"}

_ARGPARSE_REQUIRED_ARGS_PATTERN = re.compile(
    r"(?:error:\s*)?the following arguments are required:\s*(.+)",
    re.IGNORECASE,
)


def _extract_missing_argparse_arguments(error_text: str) -> list[str]:
    match = _ARGPARSE_REQUIRED_ARGS_PATTERN.search(str(error_text or ""))
    if not match:
        return []

    missing = match.group(1).strip()
    if not missing:
        return []

    # argparse usually prints comma-separated flags, but keep the parser narrow so
    # we only hand off the clearly missing names instead of guessing at aliases.
    missing = missing.replace(" and ", ", ")
    values = [part.strip(" .`'\"") for part in missing.split(",")]
    return [value for value in values if value]


def _build_argparse_missing_args_question(command: str, missing_args: list[str]) -> str:
    missing_text = ", ".join(missing_args) if missing_args else "required arguments"
    return (
        f"The command `{command}` is missing required arguments: {missing_text}. "
        "What values should I use?"
    )


def _detect_unsupported_shell_syntax(command: str) -> str | None:
    # Keep this intentionally narrow: the current failure mode is Bash here-string
    # redirection, which /bin/sh does not understand.
    if "<<<" in command:
        return (
            "Command uses Bash-only here-string redirection (`<<<`), but smallctl runs shell "
            "commands through /bin/sh on Unix. Rewrite it with POSIX syntax (for example, "
            "use `printf` piped into the command) or wrap the whole command in `bash -lc`."
        )
    return None


def _shell_execution_authoring_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    """
    Keep small-model runs in the authoring contract until they have produced an
    actual artifact to verify.

    This allows file creation/replacement to happen first, then shell execution
    can resume once there is something concrete to test.
    """
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    if plan is not None and not getattr(plan, "approved", False):
        return fail(
            "Shell execution is blocked until the spec contract is approved.",
            metadata={
                "command": command,
                "reason": "spec_not_approved",
                "plan_id": getattr(plan, "plan_id", ""),
            },
        )

    if state.contract_phase() == "author":
        if not state.files_changed_this_cycle:
            return fail(
                "Shell execution is blocked until the authoring contract has produced a target artifact.",
                metadata={
                    "command": command,
                    "reason": "authoring_target_missing",
                    "contract_phase": state.contract_phase(),
                    "files_changed_this_cycle": state.files_changed_this_cycle,
                },
            )
    return None


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    key, _value = token.split("=", 1)
    return key.isidentifier()


def _leading_command_tokens(command: str, *, max_depth: int = 3) -> list[str]:
    current = command
    for _ in range(max_depth):
        tokens = _shell_tokens(current)
        if not tokens:
            return []
        if len(tokens) >= 3 and tokens[0].lower() in _SHELL_WRAPPER_TOKENS and tokens[1] in _SHELL_WRAPPER_COMMAND_FLAGS:
            current = tokens[2]
            continue
        index = 0
        if tokens[0].lower() == "env":
            index = 1
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
        while index < len(tokens) and _looks_like_env_assignment(tokens[index]):
            index += 1
        return tokens[index:]
    return _shell_tokens(current)


def _command_uses_leading_sudo(command: str) -> bool:
    tokens = _leading_command_tokens(command)
    return bool(tokens) and tokens[0].lower() == "sudo"


def _matches_any_pattern(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


async def _run_sudo_validation(
    *,
    cwd: str,
    harness: Any = None,
    timeout_sec: int = 10,
    password: str | None = None,
) -> dict[str, Any]:
    proc = None
    try:
        command = "sudo -n -v" if password is None else "sudo -S -p '' -v"
        stdin = asyncio.subprocess.DEVNULL if password is None else asyncio.subprocess.PIPE
        proc = await _create_process(
            command=command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=stdin,
            harness=harness,
        )
        input_bytes = None if password is None else f"{password}\n".encode("utf-8")
        stdout, stderr = await asyncio.wait_for(proc.communicate(input_bytes), timeout=max(1, timeout_sec))
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        combined = "\n".join(part for part in (stderr_text.strip(), stdout_text.strip()) if part).strip()
        if proc.returncode in (0, None):
            return {"status": "ok", "stdout": stdout_text, "stderr": stderr_text}
        if _matches_any_pattern(combined, SUDO_INVALID_PASSWORD_PATTERNS):
            return {
                "status": "invalid_password",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": combined or "Incorrect sudo password.",
            }
        if _matches_any_pattern(combined, SUDO_PROMPT_PATTERNS):
            return {
                "status": "password_required",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": combined or "Sudo password is required.",
            }
        if _matches_any_pattern(combined, SUDO_PERMISSION_DENIED_PATTERNS):
            return {
                "status": "permission_denied",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": combined or "Sudo permission denied.",
            }
        return {
            "status": "error",
            "stdout": stdout_text,
            "stderr": stderr_text,
            "error": combined or f"sudo validation exited with code {proc.returncode}",
        }
    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except Exception:
                pass
        return {"status": "error", "error": f"sudo validation timed out after {max(1, timeout_sec)}s"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    finally:
        if harness and proc and hasattr(harness, "_active_processes"):
            try:
                harness._active_processes.discard(proc)
            except Exception:
                pass


async def _ensure_sudo_credentials(
    *,
    command: str,
    cwd: str,
    harness: Any = None,
    timeout_sec: int = 30,
    sudo_human_message: str,
) -> dict[str, Any] | None:
    if not _command_uses_leading_sudo(command):
        return None

    validation_timeout = max(5, min(timeout_sec, 15))
    validation = await _run_sudo_validation(cwd=cwd, harness=harness, timeout_sec=validation_timeout)
    status = str(validation.get("status") or "").strip().lower()
    if status == "ok":
        return None
    if status == "permission_denied":
        return fail(
            str(validation.get("error") or "Sudo permission denied."),
            metadata={"command": command, "reason": "sudo_permission_denied", "sudo_validation": validation},
        )
    if status not in {"password_required", "invalid_password"}:
        return fail(
            str(validation.get("error") or "Unable to validate sudo credentials."),
            metadata={"command": command, "reason": "sudo_validation_failed", "sudo_validation": validation},
        )

    password_fn = getattr(harness, "request_sudo_password", None)
    if not callable(password_fn) or getattr(harness, "event_handler", None) is None:
        return needs_human(
            f"Command requires sudo/password input: '{command}'. {sudo_human_message}",
            metadata={"command": command, "reason": "sudo_password_required", "sudo_validation": validation},
        )

    prompt_text = "Enter the sudo password to continue this command."
    validation_error = str(validation.get("error") or "").strip()
    if validation_error:
        prompt_text = f"{prompt_text}\n\n{validation_error}"

    for attempt in range(1, 4):
        password = await password_fn(command=command, prompt_text=prompt_text)
        if password is None:
            return fail(
                "Sudo password entry cancelled by user.",
                metadata={"command": command, "reason": "sudo_password_cancelled"},
            )
        validation = await _run_sudo_validation(
            cwd=cwd,
            harness=harness,
            timeout_sec=validation_timeout,
            password=password,
        )
        status = str(validation.get("status") or "").strip().lower()
        if status == "ok":
            return None
        if status == "permission_denied":
            return fail(
                str(validation.get("error") or "Sudo permission denied."),
                metadata={"command": command, "reason": "sudo_permission_denied", "sudo_validation": validation},
            )
        if status in {"password_required", "invalid_password"} and attempt < 3:
            prompt_text = "Incorrect sudo password. Try again."
            validation_error = str(validation.get("error") or "").strip()
            if validation_error:
                prompt_text = f"{prompt_text}\n\n{validation_error}"
            continue
        if status in {"password_required", "invalid_password"}:
            return fail(
                "Sudo authentication failed after 3 attempts.",
                metadata={"command": command, "reason": "sudo_password_rejected", "sudo_validation": validation},
            )
        return fail(
            str(validation.get("error") or "Unable to validate sudo credentials."),
            metadata={"command": command, "reason": "sudo_validation_failed", "sudo_validation": validation},
        )
    return None


async def shell_exec(
    command: str,
    state: LoopState,
    timeout_sec: int = 30,
    harness: Any = None,
) -> dict[str, Any]:
    proc = None
    password_prompt_detected = False
    try:
        sudo_human_message = (
            "Sudo execution requires a password. If interactive prompts are unavailable, "
            "configure passwordless sudo or ask the user for help."
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
        approval_fn = getattr(harness, "request_shell_approval", None)
        if callable(approval_fn) and getattr(harness, "event_handler", None) is not None:
            approved = await approval_fn(
                command=command,
                cwd=state.cwd,
                timeout_sec=timeout_sec,
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

        sudo_guard = await _ensure_sudo_credentials(
            command=command,
            cwd=state.cwd,
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

        async def read_stream(stream, out_list, is_stderr: bool = False):
            nonlocal password_prompt_detected, detection_buffer
            if not stream or not hasattr(stream, "read"):
                return
            while True:
                try:
                    chunk = await stream.read(2048)
                except Exception:
                    break
                if not chunk:
                    break
                chunk_str = chunk.decode("utf-8", errors="replace")
                out_list.append(chunk_str)
                
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
                
                if harness and hasattr(harness, "_emit") and getattr(harness, "event_handler", None):
                    from ..models.events import UIEvent, UIEventType
                    # Sanity check: cap emittable content to avoid UI overwhelm if chunk is massive
                    emittable = chunk_str
                    if len(emittable) > 16384:
                         emittable = emittable[:16384] + "\n[UI TRUNCATED - LARGE OUTPUT]"
                         
                    evt = UIEvent(
                        event_type=UIEventType.SHELL_STREAM,
                        content=emittable,
                    )
                    # Use await instead of create_task to provide natural backpressure
                    # so we don't spam the UI loop faster than it can render.
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            await harness._emit(harness.event_handler, evt)
                    except RuntimeError:
                        pass

        if hasattr(proc, "stdout") and hasattr(proc, "stderr") and hasattr(proc.stdout, "read"):
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(proc.stdout, stdout_data, is_stderr=False),
                    read_stream(proc.stderr, stderr_data, is_stderr=True),
                    proc.wait()
                ),
                timeout=timeout_sec
            )
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
        else:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            output = {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode,
            }
            if harness and hasattr(harness, "_emit") and getattr(harness, "event_handler", None):
                from ..models.events import UIEvent, UIEventType
                stdout_txt = output.get("stdout", "") if isinstance(output.get("stdout"), str) else ""
                stderr_txt = output.get("stderr", "") if isinstance(output.get("stderr"), str) else ""
                msg = stdout_txt + stderr_txt
                if msg:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            await harness._emit(harness.event_handler, UIEvent(event_type=UIEventType.SHELL_STREAM, content=msg))
                    except RuntimeError:
                        pass

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
        return fail(f"Command timed out after {timeout_sec}s")
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


async def shell_background(command: str, state: LoopState, harness: Any = None) -> dict[str, Any]:
    try:
        proc = await _create_process(
            command=command,
            cwd=state.cwd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            harness=harness,
        )
        job_id = str(proc.pid)
        state.background_processes[job_id] = _coerce_background_process_record(
            {
                "pid": proc.pid,
                "command": command,
                "cwd": state.cwd,
                "status": "running",
            },
            job_id=job_id,
        )
        state.touch()
        return ok({"job_id": job_id, "pid": proc.pid, "command": command})
    except Exception as exc:
        return fail(str(exc))


async def process_kill(job_id: str, state: LoopState) -> dict[str, Any]:
    proc_meta = state.background_processes.get(job_id)
    if not proc_meta:
        return fail(f"Unknown job id: {job_id}")
    pid = int(proc_meta["pid"])
    try:
        if os.name == "nt":
            await asyncio.create_subprocess_shell(f"taskkill /PID {pid} /F")
        else:
            os.kill(pid, 9)
        del state.background_processes[job_id]
        state.touch()
        return ok(f"killed {pid}")
    except Exception as exc:
        return fail(str(exc))


async def env_get(name: str) -> dict[str, Any]:
    return ok({"name": name, "value": os.getenv(name)})


async def env_set(name: str, value: str) -> dict[str, Any]:
    os.environ[name] = value
    return ok({"name": name, "value": value})


async def cwd_get(state: LoopState) -> dict[str, Any]:
    return ok({"cwd": state.cwd})


async def cwd_set(path: str, state: LoopState) -> dict[str, Any]:
    new_cwd = Path(path)
    if not new_cwd.is_absolute():
        new_cwd = Path(state.cwd) / new_cwd
    new_cwd = new_cwd.resolve()
    if not new_cwd.exists() or not new_cwd.is_dir():
        return fail(f"Invalid directory: {new_cwd}")
    state.cwd = str(new_cwd)
    state.touch()
    return ok({"cwd": state.cwd})


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
        # The caller (shell_exec/shell_background) should handle the removal or teardown.

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
