from __future__ import annotations

import asyncio
import asyncio.subprocess
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..state import LoopState, _coerce_background_process_record
from .common import fail, needs_human, ok


async def shell_exec(
    command: str,
    state: LoopState,
    timeout_sec: int = 30,
    harness: Any = None,
) -> dict[str, Any]:
    proc = None
    try:
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

        proc = await _create_process(command=command, cwd=state.cwd, harness=harness)

        stdout_data = []
        stderr_data = []

        async def read_stream(stream, out_list):
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
                    read_stream(proc.stdout, stdout_data),
                    read_stream(proc.stderr, stderr_data),
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
            
            # Sudo password detection
            sudo_prompts = ("sudo: a password is required", "[sudo] password for", "password:", "Password:")
            if any(p.lower() in error.lower() for p in sudo_prompts):
                return needs_human(
                    "Sudo execution requires a password. Please provide it:",
                    metadata={"output": output, "command": command}
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
    harness: Any = None,
) -> asyncio.subprocess.Process:
    env = _shell_env()
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
    harness: Any = None,
) -> asyncio.subprocess.Process:
    return await create_process(
        command=command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
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

    async def communicate(self) -> tuple[bytes, bytes]:
        stdout, stderr = await asyncio.to_thread(self._proc.communicate)
        return stdout or b"", stderr or b""

    def kill(self) -> None:
        self._proc.kill()
