from __future__ import annotations

import asyncio
import asyncio.subprocess
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..state import LoopState
from .common import fail
from .installer_preflight import run_installer_preflight_probes
from .process_lifecycle import register_process
from .shell_processes import cwd_get, cwd_set, env_get, env_set, process_kill, shell_job_launch, shell_job_status
from .shell_foreground import shell_exec_foreground
from .shell_support import (
    _apt_deb822_preflight_guard,
    _apt_sources_list_d_guard,
    _expose_interactive_session_tools,
    _foreground_command_guard,
    _interactive_installer_yes_pipe_guard,
    _looks_like_deb822_validator,
    _mark_deb822_preflight_clean,
    _mark_remote_installer_preflight_clean,
    _remote_installer_preflight_guard,
    _shell_workspace_destructive_delete_guard,
    _shell_write_session_artifact_delete_guard,
    _shell_write_session_target_path_guard,
    record_apt_update_result,
)


def _local_user() -> str:
    return os.environ.get("USER", "")


async def _run_local_installer_preflight_probes(
    command: str,
    *,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Run automated local probes to discover installer environment state."""

    def _build_probe_command(probe_script: str) -> str:
        return probe_script

    probes = await run_installer_preflight_probes(
        command=command,
        state=state,
        create_process=_create_process,
        harness=harness,
        host="localhost",
        user=_local_user(),
        build_probe_command=_build_probe_command,
    )
    # Local-specific wording
    if probes.get("noninteractive_flags"):
        probes["recommended_approach"] = (
            f"This installer supports non-interactive mode. "
            f"Retry with `shell_exec` and the flag `{probes['noninteractive_flags'][0]}`."
        )
    elif not probes.get("probe_error"):
        probes["recommended_approach"] = (
            "This appears to be an interactive installer with no documented non-interactive flags. "
            "If the installer must run interactively, use an explicit `printf` script with known answers, "
            "or configure a preseed/config file when available."
        )
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
    workspace_delete_guard = _shell_workspace_destructive_delete_guard(state, command)
    if workspace_delete_guard is not None:
        return workspace_delete_guard
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
    apt_guard = _apt_deb822_preflight_guard(
        command,
        tool_name="shell_exec",
        state=state,
        host="localhost",
        user=_local_user(),
    )
    if apt_guard is not None:
        if harness and hasattr(harness, "_runlog"):
            harness._runlog(
                "apt_deb822_preflight_blocked",
                "APT blocked pending deb822 validation on localhost — run the validator first.",
                host="localhost",
                user=_local_user(),
                reason="apt_deb822_preflight_required",
            )
        return apt_guard

    sources_guard = _apt_sources_list_d_guard(
        command,
        tool_name="shell_exec",
        state=state,
        host="localhost",
        user=_local_user(),
    )
    if sources_guard is not None:
        return sources_guard
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

        probes = await _run_local_installer_preflight_probes(command, state=state, harness=harness)
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

    result = await shell_exec_foreground(
        command,
        state=state,
        timeout_sec=timeout_sec,
        harness=harness,
        create_process=_create_process,
    )
    if result.get("success") and _looks_like_deb822_validator(command):
        _mark_deb822_preflight_clean(state, host="localhost", user=_local_user())
    # Record apt-get update results for the sources.list.d guard
    record_apt_update_result(
        state,
        command=command,
        success=bool(result.get("success")),
        stderr=str(result.get("stderr") or ""),
        host="localhost",
        user=_local_user(),
    )
    return result


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
                proc = await asyncio.create_subprocess_exec(
                    *candidate,
                    cwd=cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=stdin,
                )
                break
            except (FileNotFoundError, PermissionError) as exc:
                last_error = exc
        if proc is None:  # If no async process was created, try Popen
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

    register_process(harness, proc)

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
