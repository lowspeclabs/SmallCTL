from __future__ import annotations

import os
from typing import Any

from .installer_preflight import run_installer_preflight_probes
from .network_ssh_helpers import build_ssh_command as _build_ssh_command


async def _run_remote_installer_preflight_probes(
    *,
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Run automated SSH probes to discover installer environment state."""
    password_files: list[str] = []

    def _build_probe_command(probe_script: str) -> str:
        full_cmd, _env_overrides, password_file_path = _build_ssh_command(
            host=host,
            command=probe_script,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
        )
        if password_file_path:
            password_files.append(password_file_path)
        return full_cmd

    from .shell import create_process as _create_process
    try:
        probes = await run_installer_preflight_probes(
            command=command,
            state=state,
            create_process=_create_process,
            harness=harness,
            host=host,
            user=user or "",
            build_probe_command=_build_probe_command,
        )
    finally:
        for path in password_files:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
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
