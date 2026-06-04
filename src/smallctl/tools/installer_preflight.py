from __future__ import annotations

import asyncio
import shlex
from typing import Any

from .process_lifecycle import stop_process, unregister_process
from .process_streams import read_stream_chunks


async def run_installer_preflight_probes(
    *,
    command: str,
    state: Any,
    create_process: Any,
    harness: Any = None,
    host: str = "localhost",
    user: str = "",
    build_probe_command: Any = None,
) -> dict[str, Any]:
    """Shared local/SSH installer preflight probe runner.

    `build_probe_command` is an optional callable that receives the raw probe
    script string and returns the final command to execute.  For SSH it should
    wrap the script in an SSH invocation; for local execution it can be left
    as the default identity.
    """
    from .shell_support import _remote_installer_cwd_and_script

    cwd, script_path = _remote_installer_cwd_and_script(command)
    probes: dict[str, Any] = {
        "host": host,
        "user": user,
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

    if build_probe_command is not None:
        probe_command = build_probe_command(probe_command)

    try:
        proc = await create_process(
            command=probe_command,
            cwd=state.cwd if state else ".",
            harness=harness,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
                f"Retry with the flag `{probes['noninteractive_flags'][0]}`."
            )
        else:
            probes["recommended_approach"] = (
                "This appears to be an interactive installer with no documented non-interactive flags. "
                "If the installer must run interactively, use an explicit `printf` script with known answers, "
                "or configure a preseed/config file when available."
            )
    except asyncio.TimeoutError:
        await stop_process(proc, harness=harness, timeout=1.0)
        probes["probe_error"] = "Preflight probes timed out after 30s"
    except Exception as exc:
        probes["probe_error"] = str(exc)
    finally:
        unregister_process(harness, proc)

    return probes
