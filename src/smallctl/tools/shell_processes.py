from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail, ok


def _find_active_process_by_pid(harness: Any, pid: int) -> Any | None:
    active_processes = getattr(harness, "_active_processes", None)
    if not isinstance(active_processes, set):
        return None
    for proc in active_processes:
        if getattr(proc, "pid", None) == pid:
            return proc
    return None


def _background_job_record(state: LoopState, job_id: str) -> dict[str, Any] | None:
    record = state.background_processes.get(job_id)
    if not isinstance(record, dict):
        return None
    return dict(record)


async def shell_job_status(job_id: str, state: LoopState, harness: Any = None) -> dict[str, Any]:
    record = _background_job_record(state, job_id)
    if record is None:
        return fail(f"Unknown job id: {job_id}")

    pid = int(record.get("pid") or 0)
    proc = _find_active_process_by_pid(harness, pid) if harness is not None else None
    status = str(record.get("status") or "unknown").strip().lower()
    command = str(record.get("command") or "").strip()
    cwd = str(record.get("cwd") or "").strip()

    if proc is not None:
        returncode = getattr(proc, "returncode", None)
        if returncode is None:
            record["status"] = "running"
            state.background_processes[job_id] = record
            state.touch()
            return ok(
                {
                    "job_id": job_id,
                    "pid": pid,
                    "command": command,
                    "cwd": cwd,
                    "status": "running",
                    "started_at": record.get("started_at"),
                }
            )
        status = "completed" if returncode == 0 else "failed"
        record["status"] = status
        record["exit_code"] = returncode
        state.background_processes[job_id] = record
        state.touch()
        if harness is not None and hasattr(harness, "_active_processes"):
            try:
                harness._active_processes.discard(proc)
            except Exception:
                pass
        return ok(
            {
                "job_id": job_id,
                "pid": pid,
                "command": command,
                "cwd": cwd,
                "status": status,
                "exit_code": returncode,
                "started_at": record.get("started_at"),
            }
        )

    if os.name != "nt" and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            status = "unknown"
        except PermissionError:
            status = "running"
        else:
            status = "running"
    if status == "running":
        record["status"] = "running"
        state.background_processes[job_id] = record
        state.touch()
        return ok(
            {
                "job_id": job_id,
                "pid": pid,
                "command": command,
                "cwd": cwd,
                "status": "running",
                "started_at": record.get("started_at"),
            }
        )

    record["status"] = "unknown"
    state.background_processes[job_id] = record
    state.touch()
    return ok(
        {
            "job_id": job_id,
            "pid": pid,
            "command": command,
            "cwd": cwd,
            "status": "unknown",
            "started_at": record.get("started_at"),
        }
    )


async def shell_job_launch(command: str, state: LoopState, *, create_process, harness: Any = None) -> dict[str, Any]:
    try:
        proc = await create_process(
            command=command,
            cwd=state.cwd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            harness=harness,
        )
        job_id = str(proc.pid)
        state.background_processes[job_id] = {
            "pid": proc.pid,
            "command": command,
            "cwd": state.cwd,
            "status": "running",
        }
        state.touch()
        return ok({"job_id": job_id, "pid": proc.pid, "command": command})
    except Exception as exc:
        return fail(str(exc))


async def process_kill(job_id: str, state: LoopState) -> dict[str, Any]:
    record = _background_job_record(state, job_id)
    if record is None:
        return fail(f"Unknown job id: {job_id}")
    pid = int(record.get("pid") or 0)
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
    return ok({"name": name, "value": os.environ.get(name, "")})


async def env_set(name: str, value: str) -> dict[str, Any]:
    os.environ[str(name)] = str(value)
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
