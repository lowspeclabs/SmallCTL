from __future__ import annotations

import asyncio
import asyncio.subprocess
import shlex
import time
from typing import Any, TYPE_CHECKING
from pathlib import Path

from .common import fail, ok
from .shell import create_process

if TYPE_CHECKING:
    from ..state import LoopState

async def ssh_exec(
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """
    Execute a command on a remote host via SSH with live streaming support.
    """
    ssh_args = [
        "ssh",
        "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new"
    ]
    
    if identity_file:
        ssh_args.extend(["-i", identity_file])
        
    target = f"{user}@{host}" if user else host
    ssh_args.append(target)
    ssh_args.append(command)
    
    full_cmd = " ".join(shlex.quote(arg) if " " in arg else arg for arg in ssh_args)
    
    start_time = time.time()
    proc = None
    try:
        proc = await create_process(
            command=full_cmd,
            cwd=state.cwd if state else ".",
            harness=harness,
        )

        stdout_data = []
        stderr_data = []

        async def read_stream(stream, out_list):
            if not stream: return
            while True:
                chunk = await stream.read(4096)
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

        await asyncio.wait_for(
            asyncio.gather(
                read_stream(proc.stdout, stdout_data),
                read_stream(proc.stderr, stderr_data),
                proc.wait()
            ),
            timeout=timeout_sec
        )
        
        elapsed = time.time() - start_time
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
            "metrics": {
                "duration_sec": round(elapsed, 3) if isinstance(elapsed, (int, float)) else 0.0,
                "host": host,
                "user": user,
            }
        }

        if proc.returncode != 0:
            err_output = output.get("stderr", "")
            if not isinstance(err_output, str):
                err_output = str(err_output or "")
            error_msg = err_output.strip() or f"SSH failed with exit code {proc.returncode}"
            hints = []
            if "Permission denied" in error_msg:
                hints.append("Check if SSH keys are correctly configured on the remote host.")
            if "Connection timed out" in error_msg:
                hints.append("Verify the host is reachable and the port is open.")
            
            return fail(error_msg, metadata={"output": output, "hints": hints})

        return ok(output)

    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            proc.kill()
        return fail(f"SSH command timed out after {timeout_sec}s")
    except Exception as exc:
        return fail(f"SSH execution error: {str(exc)}")
    finally:
        if harness and proc and hasattr(harness, "_active_processes"):
            harness._active_processes.discard(proc)
