from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from .common import fail, ok
from .process_lifecycle import cancel_tasks, stop_process, truncate_output, unregister_process
from .process_streams import read_stream_chunks
from .network_ssh_helpers import detect_interactive_prompt as _detect_interactive_prompt
from .shell import create_process as _create_process
from .shell_support import _expose_interactive_session_tools
from .ui_streaming import BufferedUIEventEmitter

_LOCAL_INTERACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}

# Ring buffer cap for in-memory stdout/stderr to prevent OOM
_RING_BUFFER_MAX_BYTES = 256 * 1024


class _RingBuffer:
    """Fixed-size ring buffer that keeps the most recent bytes."""

    def __init__(self, max_bytes: int = _RING_BUFFER_MAX_BYTES) -> None:
        self.max_bytes = max_bytes
        self._data: list[str] = []
        self._total_bytes = 0

    def append(self, text: str) -> None:
        self._data.append(text)
        self._total_bytes += len(text.encode("utf-8", errors="replace"))
        self._trim()

    def _trim(self) -> None:
        while self._data and self._total_bytes > self.max_bytes:
            removed = self._data.pop(0)
            self._total_bytes -= len(removed.encode("utf-8", errors="replace"))

    def getvalue(self) -> str:
        return "".join(self._data)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


def _interactive_session_snapshot(
    session_id: str,
    session: dict[str, Any],
    *,
    max_chars: int = 6000,
    new_output: bool = False,
) -> dict[str, Any]:
    proc = session.get("proc")
    stdout_buf: _RingBuffer = session.get("stdout_buf")
    stderr_buf: _RingBuffer = session.get("stderr_buf")
    stdout = stdout_buf.getvalue() if stdout_buf else ""
    stderr = stderr_buf.getvalue() if stderr_buf else ""
    combined = stdout + stderr

    detected_prompt = _detect_interactive_prompt(combined)
    returncode = getattr(proc, "returncode", None)

    # Determine status
    if returncode is not None:
        status = "exited"
    elif detected_prompt is not None:
        status = "waiting_for_input"
    else:
        # Check if we have recent output flow
        last_output_at = session.get("last_output_at", 0.0)
        idle_grace_sec = session.get("idle_grace_sec", 20.0)
        if time.time() - last_output_at > idle_grace_sec:
            status = "idle"
        else:
            status = "running"

    snapshot: dict[str, Any] = {
        "session_id": session_id,
        "status": status,
        "detected_prompt": detected_prompt,
        "stdout_tail": truncate_output(stdout[-max_chars:], max_bytes=max_chars),
        "stderr_tail": truncate_output(stderr[-max_chars:], max_bytes=max_chars),
        "output_tail": truncate_output(combined[-max_chars:], max_bytes=max_chars),
        "exit_code": returncode,
        "command": session.get("command"),
        "cwd": session.get("cwd"),
        "total_bytes": (stdout_buf.total_bytes if stdout_buf else 0) + (stderr_buf.total_bytes if stderr_buf else 0),
        "elapsed_sec": round(time.time() - session.get("started_at", time.time()), 2),
    }

    if new_output:
        last_read_offset = session.get("_last_read_offset", 0)
        total_len = len(combined)
        if total_len > last_read_offset:
            snapshot["new_output"] = combined[last_read_offset:]
            snapshot["bytes_since_last_read"] = total_len - last_read_offset
        else:
            snapshot["new_output"] = ""
            snapshot["bytes_since_last_read"] = 0
        session["_last_read_offset"] = total_len

    return snapshot


async def _cleanup_interactive_session(
    session_id: str,
    session: dict[str, Any],
    *,
    harness: Any = None,
    terminate: bool = False,
) -> None:
    _LOCAL_INTERACTIVE_SESSIONS.pop(str(session_id or "").strip(), None)
    proc = session.get("proc")
    if terminate:
        await stop_process(proc, harness=harness, timeout=2.0)
    else:
        unregister_process(harness, proc)
    await cancel_tasks(list(session.get("tasks", []) or []))


def _check_session_timeout(session: dict[str, Any]) -> bool:
    """Return True if the session wall-clock timeout has been exceeded."""
    started_at = session.get("started_at")
    timeout_sec = session.get("timeout_sec", 900)
    if started_at is None:
        return False
    elapsed = time.time() - started_at
    return elapsed > timeout_sec


async def interactive_start(
    *,
    command: str,
    cwd: str | None = None,
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Start an interactive local command and keep stdin/stdout open."""
    if not command or not str(command).strip():
        return fail("interactive_start requires a command.", metadata={"reason": "missing_command"})

    proc = await _create_process(
        command=command,
        cwd=cwd or ".",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        harness=harness,
    )
    session_id = f"locint-{uuid.uuid4().hex[:12]}"
    session: dict[str, Any] = {
        "proc": proc,
        "stdout_buf": _RingBuffer(),
        "stderr_buf": _RingBuffer(),
        "command": command,
        "cwd": cwd or ".",
        "started_at": time.time(),
        "timeout_sec": 900,
        "idle_grace_sec": 20.0,
        "harness": harness,
        "last_output_at": time.time(),
        "_last_read_offset": 0,
    }
    _LOCAL_INTERACTIVE_SESSIONS[session_id] = session

    async def collect(stream: Any, key: str) -> None:
        buf: _RingBuffer = session[key]

        async def on_chunk(chunk_str: str) -> None:
            buf.append(chunk_str)
            session["last_output_at"] = time.time()

        await read_stream_chunks(stream, [], chunk_size=4096, on_chunk=on_chunk, idle_timeout_sec=None)

    session["tasks"] = [
        asyncio.create_task(collect(proc.stdout, "stdout_buf")),
        asyncio.create_task(collect(proc.stderr, "stderr_buf")),
    ]
    await asyncio.sleep(0.2)
    _expose_interactive_session_tools(state)
    return ok(
        _interactive_session_snapshot(session_id, session),
        metadata={
            "interactive_session": True,
            "next_tools": ["interactive_read", "interactive_send", "interactive_close"],
        },
    )


async def interactive_read(
    *,
    session_id: str,
    wait_sec: float = 0.0,
    until: str = "now",
    deadline_sec: float = 30.0,
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Read current output and detected prompt state from an active interactive local session.

    Args:
        session_id: The session ID returned by interactive_start.
        wait_sec: Seconds to sleep before snapshot (for until="now").
        until: One of "now", "prompt", "exit", "change".
        deadline_sec: Maximum seconds to wait for the until condition.
        max_chars: Maximum characters to include in tail fields.
    """
    session = _LOCAL_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown interactive session.", metadata={"session_id": session_id})

    # Check wall-clock timeout
    if _check_session_timeout(session):
        await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=True)
        return fail(
            "Interactive session timed out.",
            metadata={"session_id": session_id, "reason": "session_timeout", "status": "timed_out"},
        )

    proc = session.get("proc")
    until = str(until or "now").strip().lower()

    if until == "now":
        await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
        snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars, new_output=True)
        if snapshot.get("status") == "exited":
            await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
        return ok(snapshot, metadata={"interactive_session": True})

    # Poll loop for until conditions
    poll_interval = 0.1
    deadline = time.time() + max(0.0, min(float(deadline_sec or 30.0), 300.0))
    prior_offset = session.get("_last_read_offset", 0)

    while True:
        if _check_session_timeout(session):
            await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=True)
            return fail(
                "Interactive session timed out.",
                metadata={"session_id": session_id, "reason": "session_timeout", "status": "timed_out"},
            )

        snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
        status = snapshot.get("status")

        if until == "prompt" and status in {"waiting_for_input", "exited"}:
            snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars, new_output=True)
            if status == "exited":
                await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
            return ok(snapshot, metadata={"interactive_session": True})

        if until == "exit" and status == "exited":
            snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars, new_output=True)
            await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
            return ok(snapshot, metadata={"interactive_session": True})

        if until == "change":
            current_offset = len(snapshot.get("output_tail", ""))
            # Use the actual combined buffer length for comparison
            stdout_buf = session.get("stdout_buf")
            stderr_buf = session.get("stderr_buf")
            combined_len = len((stdout_buf.getvalue() if stdout_buf else "") + (stderr_buf.getvalue() if stderr_buf else ""))
            if combined_len > prior_offset or status == "exited":
                session["_last_read_offset"] = prior_offset
                snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars, new_output=True)
                if status == "exited":
                    await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
                return ok(snapshot, metadata={"interactive_session": True})

        if time.time() >= deadline:
            snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars, new_output=True)
            return ok(
                snapshot,
                metadata={"interactive_session": True, "deadline_reached": True},
            )

        await asyncio.sleep(poll_interval)


async def interactive_send(
    *,
    session_id: str,
    input: str,
    wait_sec: float = 0.5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Send input to an active interactive local session."""
    session = _LOCAL_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown interactive session.", metadata={"session_id": session_id})

    # Check wall-clock timeout
    if _check_session_timeout(session):
        await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=True)
        return fail(
            "Interactive session timed out.",
            metadata={"session_id": session_id, "reason": "session_timeout", "status": "timed_out"},
        )

    proc = session.get("proc")
    if getattr(proc, "returncode", None) is not None:
        snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
        await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
        return fail(
            "Interactive session has already exited.",
            metadata=snapshot,
        )

    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return fail("Interactive session stdin is unavailable.", metadata={"session_id": session_id})

    stdin.write(str(input or "").encode("utf-8"))
    await stdin.drain()
    await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
    return ok(_interactive_session_snapshot(session_id, session, max_chars=max_chars), metadata={"interactive_session": True})


async def interactive_close(
    *,
    session_id: str,
    terminate: bool = True,
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Close an active interactive local session."""
    session = _LOCAL_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown interactive session.", metadata={"session_id": session_id})
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=terminate)
    return ok(snapshot, metadata={"interactive_session": True})


async def interactive_run(
    *,
    command: str,
    inputs: list[str],
    cwd: str | None = None,
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Start an interactive command, send all inputs sequentially, and return the full transcript.

    This is the high-level "scripted" interaction tool for small models.
    The harness drives the session lifecycle; the model just provides the
    command and the list of inputs.
    """
    if not inputs:
        return fail("interactive_run requires at least one input.", metadata={"reason": "missing_inputs"})

    start_result = await interactive_start(command=command, cwd=cwd, state=state, harness=harness)
    if not start_result.get("success"):
        return start_result

    session_id = start_result.get("output", {}).get("session_id")
    if not session_id:
        return fail("interactive_run failed to obtain session_id.", metadata={"reason": "no_session_id"})

    session = _LOCAL_INTERACTIVE_SESSIONS.get(session_id)
    if not isinstance(session, dict):
        return fail("Session disappeared after start.", metadata={"session_id": session_id})

    transcript: list[dict[str, Any]] = []
    prompts_seen: list[dict[str, Any]] = []

    for idx, user_input in enumerate(inputs):
        # Wait for prompt or exit
        read_result = await interactive_read(
            session_id=session_id,
            until="prompt",
            deadline_sec=30.0,
        )
        if not read_result.get("success"):
            await interactive_close(session_id=session_id, terminate=True)
            return read_result

        snapshot = read_result.get("output", {})
        detected_prompt = snapshot.get("detected_prompt")
        if detected_prompt:
            prompts_seen.append(detected_prompt)

        if snapshot.get("status") == "exited":
            transcript.append({"send": user_input, "output": snapshot.get("output_tail", ""), "status": "exited"})
            break

        # Send input
        send_result = await interactive_send(session_id=session_id, input=user_input)
        if not send_result.get("success"):
            await interactive_close(session_id=session_id, terminate=True)
            return send_result

        transcript.append({
            "send": user_input,
            "output": send_result.get("output", {}).get("output_tail", ""),
            "status": send_result.get("output", {}).get("status"),
        })

    # Final read to capture any remaining output
    final_read = await interactive_read(session_id=session_id, until="exit", deadline_sec=30.0)
    final_snapshot = final_read.get("output", {}) if final_read.get("success") else {}
    exit_code = final_snapshot.get("exit_code")

    # Always close
    close_result = await interactive_close(session_id=session_id, terminate=False)
    combined_output = "\n".join(
        f"[send] {entry['send']}\n[output] {entry['output']}"
        for entry in transcript
    )
    if final_snapshot.get("output_tail"):
        combined_output += f"\n[final] {final_snapshot['output_tail']}"

    return ok(
        {
            "status": final_snapshot.get("status", "exited"),
            "transcript": transcript,
            "combined_output": combined_output,
            "exit_code": exit_code,
            "prompts_seen": prompts_seen,
        },
        metadata={"interactive_session": True},
    )
