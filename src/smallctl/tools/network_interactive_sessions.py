from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from .common import fail, ok
from .process_lifecycle import cancel_tasks, stop_process, unregister_process
from .process_streams import read_stream_chunks
from .ssh_parsing import strip_redundant_root_sudo as _strip_redundant_root_sudo
from .network_ssh_helpers import (
    build_ssh_command as _build_ssh_command,
    detect_interactive_prompt as _detect_interactive_prompt,
    ssh_execution_debug_metadata as _ssh_execution_debug_metadata,
)
from .shell_support import _expose_interactive_session_tools
from .ui_streaming import BufferedUIEventEmitter

_SSH_INTERACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}


def _snapshot_signature(snapshot: dict[str, Any]) -> tuple[str, str, str, int | None]:
    return (
        str(snapshot.get("status") or "").strip(),
        str(snapshot.get("detected_prompt") or "").strip(),
        str(snapshot.get("output_tail") or ""),
        snapshot.get("exit_code") if isinstance(snapshot.get("exit_code"), int) else None,
    )


def _annotate_interactive_snapshot(
    session: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    event: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {"interactive_session": True}
    last_signature = session.get("last_snapshot_signature")
    signature = _snapshot_signature(snapshot)
    unchanged = signature == last_signature
    if event == "read":
        unchanged_count = int(session.get("unchanged_read_count", 0) or 0) + 1 if unchanged else 0
        session["unchanged_read_count"] = unchanged_count
        if unchanged:
            snapshot["_note"] = "Interactive SSH output is unchanged since the previous read; the prompt state has not advanced."
            metadata["interactive_output_unchanged"] = True
            metadata["unchanged_read_count"] = unchanged_count
    elif event == "send":
        session["unchanged_read_count"] = 0
        if unchanged:
            snapshot["_note"] = "Interactive SSH output is unchanged after send; input may have been echoed without advancing the prompt."
            metadata["interactive_output_unchanged"] = True
    else:
        session["unchanged_read_count"] = 0
    session["last_snapshot_signature"] = signature
    return snapshot, metadata


def _interactive_session_snapshot(session_id: str, session: dict[str, Any], *, max_chars: int = 6000) -> dict[str, Any]:
    proc = session.get("proc")
    stdout = "".join(session.get("stdout", []))
    stderr = "".join(session.get("stderr", []))
    combined = stdout + stderr
    detected_prompt = _detect_interactive_prompt(combined)
    returncode = getattr(proc, "returncode", None)
    if returncode is not None:
        status = "exited"
    elif detected_prompt is not None:
        status = "waiting_for_input"
    else:
        status = "running"
    return {
        "session_id": session_id,
        "status": status,
        "detected_prompt": detected_prompt,
        "stdout_tail": stdout[-max_chars:],
        "stderr_tail": stderr[-max_chars:],
        "output_tail": combined[-max_chars:],
        "exit_code": returncode,
        "host": session.get("host"),
        "user": session.get("user"),
        "command": session.get("command"),
    }


async def _cleanup_interactive_session(
    session_id: str,
    session: dict[str, Any],
    *,
    harness: Any = None,
    terminate: bool = False,
) -> None:
    _SSH_INTERACTIVE_SESSIONS.pop(str(session_id or "").strip(), None)
    proc = session.get("proc")
    if terminate:
        await stop_process(proc, harness=harness, timeout=2.0)
    else:
        unregister_process(harness, proc)
    await cancel_tasks(list(session.get("tasks", []) or []))


async def ssh_session_start(
    *,
    host: str = "",
    target: str | None = None,
    command: str = "",
    user: str | None = None,
    username: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    timeout_sec: int = 900,
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Start an interactive SSH command and keep stdin/stdout open."""
    if password and str(password).startswith("[REDACTED"):
        return fail(
            "The SSH password provided was literally redacted. Ask the human user to provide the actual password.",
            metadata={"host": host, "command": command, "reason": "redacted_password_provided"},
        )
    try:
        from .ssh_parsing import normalize_ssh_arguments, normalize_ssh_target
        normalized = normalize_ssh_arguments(
            {
                "host": host,
                "target": target,
                "user": user,
                "username": username,
            }
        )
        host = str(normalized.get("host") or "")
        user = str(normalized.get("user") or "") or None
        host, user = normalize_ssh_target(host=host, user=user)
    except ValueError as exc:
        return fail(str(exc), metadata={"host": host, "user": user, "command": command})
    command, stripped_root_sudo = _strip_redundant_root_sudo(command, user)

    proc = None
    try:
        full_cmd, env_overrides = _build_ssh_command(
            host=host,
            command=command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
            force_tty=True,
        )
    except FileNotFoundError as exc:
        if str(exc) == "sshpass":
            return fail(
                "sshpass is required for password-based SSH but was not found.",
                metadata={"host": host, "user": user, "command": command, "reason": "sshpass_missing"},
            )
        raise

    from .shell import create_process as _create_process
    proc = await _create_process(
        command=full_cmd,
        cwd=".",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        env_overrides=env_overrides,
        harness=harness,
    )
    session_id = f"sshint-{uuid.uuid4().hex[:12]}"
    session: dict[str, Any] = {
        "proc": proc,
        "stdout": [],
        "stderr": [],
        "host": host,
        "user": user,
        "command": command,
        "started_at": time.time(),
        "timeout_sec": max(1, int(timeout_sec or 900)),
        "harness": harness,
    }
    _SSH_INTERACTIVE_SESSIONS[session_id] = session

    async def collect(stream: Any, key: str) -> None:
        await read_stream_chunks(stream, session[key], chunk_size=4096, idle_timeout_sec=None)

    session["tasks"] = [
        asyncio.create_task(collect(proc.stdout, "stdout")),
        asyncio.create_task(collect(proc.stderr, "stderr")),
    ]
    await asyncio.sleep(0.2)
    session["last_snapshot_signature"] = _snapshot_signature(
        _interactive_session_snapshot(session_id, session)
    )
    _expose_interactive_session_tools(state)
    return ok(
        _interactive_session_snapshot(session_id, session),
        metadata={
            "interactive_session": True,
            "pty_requested": True,
            "next_tools": ["ssh_session_read", "ssh_session_send", "ssh_session_close"],
            **_ssh_execution_debug_metadata(
                password=password,
                identity_file=identity_file,
                strict_host_key_checking="accept-new",
            ),
        },
    )


async def ssh_session_read(
    *,
    session_id: str,
    wait_sec: float = 1.0,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    snapshot, metadata = _annotate_interactive_snapshot(session, snapshot, event="read")
    if snapshot.get("status") == "exited":
        await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
    return ok(snapshot, metadata=metadata)


async def ssh_session_send(
    *,
    session_id: str,
    input: str,
    send_newline: bool = True,
    wait_sec: float = 0.5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    proc = session.get("proc")
    if getattr(proc, "returncode", None) is not None:
        snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
        await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=False)
        return fail(
            "SSH interactive session has already exited.",
            metadata=snapshot,
        )
    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return fail("SSH interactive session stdin is unavailable.", metadata={"session_id": session_id})
    text = str(input or "")
    if send_newline and text and not text.endswith("\n"):
        text += "\n"
    stdin.write(text.encode("utf-8"))
    await stdin.drain()
    session["last_sent_input"] = str(input or "")
    session["last_send_used_newline"] = bool(send_newline)
    await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    snapshot, metadata = _annotate_interactive_snapshot(session, snapshot, event="send")
    metadata["send_newline"] = bool(send_newline)
    return ok(snapshot, metadata=metadata)


async def ssh_session_send_and_read(
    *,
    session_id: str,
    input: str,
    send_newline: bool = True,
    wait_sec: float = 0.5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    sent = await ssh_session_send(
        session_id=session_id,
        input=input,
        send_newline=send_newline,
        wait_sec=wait_sec,
        max_chars=max_chars,
    )
    if not bool(sent.get("success")):
        return sent
    return await ssh_session_read(session_id=session_id, wait_sec=0, max_chars=max_chars)


async def ssh_session_close(
    *,
    session_id: str,
    terminate: bool = True,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    await _cleanup_interactive_session(session_id, session, harness=session.get("harness"), terminate=terminate)
    return ok(snapshot, metadata={"interactive_session": True})
