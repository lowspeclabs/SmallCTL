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

_SSH_INTERACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}


def _command_preview(command: Any, *, limit: int = 160) -> str:
    text = " ".join(str(command or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _ssh_session_registry_snapshot() -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    now = time.time()
    for active_id, active in sorted(_SSH_INTERACTIVE_SESSIONS.items()):
        if not isinstance(active, dict):
            continue
        proc = active.get("proc")
        returncode = getattr(proc, "returncode", None)
        started_at = active.get("started_at")
        age_sec = None
        if isinstance(started_at, (int, float)):
            age_sec = round(max(0.0, now - float(started_at)), 3)
        snapshot.append(
            {
                "session_id": active_id,
                "host": active.get("host"),
                "user": active.get("user"),
                "status": "exited" if returncode is not None else "running",
                "exit_code": returncode,
                "age_sec": age_sec,
                "command_preview": _command_preview(active.get("command")),
                "unchanged_read_count": int(active.get("unchanged_read_count", 0) or 0),
            }
        )
    return snapshot


def _state_registry_snapshot(state: Any) -> list[dict[str, Any]]:
    registry = _active_ssh_session_registry(state)
    rows: list[dict[str, Any]] = []
    for session_id, item in sorted(registry.items()):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "session_id": str(session_id),
                "host": item.get("host"),
                "user": item.get("user"),
                "status": item.get("status"),
                "started_at": item.get("started_at"),
                "command_preview": _command_preview(item.get("command")),
            }
        )
    return rows


def _runlog_ssh_session_registry(
    harness: Any,
    event: str,
    message: str,
    *,
    state: Any = None,
    requested_session_id: str = "",
    host: str = "",
    user: str | None = None,
    command: str = "",
    reason: str = "",
    status: str = "",
    close_terminate: bool | None = None,
) -> None:
    if harness is None or not hasattr(harness, "_runlog"):
        return
    data: dict[str, Any] = {
        "requested_session_id": str(requested_session_id or "").strip(),
        "host": host,
        "user": user,
        "reason": reason,
        "status": status,
        "active_sessions": _ssh_session_registry_snapshot(),
        "state_active_sessions": _state_registry_snapshot(state),
    }
    if command:
        data["command_preview"] = _command_preview(command)
    if close_terminate is not None:
        data["terminate"] = bool(close_terminate)
    harness._runlog(event, message, **data)


def _active_ssh_session_registry(state: Any) -> dict[str, Any]:
    """Return the mutable registry of active SSH interactive sessions on state.scratchpad."""
    if state is None:
        return {}
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    registry = scratchpad.setdefault("_active_ssh_interactive_sessions", {})
    if not isinstance(registry, dict):
        registry = {}
        scratchpad["_active_ssh_interactive_sessions"] = registry
    return registry


def _sync_active_ssh_session_to_state(
    state: Any,
    session_id: str,
    session: dict[str, Any],
    *,
    status: str | None = None,
) -> None:
    """Mirror an in-memory SSH session into state.scratchpad for prompt visibility."""
    if state is None:
        return
    registry = _active_ssh_session_registry(state)
    proc = session.get("proc")
    resolved_status = status
    if resolved_status is None:
        resolved_status = (
            "exited" if getattr(proc, "returncode", None) is not None else "running"
        )
    registry[str(session_id or "").strip()] = {
        "session_id": session_id,
        "host": session.get("host"),
        "user": session.get("user"),
        "command": session.get("command"),
        "status": resolved_status,
        "started_at": session.get("started_at"),
    }


def _remove_active_ssh_session_from_state(state: Any, session_id: str) -> None:
    """Remove a closed SSH session from state.scratchpad."""
    if state is None:
        return
    registry = _active_ssh_session_registry(state)
    registry.pop(str(session_id or "").strip(), None)


def _snapshot_signature(snapshot: dict[str, Any]) -> tuple[str, str, str, int | None]:
    return (
        str(snapshot.get("status") or "").strip(),
        str(snapshot.get("detected_prompt") or "").strip(),
        str(snapshot.get("output_tail") or ""),
        snapshot.get("exit_code")
        if isinstance(snapshot.get("exit_code"), int)
        else None,
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
        unchanged_count = (
            int(session.get("unchanged_read_count", 0) or 0) + 1 if unchanged else 0
        )
        session["unchanged_read_count"] = unchanged_count
        if unchanged:
            snapshot["_note"] = (
                "Interactive SSH output is unchanged since the previous read; the prompt state has not advanced."
            )
            metadata["interactive_output_unchanged"] = True
            metadata["unchanged_read_count"] = unchanged_count
    elif event == "send":
        session["unchanged_read_count"] = 0
        if unchanged:
            snapshot["_note"] = (
                "Interactive SSH output is unchanged after send; input may have been echoed without advancing the prompt."
            )
            metadata["interactive_output_unchanged"] = True
    else:
        session["unchanged_read_count"] = 0
    session["last_snapshot_signature"] = signature
    return snapshot, metadata


def _interactive_session_snapshot(
    session_id: str, session: dict[str, Any], *, max_chars: int = 6000
) -> dict[str, Any]:
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
    state: Any = None,
    terminate: bool = False,
) -> None:
    _SSH_INTERACTIVE_SESSIONS.pop(str(session_id or "").strip(), None)
    _remove_active_ssh_session_from_state(state, session_id)
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
            metadata={
                "host": host,
                "command": command,
                "reason": "redacted_password_provided",
            },
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
    for active_id, active in list(_SSH_INTERACTIVE_SESSIONS.items()):
        proc = active.get("proc") if isinstance(active, dict) else None
        if getattr(proc, "returncode", None) is not None:
            continue
        if str(active.get("host") or "") == host and str(
            active.get("user") or ""
        ) == str(user or ""):
            _runlog_ssh_session_registry(
                harness,
                "ssh_interactive_session_rejected",
                "rejected interactive SSH session start because target already has an active session",
                state=state,
                requested_session_id=active_id,
                host=host,
                user=user,
                command=command,
                reason="active_interactive_session_exists",
                status="blocked",
            )
            return fail(
                f"An interactive SSH session is already active for this target. "
                f"Use the existing session_id '{active_id}' with ssh_session_read/ssh_session_send, "
                f"or close it with ssh_session_close before starting a new one.",
                metadata={
                    "reason": "active_interactive_session_exists",
                    "active_session_id": active_id,
                    "host": host,
                    "user": user,
                },
            )

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
                metadata={
                    "host": host,
                    "user": user,
                    "command": command,
                    "reason": "sshpass_missing",
                },
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
    _sync_active_ssh_session_to_state(state, session_id, session, status="running")

    async def collect(stream: Any, key: str) -> None:
        await read_stream_chunks(
            stream, session[key], chunk_size=4096, idle_timeout_sec=None
        )

    session["tasks"] = [
        asyncio.create_task(collect(proc.stdout, "stdout")),
        asyncio.create_task(collect(proc.stderr, "stderr")),
    ]
    await asyncio.sleep(0.2)
    session["last_snapshot_signature"] = _snapshot_signature(
        _interactive_session_snapshot(session_id, session)
    )
    _sync_active_ssh_session_to_state(state, session_id, session)
    _expose_interactive_session_tools(state)
    _runlog_ssh_session_registry(
        harness,
        "ssh_interactive_session_started",
        "started interactive SSH session",
        state=state,
        requested_session_id=session_id,
        host=host,
        user=user,
        command=command,
        status="running",
    )
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
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        _runlog_ssh_session_registry(
            harness,
            "ssh_interactive_session_unknown",
            "interactive SSH session lookup failed",
            state=state,
            requested_session_id=session_id,
            reason="unknown_session",
            status="failed",
        )
        return fail(
            "Unknown SSH interactive session.", metadata={"session_id": session_id}
        )
    await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    snapshot, metadata = _annotate_interactive_snapshot(session, snapshot, event="read")
    _sync_active_ssh_session_to_state(state, session_id, session)
    _runlog_ssh_session_registry(
        session.get("harness") or harness,
        "ssh_interactive_session_read",
        "read interactive SSH session",
        state=state,
        requested_session_id=session_id,
        host=str(session.get("host") or ""),
        user=session.get("user"),
        command=str(session.get("command") or ""),
        status=str(snapshot.get("status") or ""),
    )
    if snapshot.get("status") == "exited":
        await _cleanup_interactive_session(
            session_id,
            session,
            harness=session.get("harness"),
            state=state,
            terminate=False,
        )
    return ok(snapshot, metadata=metadata)


async def ssh_session_send(
    *,
    session_id: str,
    input: str,
    send_newline: bool = True,
    wait_sec: float = 0.5,
    max_chars: int = 6000,
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        _runlog_ssh_session_registry(
            harness,
            "ssh_interactive_session_unknown",
            "interactive SSH session lookup failed",
            state=state,
            requested_session_id=session_id,
            reason="unknown_session",
            status="failed",
        )
        return fail(
            "Unknown SSH interactive session.", metadata={"session_id": session_id}
        )
    proc = session.get("proc")
    if getattr(proc, "returncode", None) is not None:
        snapshot = _interactive_session_snapshot(
            session_id, session, max_chars=max_chars
        )
        await _cleanup_interactive_session(
            session_id,
            session,
            harness=session.get("harness"),
            state=state,
            terminate=False,
        )
        return fail(
            "SSH interactive session has already exited.",
            metadata=snapshot,
        )
    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return fail(
            "SSH interactive session stdin is unavailable.",
            metadata={"session_id": session_id},
        )
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
    _sync_active_ssh_session_to_state(state, session_id, session)
    _runlog_ssh_session_registry(
        session.get("harness") or harness,
        "ssh_interactive_session_send",
        "sent input to interactive SSH session",
        state=state,
        requested_session_id=session_id,
        host=str(session.get("host") or ""),
        user=session.get("user"),
        command=str(session.get("command") or ""),
        status=str(snapshot.get("status") or ""),
    )
    return ok(snapshot, metadata=metadata)


async def ssh_session_send_and_read(
    *,
    session_id: str,
    input: str,
    send_newline: bool = True,
    wait_sec: float = 0.5,
    max_chars: int = 6000,
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    sent = await ssh_session_send(
        session_id=session_id,
        input=input,
        send_newline=send_newline,
        wait_sec=wait_sec,
        max_chars=max_chars,
        state=state,
        harness=harness,
    )
    if not bool(sent.get("success")):
        return sent
    return await ssh_session_read(
        session_id=session_id,
        wait_sec=0,
        max_chars=max_chars,
        state=state,
        harness=harness,
    )


async def ssh_session_close(
    *,
    session_id: str,
    terminate: bool = True,
    max_chars: int = 6000,
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        _runlog_ssh_session_registry(
            harness,
            "ssh_interactive_session_unknown",
            "interactive SSH session lookup failed",
            state=state,
            requested_session_id=session_id,
            reason="unknown_session",
            status="failed",
        )
        return fail(
            "Unknown SSH interactive session.", metadata={"session_id": session_id}
        )
    snapshot = _interactive_session_snapshot(session_id, session, max_chars=max_chars)
    _runlog_ssh_session_registry(
        session.get("harness") or harness,
        "ssh_interactive_session_closed",
        "closed interactive SSH session",
        state=state,
        requested_session_id=session_id,
        host=str(session.get("host") or ""),
        user=session.get("user"),
        command=str(session.get("command") or ""),
        status=str(snapshot.get("status") or ""),
        close_terminate=terminate,
    )
    await _cleanup_interactive_session(
        session_id,
        session,
        harness=session.get("harness"),
        state=state,
        terminate=terminate,
    )
    return ok(snapshot, metadata={"interactive_session": True})
