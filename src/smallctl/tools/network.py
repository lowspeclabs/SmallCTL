from __future__ import annotations

import asyncio
import asyncio.subprocess
import re
import shlex
import shutil
import time
import uuid
from typing import Any, TYPE_CHECKING

from ..models.events import UIEventType
from .common import fail, ok
from ..risk_policy import evaluate_risk_policy
from ..state import LoopState
from .shell import create_process
from .process_streams import read_stream_chunks
from .shell_support import (
    InvalidInputLoopDetector,
    _expose_interactive_session_tools,
    _foreground_command_guard,
    _interactive_installer_yes_pipe_guard,
    _installer_command_suggested_timeout,
    _mark_remote_installer_preflight_clean,
    _remote_installer_cwd_and_script,
    _remote_installer_preflight_guard,
)
from .ui_streaming import BufferedUIEventEmitter

if TYPE_CHECKING:
    from ..state import LoopState


_IGNORABLE_SSH_FLAGS = {
    "-4",
    "-6",
    "-A",
    "-a",
    "-C",
    "-f",
    "-G",
    "-g",
    "-K",
    "-k",
    "-M",
    "-N",
    "-n",
    "-q",
    "-s",
    "-T",
    "-t",
    "-tt",
    "-v",
    "-vv",
    "-vvv",
    "-X",
    "-x",
    "-Y",
    "-y",
}
_SAFE_SSH_OPTION_KEYS = {
    "BatchMode",
    "ConnectTimeout",
    "IdentityFile",
    "NumberOfPasswordPrompts",
    "PasswordAuthentication",
    "Port",
    "PreferredAuthentications",
    "PubkeyAuthentication",
    "StrictHostKeyChecking",
    "User",
}
_LOCAL_SHELL_CONTROL_TOKENS = {"|", "||", "&&", ";", ";&", ";;&"}
_SSH_DIAGNOSTIC_NOT_FOUND_MARKERS = (
    "not found",
    "could not be found",
    "no such file",
    "command not found",
)
_SSH_TRANSPORT_FAILURE_MARKERS = (
    "permission denied",
    "connection timed out",
    "connection refused",
    "connection closed by remote host",
    "could not resolve hostname",
    "no route to host",
    "host key verification failed",
    "kex_exchange_identification",
    "connection reset by peer",
    "operation timed out",
    "network is unreachable",
)
_INTERACTIVE_PROMPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("yes_no", re.compile(r"(?P<question>[^\n\r]{0,240}?\((?:y/n|y/N|Y/n|Y/N|yes/no|yes/NO)\))\s*$", re.IGNORECASE)),
    ("choice", re.compile(r"(?P<question>[^\n\r]{0,240}?(?:choice|selection|option)\s*:)\s*$", re.IGNORECASE)),
    ("enter", re.compile(r"(?P<question>[^\n\r]{0,240}?(?:press|hit)\s+(?:enter|return)[^\n\r]*)\s*$", re.IGNORECASE)),
    ("password", re.compile(r"(?P<question>[^\n\r]{0,240}password\s*:)\s*$", re.IGNORECASE)),
    ("free_text", re.compile(r"(?P<question>[^\n\r]{0,240}:\s*)$", re.IGNORECASE)),
)
_SSH_INTERACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}
_ROOT_SUDO_PREFIX_RE = re.compile(
    r"^\s*sudo(?:\s+-(?:n|E|H|S))*\s+(?:(?:-i|-s)\s+)?(?:--\s+)?(?P<command>.+?)\s*$",
    re.DOTALL,
)
_ROOT_SUDO_SEGMENT_RE = re.compile(
    r"(?P<prefix>^|(?:&&|\|\||;|\|)\s*)sudo(?:\s+-(?:n|E|H|S))*\s+(?:(?:-i|-s)\s+)?(?:--\s+)?",
)


def _strip_redundant_root_sudo(command: str, user: str | None) -> tuple[str, bool]:
    if str(user or "").strip().lower() != "root":
        return command, False
    text = str(command or "").strip()
    if not text.startswith("sudo"):
        return command, False
    match = _ROOT_SUDO_PREFIX_RE.match(text)
    if not match:
        return command, False
    stripped = str(match.group("command") or "").strip()
    stripped = _ROOT_SUDO_SEGMENT_RE.sub(lambda match: str(match.group("prefix") or ""), stripped)
    return stripped, bool(stripped and stripped != text)


_SSH_ACCEPT_NEW_INCOMPATIBLE_MARKERS = (
    "keyword stricthostkeychecking extra arguments at end of line",
    "bad configuration option",
)


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _normalize_optional_ssh_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_ssh_target(*, host: str, user: str | None = None) -> tuple[str, str | None]:
    host_text = str(host or "").strip()
    user_text = _normalize_optional_ssh_string(user)
    if not host_text:
        return "", user_text
    if host_text.count("@") > 1:
        raise ValueError("SSH target must contain at most one `@` separator.")
    if "@" not in host_text:
        return host_text, user_text

    embedded_user, bare_host = host_text.rsplit("@", 1)
    embedded_user = embedded_user.strip()
    bare_host = bare_host.strip()
    if not embedded_user or not bare_host:
        raise ValueError("SSH target must be either `host` plus `user` or `user@host`.")
    if user_text is not None and user_text != embedded_user:
        raise ValueError("SSH target must be either `host` plus `user` or `user@host`.")
    return bare_host, embedded_user


def normalize_ssh_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}

    normalized = dict(arguments)
    target_text = _normalize_optional_ssh_string(normalized.pop("target", None))
    explicit_host = _normalize_optional_ssh_string(normalized.get("host"))
    if target_text:
        if explicit_host and explicit_host != target_text:
            raise ValueError("Conflicting SSH targets provided via `target` and `host`.")
        normalized["host"] = target_text
    alias_user = _normalize_optional_ssh_string(normalized.pop("username", None))
    explicit_user = _normalize_optional_ssh_string(normalized.get("user"))
    if alias_user:
        if explicit_user and explicit_user != alias_user:
            raise ValueError("Conflicting SSH usernames provided via `user` and `username`.")
        normalized["user"] = alias_user
        explicit_user = alias_user
    elif explicit_user is None:
        normalized.pop("user", None)
    else:
        normalized["user"] = explicit_user

    host_text = _normalize_optional_ssh_string(normalized.get("host")) or ""
    host_text, user_text = normalize_ssh_target(host=host_text, user=explicit_user)
    if not host_text:
        raise ValueError("SSH target requires either `target` or `host`.")
    normalized["host"] = host_text
    if user_text:
        normalized["user"] = user_text
    else:
        normalized.pop("user", None)
    return normalized


def _split_ssh_option_value(option: str) -> tuple[str, str | None]:
    cleaned = str(option or "").strip()
    if "=" in cleaned:
        key, value = cleaned.split("=", 1)
        return key.strip(), value.strip() or None
    parts = cleaned.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip() or None
    return cleaned, None


def _parse_int_option(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _shell_tokens_with_spans(command: str) -> list[tuple[str, int, int]]:
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: list[tuple[str, int, int]] = []
    while True:
        start = lexer.instream.tell()
        token = lexer.get_token()
        end = lexer.instream.tell()
        if token == lexer.eof:
            break
        tokens.append((token, start, end))
    return tokens


def _is_shell_redirection_token(token: str) -> bool:
    stripped = str(token or "").strip()
    if not stripped:
        return False
    return (
        stripped.startswith((">", "<"))
        or stripped.startswith(("1>", "1<", "2>", "2<"))
        or stripped.startswith((">>", "<<", "&>", ">&", "<&"))
        or stripped.endswith((">&1", ">&2", "<&0", "<&1", "<&2"))
    )


def _join_remote_shell_tokens(
    command: str,
    tokens: list[tuple[str, int, int]],
) -> str:
    return " ".join(
        (
            token
            if _is_shell_redirection_token(token)
            else (
                token
                if idx == 0 and command[start:end].lstrip().startswith(("'", '"'))
                else shlex.quote(token)
            )
        )
        for idx, (token, start, end) in enumerate(tokens)
    )


def parse_ssh_exec_args_from_shell_command(command: str) -> dict[str, Any] | None:
    command_text = str(command or "").strip()
    if not command_text:
        return None

    try:
        token_spans = _shell_tokens_with_spans(command_text)
    except ValueError:
        return None
    tokens = [token for token, _start, _end in token_spans]
    if not tokens:
        return None

    parsed: dict[str, Any] = {}
    target: str | None = None
    index = 0

    if tokens[index] == "sshpass":
        index += 1
        password: str | None = None
        while index < len(tokens):
            token = tokens[index]
            if token == "ssh":
                break
            if token == "-p":
                index += 1
                if index >= len(tokens):
                    return None
                password = tokens[index]
            elif token.startswith("-p") and len(token) > 2:
                password = token[2:]
            else:
                # Only rewrite the simple sshpass form we can preserve safely.
                return None
            index += 1
        if index >= len(tokens) or tokens[index] != "ssh" or not password:
            return None
        parsed["password"] = password

    if tokens[index] != "ssh":
        return None
    index += 1

    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            if index >= len(tokens):
                return None
            target = tokens[index]
            index += 1
            break
        if not token.startswith("-"):
            target = token
            index += 1
            break

        option_name = token
        option_value: str | None = None
        if token in {"-i", "-l", "-o", "-p"}:
            index += 1
            if index >= len(tokens):
                return None
            option_value = tokens[index]
        elif token.startswith("-i") and len(token) > 2:
            option_name = "-i"
            option_value = token[2:]
        elif token.startswith("-l") and len(token) > 2:
            option_name = "-l"
            option_value = token[2:]
        elif token.startswith("-o") and len(token) > 2:
            option_name = "-o"
            option_value = token[2:]
        elif token.startswith("-p") and len(token) > 2:
            option_name = "-p"
            option_value = token[2:]
        elif token in _IGNORABLE_SSH_FLAGS:
            index += 1
            continue
        else:
            return None

        if option_name == "-i":
            parsed["identity_file"] = option_value
        elif option_name == "-l":
            parsed["user"] = option_value
        elif option_name == "-p":
            port_value = _parse_int_option(option_value)
            if port_value is None:
                return None
            parsed["port"] = port_value
        elif option_name == "-o":
            key, value = _split_ssh_option_value(option_value or "")
            if key not in _SAFE_SSH_OPTION_KEYS:
                return None
            if key == "IdentityFile":
                parsed["identity_file"] = value
            elif key == "Port":
                port_value = _parse_int_option(value)
                if port_value is None:
                    return None
                parsed["port"] = port_value
            elif key == "User":
                parsed["user"] = value
        index += 1

    if not target:
        return None

    remote_tokens = token_spans[index:]
    if not remote_tokens:
        parsed["host"] = target
        parsed["command"] = "whoami"
        return normalize_ssh_arguments(parsed)
    if any(token in _LOCAL_SHELL_CONTROL_TOKENS for token, _start, _end in remote_tokens):
        return None

    parsed["host"] = target
    parsed["command"] = _join_remote_shell_tokens(command_text, remote_tokens)
    return normalize_ssh_arguments(parsed)


def _build_ssh_command(
    *,
    host: str,
    command: str,
    user: str | None,
    port: int,
    identity_file: str | None,
    password: str | None,
    strict_host_key_checking: str = "accept-new",
    force_tty: bool = False,
) -> tuple[str, dict[str, str] | None]:
    host, user = normalize_ssh_target(host=host, user=user)
    ssh_args = [
        "-p", str(port),
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", f"StrictHostKeyChecking={strict_host_key_checking}",
    ]
    env_overrides: dict[str, str] | None = None

    if password:
        if shutil.which("sshpass") is None:
            raise FileNotFoundError("sshpass")
        ssh_args.extend([
            "-o", "BatchMode=no",
            "-o", "PasswordAuthentication=yes",
            "-o", "PreferredAuthentications=password,keyboard-interactive",
            "-o", "PubkeyAuthentication=no",
            "-o", "NumberOfPasswordPrompts=1",
        ])
        command_args = ["sshpass", "-e", "ssh"]
        env_overrides = {"SSHPASS": password}
    else:
        ssh_args.extend(["-o", "BatchMode=yes"])
        command_args = ["ssh"]

    if identity_file:
        ssh_args.extend(["-i", identity_file])
    if force_tty:
        ssh_args.append("-tt")

    target = f"{user}@{host}" if user else host
    ssh_args.extend([target, command])
    return _shell_join([*command_args, *ssh_args]), env_overrides


def _detect_interactive_prompt(text: str) -> dict[str, Any] | None:
    tail = str(text or "")[-4096:]
    if not tail.strip():
        return None
    normalized_tail = tail.rstrip()
    for prompt_type, pattern in _INTERACTIVE_PROMPT_PATTERNS:
        match = pattern.search(normalized_tail)
        if not match:
            continue
        question = " ".join(str(match.group("question") or "").split())
        detected: dict[str, Any] = {
            "type": prompt_type,
            "question": question,
        }
        if prompt_type == "yes_no":
            marker = question[-5:].lower()
            if "/n" in marker:
                detected["default"] = "N"
            elif "/y" in marker:
                detected["default"] = "Y"
        return detected
    return None


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


def _ssh_accept_new_is_incompatible(stderr: str) -> bool:
    lowered = str(stderr or "").strip().lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in _SSH_ACCEPT_NEW_INCOMPATIBLE_MARKERS):
        return "stricthostkeychecking" in lowered or "accept-new" in lowered
    return "accept-new" in lowered and "unsupported option" in lowered


def _ssh_failure_kind(*, exit_code: int, stderr: str) -> str:
    lowered = str(stderr or "").strip().lower()
    if any(marker in lowered for marker in _SSH_TRANSPORT_FAILURE_MARKERS):
        return "transport"
    if exit_code == 255 and not lowered:
        return "transport"
    return "remote_command"


def _ssh_error_class(*, exit_code: int, stderr: str) -> str:
    lowered = str(stderr or "").strip().lower()
    if "permission denied" in lowered:
        return "auth_permission_denied"
    if "host key verification failed" in lowered or "remote host identification has changed" in lowered:
        return "host_key_verification"
    if "could not resolve hostname" in lowered or "name or service not known" in lowered or "temporary failure in name resolution" in lowered:
        return "dns_resolution"
    if "connection refused" in lowered:
        return "connection_refused"
    if "connection timed out" in lowered or "operation timed out" in lowered or "timed out" in lowered:
        return "connection_timeout"
    return "transport_failure" if _ssh_failure_kind(exit_code=exit_code, stderr=stderr) == "transport" else "remote_exit_nonzero"


def _ssh_diagnostic_not_found(command: str, output: dict[str, Any]) -> bool:
    """Return True when an exit-1 SSH result is a diagnostic 'not found' probe."""
    stderr = str(output.get("stderr") or "").lower()
    stdout = str(output.get("stdout") or "").lower()
    combined = stdout + " " + stderr
    return any(marker in combined for marker in _SSH_DIAGNOSTIC_NOT_FOUND_MARKERS)


def _ssh_execution_debug_metadata(
    *,
    password: str | None,
    identity_file: str | None,
    strict_host_key_checking: str,
) -> dict[str, Any]:
    password_text = str(password or "").strip()
    identity_file_text = str(identity_file or "").strip()
    return {
        "ssh_auth_mode": "password" if password_text else "key",
        "ssh_auth_transport": "sshpass_env" if password_text else "ssh",
        "ssh_password_provided": bool(password_text),
        "ssh_identity_file_supplied": bool(identity_file_text),
        "ssh_strict_host_key_checking": strict_host_key_checking,
    }


async def ssh_exec(
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    timeout_sec: int = 60,
    stdin_data: str | None = None,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """
    Execute a command on a remote host via SSH with live streaming support.
    """
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    if plan is not None and not getattr(plan, "approved", False):
        return fail(
            "SSH execution is blocked until the spec contract is approved.",
            metadata={
                "host": host,
                "command": command,
                "reason": "spec_not_approved",
                "plan_id": getattr(plan, "plan_id", ""),
            },
        )
    if state is not None and state.contract_phase() == "author":
        if not state.files_changed_this_cycle:
            return fail(
                "SSH execution is blocked until the authoring contract has produced a target artifact.",
                metadata={
                    "host": host,
                    "command": command,
                    "reason": "authoring_target_missing",
                    "contract_phase": state.contract_phase(),
                    "files_changed_this_cycle": state.files_changed_this_cycle,
                },
            )
    policy_state = state if state is not None else LoopState()
    approval_fn = getattr(harness, "request_shell_approval", None)
    approval_available = callable(approval_fn) and getattr(harness, "event_handler", None) is not None
    risk_decision = evaluate_risk_policy(
        policy_state,
        tool_name="ssh_exec",
        tool_risk="high",
        phase=str(policy_state.current_phase or ""),
        action=command,
        expected_effect="Run the requested SSH command on the remote host.",
        rollback="Stop the command and revert any in-progress remote changes if needed.",
        verification="Inspect the remote command output and any follow-up verifier result.",
        approval_available=approval_available,
    )
    if not risk_decision.allowed:
        return fail(
            risk_decision.reason,
            metadata={
                "host": host,
                "command": command,
                "reason": "missing_supported_claim",
                "proof_bundle": risk_decision.proof_bundle,
            },
        )
    approval_wait_sec = 0.0
    if risk_decision.requires_approval and callable(approval_fn) and approval_available:
        approval_start = time.monotonic()
        approved = await approval_fn(
            command=command,
            cwd=str(getattr(policy_state, "cwd", ".") or "."),
            timeout_sec=timeout_sec,
            proof_bundle=risk_decision.proof_bundle,
        )
        approval_wait_sec = time.monotonic() - approval_start
        if not approved:
            denied = fail(
                "SSH execution denied by user.",
                metadata={
                    "approval_denied": True,
                    "command": command,
                    "cwd": str(getattr(policy_state, "cwd", ".") or "."),
                    "timeout_sec": timeout_sec,
                    "host": host,
                },
            )
            denied["status"] = "denied"
            return denied

    result = await run_ssh_command(
        host=host,
        command=command,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        stdin_data=stdin_data,
        state=state,
        harness=harness,
    )
    if isinstance(result, dict) and isinstance(result.get("metadata"), dict):
        if approval_wait_sec > 0:
            result["metadata"]["approval_wait_sec"] = round(approval_wait_sec, 3)
    return result


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
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Start an interactive SSH command and keep stdin/stdout open."""
    if password and str(password).startswith("[REDACTED"):
        return fail(
            "The SSH password provided was literally redacted. Ask the human user to provide the actual password.",
            metadata={"host": host, "command": command, "reason": "redacted_password_provided"},
        )
    try:
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
    # ssh_session_start is the designated tool for interactive installers;
    # do not block it with the remote-installer preflight guard.

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

    proc = await create_process(
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
    }
    _SSH_INTERACTIVE_SESSIONS[session_id] = session

    async def collect(stream: Any, key: str) -> None:
        await read_stream_chunks(stream, session[key], chunk_size=4096, idle_timeout_sec=None)

    session["tasks"] = [
        asyncio.create_task(collect(proc.stdout, "stdout")),
        asyncio.create_task(collect(proc.stderr, "stderr")),
    ]
    await asyncio.sleep(0.2)
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
    return ok(_interactive_session_snapshot(session_id, session, max_chars=max_chars), metadata={"interactive_session": True})


async def ssh_session_send(
    *,
    session_id: str,
    input: str,
    wait_sec: float = 0.5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.get(str(session_id or "").strip())
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    proc = session.get("proc")
    if getattr(proc, "returncode", None) is not None:
        return fail(
            "SSH interactive session has already exited.",
            metadata=_interactive_session_snapshot(session_id, session, max_chars=max_chars),
        )
    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return fail("SSH interactive session stdin is unavailable.", metadata={"session_id": session_id})
    stdin.write(str(input or "").encode("utf-8"))
    await stdin.drain()
    await asyncio.sleep(max(0.0, min(float(wait_sec or 0), 30.0)))
    return ok(_interactive_session_snapshot(session_id, session, max_chars=max_chars), metadata={"interactive_session": True})


async def ssh_session_close(
    *,
    session_id: str,
    terminate: bool = True,
    max_chars: int = 6000,
) -> dict[str, Any]:
    session = _SSH_INTERACTIVE_SESSIONS.pop(str(session_id or "").strip(), None)
    if not isinstance(session, dict):
        return fail("Unknown SSH interactive session.", metadata={"session_id": session_id})
    proc = session.get("proc")
    if terminate and getattr(proc, "returncode", None) is None:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
    for task in session.get("tasks", []):
        if hasattr(task, "cancel"):
            task.cancel()
    return ok(_interactive_session_snapshot(session_id, session, max_chars=max_chars), metadata={"interactive_session": True})


async def _run_remote_installer_preflight_probes(
    *,
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    """Run automated SSH probes to discover installer environment state."""
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

    try:
        full_cmd, env_overrides = _build_ssh_command(
            host=host,
            command=probe_command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
        )
        proc = await create_process(
            command=full_cmd,
            cwd=state.cwd if state else ".",
            env_overrides=env_overrides,
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
                f"Retry with `ssh_exec` and the flag `{probes['noninteractive_flags'][0]}`."
            )
        else:
            probes["recommended_approach"] = (
                "This appears to be an interactive installer. "
                "Use `ssh_session_start` to run it with a pseudo-terminal, "
                "then answer prompts with `ssh_session_send`."
            )
    except asyncio.TimeoutError:
        probes["probe_error"] = "Preflight probes timed out after 30s"
    except Exception as exc:
        probes["probe_error"] = str(exc)

    return probes


async def run_ssh_command(
    *,
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
    stdin_data: str | None = None,
) -> dict[str, Any]:
    """Run a generated SSH command and return the same result shape as ssh_exec."""
    if password and str(password).startswith("[REDACTED"):
        return fail(
            "The SSH password provided was literally redacted. This means you have lost access to the real password due to security scrubbing. "
            "You MUST ask the human user to provide the actual password in plain text. Do NOT retry this command blindly.",
            metadata={
                "host": host,
                "command": command,
                "reason": "redacted_password_provided",
            },
        )
    strict_host_key_mode = "accept-new"
    try:
        host, user = normalize_ssh_target(host=host, user=user)
    except ValueError as exc:
        return fail(
            str(exc),
            metadata={
                "host": host,
                "command": command,
                "user": user,
                "reason": "invalid_ssh_target",
            },
        )
    command, stripped_root_sudo = _strip_redundant_root_sudo(command, user)
    requested_timeout_sec = timeout_sec
    timeout_sec = _installer_command_suggested_timeout(command, timeout_sec)

    yes_pipe_guard = _interactive_installer_yes_pipe_guard(command, tool_name="ssh_exec")
    if yes_pipe_guard is not None:
        metadata = dict(yes_pipe_guard.get("metadata") or {})
        metadata.update(
            {
                "host": host,
                "user": user,
                **_ssh_execution_debug_metadata(
                    password=password,
                    identity_file=identity_file,
                    strict_host_key_checking=strict_host_key_mode,
                ),
            }
        )
        yes_pipe_guard["metadata"] = metadata
        return yes_pipe_guard

    foreground_guard = _foreground_command_guard(command, tool_name="ssh_exec")
    if foreground_guard is not None:
        metadata = dict(foreground_guard.get("metadata") or {})
        if stripped_root_sudo:
            metadata["stripped_redundant_root_sudo"] = True
        metadata.update(
            {
                "host": host,
                "user": user,
                **_ssh_execution_debug_metadata(
                    password=password,
                    identity_file=identity_file,
                    strict_host_key_checking=strict_host_key_mode,
                ),
            }
        )
        foreground_guard["metadata"] = metadata
        return foreground_guard

    preflight_guard = None
    if stdin_data is None:
        preflight_guard = _remote_installer_preflight_guard(
            command,
            host=host,
            user=user,
            state=state,
        )
    if preflight_guard is not None:
        metadata = dict(preflight_guard.get("metadata") or {})
        # If the guard already detected a hard failure (missing/corrupt files),
        # preserve that actionable error instead of running probes.
        if metadata.get("reason") == "remote_installer_preflight_failed":
            if stripped_root_sudo:
                metadata["stripped_redundant_root_sudo"] = True
            metadata.update(
                {
                    "host": host,
                    "user": user,
                    **_ssh_execution_debug_metadata(
                        password=password,
                        identity_file=identity_file,
                        strict_host_key_checking=strict_host_key_mode,
                    ),
                }
            )
            preflight_guard["metadata"] = metadata
            return preflight_guard

        # Run automatic environment probes before returning the block
        probes = await _run_remote_installer_preflight_probes(
            host=host,
            command=command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
            state=state,
            harness=harness,
        )
        if stripped_root_sudo:
            metadata["stripped_redundant_root_sudo"] = True
        metadata.update(
            {
                "host": host,
                "user": user,
                "preflight_probes": probes,
                "suggested_tool_after_preflight": (
                    "ssh_session_start" if probes.get("is_interactive") else "ssh_exec"
                ),
                **_ssh_execution_debug_metadata(
                    password=password,
                    identity_file=identity_file,
                    strict_host_key_checking=strict_host_key_mode,
                ),
            }
        )

        # Auto-clear preflight if basic checks pass so the model can retry immediately
        if probes.get("script_exists") and probes.get("script_executable"):
            _mark_remote_installer_preflight_clean(
                state, host=host, user=user, cwd=probes.get("cwd", "")
            )

        # Expose interactive session tools when the installer requires interactivity
        if probes.get("is_interactive"):
            _expose_interactive_session_tools(state)

        # Build enriched, actionable error text
        parts: list[str] = []
        parts.append("Remote installer environment scan completed.")
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
            parts.append(f"- Script: {probes['script_path']} (NOT FOUND)")
            if probes.get("cwd"):
                parts.append(f"  Check the correct path in `{probes['cwd']}/` and retry.")

        if probes.get("repo_clean"):
            parts.append("- Git repo: clean")
        elif probes.get("cwd"):
            parts.append("- Git repo: dirty or not a git repository")

        if probes.get("noninteractive_flags"):
            parts.append(
                f"- Non-interactive flags detected: {', '.join(probes['noninteractive_flags'])}"
            )
        if probes.get("preseed_files"):
            parts.append(
                f"- Preseed/config files found: {', '.join(probes['preseed_files'])}"
            )

        parts.append(f"\n{probes['recommended_approach']}")

        if not probes.get("script_exists"):
            parts.append(
                "\nAction required: Verify the installer path before retrying."
            )

        preflight_guard["error"] = "\n".join(parts)
        preflight_guard["metadata"] = metadata
        return preflight_guard

    try:
        full_cmd, env_overrides = _build_ssh_command(
            host=host,
            command=command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
        )
    except FileNotFoundError as exc:
        if str(exc) == "sshpass":
            return fail(
                "Password authentication requires `sshpass`, but it is not installed.",
                metadata={
                    "host": host,
                    "command": command,
                    "user": user,
                    "reason": "sshpass_missing",
                    **_ssh_execution_debug_metadata(
                        password=password,
                        identity_file=identity_file,
                        strict_host_key_checking="accept-new",
                    ),
                },
            )
        raise

    execution_debug_metadata = _ssh_execution_debug_metadata(
        password=password,
        identity_file=identity_file,
        strict_host_key_checking=strict_host_key_mode,
    )
    if timeout_sec != requested_timeout_sec:
        execution_debug_metadata["timeout_sec_auto_extended"] = {
            "from": requested_timeout_sec,
            "to": timeout_sec,
            "reason": "installer_like_command",
        }
    if stripped_root_sudo:
        execution_debug_metadata["stripped_redundant_root_sudo"] = True

    last_process_output: dict[str, Any] | None = None
    invalid_input_loop: dict[str, Any] | None = None

    def _build_process_output(
        *,
        stdout: str,
        stderr: str,
        exit_code: int | None,
        elapsed: float,
    ) -> dict[str, Any]:
        max_final_result = 256 * 1024
        final_stdout = stdout
        final_stderr = stderr
        if len(final_stdout) > max_final_result:
            final_stdout = final_stdout[:max_final_result] + "\n[OUTPUT TRUNCATED - TOO LARGE]"
        if len(final_stderr) > max_final_result:
            final_stderr = final_stderr[:max_final_result] + "\n[OUTPUT TRUNCATED - TOO LARGE]"
        return {
            "stdout": final_stdout,
            "stderr": final_stderr,
            "exit_code": exit_code,
            "metrics": {
                "duration_sec": round(elapsed, 3) if isinstance(elapsed, (int, float)) else 0.0,
                "host": host,
                "user": user,
            },
        }

    async def _run_ssh_process(command_text: str, stdin_payload: str | None = None) -> tuple[dict[str, Any], asyncio.subprocess.Process | None]:
        nonlocal last_process_output
        start_time = time.time()
        proc = await create_process(
            command=command_text,
            cwd=state.cwd if state else ".",
            env_overrides=env_overrides,
            harness=harness,
            stdin=asyncio.subprocess.PIPE if stdin_payload is not None else asyncio.subprocess.DEVNULL,
        )
        if stdin_payload is not None and proc.stdin is not None:
            proc.stdin.write(stdin_payload.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

        stdout_data: list[str] = []
        stderr_data: list[str] = []
        invalid_input_detector = InvalidInputLoopDetector()
        stream_emitter = BufferedUIEventEmitter(
            harness=harness,
            event_type=UIEventType.SHELL_STREAM,
        )

        async def read_stream(stream: Any, out_list: list[str]) -> None:
            async def handle_chunk(chunk_str: str) -> None:
                nonlocal invalid_input_loop
                if invalid_input_loop is None:
                    loop_metadata = invalid_input_detector.observe(chunk_str)
                    if loop_metadata is not None:
                        invalid_input_loop = {
                            **loop_metadata,
                            "command": command,
                            "tool_name": "ssh_exec",
                            "host": host,
                            "user": user,
                        }
                        if proc.returncode is None:
                            try:
                                proc.terminate()
                            except Exception:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                await stream_emitter.emit_text(chunk_str)

            await read_stream_chunks(stream, out_list, chunk_size=4096, on_chunk=handle_chunk, idle_timeout_sec=30)

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(proc.stdout, stdout_data),
                    read_stream(proc.stderr, stderr_data),
                    proc.wait(),
                ),
                timeout=timeout_sec,
            )
        finally:
            elapsed = time.time() - start_time
            last_process_output = _build_process_output(
                stdout="".join(stdout_data),
                stderr="".join(stderr_data),
                exit_code=proc.returncode,
                elapsed=elapsed,
            )
            await stream_emitter.flush()

        return last_process_output, proc

    proc = None
    try:
        output, proc = await _run_ssh_process(full_cmd, stdin_data)
        retry_metadata: dict[str, Any] = {}
        if int(output.get("exit_code") or 0) != 0 and _ssh_accept_new_is_incompatible(str(output.get("stderr") or "")):
            strict_host_key_mode = "no"
            full_cmd, env_overrides = _build_ssh_command(
                host=host,
                command=command,
                user=user,
                port=port,
                identity_file=identity_file,
                password=password,
                strict_host_key_checking=strict_host_key_mode,
            )
            execution_debug_metadata = _ssh_execution_debug_metadata(
                password=password,
                identity_file=identity_file,
                strict_host_key_checking=strict_host_key_mode,
            )
            if stripped_root_sudo:
                execution_debug_metadata["stripped_redundant_root_sudo"] = True
            output, proc = await _run_ssh_process(full_cmd, stdin_data)
            retry_metadata = {
                "ssh_option_retry": "strict_host_key_checking_no",
                "ssh_option_retry_reason": "accept_new_incompatible",
            }

        if invalid_input_loop is not None:
            return fail(
                "SSH command stopped after repeated invalid interactive input. "
                "Use documented non-interactive flags, a config/preseed file, or an explicit prompt answer script.",
                metadata={
                    "output": output,
                    "output_received": bool(
                        str(output.get("stdout") or "").strip()
                        or str(output.get("stderr") or "").strip()
                    ),
                    "failure_kind": "interactive_input_loop",
                    "ssh_error_class": "interactive_invalid_input_loop",
                    "ssh_transport_succeeded": True,
                    **execution_debug_metadata,
                    **retry_metadata,
                    **invalid_input_loop,
                },
            )

        if proc.returncode != 0:
            err_output = output.get("stderr", "")
            if not isinstance(err_output, str):
                err_output = str(err_output or "")
            # Diagnostic probes that explicitly report "not found" are informational
            # successes, not execution failures. Treating them as ok prevents the
            # harness from entering a repair loop over a negative result.
            if (
                int(proc.returncode) == 1
                and _ssh_diagnostic_not_found(command, output)
            ):
                return ok(output, metadata={**execution_debug_metadata, **retry_metadata})
            failure_kind = _ssh_failure_kind(
                exit_code=int(proc.returncode),
                stderr=err_output,
            )
            ssh_error_class = _ssh_error_class(
                exit_code=int(proc.returncode),
                stderr=err_output,
            )
            hints = []
            if failure_kind == "transport":
                error_msg = err_output.strip() or f"SSH transport failed with exit code {proc.returncode}"
            else:
                error_msg = err_output.strip() or f"Remote SSH command exited with code {proc.returncode}"
                hints.append(
                    "SSH transport appears to have succeeded; inspect the remote command, stdout, and exit code to decide whether the probe simply returned a non-zero status."
                )
            if "Permission denied" in error_msg:
                if password:
                    hints.append("Check the SSH username/password and verify that password authentication is enabled on the remote host.")
                else:
                    hints.append("Check if SSH keys are correctly configured on the remote host.")
            if "Connection timed out" in error_msg:
                hints.append("Verify the host is reachable and the port is open.")

            return fail(
                error_msg,
                metadata={
                    "output": output,
                    "output_received": bool(
                        str(output.get("stdout") or "").strip()
                        or str(output.get("stderr") or "").strip()
                    ),
                    "hints": hints,
                    "failure_kind": failure_kind,
                    "failure_mode": ssh_error_class,
                    "ssh_error_class": ssh_error_class,
                    "ssh_transport_succeeded": failure_kind == "remote_command",
                    **execution_debug_metadata,
                    **retry_metadata,
                },
            )

        return ok(output, metadata={**execution_debug_metadata, **retry_metadata})

    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        if proc is not None:
            for pipe in (proc.stdout, proc.stderr, proc.stdin):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass
        output = last_process_output if isinstance(last_process_output, dict) else {}
        combined_output = f"{output.get('stdout', '')}{output.get('stderr', '')}"
        detected_prompt = _detect_interactive_prompt(combined_output)
        if detected_prompt is not None:
            return fail(
                "SSH command appears to be waiting for interactive input. "
                "Use documented non-interactive flags/config when available, or retry with `ssh_session_start` and answer prompts with `ssh_session_send`.",
                metadata={
                    "output": output,
                    "output_received": True,
                    "failure_kind": "interactive_prompt_wait",
                    "ssh_error_class": "interactive_prompt_wait",
                    "ssh_transport_succeeded": True,
                    "detected_prompt": detected_prompt,
                    "suggested_tools": ["ssh_session_start", "ssh_session_read", "ssh_session_send", "ssh_session_close"],
                    **execution_debug_metadata,
                },
            )
        return fail(
            f"SSH command timed out after {timeout_sec}s",
            metadata={
                "output": output,
                "output_received": bool(
                    str(output.get("stdout") or "").strip()
                    or str(output.get("stderr") or "").strip()
                ),
                "failure_kind": "timeout",
                "ssh_error_class": "command_timeout",
                "ssh_transport_succeeded": bool(
                    str(output.get("stdout") or "").strip()
                    or str(output.get("stderr") or "").strip()
                ),
                **execution_debug_metadata,
            },
        )
    except Exception as exc:
        return fail(
            f"SSH execution error: {str(exc)}",
            metadata=execution_debug_metadata,
        )
    finally:
        if proc is not None:
            for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None), getattr(proc, "stdin", None)):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass
        if harness and proc and hasattr(harness, "_active_processes"):
            harness._active_processes.discard(proc)
