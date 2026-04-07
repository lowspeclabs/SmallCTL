from __future__ import annotations

import asyncio
import asyncio.subprocess
import shlex
import shutil
import time
from typing import Any, TYPE_CHECKING

from .common import fail, ok
from .shell import create_process

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
    if user_text is not None:
        raise ValueError("SSH target must be either `host` plus `user` or `user@host`.")
    return bare_host, embedded_user


def normalize_ssh_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}

    normalized = dict(arguments)
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
        return None
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
) -> tuple[str, dict[str, str] | None]:
    host, user = normalize_ssh_target(host=host, user=user)
    ssh_args = [
        "-p", str(port),
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
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

    target = f"{user}@{host}" if user else host
    ssh_args.extend([target, command])
    return _shell_join([*command_args, *ssh_args]), env_overrides


async def ssh_exec(
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    timeout_sec: int = 60,
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
    try:
        host, user = normalize_ssh_target(host=host, user=user)
        full_cmd, env_overrides = _build_ssh_command(
            host=host,
            command=command,
            user=user,
            port=port,
            identity_file=identity_file,
            password=password,
        )
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
    except FileNotFoundError as exc:
        if str(exc) == "sshpass":
            return fail(
                "Password authentication requires `sshpass`, but it is not installed.",
                metadata={
                    "host": host,
                    "command": command,
                    "user": user,
                    "reason": "sshpass_missing",
                },
            )
        raise

    start_time = time.time()
    proc = None
    try:
        proc = await create_process(
            command=full_cmd,
            cwd=state.cwd if state else ".",
            env_overrides=env_overrides,
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
                if password:
                    hints.append("Check the SSH username/password and verify that password authentication is enabled on the remote host.")
                else:
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
