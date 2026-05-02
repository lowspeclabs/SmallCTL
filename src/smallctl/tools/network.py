from __future__ import annotations

import asyncio
import asyncio.subprocess
import shlex
import shutil
import time
from typing import Any, TYPE_CHECKING

from ..models.events import UIEvent, UIEventType
from .common import fail, ok
from ..risk_policy import evaluate_risk_policy
from ..state import LoopState
from .shell import create_process
from .process_streams import read_stream_chunks
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
    if user_text is not None:
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

    target = f"{user}@{host}" if user else host
    ssh_args.extend([target, command])
    return _shell_join([*command_args, *ssh_args]), env_overrides


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
    if risk_decision.requires_approval and callable(approval_fn) and approval_available:
        approved = await approval_fn(
            command=command,
            cwd=str(getattr(policy_state, "cwd", ".") or "."),
            timeout_sec=timeout_sec,
            proof_bundle=risk_decision.proof_bundle,
        )
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

    return await run_ssh_command(
        host=host,
        command=command,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )


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

    async def _run_ssh_process(command_text: str, stdin_payload: str | None = None) -> tuple[dict[str, Any], asyncio.subprocess.Process | None]:
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
        stream_emitter = BufferedUIEventEmitter(
            harness=harness,
            event_type=UIEventType.SHELL_STREAM,
        )

        async def read_stream(stream: Any, out_list: list[str]) -> None:
            async def handle_chunk(chunk_str: str) -> None:
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
            await stream_emitter.flush()

        elapsed = time.time() - start_time
        final_stdout = "".join(stdout_data)
        final_stderr = "".join(stderr_data)

        max_final_result = 256 * 1024
        if len(final_stdout) > max_final_result:
            final_stdout = final_stdout[:max_final_result] + "\n[OUTPUT TRUNCATED - TOO LARGE]"
        if len(final_stderr) > max_final_result:
            final_stderr = final_stderr[:max_final_result] + "\n[OUTPUT TRUNCATED - TOO LARGE]"

        return {
            "stdout": final_stdout,
            "stderr": final_stderr,
            "exit_code": proc.returncode,
            "metrics": {
                "duration_sec": round(elapsed, 3) if isinstance(elapsed, (int, float)) else 0.0,
                "host": host,
                "user": user,
            },
        }, proc

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
            output, proc = await _run_ssh_process(full_cmd, stdin_data)
            retry_metadata = {
                "ssh_option_retry": "strict_host_key_checking_no",
                "ssh_option_retry_reason": "accept_new_incompatible",
            }

        if proc.returncode != 0:
            err_output = output.get("stderr", "")
            if not isinstance(err_output, str):
                err_output = str(err_output or "")
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
                    "hints": hints,
                    "failure_kind": failure_kind,
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
        for pipe in (proc.stdout, proc.stderr, proc.stdin):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass
        return fail(
            f"SSH command timed out after {timeout_sec}s",
            metadata=execution_debug_metadata,
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
