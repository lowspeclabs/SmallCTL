from __future__ import annotations

import re
import shutil
from typing import Any

from .ssh_parsing import normalize_ssh_target, shell_join
from .ansi_utils import detect_tui_application

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
    ("choice", re.compile(r"Choice:\s*\[(?P<choices>[^\]]+)\]", re.IGNORECASE)),
    ("yes_no", re.compile(r"(?P<question>[^\n\r]{0,120}\?)\s*\(?(?P<default>[yYnN])/?[yYnN]\)?", re.IGNORECASE)),
    ("free_text", re.compile(r"(?P<question>[^\n\r]{0,240}:\s*)$", re.IGNORECASE)),
)
_SSH_ACCEPT_NEW_INCOMPATIBLE_MARKERS = (
    "keyword stricthostkeychecking extra arguments at end of line",
    "bad configuration option",
)


def build_ssh_command(
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
    return shell_join([*command_args, *ssh_args]), env_overrides


def detect_interactive_prompt(text: str) -> dict[str, Any] | None:
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


def ssh_accept_new_is_incompatible(stderr: str) -> bool:
    lowered = str(stderr or "").strip().lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in _SSH_ACCEPT_NEW_INCOMPATIBLE_MARKERS):
        return "stricthostkeychecking" in lowered or "accept-new" in lowered
    return "accept-new" in lowered and "unsupported option" in lowered


def ssh_failure_kind(*, exit_code: int, stderr: str) -> str:
    lowered = str(stderr or "").strip().lower()
    if any(marker in lowered for marker in _SSH_TRANSPORT_FAILURE_MARKERS):
        return "transport"
    if exit_code == 255 and not lowered:
        return "transport"
    return "remote_command"


def ssh_error_class(*, exit_code: int, stderr: str) -> str:
    lowered = str(stderr or "").strip().lower()
    if detect_tui_application(stderr) is not None and (
        "installer" in lowered or "dialog" in lowered or "error opening terminal" in lowered
    ):
        return "interactive_installer_blocked"
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
    return "transport_failure" if ssh_failure_kind(exit_code=exit_code, stderr=stderr) == "transport" else "remote_exit_nonzero"


def ssh_diagnostic_not_found(command: str, output: dict[str, Any]) -> bool:
    """Return True when an exit-1 SSH result is a diagnostic 'not found' probe."""
    stderr = str(output.get("stderr") or "").lower()
    stdout = str(output.get("stdout") or "").lower()
    combined = stdout + " " + stderr
    return any(marker in combined for marker in _SSH_DIAGNOSTIC_NOT_FOUND_MARKERS)


def ssh_execution_debug_metadata(
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
