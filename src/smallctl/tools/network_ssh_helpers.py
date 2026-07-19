from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import Any

from .ssh_parsing import normalize_ssh_target, shell_join
from .ansi_utils import detect_tui_application

_SSH_DIAGNOSTIC_NOT_FOUND_MARKERS = (
    "not found",
    "could not be found",
    "no such file",
    "command not found",
)
_KNOWN_HOSTS_ADDED_RE = re.compile(
    r"Warning: Permanently added '[^']+' \([^)]+\) to the list of known hosts\.\r?\n?",
    re.IGNORECASE,
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
_SSH_COMPOSITE_ERROR_MARKERS = (
    "command line error:",
    "error opening terminal:",
    "error: problems in request:",
    "error: unable to find a match:",
    "dnf search: error:",
    "unit ",
    "could not be found",
    "cannot access",
    "no such file or directory",
)
_SSH_COMPOSITE_SHELL_TOKENS = ("&&", "||", ";", "|", "tee ", "head ", "tail ")
_SSH_PACKAGE_OR_INSTALL_TOKENS = (
    " apt ",
    " apt-get ",
    " dnf ",
    " yum ",
    " apk ",
    " pacman ",
    " zypper ",
    " install ",
    " installer",
    " bash ",
    " systemctl ",
    " service ",
)


def ssh_command_is_package_manager_install(command: str) -> bool:
    """Return True when the command runs a package manager install/reinstall."""
    normalized = f" {str(command or '').lower()} "
    manager_tokens = {" apt ", " apt-get ", " dnf ", " yum ", " apk ", " pacman ", " zypper "}
    has_manager = any(token in normalized for token in manager_tokens)
    if not has_manager:
        return False
    return any(token in normalized for token in (" install ", " reinstall ", " groupinstall "))


def ssh_timeout_suggests_verify(command: str) -> str:
    """Return a short hint telling the model to verify before retrying a timed-out install."""
    if ssh_command_is_package_manager_install(command):
        return (
            "The package-manager install timed out locally, but the remote process may still be running. "
            "Before retrying, verify the current state with package-presence and service checks "
            "(e.g., `dnf list installed <pkg>`, `systemctl status <svc>`)."
        )
    return ""


def filter_ssh_known_hosts_warning(stderr: str) -> tuple[str, bool]:
    """Strip the noisy 'Permanently added ... to the list of known hosts' line from SSH stderr.

    Returns the cleaned stderr and a flag indicating whether the warning was removed.
    Callers may use the flag to record that a new host key was accepted.
    """
    if not stderr:
        return stderr, False
    filtered = _KNOWN_HOSTS_ADDED_RE.sub("", stderr)
    return filtered, len(filtered) < len(stderr)


_INTERACTIVE_PROMPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("choice", re.compile(r"Choice:\s*\[(?P<choices>[^\]]+)\]", re.IGNORECASE)),
    ("yes_no", re.compile(r"(?P<question>[^\n\r]{0,120}\?)\s*\(?(?P<default>[yYnN])/?[yYnN]\)?", re.IGNORECASE)),
    ("free_text", re.compile(r"(?P<question>[^\n\r]{0,240}:\s*)$", re.IGNORECASE)),
)
_SSH_ACCEPT_NEW_INCOMPATIBLE_MARKERS = (
    "keyword stricthostkeychecking extra arguments at end of line",
    "bad configuration option",
)
_SSH_STRICT_HOST_KEY_MODES = {"accept-new", "yes", "no", "off"}
_DEFAULT_SSH_STRICT_HOST_KEY_MODE = "accept-new"


class SSHStrictHostKeyConfigError(ValueError):
    """Raised when ssh_strict_host_key_checking holds an unsupported value."""


def normalize_strict_host_key_mode(value: Any) -> str:
    """Return a supported StrictHostKeyChecking mode.

    An empty/None value falls back to the default (accept-new). Any other
    unsupported value raises a typed configuration error instead of silently
    downgrading to a default the operator did not ask for.
    """
    mode = str(value or "").strip().lower()
    if not mode:
        return _DEFAULT_SSH_STRICT_HOST_KEY_MODE
    if mode == "off":
        return "no"
    if mode in _SSH_STRICT_HOST_KEY_MODES:
        return mode
    supported = sorted(_SSH_STRICT_HOST_KEY_MODES)
    raise SSHStrictHostKeyConfigError(
        f"Invalid ssh_strict_host_key_checking value {value!r}; "
        f"expected one of: {', '.join(supported)}."
    )


def resolve_ssh_strict_host_key_mode(harness: Any) -> str:
    """Resolve the configured StrictHostKeyChecking mode from harness config."""
    config = getattr(harness, "config", None)
    configured = getattr(config, "ssh_strict_host_key_checking", "")
    return normalize_strict_host_key_mode(configured)


def build_ssh_command(
    *,
    host: str,
    command: str,
    user: str | None,
    port: int,
    identity_file: str | None,
    password: str | None,
    strict_host_key_checking: str = _DEFAULT_SSH_STRICT_HOST_KEY_MODE,
    force_tty: bool = False,
) -> tuple[str, dict[str, str] | None, str | None]:
    """Build an SSH command string.

    When a password is supplied, the password is written to a temporary file
    with mode ``0600`` and ``sshpass -f`` is used so the secret is not exposed
    in process environment variables. Callers are responsible for deleting the
    returned ``password_file_path`` once the process has completed.
    """
    host, user = normalize_ssh_target(host=host, user=user)
    strict_host_key_checking = normalize_strict_host_key_mode(strict_host_key_checking)
    ssh_args = [
        "-p", str(port),
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", f"StrictHostKeyChecking={strict_host_key_checking}",
    ]
    env_overrides: dict[str, str] | None = None
    password_file_path: str | None = None

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
        fd, password_file_path = tempfile.mkstemp(prefix="sshpass_", suffix=".txt", text=True)
        try:
            os.chmod(password_file_path, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(password)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(password_file_path)
            except FileNotFoundError:
                pass
            raise
        command_args = ["sshpass", "-f", password_file_path, "ssh"]
    else:
        ssh_args.extend(["-o", "BatchMode=yes"])
        command_args = ["ssh"]

    if identity_file:
        ssh_args.extend(["-i", identity_file])
    if force_tty:
        ssh_args.append("-tt")

    target = f"{user}@{host}" if user else host
    ssh_args.extend(["--", target, command])
    return shell_join([*command_args, *ssh_args]), env_overrides, password_file_path


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


def is_purely_diagnostic(command: str) -> bool:
    """Return True if the command is purely diagnostic and not mutating."""
    cmd_lower = str(command or "").lower()
    
    # Handle chained commands using ;, &&, || by checking each component
    if any(chain_token in cmd_lower for chain_token in (";", "&&", "||")):
        parts = re.split(r'&&|\|\||;', command)
        return all(is_purely_diagnostic(part.strip()) for part in parts if part.strip())
        
    # Exclude redirection to files (which writes/mutates)
    # Filter out standard error redirections like '2>&1' or '2>/dev/null' or '2>&2'
    clean_cmd = re.sub(r'2\s*>\s*&\s*1|2\s*>\s*/dev/null|2\s*>\s*&\s*2|2\s*>\s*&\s*-\s*|>\s*/dev/null', '', cmd_lower)
    if '>' in clean_cmd:
        return False
        
    # Split into words to check for single-word mutating keywords
    words = set(re.findall(r'\b[a-z0-9_-]+\b', cmd_lower))
    mutating_single_words = {
        "chown", "chmod", "rm", "rmdir", "mkdir", "mv", "cp", "touch", 
        "dd", "tee", "ln", "tar", "unzip", "zip"
    }
    if words.intersection(mutating_single_words):
        return False
        
    # For package managers, exclude if they are running a mutating action
    pkg_mutating_patterns = [
        r'\b(apt|apt-get|dnf|yum|apk|pacman|zypper)\b.*\b(install|reinstall|remove|autoremove|purge|upgrade|dist-upgrade|full-upgrade|update|clean)\b',
    ]
    for pattern in pkg_mutating_patterns:
        if re.search(pattern, cmd_lower):
            return False

    # For git, exclude if it runs a mutating subcommand
    git_mutating_patterns = [
        r'\bgit\s+(clone|pull|push|checkout|commit|add|reset|merge|rebase|init)\b'
    ]
    for pattern in git_mutating_patterns:
        if re.search(pattern, cmd_lower):
            return False

    # Multi-word mutating check
    mutating_multi_patterns = [
        r'\bdocker\s+(run|rm|restart|stop|start|exec|create|pull|push|build)\b',
        r'\bsystemctl\s+(start|stop|restart|enable|disable|reload)\b',
        r'\bservice\s+[a-z0-9_-]+\s+(start|stop|restart|reload)\b',
    ]
    for pattern in mutating_multi_patterns:
        if re.search(pattern, cmd_lower):
            return False
            
    # List of diagnostic command keywords/roots
    diagnostic_keywords = {
        "ls", "cat", "grep", "egrep", "fgrep", "stat", "test", "file", "find",
        "du", "df", "head", "tail", "wc", "which", "type", "hostname", "whoami",
        "id", "uname", "ps", "md5sum", "sha256sum", "apt", "dnf", "yum", "apk",
        "pacman", "zypper", "git", "systemctl", "service", "echo", "getenforce",
        "sestatus", "aa-status",
    }
    if words.intersection(diagnostic_keywords):
        return True
        
    if "docker" in words:
        diag_docker_actions = {"ps", "inspect", "images", "logs", "diff"}
        docker_match = re.search(r'\bdocker\s+([a-z]+)\b', cmd_lower)
        if docker_match and docker_match.group(1) in diag_docker_actions:
            return True
            
    return False


def ssh_diagnostic_not_found(command: str, output: dict[str, Any]) -> bool:
    """Return True when an SSH result is a diagnostic 'not found' probe.

    Covers the common non-zero statuses produced by informational probes:
    exit 1/2 (grep no match, missing file) and exit 127 (command not found
    because the tool being probed is not installed).
    """
    if not is_purely_diagnostic(command):
        return False

    stderr = str(output.get("stderr") or "").lower()
    stdout = str(output.get("stdout") or "").lower()
    combined = stdout + " " + stderr

    if any(marker in combined for marker in _SSH_DIAGNOSTIC_NOT_FOUND_MARKERS):
        return True

    try:
        exit_code = int(output.get("exit_code") or 0)
    except (TypeError, ValueError):
        exit_code = 0

    if exit_code in (1, 2):
        cmd_lower = command.lower()
        if any(g in cmd_lower for g in ("grep", "egrep", "fgrep")):
            if not stdout.strip() and not stderr.strip():
                return True

    if exit_code == 127:
        return True

    return False


def ssh_semantic_failure(command: str, output: dict[str, Any]) -> str:
    """Return a high-confidence remote failure reason hidden behind exit 0.

    Many model-generated SSH commands use pipelines or `|| echo`, causing the
    outer shell to return 0 even when the meaningful command failed. Keep this
    intentionally narrow so ordinary diagnostic probes remain informational.
    """
    try:
        exit_code = int(output.get("exit_code") or 0)
    except (TypeError, ValueError):
        exit_code = 0
    if exit_code != 0:
        return ""
    command_text = str(command or "")
    normalized_command = f" {command_text.lower()} "
    if not any(token in normalized_command for token in _SSH_COMPOSITE_SHELL_TOKENS):
        return ""
    if not any(token in normalized_command for token in _SSH_PACKAGE_OR_INSTALL_TOKENS):
        return ""
    stdout = str(output.get("stdout") or "")
    stderr = str(output.get("stderr") or "")
    combined = f"{stdout}\n{stderr}".strip()
    lowered = combined.lower()
    if not lowered:
        return ""
    for marker in _SSH_COMPOSITE_ERROR_MARKERS:
        if marker not in lowered:
            continue
        if marker == "unit " and "could not be found" not in lowered:
            continue
        first_line = next((line.strip() for line in combined.splitlines() if marker in line.lower()), "")
        return first_line or marker.strip(":")
    return ""


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
        "ssh_auth_transport": "sshpass_file" if password_text else "ssh",
        "ssh_password_provided": bool(password_text),
        "ssh_identity_file_supplied": bool(identity_file_text),
        "ssh_strict_host_key_checking": strict_host_key_checking,
    }
