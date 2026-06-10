from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreflightDecision:
    action: str = "allow"
    risk: str = ""
    findings: list[str] = field(default_factory=list)
    shell_required: bool = False
    executable: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "risk": self.risk,
            "findings": list(self.findings),
            "shell_required": self.shell_required,
            "executable": self.executable,
            "reason": self.reason,
        }


_DESTRUCTIVE_COMMANDS = frozenset({
    "rm", "rmdir", "shred", "dd", "mkfs", "mkswap", "fdisk", "parted",
    "wipefs", "truncate", "shred",
})

_PACKAGE_MANAGERS = frozenset({
    "apt", "apt-get", "dnf", "yum", "pacman", "zypper", "apk",
    "brew", "port", "pkg", "emerge",
})

_SERVICE_MANAGERS = frozenset({
    "systemctl", "service", "supervisorctl", "rc-service", "launchctl",
    "initctl", "sv", "s6-svc",
})

_LARGE_OUTPUT_COMMANDS = frozenset({
    "find", "locate", "grep", "rg", "ag", "ack",
})


def _extract_executable(command: str) -> str:
    stripped = command.strip()
    if not stripped:
        return ""
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    for token in tokens:
        token = token.strip()
        if not token or token.startswith("-") or token.startswith("$"):
            continue
        if token in ("sudo", "su", "doas", "pkexec", "env", "nice", "nohup", "time"):
            continue
        return token.split("/")[-1]
    return ""


def _looks_like_destructive(words: list[str]) -> bool:
    for word in words:
        base = word.split("/")[-1]
        if base in _DESTRUCTIVE_COMMANDS:
            return True
        if base == "git" and any(w in ("clean", "reset", "checkout") for w in words):
            return True
        if base == "docker" and any(w in ("rmi", "rm", "prune", "kill") for w in words):
            return True
    return False


def _looks_like_sudo(words: list[str]) -> bool:
    return any(w in ("sudo", "su", "doas", "pkexec") for w in words)


def _looks_like_package_manager(words: list[str]) -> bool:
    for word in words:
        base = word.split("/")[-1]
        if base in _PACKAGE_MANAGERS:
            return True
    return False


def _looks_like_service_manager(words: list[str]) -> bool:
    for word in words:
        base = word.split("/")[-1]
        if base in _SERVICE_MANAGERS:
            return True
    return False


def _looks_like_large_output(words: list[str]) -> bool:
    for word in words:
        base = word.split("/")[-1]
        if base in _LARGE_OUTPUT_COMMANDS:
            return True
    return False


def _tokenize(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _make_cache_key(command: str, level: int, block_destructive: bool,
                    require_approval_for_sudo: bool, require_approval_for_package_changes: bool,
                    require_approval_for_service_changes: bool, warn_large_output: bool,
                    cwd: str) -> str:
    return (
        f"cmd:{command}|lvl:{level}|bd:{block_destructive}|"
        f"ras:{require_approval_for_sudo}|rapc:{require_approval_for_package_changes}|"
        f"rasc:{require_approval_for_service_changes}|wlo:{warn_large_output}|cwd:{cwd}"
    )


def cached_shell_preflight(
    command: str,
    *,
    env: dict[str, str],
    level: int,
    block_destructive: bool,
    require_approval_for_sudo: bool,
    require_approval_for_package_changes: bool,
    require_approval_for_service_changes: bool,
    warn_large_output: bool,
    cache: dict[str, Any] | None,
    cwd: str,
) -> PreflightDecision:
    if level <= 0:
        return PreflightDecision(action="allow")

    cache_key = _make_cache_key(
        command, level, block_destructive,
        require_approval_for_sudo, require_approval_for_package_changes,
        require_approval_for_service_changes, warn_large_output, cwd,
    )
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, PreflightDecision):
            return cached
        if isinstance(cached, dict):
            return PreflightDecision(**cached)

    words = _tokenize(command)
    executable = _extract_executable(command)
    findings: list[str] = []
    risk = ""
    action = "allow"
    reason = ""

    if block_destructive and _looks_like_destructive(words):
        risk = "destructive"
        findings.append(f"Detected potentially destructive command: {executable}")
        action = "block" if level >= 2 else "approval_required"
        reason = "Command may delete or destroy data"
    elif require_approval_for_sudo and _looks_like_sudo(words):
        risk = "sudo"
        findings.append("Command requires elevated privileges")
        action = "approval_required"
        reason = "Command uses sudo or equivalent"
    elif require_approval_for_package_changes and _looks_like_package_manager(words):
        risk = "package_change"
        findings.append("Command may install, remove, or upgrade packages")
        action = "approval_required"
        reason = "Package manager detected"
    elif require_approval_for_service_changes and _looks_like_service_manager(words):
        risk = "service_change"
        findings.append("Command may start, stop, or modify services")
        action = "approval_required"
        reason = "Service manager detected"
    elif warn_large_output and _looks_like_large_output(words):
        risk = "large_output"
        findings.append("Command may produce large output")
        action = "warn"
        reason = "Potentially large output"

    decision = PreflightDecision(
        action=action,
        risk=risk,
        findings=findings,
        shell_required="sudo" in words or executable in _SERVICE_MANAGERS,
        executable=executable,
        reason=reason,
    )

    if cache is not None:
        cache[cache_key] = decision.to_dict()

    return decision


def to_compact_message(decision: PreflightDecision) -> str:
    if not decision.risk:
        return "Shell preflight: no risk detected."
    parts = [f"Shell preflight {decision.action}: {decision.risk}"]
    if decision.reason:
        parts.append(decision.reason)
    if decision.findings:
        parts.extend(f"- {f}" for f in decision.findings)
    return "\n".join(parts)
