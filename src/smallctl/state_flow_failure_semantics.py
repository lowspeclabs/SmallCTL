from __future__ import annotations

from typing import Any

_FAILURE_TEXT_MARKERS = (
    "error:",
    "failed",
    "failure",
    "missing",
    "does not exist",
    "not found",
    "verifier_failed",
)

_READ_COMMAND_MARKERS = (
    "grep ",
    "rg ",
    "find ",
    "findstr ",
    "journalctl",
    "tail ",
    "head ",
    "cat ",
    "ls ",
    "dmesg",
    "systemctl status",
)


def evidence_has_failure_semantics(evidence: Any) -> bool:
    if getattr(evidence, "negative", False):
        return True
    metadata = getattr(evidence, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if metadata.get("success") is False:
        return True
    exit_code = metadata.get("exit_code")
    if exit_code is not None and int(exit_code) != 0:
        return True
    if metadata.get("failure_mode"):
        return True
    if str(metadata.get("verifier_verdict") or "").strip().lower() == "fail":
        return True
    if metadata.get("stderr"):
        return True
    statement = str(getattr(evidence, "statement", "") or "").lower()
    if any(marker in statement for marker in _FAILURE_TEXT_MARKERS):
        return True
    tool_name = str(getattr(evidence, "tool_name", "") or "").strip().lower()
    if tool_name in {"shell_exec", "ssh_exec"}:
        command = str(metadata.get("command") or "").lower()
        exit_code = metadata.get("exit_code")
        if (
            any(marker in command for marker in _READ_COMMAND_MARKERS)
            and exit_code is not None
            and int(exit_code) != 0
        ):
            return True
    return False


def artifact_has_failure_semantics(artifact: Any) -> bool:
    metadata = getattr(artifact, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if metadata.get("success") is False:
        return True
    exit_code = metadata.get("exit_code")
    if exit_code is not None and int(exit_code) != 0:
        return True
    if metadata.get("failure_mode"):
        return True
    if str(metadata.get("verifier_verdict") or "").strip().lower() == "fail":
        return True
    if metadata.get("stderr"):
        return True
    summary = str(getattr(artifact, "summary", "") or "").lower()
    if any(marker in summary for marker in _FAILURE_TEXT_MARKERS):
        return True
    tool_name = str(getattr(artifact, "tool_name", "") or "").strip().lower()
    if tool_name in {"shell_exec", "ssh_exec"}:
        command = str(metadata.get("command") or "").lower()
        exit_code = metadata.get("exit_code")
        if (
            any(marker in command for marker in _READ_COMMAND_MARKERS)
            and exit_code is not None
            and int(exit_code) != 0
        ):
            return True
    return False
