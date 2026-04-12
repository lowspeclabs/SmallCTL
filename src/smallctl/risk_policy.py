from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from .reasoning_policy import (
    build_claim_proof_bundle,
    classify_task as _classify_task,
    has_supported_claim,
    task_requires_claim_support,
)
from .state import LoopState

_READ_ONLY_PHASES = {"explore", "verify"}
_SHELL_TOOLS = {"shell_exec", "ssh_exec"}
_SHELL_WRAPPER_TOKENS = {
    "bash",
    "sh",
    "zsh",
    "dash",
    "ksh",
    "pwsh",
    "powershell",
    "cmd",
    "cmd.exe",
}
_SHELL_WRAPPER_COMMAND_FLAGS = {"-c", "-lc", "/c", "-Command", "-command"}
_READ_ONLY_ROOT_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "grep",
    "rg",
    "cat",
    "head",
    "tail",
    "awk",
    "wc",
    "stat",
    "which",
    "pytest",
    "apt-cache",
    "journalctl",
    "uname",
    "id",
    "whoami",
}


@dataclass(frozen=True)
class RiskPolicyDecision:
    allowed: bool
    requires_approval: bool = False
    reason: str = ""
    proof_bundle: dict[str, Any] | None = None
    tool_risk: str = "medium"
    task_classification: str = "implementation"
    approval_kind: str = ""


def classify_task(state: LoopState) -> str:
    return _classify_task(state)


def build_risk_proof_bundle(
    state: LoopState,
    *,
    tool_name: str,
    tool_risk: str,
    phase: str,
    action: str = "",
    expected_effect: str = "",
    rollback: str = "",
    verification: str = "",
) -> dict[str, Any]:
    task_classification = classify_task(state)
    bundle = build_claim_proof_bundle(
        state,
        tool_name=tool_name,
        action=action,
        expected_effect=expected_effect,
        rollback=rollback,
        verification=verification,
    )
    bundle.update(
        {
            "phase": phase,
            "tool_risk": tool_risk,
            "task_classification": task_classification,
            "approval_kind": "shell" if tool_name in _SHELL_TOOLS else "",
            "read_only_phase": phase in _READ_ONLY_PHASES,
        }
    )
    return bundle


def evaluate_risk_policy(
    state: LoopState,
    *,
    tool_name: str,
    tool_risk: str,
    phase: str,
    action: str = "",
    expected_effect: str = "",
    rollback: str = "",
    verification: str = "",
    approval_available: bool = False,
) -> RiskPolicyDecision:
    task_classification = classify_task(state)
    proof_bundle = build_risk_proof_bundle(
        state,
        tool_name=tool_name,
        tool_risk=tool_risk,
        phase=phase,
        action=action,
        expected_effect=expected_effect,
        rollback=rollback,
        verification=verification,
    )
    if tool_risk == "low":
        return RiskPolicyDecision(
            allowed=True,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
        )

    if (
        task_classification == "diagnosis_remediation"
        and tool_name in _SHELL_TOOLS
        and _is_read_only_evidence_action(action)
    ):
        requires_approval = approval_available and tool_name in _SHELL_TOOLS and tool_risk in {"medium", "high"}
        return RiskPolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell" if tool_name in _SHELL_TOOLS else "",
        )

    if task_classification == "diagnosis_remediation" and not has_supported_claim(state):
        return RiskPolicyDecision(
            allowed=False,
            reason=(
                f"{tool_name} is blocked for diagnosis/remediation work until a supported claim exists. "
                "Record a confirmed claim with supporting evidence first."
            ),
            proof_bundle=proof_bundle,
            tool_risk=tool_risk,
            task_classification=task_classification,
            approval_kind="shell" if tool_name in _SHELL_TOOLS else "generic",
        )

    requires_approval = approval_available and tool_name in _SHELL_TOOLS and tool_risk in {"medium", "high"}
    return RiskPolicyDecision(
        allowed=True,
        requires_approval=requires_approval,
        proof_bundle=proof_bundle,
        tool_risk=tool_risk,
        task_classification=task_classification,
        approval_kind="shell" if tool_name in _SHELL_TOOLS else "generic" if tool_risk == "high" else "",
    )


def _is_read_only_evidence_action(action: str) -> bool:
    command = str(action or "").strip().lower()
    if not command:
        return False
    segments = _split_shell_segments(str(action or "").strip())
    return bool(segments) and all(_segment_is_read_only(segment) for segment in segments)


def _split_shell_segments(command: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    escape = False
    index = 0

    while index < len(command):
        char = command[index]

        if escape:
            current.append(char)
            escape = False
            index += 1
            continue

        if char == "\\" and quote != "'":
            current.append(char)
            escape = True
            index += 1
            continue

        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue

        if char == "`" or command.startswith("$(", index):
            return []

        if char in {"<", ">", "(", ")"}:
            return []

        if command.startswith("&&", index) or command.startswith("||", index):
            segment = "".join(current).strip()
            if not segment:
                return []
            segments.append(segment)
            current = []
            index += 2
            continue

        if char in {";", "|"}:
            segment = "".join(current).strip()
            if not segment:
                return []
            segments.append(segment)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    if quote or escape:
        return []

    segment = "".join(current).strip()
    if not segment:
        return []
    segments.append(segment)
    return segments


def _segment_is_read_only(segment: str) -> bool:
    command = str(segment or "").strip()
    if not command:
        return False

    unwrapped = _unwrap_shell_wrapper_command(command)
    if unwrapped and unwrapped != command:
        return _is_read_only_evidence_action(unwrapped)

    tokens = _leading_command_tokens(command)
    if not tokens:
        return False

    root = tokens[0].lower()
    if root in _READ_ONLY_ROOT_COMMANDS:
        return True
    if root == "sed":
        return len(tokens) >= 2 and tokens[1] == "-n"
    if root == "command":
        return len(tokens) >= 2 and tokens[1] == "-v"
    if root == "git":
        return len(tokens) >= 2 and tokens[1] in {"status", "diff", "show", "log"}
    if root in {"python", "python3"}:
        return len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest"
    if root == "dpkg":
        return len(tokens) >= 2 and tokens[1] in {"-l", "--list"}
    if root == "systemctl":
        return len(tokens) >= 2 and tokens[1] in {"status", "show", "is-active", "is-enabled", "list-units", "list-unit-files"}
    return False


def _unwrap_shell_wrapper_command(command: str) -> str | None:
    tokens = _shell_tokens(command)
    if not tokens:
        return None

    if tokens[0].lower() == "sudo":
        inner_tokens = tokens[1:]
        while inner_tokens and inner_tokens[0].startswith("-"):
            inner_tokens = inner_tokens[1:]
        return " ".join(inner_tokens) if inner_tokens else None

    if tokens[0].lower() == "env":
        inner_tokens = tokens[1:]
        while inner_tokens and (inner_tokens[0].startswith("-") or _looks_like_env_assignment(inner_tokens[0])):
            inner_tokens = inner_tokens[1:]
        return " ".join(inner_tokens) if inner_tokens else None

    if len(tokens) >= 3 and tokens[0].lower() in _SHELL_WRAPPER_TOKENS and tokens[1] in _SHELL_WRAPPER_COMMAND_FLAGS:
        return tokens[2]

    return None


def _leading_command_tokens(command: str) -> list[str]:
    current = command
    for _ in range(4):
        unwrapped = _unwrap_shell_wrapper_command(current)
        if unwrapped and unwrapped != current:
            current = unwrapped
            continue

        tokens = _shell_tokens(current)
        if not tokens:
            return []

        index = 0
        while index < len(tokens) and _looks_like_env_assignment(tokens[index]):
            index += 1
        return tokens[index:]

    return _shell_tokens(current)


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    key, _value = token.split("=", 1)
    return key.isidentifier()
