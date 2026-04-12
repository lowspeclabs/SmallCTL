from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PHASES = ("explore", "plan", "author", "execute", "verify", "repair")
STAGED_THOUGHT_ARCHITECTURES = {"multi_phase_discovery", "staged_reasoning"}


@dataclass(frozen=True)
class PhaseContract:
    phase: str
    focus: str
    prompt_priority: str
    blocked_tools: tuple[str, ...] = ()
    required_handoffs: tuple[str, ...] = ()
    allow_tool_reuse: bool = True

    def blocks(self, tool_name: str) -> bool:
        return tool_name in self.blocked_tools

    def summary(self) -> str:
        return f"{self.phase}: {self.focus}"


_PHASE_CONTRACTS: dict[str, PhaseContract] = {
    "explore": PhaseContract(
        phase="explore",
        focus="gather observations, verify facts, and collect open questions",
        prompt_priority="verified observations and unanswered questions",
        blocked_tools=("file_write", "file_append", "file_patch", "shell_exec", "ssh_exec", "task_complete", "task_fail"),
        required_handoffs=("ContextBrief",),
    ),
    "plan": PhaseContract(
        phase="plan",
        focus="turn evidence into hypotheses and an executable plan",
        prompt_priority="compressed evidence and candidate causes",
        blocked_tools=("file_write", "file_append", "file_patch", "shell_exec", "ssh_exec", "task_complete", "task_fail"),
        required_handoffs=("ContextBrief",),
    ),
    "author": PhaseContract(
        phase="author",
        focus="make bounded implementation changes from an approved plan",
        prompt_priority="approved execution plan and target files",
        blocked_tools=("task_complete", "task_fail"),
        required_handoffs=("ExecutionPlan",),
    ),
    "execute": PhaseContract(
        phase="execute",
        focus="run approved actions and verify their effect",
        prompt_priority="approved plan, evidence support, and approval state",
        blocked_tools=("task_fail",),
        required_handoffs=("ExecutionPlan",),
    ),
    "verify": PhaseContract(
        phase="verify",
        focus="compare observed state against expected outcomes",
        prompt_priority="acceptance criteria and recent verification evidence",
        blocked_tools=("file_write", "file_append", "file_patch", "task_complete", "task_fail"),
        required_handoffs=("ExecutionPlan",),
    ),
    "repair": PhaseContract(
        phase="repair",
        focus="recover from a failed verifier or execution step",
        prompt_priority="failure evidence, changed files, and the active write session",
        blocked_tools=("task_complete", "task_fail"),
        required_handoffs=("ExecutionPlan",),
    ),
}


def normalize_phase(value: str | None) -> str:
    phase = (value or "explore").strip().lower()
    if phase not in PHASES:
        return "explore"
    return phase


def phase_contract(phase: str | None) -> PhaseContract:
    return _PHASE_CONTRACTS[normalize_phase(phase)]


def is_staged_thought_architecture(strategy: dict[str, Any] | None) -> bool:
    if not isinstance(strategy, dict):
        return False
    architecture = str(strategy.get("thought_architecture") or "").strip().lower()
    return architecture in STAGED_THOUGHT_ARCHITECTURES


def is_phase_contract_active(strategy: dict[str, Any] | None) -> bool:
    return is_staged_thought_architecture(strategy)


def filter_phase_blocked_tools(
    pending_tool_calls: list[Any],
    *,
    phase: str,
) -> tuple[list[Any], list[str]]:
    contract = phase_contract(phase)
    if not pending_tool_calls:
        return [], []
    allowed: list[Any] = []
    blocked: list[str] = []
    for call in pending_tool_calls:
        tool_name = str(getattr(call, "tool_name", "") or "")
        if tool_name and contract.blocks(tool_name):
            blocked.append(tool_name)
            continue
        allowed.append(call)
    return allowed, blocked
