from __future__ import annotations

from typing import Any

from ..shell_utils import is_read_only_shell_evidence_action
from .scaling_constants import (
    HIGH_RISK_TOOLS,
    LOCAL_FILE_MUTATION_TOOLS,
    READ_ONLY_STAGED_TOOLS,
    REMOTE_MUTATION_TOOLS,
    SHELL_EXECUTION_TOOLS,
)


def is_read_only_tool_call(call: Any) -> bool:
    tool_name = str(getattr(call, "tool_name", "") or "").strip()
    if not tool_name:
        return False
    if tool_name in READ_ONLY_STAGED_TOOLS:
        return True
    if tool_name in {"shell_exec", "ssh_exec"}:
        args = getattr(call, "args", {}) or {}
        if not isinstance(args, dict):
            return False
        return is_read_only_shell_evidence_action(str(args.get("command") or ""))
    return False


def candidate_uses_only_read_only_tools(candidate: Any) -> bool:
    return bool(getattr(candidate, "pending_tool_calls", [])) and all(
        is_read_only_tool_call(call) for call in candidate.pending_tool_calls
    )


def candidate_uses_local_file_mutation_tools(candidate: Any) -> bool:
    return any(
        str(getattr(call, "tool_name", "") or "").strip() in LOCAL_FILE_MUTATION_TOOLS
        for call in getattr(candidate, "pending_tool_calls", [])
    )


def unsafe_branch_execution_reason(candidate: Any) -> str:
    remote_mutation_tools = [
        str(getattr(call, "tool_name", "") or "").strip()
        for call in getattr(candidate, "pending_tool_calls", [])
        if str(getattr(call, "tool_name", "") or "").strip() in REMOTE_MUTATION_TOOLS
    ]
    if remote_mutation_tools:
        return "unsafe_branch_tool:" + ",".join(sorted(set(remote_mutation_tools)))
    unsafe_shell_tools = [
        str(getattr(call, "tool_name", "") or "").strip()
        for call in getattr(candidate, "pending_tool_calls", [])
        if str(getattr(call, "tool_name", "") or "").strip() in SHELL_EXECUTION_TOOLS
        and not is_read_only_tool_call(call)
    ]
    if unsafe_shell_tools:
        return "unsafe_branch_tool:" + ",".join(sorted(set(unsafe_shell_tools)))
    return ""


def _risk_count(pending_tool_calls: list[Any]) -> int:
    return sum(1 for call in pending_tool_calls if str(getattr(call, "tool_name", "") or "").strip() in HIGH_RISK_TOOLS)


def score_proposal(
    pending_tool_calls: list[Any],
    *,
    allowed_tool_names: set[str],
    assistant_text: str = "",
) -> tuple[float, list[str]]:
    failed: list[str] = []
    if not pending_tool_calls:
        failed.append("no_tool_calls")
    unknown = [
        str(getattr(call, "tool_name", "") or "").strip()
        for call in pending_tool_calls
        if str(getattr(call, "tool_name", "") or "").strip() not in allowed_tool_names
    ]
    if unknown:
        failed.append("tool_not_allowed:" + ",".join(sorted(set(unknown))))
    missing_args = [
        str(getattr(call, "tool_name", "") or "").strip()
        for call in pending_tool_calls
        if not isinstance(getattr(call, "args", None), dict)
    ]
    if missing_args:
        failed.append("invalid_arguments:" + ",".join(sorted(set(missing_args))))

    score = 1.0
    if "no_tool_calls" in failed:
        score -= 0.55
    if unknown:
        score -= 0.75
    if missing_args:
        score -= 0.3
    risk_count = sum(1 for call in pending_tool_calls if str(getattr(call, "tool_name", "") or "").strip() in HIGH_RISK_TOOLS)
    score -= min(0.18, 0.06 * risk_count)
    if assistant_text.strip():
        score += 0.03
    return max(0.0, min(1.0, score)), failed


def select_best_proposal(candidates: list[Any]) -> Any | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (
            getattr(candidate, "score", 0.0),
            -len(getattr(candidate, "failed_criteria", [])),
            -_risk_count(getattr(candidate, "pending_tool_calls", [])),
            -getattr(candidate, "candidate_idx", 0),
        ),
        reverse=True,
    )[0]
