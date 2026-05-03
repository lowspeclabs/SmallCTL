from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..state import clip_text_value, json_safe_value

TurnType = Literal["NEW_TASK", "ITERATION", "CORRECTION", "CLARIFICATION", "RETRY"]
PreviousTaskRelevance = Literal["none", "low", "medium", "high"]


@dataclass
class TaskResetPolicy:
    keep_prior_result: bool
    keep_raw_tool_history: bool = False
    force_fresh_plan: bool = True
    preserve_guard_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep_prior_result": self.keep_prior_result,
            "keep_raw_tool_history": self.keep_raw_tool_history,
            "force_fresh_plan": self.force_fresh_plan,
            "preserve_guard_context": self.preserve_guard_context,
        }


@dataclass
class FollowupClassification:
    status: str = "active"
    turn_type: TurnType = "NEW_TASK"
    previous_task_relevance: PreviousTaskRelevance = "none"
    user_goal: str = ""
    success_condition: str = ""
    reset_policy: TaskResetPolicy = field(
        default_factory=lambda: TaskResetPolicy(keep_prior_result=False)
    )
    allowed_paths: list[str] = field(default_factory=list)
    allowed_artifacts: list[str] = field(default_factory=list)
    ignored_context: list[str] = field(default_factory=list)
    failure_summary: str = ""
    verification_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "turn_type": self.turn_type,
            "previous_task_relevance": self.previous_task_relevance,
            "user_goal": _clip(self.user_goal, 320),
            "success_condition": _clip(self.success_condition, 220),
            "reset_policy": self.reset_policy.to_dict(),
            "allowed_paths": _clip_list(self.allowed_paths, limit=12, item_limit=160),
            "allowed_artifacts": _clip_list(self.allowed_artifacts, limit=8, item_limit=40),
            "ignored_context": _clip_list(self.ignored_context, limit=6, item_limit=120),
            "failure_summary": _clip(self.failure_summary, 220),
            "verification_hint": _clip(self.verification_hint, 160),
        }
        return json_safe_value(payload)


@dataclass
class FollowupSignals:
    has_prior_task: bool
    has_overlap: bool = False
    explicit_conflicting_target: bool = False
    selected_action_option: bool = False
    contextual_reference: bool = False
    same_target_delta: bool = False
    corrective_resteer: bool = False
    quality_followup: bool = False
    remote_live_correction: bool = False
    remote_clarification: bool = False
    guard_failure_context: bool = False
    retry_language: bool = False


def classify_followup_transaction(
    *,
    raw_task: str,
    effective_task: str,
    previous_task: str,
    signals: FollowupSignals,
    task_mode: str = "",
    allowed_paths: list[str] | None = None,
    allowed_artifacts: list[str] | None = None,
    failure_summary: str = "",
    verification_hint: str = "",
) -> FollowupClassification:
    raw = _clip(raw_task, 320)
    effective = _clip(effective_task or raw_task, 320)
    prior = _clip(previous_task, 240)

    turn_type: TurnType
    relevance: PreviousTaskRelevance
    if not signals.has_prior_task:
        turn_type = "NEW_TASK"
        relevance = "none"
    elif signals.explicit_conflicting_target:
        turn_type = "NEW_TASK"
        relevance = "low"
    elif signals.guard_failure_context and signals.retry_language:
        turn_type = "RETRY"
        relevance = "high"
    elif signals.remote_clarification:
        turn_type = "CLARIFICATION"
        relevance = "high"
    elif (
        signals.corrective_resteer
        or signals.quality_followup
        or signals.remote_live_correction
    ):
        turn_type = "CORRECTION"
        relevance = "high"
    elif (
        signals.selected_action_option
        or signals.contextual_reference
        or signals.same_target_delta
        or signals.has_overlap
    ):
        turn_type = "ITERATION"
        relevance = "high"
    else:
        turn_type = "NEW_TASK"
        relevance = "low" if prior else "none"

    return FollowupClassification(
        status="active",
        turn_type=turn_type,
        previous_task_relevance=relevance,
        user_goal=effective or raw,
        success_condition=_success_condition_for(turn_type, effective or raw, task_mode=task_mode),
        reset_policy=_reset_policy_for(turn_type),
        allowed_paths=_clip_list(allowed_paths or [], limit=12, item_limit=160),
        allowed_artifacts=_clip_list(allowed_artifacts or [], limit=8, item_limit=40),
        ignored_context=_ignored_context_for(turn_type),
        failure_summary=_clip(failure_summary, 220),
        verification_hint=_clip(verification_hint, 160),
    )


def transaction_from_scratchpad(scratchpad: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scratchpad, dict):
        return {}
    tx = scratchpad.get("_task_transaction")
    if isinstance(tx, dict) and tx:
        return tx
    handoff = scratchpad.get("_last_task_handoff")
    if isinstance(handoff, dict) and handoff:
        if str(handoff.get("status") or "").strip().lower() in {
            "closed",
            "failed",
            "aborted",
            "superseded",
        }:
            return {}
        return handoff
    return {}


def recovery_context_lines(transaction: dict[str, Any]) -> list[str]:
    if not isinstance(transaction, dict) or not transaction:
        return []
    lines: list[str] = []
    turn_type = str(transaction.get("turn_type") or "").strip()
    if turn_type:
        lines.append(f"Current turn type: {turn_type}.")
    success = str(transaction.get("success_condition") or "").strip()
    if success:
        lines.append(f"Current success condition: {success}.")
    return lines[:2]


def _reset_policy_for(turn_type: TurnType) -> TaskResetPolicy:
    if turn_type == "CLARIFICATION":
        return TaskResetPolicy(
            keep_prior_result=True,
            keep_raw_tool_history=False,
            force_fresh_plan=False,
            preserve_guard_context=False,
        )
    if turn_type in {"CORRECTION", "RETRY"}:
        return TaskResetPolicy(
            keep_prior_result=True,
            keep_raw_tool_history=False,
            force_fresh_plan=True,
            preserve_guard_context=True,
        )
    if turn_type == "ITERATION":
        return TaskResetPolicy(
            keep_prior_result=True,
            keep_raw_tool_history=False,
            force_fresh_plan=True,
            preserve_guard_context=False,
        )
    return TaskResetPolicy(
        keep_prior_result=False,
        keep_raw_tool_history=False,
        force_fresh_plan=True,
        preserve_guard_context=False,
    )


def _success_condition_for(turn_type: TurnType, goal: str, *, task_mode: str = "") -> str:
    if turn_type == "CLARIFICATION":
        return "Clarified constraints are recorded; do not execute unless the user asks."
    if turn_type == "CORRECTION":
        return "The corrected approach is applied and focused verification is complete."
    if turn_type == "RETRY":
        return "The previous failure is addressed and focused verification is complete."
    if turn_type == "ITERATION":
        return "Requested delta is applied and focused verification is complete."
    if task_mode == "chat":
        return "The new request is answered."
    return "The new task is completed and verified where practical."


def _ignored_context_for(turn_type: TurnType) -> list[str]:
    if turn_type == "NEW_TASK":
        return ["previous task plan", "previous verifier commands", "old tool attempts"]
    if turn_type in {"ITERATION", "CORRECTION", "RETRY"}:
        return ["old plan steps", "old verifier commands", "raw tool history"]
    return []


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return clip_text_value(text, limit=limit)[0]


def _clip_list(values: list[Any], *, limit: int, item_limit: int) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = _clip(value, item_limit)
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned
