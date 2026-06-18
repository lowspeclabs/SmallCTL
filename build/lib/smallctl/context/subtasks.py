from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..state import EpisodicSummary, LoopState


@dataclass
class ChildRunRequest:
    brief: str
    phase: str
    depth: int = 1
    max_prompt_tokens: int | None = None
    recent_message_limit: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildRunResult:
    status: str
    summary: str
    artifact_ids: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    remaining_plan: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SubtaskRunner:
    def __init__(self, *, max_child_depth: int = 1) -> None:
        self.max_child_depth = max_child_depth

    def create_child_state(self, *, parent_state: LoopState, request: ChildRunRequest) -> LoopState:
        if request.depth > self.max_child_depth:
            raise ValueError(f"Child depth {request.depth} exceeds max_child_depth {self.max_child_depth}")
        child_state = LoopState(
            current_phase=request.phase,
            cwd=parent_state.cwd,
            recent_message_limit=request.recent_message_limit,
        )
        parent_thread_id = parent_state.thread_id or "thread"
        child_state.thread_id = (
            f"{parent_thread_id}.child.{len(parent_state.episodic_summaries) + 1:04d}"
        )
        child_state.run_brief.original_task = request.brief
        child_state.run_brief.current_phase_objective = request.phase
        child_state.run_brief.constraints = _coerce_string_list(request.metadata.get("constraints"))
        child_state.run_brief.acceptance_criteria = _coerce_string_list(
            request.metadata.get("acceptance_criteria")
        )
        child_state.working_memory.known_facts = _coerce_string_list(request.metadata.get("known_facts"))
        transaction = _transaction_for_child(parent_state, request)
        if transaction:
            user_goal = str(transaction.get("user_goal") or "").strip()
            if user_goal:
                child_state.run_brief.original_task = user_goal
            success_condition = str(transaction.get("success_condition") or "").strip()
            if success_condition:
                child_state.run_brief.acceptance_criteria = _dedupe(
                    child_state.run_brief.acceptance_criteria + [success_condition]
                )[-6:]
            turn_type = str(transaction.get("turn_type") or "").strip()
            allowed_paths = _coerce_string_list(transaction.get("allowed_paths"))
            if turn_type:
                child_state.run_brief.constraints = _dedupe(
                    child_state.run_brief.constraints + [f"turn_type={turn_type}"]
                )[-8:]
            if allowed_paths:
                child_state.run_brief.constraints = _dedupe(
                    child_state.run_brief.constraints
                    + [f"allowed_paths={', '.join(allowed_paths[:4])}"]
                )[-8:]
            failure_summary = str(transaction.get("failure_summary") or "").strip()
            if failure_summary:
                child_state.working_memory.failures = [failure_summary]
            verification_hint = str(transaction.get("verification_hint") or "").strip()
            if verification_hint:
                child_state.working_memory.next_actions = [verification_hint]
            allowed_artifacts = _coerce_string_list(transaction.get("allowed_artifacts"))
            for artifact_id in allowed_artifacts:
                record = parent_state.artifacts.get(artifact_id)
                if record is not None:
                    child_state.artifacts[artifact_id] = record
            child_state.scratchpad["_task_transaction"] = transaction
            child_state.scratchpad["allowed_artifacts"] = allowed_artifacts
            child_state.scratchpad["allowed_paths"] = allowed_paths
        child_state.scratchpad["subtask_depth"] = request.depth
        child_state.scratchpad["parent_phase"] = parent_state.current_phase
        child_state.scratchpad["parent_task"] = parent_state.run_brief.original_task
        child_state.scratchpad["parent_thread_id"] = parent_thread_id
        if request.max_prompt_tokens is not None:
            child_state.scratchpad["max_prompt_tokens"] = request.max_prompt_tokens
        return child_state

    def run(
        self,
        *,
        parent_state: LoopState,
        request: ChildRunRequest,
        executor: Callable[[LoopState, ChildRunRequest], ChildRunResult],
    ) -> ChildRunResult:
        child_state = self.create_child_state(parent_state=parent_state, request=request)
        result = executor(child_state, request)
        self.merge_result(parent_state=parent_state, request=request, result=result)
        return result

    def merge_result(
        self,
        *,
        parent_state: LoopState,
        request: ChildRunRequest,
        result: ChildRunResult,
    ) -> None:
        summary = EpisodicSummary(
            summary_id=f"child-{len(parent_state.episodic_summaries) + 1:04d}",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            decisions=result.decisions[:6],
            files_touched=result.files_touched[:8],
            remaining_plan=result.remaining_plan[:6],
            artifact_ids=result.artifact_ids[:8],
            notes=[f"subtask phase={request.phase}", f"status={result.status}", result.summary],
        )
        parent_state.episodic_summaries.append(summary)
        if result.summary:
            parent_state.working_memory.known_facts = _dedupe(
                parent_state.working_memory.known_facts + [result.summary]
            )[-12:]
        if result.remaining_plan:
            parent_state.working_memory.next_actions = _dedupe(
                parent_state.working_memory.next_actions + result.remaining_plan
            )[-8:]
        if result.decisions:
            parent_state.working_memory.decisions = _dedupe(
                parent_state.working_memory.decisions + result.decisions
            )[-8:]
        if result.artifacts:
            for aid, record in result.artifacts.items():
                if aid not in parent_state.artifacts:
                    parent_state.artifacts[aid] = record
        parent_state.touch()


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _transaction_for_child(parent_state: LoopState, request: ChildRunRequest) -> dict[str, Any]:
    metadata_tx = request.metadata.get("task_transaction")
    if isinstance(metadata_tx, dict) and metadata_tx:
        return dict(metadata_tx)
    scratchpad = parent_state.scratchpad if isinstance(getattr(parent_state, "scratchpad", None), dict) else {}
    tx = scratchpad.get("_task_transaction")
    if isinstance(tx, dict) and tx:
        return dict(tx)
    return {}
