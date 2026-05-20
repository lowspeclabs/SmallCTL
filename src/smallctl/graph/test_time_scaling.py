from __future__ import annotations

import asyncio
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from ..models.events import UIEvent, UIEventType
from ..recovery_metrics import increment_metric, recovery_metrics
from ..state import json_safe_value
from ..shell_utils import is_read_only_shell_evidence_action
from ..tools import ToolDispatcher, build_registry
from .model_call_nodes import _conversation_tool_calls_from_pending, model_call
from .plan_execution import PlanExecutionEngine
from .plan_verification import StepCompletionGate, compact_step_evidence
from .state import GraphRunState, PendingToolCall
from .tool_plan_schema import READONLY_TOOL_PLAN_TOOLS
from .tool_execution_nodes import dispatch_tools
from .tool_execution_persistence import persist_tool_results
from .tool_outcomes import apply_tool_outcomes


PROMPT_VARIANTS = (
    "Use the standard staged-step strategy. Prefer concrete tool progress and call step_complete only when verified.",
    "Be conservative: inspect or verify before mutation, prefer the smallest change, and avoid unrelated edits.",
    "Try an alternate route from the obvious one while still obeying the active step contract and tool allowlist.",
    "Debug first: identify the likely failure mode, gather targeted evidence, then act.",
    "Minimize risk: avoid shell or remote tools unless they are necessary for this exact step.",
)

HIGH_RISK_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "shell_exec",
    "ssh_exec",
    "ssh_file_write",
    "ssh_file_patch",
}

READ_ONLY_CONTROL_TOOLS = {"loop_status"}

READ_ONLY_STAGED_TOOLS = frozenset(set(READONLY_TOOL_PLAN_TOOLS) | READ_ONLY_CONTROL_TOOLS)

LOCAL_FILE_MUTATION_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
}

SHELL_EXECUTION_TOOLS = {"shell_exec", "ssh_exec"}
REMOTE_MUTATION_TOOLS = {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}


@dataclass(slots=True)
class ProposalCandidate:
    candidate_idx: int
    prompt_variant: str
    assistant_text: str = ""
    thinking_text: str = ""
    pending_tool_calls: list[PendingToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    score: float = 0.0
    failed_criteria: list[str] = field(default_factory=list)
    isolated: bool = False


@dataclass(slots=True)
class BranchExecutionResult:
    candidate: ProposalCandidate
    state: Any
    verification: Any = None
    score_value: float = 0.0
    passed_threshold: bool = False
    error: str = ""


@dataclass(slots=True)
class CandidateStateGuard:
    active_step_run_id: str
    step_sandbox_history: list[Any]
    tool_execution_records: dict[str, Any]
    recent_messages: list[Any]
    transcript_messages: list[Any]
    scratchpad: dict[str, Any]
    files_changed_this_cycle: list[str]
    write_session: Any
    artifacts: dict[str, Any]
    pending_interrupt: dict[str, Any] | None
    step_verification_result: Any
    tool_history: list[str]
    token_usage: int
    last_completion_tokens: int

    @classmethod
    def capture(cls, state: Any) -> "CandidateStateGuard":
        return cls(
            active_step_run_id=str(getattr(state, "active_step_run_id", "") or ""),
            step_sandbox_history=deepcopy(getattr(state, "step_sandbox_history", []) or []),
            tool_execution_records=deepcopy(getattr(state, "tool_execution_records", {}) or {}),
            recent_messages=deepcopy(getattr(state, "recent_messages", []) or []),
            transcript_messages=deepcopy(getattr(state, "transcript_messages", []) or []),
            scratchpad=deepcopy(getattr(state, "scratchpad", {}) or {}),
            files_changed_this_cycle=deepcopy(getattr(state, "files_changed_this_cycle", []) or []),
            write_session=deepcopy(getattr(state, "write_session", None)),
            artifacts=deepcopy(getattr(state, "artifacts", {}) or {}),
            pending_interrupt=deepcopy(getattr(state, "pending_interrupt", None)),
            step_verification_result=deepcopy(getattr(state, "step_verification_result", None)),
            tool_history=deepcopy(getattr(state, "tool_history", []) or []),
            token_usage=int(getattr(state, "token_usage", 0) or 0),
            last_completion_tokens=int(getattr(state, "last_completion_tokens", 0) or 0),
        )

    def restore(self, state: Any, *, keep_accounting: bool = True) -> None:
        current_token_usage = int(getattr(state, "token_usage", 0) or 0)
        current_last_completion_tokens = int(getattr(state, "last_completion_tokens", 0) or 0)
        state.active_step_run_id = self.active_step_run_id
        state.step_sandbox_history = deepcopy(self.step_sandbox_history)
        state.tool_execution_records = deepcopy(self.tool_execution_records)
        state.recent_messages = deepcopy(self.recent_messages)
        state.transcript_messages = deepcopy(self.transcript_messages)
        state.scratchpad = deepcopy(self.scratchpad)
        state.files_changed_this_cycle = deepcopy(self.files_changed_this_cycle)
        state.write_session = deepcopy(self.write_session)
        state.artifacts = deepcopy(self.artifacts)
        state.pending_interrupt = deepcopy(self.pending_interrupt)
        state.step_verification_result = deepcopy(self.step_verification_result)
        state.tool_history = deepcopy(self.tool_history)
        if keep_accounting:
            state.token_usage = current_token_usage
            state.last_completion_tokens = current_last_completion_tokens
        else:
            state.token_usage = self.token_usage
            state.last_completion_tokens = self.last_completion_tokens


@dataclass(slots=True)
class FileSnapshotGuard:
    cwd: Path
    snapshots: dict[Path, bytes | None]
    directories_existed: set[Path] = field(default_factory=set)

    @classmethod
    def capture(cls, *, cwd: str | Path, paths: list[str | Path]) -> "FileSnapshotGuard":
        root = Path(cwd).resolve()
        snapshots: dict[Path, bytes | None] = {}
        directories_existed: set[Path] = {root}
        for raw_path in paths:
            path = _resolve_snapshot_path(root, raw_path)
            if path is None:
                continue
            if path in snapshots:
                continue
            for parent in _snapshot_parent_dirs(root, path):
                if parent.exists() and parent.is_dir():
                    directories_existed.add(parent)
            try:
                snapshots[path] = path.read_bytes() if path.exists() and path.is_file() else None
            except OSError:
                snapshots[path] = None
        return cls(cwd=root, snapshots=snapshots, directories_existed=directories_existed)

    def restore(self) -> None:
        for path, content in self.snapshots.items():
            if content is None:
                try:
                    if path.exists() and path.is_file():
                        path.unlink()
                except OSError:
                    continue
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            except OSError:
                continue
        self._remove_new_empty_directories()

    def _remove_new_empty_directories(self) -> None:
        candidates: set[Path] = set()
        for path in self.snapshots:
            candidates.update(_snapshot_parent_dirs(self.cwd, path))
            if path.exists() and path.is_dir():
                candidates.add(path)
        for directory in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
            if directory == self.cwd or directory in self.directories_existed:
                continue
            try:
                directory.relative_to(self.cwd)
            except ValueError:
                continue
            try:
                directory.rmdir()
            except OSError:
                continue


def collect_step_snapshot_paths(step: Any) -> list[str]:
    paths: list[str] = []
    for output in getattr(step, "outputs_expected", []) or []:
        if getattr(output, "kind", "") == "file":
            ref = str(getattr(output, "ref", "") or "").strip()
            if ref:
                paths.append(ref)
    for spec in getattr(step, "verifiers", []) or []:
        if getattr(spec, "kind", "") not in {"file_exists", "file_changed", "syntax_ok", "file_count"}:
            continue
        args = getattr(spec, "args", {}) or {}
        if not isinstance(args, dict):
            continue
        ref = str(args.get("path") or args.get("ref") or "").strip()
        if ref:
            paths.append(ref)
    return list(dict.fromkeys(paths))


def collect_candidate_snapshot_paths(candidates: list["ProposalCandidate"]) -> list[str]:
    paths: list[str] = []
    for candidate in candidates:
        for call in candidate.pending_tool_calls:
            tool_name = str(getattr(call, "tool_name", "") or "").strip()
            if tool_name not in LOCAL_FILE_MUTATION_TOOLS:
                continue
            args = getattr(call, "args", {}) or {}
            if not isinstance(args, dict):
                continue
            for key in ("path", "target_path"):
                ref = str(args.get(key) or "").strip()
                if ref:
                    paths.append(ref)
    return list(dict.fromkeys(paths))


def build_candidate_prompts(
    base_messages: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    count = max(1, min(int(max_candidates or 1), len(PROMPT_VARIANTS)))
    prompts: list[tuple[str, list[dict[str, Any]]]] = []
    for variant in PROMPT_VARIANTS[:count]:
        messages = [json_safe_value(message) for message in base_messages]
        messages.append(
            {
                "role": "system",
                "content": "[TEST-TIME SCALING CANDIDATE]\n" + variant,
            }
        )
        prompts.append((variant, messages))
    return prompts


def score_proposal(
    pending_tool_calls: list[PendingToolCall],
    *,
    allowed_tool_names: set[str],
    assistant_text: str = "",
) -> tuple[float, list[str]]:
    failed: list[str] = []
    if not pending_tool_calls:
        failed.append("no_tool_calls")
    unknown = [
        call.tool_name
        for call in pending_tool_calls
        if str(call.tool_name or "").strip() not in allowed_tool_names
    ]
    if unknown:
        failed.append("tool_not_allowed:" + ",".join(sorted(set(unknown))))
    missing_args = [
        call.tool_name
        for call in pending_tool_calls
        if not isinstance(call.args, dict)
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
    risk_count = sum(1 for call in pending_tool_calls if call.tool_name in HIGH_RISK_TOOLS)
    score -= min(0.18, 0.06 * risk_count)
    if assistant_text.strip():
        score += 0.03
    return max(0.0, min(1.0, score)), failed


def select_best_proposal(candidates: list[ProposalCandidate]) -> ProposalCandidate | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            -len(candidate.failed_criteria),
            -_risk_count(candidate.pending_tool_calls),
            -candidate.candidate_idx,
        ),
        reverse=True,
    )[0]


async def run_proposal_scaling(
    graph_state: GraphRunState,
    deps: Any,
    *,
    base_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> ProposalCandidate | None:
    harness = deps.harness
    config = getattr(harness, "config", None)
    policy = str(getattr(config, "test_time_scaling_policy", "proposal_then_execute") or "").strip()
    if not policy:
        policy = "proposal_then_execute"
    increment_metric(harness.state, "test_time_scaling_attempts")
    await _emit_scaling_status(
        deps,
        "Scaling test-time proposals...",
        policy=policy,
        phase="proposal_start",
    )
    candidates = await generate_proposal_candidates(
        graph_state,
        deps,
        base_messages=base_messages,
        tools=tools,
    )
    if graph_state.final_result is not None:
        return None
    selected = select_best_proposal(candidates)
    if selected is None:
        return None
    increment_metric(harness.state, "test_time_scaling_candidates", len(candidates))
    if selected.failed_criteria:
        increment_metric(harness.state, "test_time_scaling_selected_with_warnings")
    else:
        increment_metric(harness.state, "test_time_scaling_clean_selection")
    metrics = recovery_metrics(harness.state)
    metrics["test_time_scaling_last"] = {
        "policy": policy,
        "candidate_count": len(candidates),
        "selected_candidate": selected.candidate_idx,
        "selected_score": round(selected.score, 3),
        "selected_latency_ms": round(selected.latency_ms, 3),
        "selected_token_cost": _usage_total_tokens(selected.usage),
        "failed_criteria": list(selected.failed_criteria),
    }
    _commit_selected_proposal(graph_state, harness, selected)
    harness.state.scratchpad["_test_time_scaling_last"] = {
        "policy": policy,
        "candidate_count": len(candidates),
        "selected_candidate": selected.candidate_idx,
        "selected_score": round(selected.score, 3),
        "failed_criteria": list(selected.failed_criteria),
    }
    await _emit_scaling_status(
        deps,
        f"Scaled {len(candidates)} candidates; selected #{selected.candidate_idx}.",
        policy=policy,
        phase="proposal_selected",
        candidate_count=len(candidates),
        selected_candidate=selected.candidate_idx,
        selected_score=round(selected.score, 3),
        candidate_history=_candidate_history(candidates, selected=selected),
    )
    return selected


async def run_sequential_branch_scaling(
    graph_state: GraphRunState,
    deps: Any,
    *,
    plan: Any,
    step: Any,
    base_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> ProposalCandidate | None:
    harness = deps.harness
    config = getattr(harness, "config", None)
    candidates = await generate_proposal_candidates(
        graph_state,
        deps,
        base_messages=base_messages,
        tools=tools,
    )
    if graph_state.final_result is not None or not candidates:
        return None
    await _emit_scaling_status(
        deps,
        f"Executing {len(candidates)} scaled branch candidates...",
        policy="sequential_branch",
        phase="branch_start",
        candidate_count=len(candidates),
        read_only_candidate_count=sum(1 for candidate in candidates if candidate_uses_only_read_only_tools(candidate)),
        candidate_history=_candidate_history(candidates),
    )

    state_guard = CandidateStateGuard.capture(harness.state)
    file_guard = FileSnapshotGuard.capture(
        cwd=getattr(harness.state, "cwd", ".") or ".",
        paths=collect_step_snapshot_paths(step) + collect_candidate_snapshot_paths(candidates),
    )
    original_step_run_id = str(getattr(harness.state, "active_step_run_id", "") or "")
    engine = PlanExecutionEngine(harness.state)
    gate = StepCompletionGate()
    best_candidate: ProposalCandidate | None = None
    best_score = -1.0
    threshold = float(getattr(config, "test_time_scaling_score_threshold", 0.85) or 0.85)
    isolated_branch_attempts = 0

    increment_metric(harness.state, "test_time_scaling_branch_attempts")
    read_only_candidates = [
        candidate
        for candidate in sorted(candidates, key=lambda item: item.candidate_idx)
        if candidate_uses_only_read_only_tools(candidate)
    ]
    read_only_results: dict[int, BranchExecutionResult] = {}
    if read_only_candidates:
        read_only_results = await _run_read_only_branch_candidates(
            graph_state,
            deps,
            candidates=read_only_candidates,
            step=step,
            original_step_run_id=original_step_run_id,
            threshold=threshold,
        )
        for result in sorted(read_only_results.values(), key=lambda item: item.candidate.candidate_idx):
            candidate = result.candidate
            if result.score_value > best_score:
                best_score = result.score_value
                best_candidate = candidate
            if result.passed_threshold:
                state_guard.restore(harness.state)
                _merge_branch_state(harness.state, result.state)
                verify_result = result.verification or await gate.verify_step(harness, step)
                evidence = compact_step_evidence(harness, step, verify_result)
                engine.complete_step(plan, step.step_id, evidence)
                _record_branch_metrics(
                    harness,
                    candidates,
                    candidate,
                    parallel_read_only_count=len(read_only_results),
                )
                graph_state.pending_tool_calls = []
                graph_state.last_tool_results = []
                graph_state.recorded_tool_call_ids = []
                graph_state.final_result = None
                graph_state.interrupt_payload = None
                graph_state.error = None
                await _emit_scaling_status(
                    deps,
                    f"Scaled {len(candidates)} branches; selected #{candidate.candidate_idx}.",
                    policy="sequential_branch",
                    phase="branch_selected",
                    candidate_count=len(candidates),
                    selected_candidate=candidate.candidate_idx,
                    selected_score=round(candidate.score, 3),
                    read_only_branch_parallel_count=len(read_only_results),
                    candidate_history=_candidate_history(candidates, selected=candidate),
                )
                return candidate

    for candidate in sorted(candidates, key=lambda item: item.candidate_idx):
        if candidate.candidate_idx in read_only_results:
            continue
        unsafe_reason = unsafe_branch_execution_reason(candidate)
        if unsafe_reason:
            candidate.score = 0.0
            candidate.failed_criteria = [unsafe_reason]
            if candidate.score > best_score:
                best_score = candidate.score
                best_candidate = candidate
            continue
        if candidate_uses_local_file_mutation_tools(candidate):
            isolated_result = await _run_isolated_local_branch_candidate(
                graph_state,
                deps,
                candidate=candidate,
                step=step,
                original_step_run_id=original_step_run_id,
                threshold=threshold,
            )
            isolated_branch_attempts += 1
            if isolated_result.score_value > best_score:
                best_score = isolated_result.score_value
                best_candidate = candidate
            if not isolated_result.passed_threshold:
                continue
        state_guard.restore(harness.state)
        file_guard.restore()
        branch_run_id = f"{original_step_run_id}-cand{candidate.candidate_idx}"
        harness.state.active_step_run_id = branch_run_id
        graph_state.pending_tool_calls = list(candidate.pending_tool_calls)
        graph_state.last_assistant_text = candidate.assistant_text
        graph_state.last_thinking_text = candidate.thinking_text
        graph_state.last_usage = dict(candidate.usage or {})
        graph_state.last_tool_results = []
        graph_state.final_result = None
        graph_state.interrupt_payload = None
        graph_state.error = None
        _commit_selected_proposal(graph_state, harness, candidate)
        await dispatch_tools(graph_state, deps)
        await persist_tool_results(graph_state, deps)
        await apply_tool_outcomes(graph_state, deps)
        if graph_state.final_result is not None or graph_state.interrupt_payload is not None:
            score_value = 0.0
        else:
            score = await gate.score_step(harness, step)
            score_value = score.score
            candidate.score = score.score
            candidate.failed_criteria = list(score.failed_criteria)
        if score_value > best_score:
            best_score = score_value
            best_candidate = candidate
        if score_value >= threshold:
            result = await gate.verify_step(harness, step)
            evidence = compact_step_evidence(harness, step, result)
            engine.complete_step(plan, step.step_id, evidence)
            _record_branch_metrics(harness, candidates, candidate)
            if isolated_branch_attempts:
                increment_metric(harness.state, "test_time_scaling_isolated_branch_attempts", isolated_branch_attempts)
            await _emit_scaling_status(
                deps,
                f"Scaled {len(candidates)} branches; selected #{candidate.candidate_idx}.",
                policy="sequential_branch",
                phase="branch_selected",
                candidate_count=len(candidates),
                selected_candidate=candidate.candidate_idx,
                selected_score=round(candidate.score, 3),
                candidate_history=_candidate_history(candidates, selected=candidate),
            )
            return candidate

    state_guard.restore(harness.state)
    file_guard.restore()
    graph_state.pending_tool_calls = []
    graph_state.last_tool_results = []
    graph_state.recorded_tool_call_ids = []
    graph_state.final_result = None
    graph_state.interrupt_payload = None
    graph_state.error = None
    all_fail_action = _all_fail_action(config)
    if best_candidate is not None:
        _record_branch_metrics(
            harness,
            candidates,
            best_candidate,
            failed=True,
            parallel_read_only_count=len(read_only_results),
            all_failed_action=all_fail_action,
        )
        if isolated_branch_attempts:
            increment_metric(harness.state, "test_time_scaling_isolated_branch_attempts", isolated_branch_attempts)
    if all_fail_action == "fail_step":
        reason = "Test-time scaling failed all candidate branches."
        if best_candidate is not None and best_candidate.failed_criteria:
            reason += " Best candidate failed: " + ", ".join(best_candidate.failed_criteria)
        engine.fail_step(plan, step.step_id, reason)
        if harness.state.pending_interrupt:
            graph_state.interrupt_payload = harness.state.pending_interrupt
    await _emit_scaling_status(
        deps,
        f"All {len(candidates)} scaled branches failed; action={all_fail_action}.",
        policy="sequential_branch",
        phase="branch_all_failed",
        candidate_count=len(candidates),
        selected_candidate=getattr(best_candidate, "candidate_idx", None),
        all_failed_action=all_fail_action,
        candidate_history=_candidate_history(candidates, selected=best_candidate),
    )
    return None


async def _run_read_only_branch_candidates(
    graph_state: GraphRunState,
    deps: Any,
    *,
    candidates: list[ProposalCandidate],
    step: Any,
    original_step_run_id: str,
    threshold: float,
) -> dict[int, BranchExecutionResult]:
    harness = deps.harness
    config = getattr(harness, "config", None)
    parallel_max = max(1, int(getattr(config, "test_time_scaling_parallel_max", 1) or 1))
    if parallel_max <= 1 or len(candidates) <= 1:
        results: dict[int, BranchExecutionResult] = {}
        for candidate in candidates:
            result = await _run_one_read_only_branch_candidate(
                graph_state,
                deps,
                candidate=candidate,
                step=step,
                original_step_run_id=original_step_run_id,
                semaphore=None,
                threshold=threshold,
            )
            results[candidate.candidate_idx] = result
            if result.passed_threshold:
                break
        return results

    semaphore = asyncio.Semaphore(parallel_max)
    tasks = [
        asyncio.create_task(
            _run_one_read_only_branch_candidate(
                graph_state,
                deps,
                candidate=candidate,
                step=step,
                original_step_run_id=original_step_run_id,
                semaphore=semaphore,
                threshold=threshold,
            )
        )
        for candidate in candidates
    ]
    results: dict[int, BranchExecutionResult] = {}
    increment_metric(harness.state, "test_time_scaling_parallel_read_only_branch_batches")
    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            results[result.candidate.candidate_idx] = result
            if result.passed_threshold:
                for pending in tasks:
                    if not pending.done():
                        pending.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                break
    finally:
        for pending in tasks:
            if not pending.done():
                pending.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def _run_one_read_only_branch_candidate(
    graph_state: GraphRunState,
    deps: Any,
    *,
    candidate: ProposalCandidate,
    step: Any,
    original_step_run_id: str,
    semaphore: asyncio.Semaphore | None,
    threshold: float,
) -> BranchExecutionResult:
    async def _execute() -> BranchExecutionResult:
        harness = deps.harness
        branch_harness = _clone_harness_for_candidate(harness)
        branch_deps = copy(deps)
        branch_deps.harness = branch_harness
        branch_deps.event_handler = None
        branch_run_id = f"{original_step_run_id}-cand{candidate.candidate_idx}"
        branch_harness.state.active_step_run_id = branch_run_id
        branch_graph_state = GraphRunState(
            loop_state=branch_harness.state,
            thread_id=graph_state.thread_id,
            run_mode=graph_state.run_mode,
            pending_tool_calls=list(candidate.pending_tool_calls),
            last_assistant_text=candidate.assistant_text,
            last_thinking_text=candidate.thinking_text,
            last_usage=dict(candidate.usage or {}),
        )
        _commit_selected_proposal(branch_graph_state, branch_harness, candidate)
        try:
            await dispatch_tools(branch_graph_state, branch_deps)
            await persist_tool_results(branch_graph_state, branch_deps)
            await apply_tool_outcomes(branch_graph_state, branch_deps)
            if branch_graph_state.final_result is not None or branch_graph_state.interrupt_payload is not None:
                candidate.score = 0.0
                candidate.failed_criteria = ["branch_interrupted_or_final"]
                return BranchExecutionResult(candidate=candidate, state=branch_harness.state)
            score = await StepCompletionGate().score_step(branch_harness, step)
            candidate.score = score.score
            candidate.failed_criteria = list(score.failed_criteria)
            verification = await StepCompletionGate().verify_step(branch_harness, step) if score.score >= threshold else None
            return BranchExecutionResult(
                candidate=candidate,
                state=branch_harness.state,
                verification=verification,
                score_value=score.score,
                passed_threshold=score.score >= threshold,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            candidate.score = 0.0
            candidate.failed_criteria = ["branch_error"]
            return BranchExecutionResult(
                candidate=candidate,
                state=branch_harness.state,
                error=str(exc),
            )

    if semaphore is None:
        return await _execute()
    async with semaphore:
        return await _execute()


async def generate_proposal_candidates(
    graph_state: GraphRunState,
    deps: Any,
    *,
    base_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[ProposalCandidate]:
    harness = deps.harness
    config = getattr(harness, "config", None)
    max_candidates = int(getattr(config, "test_time_scaling_max_candidates", 3) or 3)
    min_candidates = int(getattr(config, "test_time_scaling_min_candidates", 2) or 2)
    max_candidates = max(min_candidates, max_candidates)
    policy = str(getattr(config, "test_time_scaling_policy", "proposal_then_execute") or "").strip()
    if not policy:
        policy = "proposal_then_execute"
    threshold = float(getattr(config, "test_time_scaling_score_threshold", 0.85) or 0.85)
    tool_names = _tool_names(tools)
    prompts = build_candidate_prompts(base_messages, max_candidates=max_candidates)
    candidates: list[ProposalCandidate] = []
    parallel_max = max(1, int(getattr(config, "test_time_scaling_parallel_max", 1) or 1))
    if parallel_max > 1 and len(prompts) > 1:
        return await _generate_proposal_candidates_parallel(
            graph_state,
            deps,
            prompts=prompts,
            tools=tools,
            tool_names=tool_names,
            threshold=threshold,
            first_pass=policy == "first_pass",
            parallel_max=parallel_max,
        )
    snapshot = CandidateStateGuard.capture(harness.state)
    original_recorded = list(graph_state.recorded_tool_call_ids)
    for idx, (variant, messages) in enumerate(prompts, start=1):
        snapshot.restore(harness.state)
        graph_state.pending_tool_calls = []
        graph_state.recorded_tool_call_ids = []
        graph_state.last_assistant_text = ""
        graph_state.last_thinking_text = ""
        graph_state.last_usage = {}
        started = time.perf_counter()
        await model_call(graph_state, deps, messages=messages, tools=tools)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if graph_state.final_result is not None:
            snapshot.restore(harness.state)
            graph_state.recorded_tool_call_ids = original_recorded
            return []
        score, failed = score_proposal(
            graph_state.pending_tool_calls,
            allowed_tool_names=tool_names,
            assistant_text=graph_state.last_assistant_text,
        )
        candidates.append(
            ProposalCandidate(
                candidate_idx=idx,
                prompt_variant=variant,
                assistant_text=graph_state.last_assistant_text,
                thinking_text=graph_state.last_thinking_text,
                pending_tool_calls=list(graph_state.pending_tool_calls),
                usage=dict(graph_state.last_usage or {}),
                latency_ms=elapsed_ms,
                score=score,
                failed_criteria=failed,
            )
        )
        if policy == "first_pass" and score >= threshold and not failed:
            break
    snapshot.restore(harness.state)
    graph_state.recorded_tool_call_ids = original_recorded
    return candidates


async def _generate_proposal_candidates_parallel(
    graph_state: GraphRunState,
    deps: Any,
    *,
    prompts: list[tuple[str, list[dict[str, Any]]]],
    tools: list[dict[str, Any]],
    tool_names: set[str],
    threshold: float,
    first_pass: bool,
    parallel_max: int,
) -> list[ProposalCandidate]:
    harness = deps.harness
    snapshot = CandidateStateGuard.capture(harness.state)
    original_recorded = list(graph_state.recorded_tool_call_ids)
    semaphore = asyncio.Semaphore(max(1, parallel_max))
    candidates: list[ProposalCandidate] = []
    tasks = [
        asyncio.create_task(
            _generate_one_proposal_candidate(
                graph_state,
                deps,
                idx=idx,
                variant=variant,
                messages=messages,
                tools=tools,
                tool_names=tool_names,
                semaphore=semaphore,
            )
        )
        for idx, (variant, messages) in enumerate(prompts, start=1)
    ]
    try:
        if first_pass:
            for task in asyncio.as_completed(tasks):
                candidate = await task
                candidates.append(candidate)
                if candidate.score >= threshold and not candidate.failed_criteria:
                    for pending in tasks:
                        if not pending.done():
                            pending.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
        else:
            candidates = list(await asyncio.gather(*tasks))
    finally:
        snapshot.restore(harness.state)
        graph_state.recorded_tool_call_ids = original_recorded
        graph_state.pending_tool_calls = []
        graph_state.last_assistant_text = ""
        graph_state.last_thinking_text = ""
        graph_state.last_usage = {}
    increment_metric(harness.state, "test_time_scaling_parallel_proposal_batches")
    for candidate in candidates:
        if candidate.usage:
            apply_usage = getattr(harness, "_apply_usage", None)
            if callable(apply_usage):
                apply_usage(candidate.usage)
    return sorted(candidates, key=lambda item: item.candidate_idx)


async def _generate_one_proposal_candidate(
    graph_state: GraphRunState,
    deps: Any,
    *,
    idx: int,
    variant: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_names: set[str],
    semaphore: asyncio.Semaphore,
) -> ProposalCandidate:
    async with semaphore:
        harness = deps.harness
        branch_harness = _clone_harness_for_candidate(harness)
        branch_deps = copy(deps)
        branch_deps.harness = branch_harness
        branch_deps.event_handler = None
        branch_graph_state = GraphRunState(
            loop_state=branch_harness.state,
            thread_id=graph_state.thread_id,
            run_mode=graph_state.run_mode,
        )
        started = time.perf_counter()
        await model_call(branch_graph_state, branch_deps, messages=messages, tools=tools)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if branch_graph_state.final_result is not None:
            return ProposalCandidate(
                candidate_idx=idx,
                prompt_variant=variant,
                latency_ms=elapsed_ms,
                score=0.0,
                failed_criteria=["model_call_failed"],
            )
        score, failed = score_proposal(
            branch_graph_state.pending_tool_calls,
            allowed_tool_names=tool_names,
            assistant_text=branch_graph_state.last_assistant_text,
        )
        return ProposalCandidate(
            candidate_idx=idx,
            prompt_variant=variant,
            assistant_text=branch_graph_state.last_assistant_text,
            thinking_text=branch_graph_state.last_thinking_text,
            pending_tool_calls=list(branch_graph_state.pending_tool_calls),
            usage=dict(branch_graph_state.last_usage or {}),
            latency_ms=elapsed_ms,
            score=score,
            failed_criteria=failed,
        )


def is_read_only_tool_call(call: PendingToolCall) -> bool:
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


def candidate_uses_only_read_only_tools(candidate: ProposalCandidate) -> bool:
    return bool(candidate.pending_tool_calls) and all(is_read_only_tool_call(call) for call in candidate.pending_tool_calls)


def candidate_uses_local_file_mutation_tools(candidate: ProposalCandidate) -> bool:
    return any(
        str(getattr(call, "tool_name", "") or "").strip() in LOCAL_FILE_MUTATION_TOOLS
        for call in candidate.pending_tool_calls
    )


def unsafe_branch_execution_reason(candidate: ProposalCandidate) -> str:
    remote_mutation_tools = [
        str(getattr(call, "tool_name", "") or "").strip()
        for call in candidate.pending_tool_calls
        if str(getattr(call, "tool_name", "") or "").strip() in REMOTE_MUTATION_TOOLS
    ]
    if remote_mutation_tools:
        return "unsafe_branch_tool:" + ",".join(sorted(set(remote_mutation_tools)))
    unsafe_shell_tools = [
        str(getattr(call, "tool_name", "") or "").strip()
        for call in candidate.pending_tool_calls
        if str(getattr(call, "tool_name", "") or "").strip() in SHELL_EXECUTION_TOOLS
        and not is_read_only_tool_call(call)
    ]
    if unsafe_shell_tools:
        return "unsafe_branch_tool:" + ",".join(sorted(set(unsafe_shell_tools)))
    return ""


def _candidate_history(
    candidates: list[ProposalCandidate],
    *,
    selected: ProposalCandidate | None = None,
) -> list[dict[str, Any]]:
    selected_idx = getattr(selected, "candidate_idx", None)
    history: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_idx):
        tools = [str(getattr(call, "tool_name", "") or "").strip() for call in candidate.pending_tool_calls]
        tools = [tool for tool in tools if tool]
        unsafe_reason = unsafe_branch_execution_reason(candidate)
        record = {
            "candidate": candidate.candidate_idx,
            "selected": selected_idx is not None and candidate.candidate_idx == selected_idx,
            "prompt_variant": candidate.prompt_variant,
            "score": round(float(candidate.score or 0.0), 3),
            "failed_criteria": list(candidate.failed_criteria),
            "tools": tools,
            "read_only": candidate_uses_only_read_only_tools(candidate),
            "unsafe_reason": unsafe_reason,
            "isolated": bool(candidate.isolated),
            "latency_ms": round(float(candidate.latency_ms or 0.0), 3),
            "token_cost": _usage_total_tokens(candidate.usage),
        }
        history.append(record)
    return history


async def _run_isolated_local_branch_candidate(
    graph_state: GraphRunState,
    deps: Any,
    *,
    candidate: ProposalCandidate,
    step: Any,
    original_step_run_id: str,
    threshold: float,
) -> BranchExecutionResult:
    harness = deps.harness
    candidate.isolated = True
    root = Path(str(getattr(harness.state, "cwd", ".") or ".")).resolve()
    try:
        with tempfile.TemporaryDirectory(prefix="smallctl-tts-candidate-") as tmpdir:
            sandbox = Path(tmpdir) / "workspace"
            _copy_workspace_for_candidate(root, sandbox)
            branch_harness = _clone_harness_for_candidate(harness)
            branch_harness.state.cwd = str(sandbox)
            branch_deps = copy(deps)
            branch_deps.harness = branch_harness
            branch_deps.event_handler = None
            branch_run_id = f"{original_step_run_id}-cand{candidate.candidate_idx}-isolated"
            branch_harness.state.active_step_run_id = branch_run_id
            branch_graph_state = GraphRunState(
                loop_state=branch_harness.state,
                thread_id=graph_state.thread_id,
                run_mode=graph_state.run_mode,
                pending_tool_calls=list(candidate.pending_tool_calls),
                last_assistant_text=candidate.assistant_text,
                last_thinking_text=candidate.thinking_text,
                last_usage=dict(candidate.usage or {}),
            )
            _commit_selected_proposal(branch_graph_state, branch_harness, candidate)
            await dispatch_tools(branch_graph_state, branch_deps)
            await persist_tool_results(branch_graph_state, branch_deps)
            await apply_tool_outcomes(branch_graph_state, branch_deps)
            if branch_graph_state.final_result is not None or branch_graph_state.interrupt_payload is not None:
                candidate.score = 0.0
                candidate.failed_criteria = ["branch_interrupted_or_final"]
                return BranchExecutionResult(candidate=candidate, state=branch_harness.state)
            score = await StepCompletionGate().score_step(branch_harness, step)
            candidate.score = score.score
            candidate.failed_criteria = list(score.failed_criteria)
            return BranchExecutionResult(
                candidate=candidate,
                state=branch_harness.state,
                score_value=score.score,
                passed_threshold=score.score >= threshold,
            )
    except Exception as exc:
        candidate.score = 0.0
        candidate.failed_criteria = ["isolated_branch_error"]
        return BranchExecutionResult(candidate=candidate, state=harness.state, error=str(exc))


def _copy_workspace_for_candidate(root: Path, sandbox: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".smallctl",
        "node_modules",
    )
    shutil.copytree(root, sandbox, ignore=ignore)


def _all_fail_action(config: Any) -> str:
    action = str(getattr(config, "test_time_scaling_all_fail_action", "fallback_normal_retry") or "").strip().lower()
    if action in {"fail", "fail_step", "block"}:
        return "fail_step"
    return "fallback_normal_retry"


async def _emit_scaling_status(deps: Any, content: str, **data: Any) -> None:
    harness = deps.harness
    emit = getattr(harness, "_emit", None)
    if not callable(emit):
        return
    event_data = {
        "kind": "test_time_scaling",
        "status_activity": str(content or "").strip(),
        **json_safe_value(data),
    }
    if not isinstance(event_data, dict):
        event_data = {"kind": "test_time_scaling", "status_activity": str(content or "").strip()}
    await emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.SYSTEM,
            content=content,
            data=event_data,
        ),
    )


def _clone_harness_for_candidate(harness: Any) -> Any:
    branch_harness = copy(harness)
    branch_harness.state = deepcopy(harness.state)
    dispatcher = getattr(harness, "dispatcher", None)
    try:
        branch_harness.registry = build_registry(branch_harness)
        branch_harness.dispatcher = ToolDispatcher(
            registry=branch_harness.registry,
            state=branch_harness.state,
            phase=str(
                getattr(dispatcher, "phase", "")
                or getattr(harness.state, "current_phase", "execute")
                or "execute"
            ),
            run_logger=None,
        )
    except Exception:
        if dispatcher is not None:
            branch_dispatcher = copy(dispatcher)
            try:
                branch_dispatcher.state = branch_harness.state
            except Exception:
                pass
            branch_harness.dispatcher = branch_dispatcher
    return branch_harness


def _merge_branch_state(target_state: Any, branch_state: Any) -> None:
    active_plan = getattr(target_state, "active_plan", None)
    draft_plan = getattr(target_state, "draft_plan", None)
    target_state.active_step_run_id = str(getattr(branch_state, "active_step_run_id", "") or "")
    target_state.step_sandbox_history = deepcopy(getattr(branch_state, "step_sandbox_history", []) or [])
    target_state.tool_execution_records = deepcopy(getattr(branch_state, "tool_execution_records", {}) or {})
    target_state.recent_messages = deepcopy(getattr(branch_state, "recent_messages", []) or [])
    target_state.transcript_messages = deepcopy(getattr(branch_state, "transcript_messages", []) or [])
    target_state.scratchpad = deepcopy(getattr(branch_state, "scratchpad", {}) or {})
    target_state.files_changed_this_cycle = deepcopy(getattr(branch_state, "files_changed_this_cycle", []) or [])
    target_state.write_session = deepcopy(getattr(branch_state, "write_session", None))
    target_state.artifacts = deepcopy(getattr(branch_state, "artifacts", {}) or {})
    target_state.pending_interrupt = deepcopy(getattr(branch_state, "pending_interrupt", None))
    target_state.step_verification_result = deepcopy(getattr(branch_state, "step_verification_result", None))
    target_state.tool_history = deepcopy(getattr(branch_state, "tool_history", []) or [])
    target_state.active_plan = active_plan
    target_state.draft_plan = draft_plan


def _commit_selected_proposal(
    graph_state: GraphRunState,
    harness: Any,
    selected: ProposalCandidate,
) -> None:
    graph_state.pending_tool_calls = list(selected.pending_tool_calls)
    graph_state.last_assistant_text = selected.assistant_text
    graph_state.last_thinking_text = selected.thinking_text
    graph_state.last_usage = dict(selected.usage or {})
    conversation_tool_calls = _conversation_tool_calls_from_pending(
        graph_state.pending_tool_calls,
        thread_id=graph_state.thread_id,
        step_count=harness.state.step_count,
    )
    if selected.assistant_text or conversation_tool_calls:
        harness._record_assistant_message(
            assistant_text=selected.assistant_text,
            tool_calls=conversation_tool_calls,
            speaker=None,
            hidden_from_prompt=False,
        )
        log_conversation_state = getattr(harness, "_log_conversation_state", None)
        if callable(log_conversation_state):
            log_conversation_state("assistant_message")
        graph_state.recorded_tool_call_ids = [tc["id"] for tc in conversation_tool_calls]


def _record_branch_metrics(
    harness: Any,
    candidates: list[ProposalCandidate],
    selected: ProposalCandidate,
    *,
    failed: bool = False,
    parallel_read_only_count: int = 0,
    all_failed_action: str = "",
) -> None:
    increment_metric(harness.state, "test_time_scaling_candidates", len(candidates))
    increment_metric(
        harness.state,
        "test_time_scaling_branch_failures" if failed else "test_time_scaling_branch_successes",
    )
    metrics = recovery_metrics(harness.state)
    metrics["test_time_scaling_last"] = {
        "policy": "sequential_branch",
        "candidate_count": len(candidates),
        "read_only_candidate_count": sum(1 for candidate in candidates if candidate_uses_only_read_only_tools(candidate)),
        "read_only_branch_parallel_count": int(parallel_read_only_count),
        "selected_candidate": selected.candidate_idx,
        "selected_score": round(selected.score, 3),
        "failed_criteria": list(selected.failed_criteria),
        "failed": failed,
        "all_failed_action": all_failed_action if failed else "",
    }
    harness.state.scratchpad["_test_time_scaling_last"] = dict(metrics["test_time_scaling_last"])


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for schema in tools:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if name:
            names.add(name)
    return names


def _risk_count(calls: list[PendingToolCall]) -> int:
    return sum(1 for call in calls if call.tool_name in HIGH_RISK_TOOLS)


def _resolve_snapshot_path(root: Path, raw_path: str | Path) -> Path | None:
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _snapshot_parent_dirs(root: Path, path: Path) -> list[Path]:
    dirs: list[Path] = []
    current = path.parent
    while True:
        try:
            current.relative_to(root)
        except ValueError:
            break
        dirs.append(current)
        if current == root:
            break
        current = current.parent
    return dirs


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    try:
        return int((usage or {}).get("total_tokens") or 0)
    except (TypeError, ValueError):
        return 0
