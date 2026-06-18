from __future__ import annotations

import asyncio
from copy import copy
from pathlib import Path
import tempfile
import time
from typing import Any

from ..recovery_metrics import increment_metric, recovery_metrics
from ..shell_utils import is_read_only_shell_evidence_action
from .model_call_nodes import model_call
from .plan_execution import PlanExecutionEngine
from .plan_verification import StepCompletionGate, compact_step_evidence
from .state import GraphRunState
from .scaling_helpers import (
    candidate_uses_local_file_mutation_tools,
    candidate_uses_only_read_only_tools,
    is_read_only_tool_call,
    score_proposal,
    select_best_proposal,
    unsafe_branch_execution_reason,
)
from .tool_plan_schema import READONLY_TOOL_PLAN_TOOLS
from .tool_execution_nodes import dispatch_tools
from .tool_execution_persistence import persist_tool_results
from .tool_outcomes import apply_tool_outcomes
from .test_time_scaling_support import (
    CandidateStateGuard,
    FileSnapshotGuard,
    BranchExecutionResult,
    ProposalCandidate,
    _all_fail_action,
    _candidate_history,
    _clone_harness_for_candidate,
    _commit_selected_proposal,
    _copy_workspace_for_candidate,
    _emit_scaling_status,
    _merge_branch_state,
    _record_branch_metrics,
    build_candidate_prompts,
    collect_candidate_snapshot_paths,
    collect_step_snapshot_paths,
    _tool_names,
    _usage_total_tokens,
)


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
    increment_metric(harness.state, "test_time_scaling_attempts")
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
