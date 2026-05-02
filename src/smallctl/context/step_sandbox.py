from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..state import ArtifactSnippet, ExecutionPlan, PlanStep, StepEvidenceArtifact, json_safe_value


def collect_dependency_evidence(
    state: Any,
    plan: ExecutionPlan,
    step: PlanStep,
) -> list[StepEvidenceArtifact]:
    seen: set[str] = set()
    ordered: list[StepEvidenceArtifact] = []

    def visit(step_id: str) -> None:
        if step_id in seen:
            return
        seen.add(step_id)
        dependency = plan.find_step(step_id)
        if dependency is not None:
            for parent_id in dependency.depends_on:
                visit(parent_id)
        evidence = getattr(state, "step_evidence", {}).get(step_id)
        if evidence is not None:
            ordered.append(evidence)

    for dependency_id in step.depends_on:
        visit(dependency_id)
    return ordered


def select_step_artifact_snippets(
    state: Any,
    evidence: list[StepEvidenceArtifact],
) -> list[ArtifactSnippet]:
    artifact_ids: list[str] = []
    for item in evidence:
        for artifact_id in item.artifact_ids:
            if artifact_id and artifact_id not in artifact_ids:
                artifact_ids.append(artifact_id)

    snippets: list[ArtifactSnippet] = []
    artifacts = getattr(state, "artifacts", {})
    if not isinstance(artifacts, dict):
        return snippets
    for artifact_id in artifact_ids:
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            continue
        text = str(getattr(artifact, "preview_text", "") or getattr(artifact, "summary", "") or "")
        if not text and getattr(artifact, "inline_content", None):
            text = str(artifact.inline_content or "")[:1200]
        if text:
            snippets.append(ArtifactSnippet(artifact_id=artifact_id, text=text[:1200], score=1.0))
    return snippets


def compact_step_sandbox_history(state: Any, step: PlanStep) -> StepEvidenceArtifact | None:
    history = getattr(state, "step_sandbox_history", None)
    if not isinstance(history, list) or len(history) < 8:
        return None
    compacted = history[:-4]
    state.step_sandbox_history = history[-4:]
    summaries = []
    for message in compacted[-8:]:
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            summaries.append(content[:200])
    if not summaries:
        return None
    return StepEvidenceArtifact(
        step_id=step.step_id,
        step_run_id=str(getattr(state, "active_step_run_id", "") or ""),
        attempt=int(getattr(step, "retry_count", 0) or 0) + 1,
        summary="Compacted intra-step history: " + " | ".join(summaries),
    )


def build_step_sandbox_prompt(harness: Any, step: PlanStep) -> list[dict[str, Any]]:
    state = harness.state
    plan = state.active_plan or state.draft_plan
    if plan is None:
        raise RuntimeError("Cannot build staged prompt without an active plan.")

    dependency_evidence = collect_dependency_evidence(state, plan, step)
    snippets = select_step_artifact_snippets(state, dependency_evidence)
    compacted = compact_step_sandbox_history(state, step)
    if compacted is not None:
        dependency_evidence.append(compacted)

    system_lines = [
        "You are executing one isolated staged plan step.",
        "The harness owns the full plan and decides when to advance.",
        "Use only the active step contract and dependency evidence below.",
        "When the active step is ready, call `step_complete(message='...')`; do not call `task_complete`.",
        f"Overall goal: {plan.goal}",
        f"Plan ID: {plan.plan_id}",
        f"Active step: {step.step_id} - {step.title}",
        f"Step run ID: {state.active_step_run_id}",
        f"Attempt: {int(step.retry_count or 0) + 1} of {max(1, int(step.max_retries or 0))}",
    ]
    task = step.task or step.description or step.title
    if task:
        system_lines.append(f"Task: {task}")
    if step.acceptance:
        system_lines.append("Acceptance: " + "; ".join(step.acceptance))
    if step.verifiers:
        verifier_bits = [
            f"{spec.kind}({'required' if spec.required else 'optional'})"
            for spec in step.verifiers
        ]
        system_lines.append("Verifiers: " + ", ".join(verifier_bits))
    if step.outputs_expected:
        output_bits = [
            f"{spec.kind}:{spec.ref or spec.description}"
            for spec in step.outputs_expected
        ]
        system_lines.append("Expected outputs: " + "; ".join(output_bits))
    if step.failure_reasons:
        system_lines.append("Recent step failures: " + " | ".join(step.failure_reasons[-3:]))

    messages: list[dict[str, Any]] = [{"role": "system", "content": "\n".join(system_lines)}]
    if dependency_evidence:
        evidence_lines = []
        for item in dependency_evidence:
            evidence_lines.append(
                f"- {item.step_id} run={item.step_run_id} summary={item.summary} "
                f"artifacts={','.join(item.artifact_ids)} files={','.join(item.files_touched)}"
            )
        messages.append({"role": "system", "content": "Dependency evidence:\n" + "\n".join(evidence_lines)})
    if snippets:
        snippet_lines = [f"[{snippet.artifact_id}] {snippet.text}" for snippet in snippets]
        messages.append({"role": "system", "content": "Referenced artifacts:\n" + "\n\n".join(snippet_lines)})

    for message in getattr(state, "step_sandbox_history", []) or []:
        if isinstance(message, ConversationMessage):
            messages.append(json_safe_value(message.to_dict(include_retrieval_safe_text=True)))
        elif isinstance(message, dict):
            messages.append(json_safe_value(message))

    budget = int(getattr(step, "prompt_token_budget", 0) or 0)
    if budget <= 0:
        budget = int(getattr(getattr(harness, "config", None), "staged_step_prompt_tokens", 0) or 0)
    if budget > 0:
        messages = _enforce_budget(messages, budget, state=state, step=step)
    return messages


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(max(1, len(str(message.get("content", ""))) // 4) for message in messages)


def _enforce_budget(
    messages: list[dict[str, Any]],
    budget: int,
    *,
    state: Any,
    step: PlanStep,
) -> list[dict[str, Any]]:
    if _estimate_tokens(messages) <= budget:
        return messages
    kept = list(messages)
    dropped: list[str] = []
    while len(kept) > 1 and _estimate_tokens(kept) > budget:
        index = 1 if len(kept) > 2 else len(kept) - 1
        dropped.append(str(kept[index].get("content", ""))[:60])
        kept.pop(index)
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad["_staged_prompt_dropped_lanes"] = {
            "step_id": step.step_id,
            "budget": budget,
            "dropped": dropped,
        }
    return kept
