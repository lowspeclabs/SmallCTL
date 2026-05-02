from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.step_sandbox import build_step_sandbox_prompt
from smallctl.models.conversation import ConversationMessage
from smallctl.state import (
    ArtifactRecord,
    ExecutionPlan,
    LoopState,
    PlanStep,
    StepEvidenceArtifact,
    StepOutputSpec,
    StepVerifierSpec,
)


def _harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(state=state, config=SimpleNamespace(staged_step_prompt_tokens=0))


def test_prompt_includes_step_contract_fields() -> None:
    state = LoopState(active_step_run_id="run-1")
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="ship feature",
        steps=[
            PlanStep(
                step_id="S1",
                title="Build API",
                task="Create REST endpoint",
                acceptance=["returns 200", "has tests"],
                verifiers=[
                    StepVerifierSpec(kind="file_exists", args={"path": "api.py"}),
                    StepVerifierSpec(kind="syntax_ok", args={"path": "api.py"}, required=False),
                ],
                outputs_expected=[
                    StepOutputSpec(kind="file", ref="api.py"),
                    StepOutputSpec(kind="artifact", ref="test-results"),
                ],
                max_retries=3,
                retry_count=1,
                failure_reasons=["lint error"],
            )
        ],
    )
    state.active_step_id = "S1"

    prompt = build_step_sandbox_prompt(_harness(state), state.active_plan.find_step("S1"))
    rendered = "\n".join(str(m.get("content", "")) for m in prompt)

    assert "ship feature" in rendered
    assert "S1" in rendered
    assert "Build API" in rendered
    assert "Create REST endpoint" in rendered
    assert "returns 200" in rendered
    assert "has tests" in rendered
    assert "file_exists(required)" in rendered
    assert "syntax_ok(optional)" in rendered
    assert "file:api.py" in rendered
    assert "artifact:test-results" in rendered
    assert "Attempt: 2 of 3" in rendered
    assert "lint error" in rendered


def test_prompt_includes_dependency_evidence() -> None:
    state = LoopState(active_step_run_id="run-2")
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="First"),
            PlanStep(step_id="S2", title="Second", depends_on=["S1"]),
        ],
    )
    state.active_step_id = "S2"
    state.step_evidence["S1"] = StepEvidenceArtifact(
        step_id="S1",
        step_run_id="run-1",
        summary="first done",
        artifact_ids=["art-1"],
        files_touched=["src/a.py"],
    )

    prompt = build_step_sandbox_prompt(_harness(state), state.active_plan.find_step("S2"))
    rendered = "\n".join(str(m.get("content", "")) for m in prompt)

    assert "first done" in rendered
    assert "art-1" in rendered
    assert "src/a.py" in rendered


def test_prompt_excludes_non_dependency_step_text() -> None:
    state = LoopState(active_step_run_id="run-1")
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="Dependency"),
            PlanStep(step_id="S2", title="Active", depends_on=["S1"]),
            PlanStep(step_id="S3", title="FORBIDDEN non-dependency"),
        ],
    )
    state.active_step_id = "S2"
    state.step_evidence["S1"] = StepEvidenceArtifact(step_id="S1", summary="dep done")

    prompt = build_step_sandbox_prompt(_harness(state), state.active_plan.find_step("S2"))
    rendered = "\n".join(str(m.get("content", "")) for m in prompt)

    assert "dep done" in rendered
    assert "FORBIDDEN non-dependency" not in rendered


def test_prompt_excludes_global_recent_messages() -> None:
    state = LoopState(active_step_run_id="run-1")
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[PlanStep(step_id="S1", title="Step")],
    )
    state.active_step_id = "S1"
    state.append_message(ConversationMessage(role="user", content="FORBIDDEN global message"))

    prompt = build_step_sandbox_prompt(_harness(state), state.active_plan.find_step("S1"))
    rendered = "\n".join(str(m.get("content", "")) for m in prompt)

    assert "FORBIDDEN global message" not in rendered


def test_prompt_includes_step_sandbox_history() -> None:
    state = LoopState(active_step_run_id="run-1")
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[PlanStep(step_id="S1", title="Step")],
    )
    state.active_step_id = "S1"
    state.step_sandbox_history.append(ConversationMessage(role="assistant", content="sandbox message"))

    prompt = build_step_sandbox_prompt(_harness(state), state.active_plan.find_step("S1"))
    rendered = "\n".join(str(m.get("content", "")) for m in prompt)

    assert "sandbox message" in rendered


def test_prompt_excludes_unrelated_artifacts() -> None:
    state = LoopState(active_step_run_id="run-1")
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="First"),
            PlanStep(step_id="S2", title="Active", depends_on=["S1"]),
        ],
    )
    state.active_step_id = "S2"
    state.step_evidence["S1"] = StepEvidenceArtifact(
        step_id="S1",
        summary="done",
        artifact_ids=["related-art"],
    )
    state.artifacts["related-art"] = ArtifactRecord(
        artifact_id="related-art",
        kind="test",
        source="test",
        created_at="2024-01-01T00:00:00",
        size_bytes=0,
        summary="related artifact",
        preview_text="related content",
    )
    state.artifacts["unrelated-art"] = ArtifactRecord(
        artifact_id="unrelated-art",
        kind="test",
        source="test",
        created_at="2024-01-01T00:00:00",
        size_bytes=0,
        summary="unrelated artifact",
        preview_text="FORBIDDEN unrelated content",
    )

    prompt = build_step_sandbox_prompt(_harness(state), state.active_plan.find_step("S2"))
    rendered = "\n".join(str(m.get("content", "")) for m in prompt)

    assert "related content" in rendered
    assert "FORBIDDEN unrelated content" not in rendered


def test_prompt_respects_token_budget() -> None:
    state = LoopState(active_step_run_id="run-1")
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[PlanStep(step_id="S1", title="Step", prompt_token_budget=10)],
    )
    state.active_step_id = "S1"
    # Add enough content to exceed the tiny budget
    state.step_sandbox_history.append(ConversationMessage(role="assistant", content="a" * 100))
    state.step_sandbox_history.append(ConversationMessage(role="assistant", content="b" * 100))

    prompt = build_step_sandbox_prompt(_harness(state), state.active_plan.find_step("S1"))

    # Budget enforcement should drop lanes and record it in scratchpad
    assert state.scratchpad["_staged_prompt_dropped_lanes"]["step_id"] == "S1"
    assert state.scratchpad["_staged_prompt_dropped_lanes"]["budget"] == 10
    assert len(state.scratchpad["_staged_prompt_dropped_lanes"]["dropped"]) > 0
