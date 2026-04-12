from __future__ import annotations

from smallctl.state import (
    ClaimRecord,
    ContextBrief,
    DecisionRecord,
    EvidenceRecord,
    ExecutionPlan,
    LoopState,
    PlanStep,
    ReasoningGraph,
)


def test_reasoning_graph_round_trip_preserves_records() -> None:
    state = LoopState(cwd="/tmp")
    state.reasoning_graph = ReasoningGraph(
        evidence_records=[
            EvidenceRecord(
                evidence_id="E1",
                kind="observation",
                statement="file_read confirmed README.md exists",
                phase="explore",
                tool_name="file_read",
                operation_id="op-1",
                artifact_id="A1",
                confidence=0.9,
            )
        ],
        decision_records=[
            DecisionRecord(
                decision_id="D1",
                phase="plan",
                intent_label="inspect",
                requested_tool="file_read",
                argument_fingerprint="fp-1",
                plan_step_id="P1",
                evidence_refs=["E1"],
                rationale_summary="Need direct observation before planning.",
            )
        ],
        claim_records=[
            ClaimRecord(
                claim_id="C1",
                kind="hypothesis",
                statement="README.md likely contains the task summary.",
                supporting_evidence_ids=["E1"],
                missing_evidence=["full README read"],
                alternative_explanations=["Need to verify the latest README revision."],
                confidence=0.4,
                decision_ids=["D1"],
            )
        ],
    )
    state.context_briefs = [
        ContextBrief(
            brief_id="B1",
            created_at="2026-04-09T00:00:00+00:00",
            tier="warm",
            step_range=(1, 2),
            task_goal="Summarize README",
            current_phase="plan",
            key_discoveries=["README.md exists"],
            tools_tried=["file_read"],
            blockers=[],
            files_touched=["README.md"],
            artifact_ids=["A1"],
            next_action_hint="Draft the summary",
            staleness_step=2,
            facts_confirmed=["README.md exists"],
            facts_unconfirmed=["README.md contents not yet read"],
            open_questions=["What does README describe?"],
            candidate_causes=["Need to inspect README"],
            disproven_causes=[],
            next_observations_needed=["Read README.md"],
            evidence_refs=["E1"],
            claim_refs=["C1"],
        )
    ]
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="Summarize README",
        claim_refs=["C1"],
        steps=[
            PlanStep(
                step_id="P1",
                title="Read README",
                claim_refs=["C1"],
                evidence_refs=["E1"],
            )
        ],
    )

    payload = state.to_dict()
    restored = LoopState.from_dict(payload)

    assert restored.schema_version == payload["schema_version"]
    assert restored.reasoning_graph.evidence_ids == ["E1"]
    assert restored.reasoning_graph.decision_ids == ["D1"]
    assert restored.reasoning_graph.claim_ids == ["C1"]
    assert restored.reasoning_graph.evidence_records[0].statement == "file_read confirmed README.md exists"
    assert restored.reasoning_graph.decision_records[0].rationale_summary == "Need direct observation before planning."
    assert restored.reasoning_graph.claim_records[0].supporting_evidence_ids == ["E1"]
    assert restored.context_briefs[0].facts_confirmed == ["README.md exists"]
    assert restored.context_briefs[0].claim_refs == ["C1"]
    assert restored.active_plan is not None
    assert restored.active_plan.claim_refs == ["C1"]
    assert restored.active_plan.steps[0].claim_refs == ["C1"]


def test_legacy_payload_without_reasoning_graph_loads_with_empty_graph() -> None:
    restored = LoopState.from_dict(
        {
            "schema_version": 1,
            "current_phase": "explore",
            "thread_id": "legacy-thread",
            "recent_messages": [],
            "run_brief": {"original_task": "Read README"},
            "working_memory": {"current_goal": "Read README"},
        }
    )

    assert restored.schema_version == 2
    assert restored.reasoning_graph.evidence_records == []
    assert restored.reasoning_graph.decision_records == []
    assert restored.reasoning_graph.claim_records == []
    assert restored.reasoning_graph.evidence_ids == []
    assert restored.reasoning_graph.decision_ids == []
    assert restored.reasoning_graph.claim_ids == []
