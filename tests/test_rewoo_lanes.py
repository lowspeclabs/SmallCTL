from __future__ import annotations

from smallctl.context.rewoo_lanes import ReWOOLaneCompiler
from smallctl.config import SmallctlConfig
from smallctl.graph.tool_plan_observations import ToolPlanObservation
from smallctl.recovery_schema import Subtask, SubtaskLedger
from smallctl.state import DecisionRecord, EvidenceRecord, ExperienceMemory, LoopState


def test_rewoo_rollout_flags_default_off() -> None:
    config = SmallctlConfig()

    assert config.rewoo_lane_frames_enabled is False
    assert config.rewoo_planner_frame_enabled is False
    assert config.rewoo_solver_frame_enabled is False
    assert config.rewoo_refiner_frame_enabled is False


def test_rewoo_planner_frame_renders_plan_state_without_tool_observations() -> None:
    state = LoopState()
    state.run_brief.original_task = "Fix the parser"
    state.run_brief.task_contract = "Only inspect before patching"
    state.run_brief.constraints = ["read-only planning"]
    state.run_brief.acceptance_criteria = ["parser test passes"]
    state.working_memory.failures = ["Tried the wrong path"]
    state.working_memory.open_questions = ["Where is the parser?"]
    state.scratchpad["_tool_plan_repair_nudge"] = "Return valid JSON"
    state.warm_experiences = [
        ExperienceMemory(memory_id="M1", notes="Prefer targeted grep first", confidence=0.9, reuse_count=2)
    ]

    compiler = ReWOOLaneCompiler()
    frame = compiler.compile(
        state=state,
        role="planner",
        tool_plan_observations=[
            ToolPlanObservation("E1", "file_read", True, "raw output", path="src/parser.py")
        ],
    )
    rendered = compiler.render(frame)

    assert "REWOO PLAN STATE" in rendered
    assert "Fix the parser" in rendered
    assert "Tried the wrong path" in rendered
    assert "Where is the parser?" in rendered
    assert "Return valid JSON" in rendered
    assert "Prefer targeted grep first" in rendered
    assert "raw output" not in rendered
    assert "TOOL PLAN OBSERVATIONS" not in rendered


def test_rewoo_solver_frame_renders_evidence_decisions_and_drop_log() -> None:
    state = LoopState()
    state.run_brief.original_task = "Find dispatch seam"
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id=f"E{i}",
            kind="tool_plan_observation",
            statement=f"finding {i}",
            tool_name="file_read",
            phase="tool_plan",
            source=f"src/{i}.py",
            confidence=0.8,
            metadata={"observation_adapter": "tool_plan_observation", "path": f"src/{i}.py"},
        )
        for i in range(8)
    ]
    state.reasoning_graph.decision_records = [
        DecisionRecord(decision_id="D1", rationale_summary="Use the dispatch helper", evidence_refs=["E7"])
    ]

    compiler = ReWOOLaneCompiler()
    frame = compiler.compile(
        state=state,
        role="solver",
        tool_plan_observations=[
            ToolPlanObservation(f"TP{i}", "file_read", True, f"summary {i}", path=f"src/{i}.py")
            for i in range(20)
        ],
        token_budget=1000,
    )
    rendered = compiler.render(frame, token_budget=1000)

    assert "REWOO EVIDENCE" in rendered
    assert "D1" in rendered
    assert any(drop.lane.startswith("rewoo.solver.") for drop in frame.drop_log)


def test_rewoo_planner_frame_includes_subtask_and_fama_hard_route_context() -> None:
    state = LoopState()
    state.run_brief.original_task = "Repair the remote verifier"
    state.subtask_ledger = SubtaskLedger(
        task_id="task-1",
        active_subtask_id="S1",
        subtasks=[
            Subtask(
                subtask_id="S1",
                title="Verify remote patch",
                goal="Read back the remote file",
                status="active",
                evidence=["TP-E4-E1"],
                blockers=["remote read-back missing"],
                next_action="ssh read the changed file",
            )
        ],
    )
    state.scratchpad["_fama_config"] = {"enabled": True, "mode": "lite", "capsule_token_budget": 180}
    state.scratchpad["_fama"] = {
        "version": 1,
        "signals": [],
        "seen_signatures": [],
        "last_observed_step": 4,
        "active_mitigations": [
            {
                "name": "remote_verification_pending_capsule",
                "reason": "remote mutation requires verification",
                "source_signal": "remote",
                "activated_step": 4,
                "expires_after_step": 8,
                "priority": 10,
            }
        ],
    }

    rendered = ReWOOLaneCompiler().render(ReWOOLaneCompiler().compile(state=state, role="planner"))

    assert "Subtask context" in rendered
    assert "Verify remote patch" in rendered
    assert "remote read-back missing" in rendered
    assert "Hard-route reasons" in rendered
    assert "remote mutation requires verification" in rendered
