from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from smallctl.graph.plan_verification import StepCompletionGate
from smallctl.graph.state import PendingToolCall
from smallctl.graph.test_time_scaling import (
    CandidateStateGuard,
    FileSnapshotGuard,
    ProposalCandidate,
    candidate_uses_only_read_only_tools,
    collect_candidate_snapshot_paths,
    collect_step_snapshot_paths,
    is_read_only_tool_call,
    score_proposal,
    select_best_proposal,
    unsafe_branch_execution_reason,
)
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState, PlanStep, StepOutputSpec, StepVerifierSpec


def _harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def test_score_step_passes_required_and_penalizes_optional_failure() -> None:
    step = PlanStep(
        step_id="S1",
        title="Step",
        verifiers=[
            StepVerifierSpec(kind="file_exists", args={"path": "."}, required=True),
            StepVerifierSpec(kind="file_exists", args={"path": "missing.txt"}, required=False),
        ],
    )

    score = asyncio.run(StepCompletionGate().score_step(_harness(LoopState()), step))

    assert score.passed is True
    assert 0.7 <= score.score < 1.0
    assert "file_exists" in score.failed_criteria


def test_score_step_required_failure_cannot_score_as_passing() -> None:
    step = PlanStep(
        step_id="S1",
        title="Step",
        verifiers=[StepVerifierSpec(kind="file_exists", args={"path": "missing.txt"}, required=True)],
    )

    score = asyncio.run(StepCompletionGate().score_step(_harness(LoopState()), step))

    assert score.passed is False
    assert score.score <= 0.6
    assert "file_exists" in score.failed_criteria


def test_proposal_scoring_rejects_unavailable_tool() -> None:
    pending = [PendingToolCall(tool_name="shell_exec", args={"command": "pytest"})]

    score, failed = score_proposal(pending, allowed_tool_names={"file_read"})

    assert score == pytest.approx(0.19)
    assert failed == ["tool_not_allowed:shell_exec"]


def test_select_best_proposal_prefers_allowed_lower_risk_candidate() -> None:
    bad = ProposalCandidate(
        candidate_idx=1,
        prompt_variant="bad",
        pending_tool_calls=[PendingToolCall(tool_name="shell_exec", args={"command": "pytest"})],
        score=0.2,
        failed_criteria=["tool_not_allowed:shell_exec"],
    )
    good = ProposalCandidate(
        candidate_idx=2,
        prompt_variant="good",
        pending_tool_calls=[PendingToolCall(tool_name="file_read", args={"path": "README.md"})],
        score=1.0,
    )

    assert select_best_proposal([bad, good]) is good


def test_read_only_tool_call_classifier_allows_safe_evidence_tools() -> None:
    assert is_read_only_tool_call(PendingToolCall(tool_name="file_read", args={"path": "README.md"}))
    assert is_read_only_tool_call(PendingToolCall(tool_name="shell_exec", args={"command": "git status"}))
    assert is_read_only_tool_call(PendingToolCall(tool_name="ssh_exec", args={"command": "systemctl status nginx"}))
    assert not is_read_only_tool_call(PendingToolCall(tool_name="file_write", args={"path": "x", "content": "y"}))
    assert not is_read_only_tool_call(PendingToolCall(tool_name="shell_exec", args={"command": "touch changed.txt"}))


def test_candidate_read_only_classifier_requires_all_calls_to_be_safe() -> None:
    readonly = ProposalCandidate(
        candidate_idx=1,
        prompt_variant="read",
        pending_tool_calls=[
            PendingToolCall(tool_name="dir_list", args={"path": "."}),
            PendingToolCall(tool_name="shell_exec", args={"command": "git diff"}),
        ],
    )
    mutating = ProposalCandidate(
        candidate_idx=2,
        prompt_variant="write",
        pending_tool_calls=[
            PendingToolCall(tool_name="dir_list", args={"path": "."}),
            PendingToolCall(tool_name="file_write", args={"path": "x", "content": "y"}),
        ],
    )

    assert candidate_uses_only_read_only_tools(readonly)
    assert not candidate_uses_only_read_only_tools(mutating)


def test_collect_candidate_snapshot_paths_includes_local_file_mutations() -> None:
    candidates = [
        ProposalCandidate(
            candidate_idx=1,
            prompt_variant="write",
            pending_tool_calls=[
                PendingToolCall(tool_name="file_write", args={"path": "side.txt", "content": "x"}),
                PendingToolCall(tool_name="file_patch", args={"target_path": "patched.py", "target_text": "a"}),
                PendingToolCall(tool_name="shell_exec", args={"command": "touch unknown.txt"}),
            ],
        ),
        ProposalCandidate(
            candidate_idx=2,
            prompt_variant="duplicate",
            pending_tool_calls=[PendingToolCall(tool_name="file_append", args={"path": "side.txt", "content": "y"})],
        ),
    ]

    assert collect_candidate_snapshot_paths(candidates) == ["side.txt", "patched.py"]


def test_unsafe_branch_execution_reason_blocks_mutating_shell() -> None:
    mutating = ProposalCandidate(
        candidate_idx=1,
        prompt_variant="shell",
        pending_tool_calls=[PendingToolCall(tool_name="shell_exec", args={"command": "touch side.txt"})],
    )
    readonly = ProposalCandidate(
        candidate_idx=2,
        prompt_variant="shell read",
        pending_tool_calls=[PendingToolCall(tool_name="shell_exec", args={"command": "git status"})],
    )

    assert unsafe_branch_execution_reason(mutating) == "unsafe_branch_tool:shell_exec"
    assert unsafe_branch_execution_reason(readonly) == ""


def test_candidate_state_guard_restores_branch_sensitive_state() -> None:
    state = LoopState(active_step_run_id="run-base", token_usage=10, last_completion_tokens=2)
    state.scratchpad["keep"] = "base"
    state.transcript_messages = [ConversationMessage(role="assistant", content="base")]
    state.recent_messages = list(state.transcript_messages)
    state.step_sandbox_history = [ConversationMessage(role="assistant", content="sandbox-base")]
    state.tool_execution_records = {"op-base": {"tool_name": "file_read"}}
    state.files_changed_this_cycle = ["base.py"]
    state.artifacts = {"A-base": {"summary": "base"}}
    state.pending_interrupt = {"kind": "base"}
    state.tool_history = ["base-tool"]

    guard = CandidateStateGuard.capture(state)
    state.active_step_run_id = "run-candidate"
    state.scratchpad["candidate"] = "loser"
    state.transcript_messages.append(ConversationMessage(role="assistant", content="loser"))
    state.step_sandbox_history = []
    state.tool_execution_records["op-loser"] = {"tool_name": "file_write"}
    state.files_changed_this_cycle.append("loser.py")
    state.artifacts["A-loser"] = {"summary": "loser"}
    state.pending_interrupt = {"kind": "loser"}
    state.tool_history.append("loser-tool")
    state.token_usage = 42
    state.last_completion_tokens = 9

    guard.restore(state)

    assert state.active_step_run_id == "run-base"
    assert state.scratchpad == {"keep": "base"}
    assert [message.content for message in state.transcript_messages] == ["base"]
    assert [message.content for message in state.step_sandbox_history] == ["sandbox-base"]
    assert state.tool_execution_records == {"op-base": {"tool_name": "file_read"}}
    assert state.files_changed_this_cycle == ["base.py"]
    assert state.artifacts == {"A-base": {"summary": "base"}}
    assert state.pending_interrupt == {"kind": "base"}
    assert state.tool_history == ["base-tool"]
    assert state.token_usage == 42
    assert state.last_completion_tokens == 9


def test_candidate_state_guard_can_restore_accounting_when_requested() -> None:
    state = LoopState(token_usage=10, last_completion_tokens=2)
    guard = CandidateStateGuard.capture(state)
    state.token_usage = 99
    state.last_completion_tokens = 12

    guard.restore(state, keep_accounting=False)

    assert state.token_usage == 10
    assert state.last_completion_tokens == 2


def test_file_snapshot_guard_restores_declared_paths_without_touching_unrelated_dirty_files(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("base", encoding="utf-8")
    created = tmp_path / "created.txt"
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("dirty before", encoding="utf-8")

    guard = FileSnapshotGuard.capture(cwd=tmp_path, paths=["target.txt", "created.txt", "../outside.txt"])
    target.write_text("loser", encoding="utf-8")
    created.write_text("loser created", encoding="utf-8")
    unrelated.write_text("dirty after", encoding="utf-8")

    guard.restore()

    assert target.read_text(encoding="utf-8") == "base"
    assert not created.exists()
    assert unrelated.read_text(encoding="utf-8") == "dirty after"


def test_collect_step_snapshot_paths_uses_outputs_and_file_verifiers() -> None:
    step = PlanStep(
        step_id="S1",
        title="Step",
        outputs_expected=[StepOutputSpec(kind="file", ref="out.txt")],
        verifiers=[
            StepVerifierSpec(kind="file_exists", args={"path": "out.txt"}),
            StepVerifierSpec(kind="syntax_ok", args={"path": "src/app.py"}),
            StepVerifierSpec(kind="last_command_passed", args={}),
        ],
    )

    assert collect_step_snapshot_paths(step) == ["out.txt", "src/app.py"]
