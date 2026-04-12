from __future__ import annotations

from smallctl.risk_policy import classify_task, evaluate_risk_policy
from smallctl.state import ClaimRecord, LoopState


def test_classify_task_distinguishes_diagnosis_work() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    assert classify_task(state) == "diagnosis_remediation"


def test_diagnosis_task_blocks_mutating_high_risk_shell_without_supported_claim() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="shell_exec",
        tool_risk="high",
        phase="execute",
        action="chmod +x ./script.sh",
        expected_effect="Run the requested shell command.",
        rollback="Stop the command if needed.",
        verification="Inspect the result.",
        approval_available=True,
    )

    assert decision.allowed is False
    assert decision.reason
    assert decision.proof_bundle["phase"] == "execute"
    assert decision.proof_bundle["tool_risk"] == "high"
    assert decision.proof_bundle["task_classification"] == "diagnosis_remediation"


def test_implementation_task_allows_high_risk_shell_with_approval() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.scratchpad["_task_classification"] = "implementation"
    state.reasoning_graph.claim_records.append(
        ClaimRecord(
            claim_id="C1",
            kind="causal",
            statement="The failing suite is caused by an outdated fixture.",
            supporting_evidence_ids=["E1"],
            status="confirmed",
        )
    )

    decision = evaluate_risk_policy(
        state,
        tool_name="shell_exec",
        tool_risk="high",
        phase="execute",
        action="pytest -q",
        expected_effect="Run the requested shell command.",
        rollback="Stop the command if needed.",
        verification="Inspect the result.",
        approval_available=True,
    )

    assert decision.allowed is True
    assert decision.requires_approval is True
    assert decision.proof_bundle["supported_claim_ids"] == ["C1"]
    assert decision.approval_kind == "shell"


def test_diagnosis_task_allows_read_only_shell_evidence_without_supported_claim() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="shell_exec",
        tool_risk="high",
        phase="execute",
        action="pytest -q",
        expected_effect="Collect failing test evidence.",
        rollback="No rollback required.",
        verification="Inspect the test output.",
        approval_available=True,
    )

    assert decision.allowed is True
    assert decision.requires_approval is True
    assert decision.proof_bundle["task_classification"] == "diagnosis_remediation"


def test_diagnosis_task_allows_read_only_ssh_diagnostics_without_supported_claim() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="ssh_exec",
        tool_risk="high",
        phase="repair",
        action="apt-cache search guacamole",
        expected_effect="Inspect available packages on the remote host.",
        rollback="No rollback required.",
        verification="Inspect the package search output.",
        approval_available=True,
    )

    assert decision.allowed is True
    assert decision.requires_approval is True


def test_diagnosis_task_allows_segmented_read_only_ssh_pipeline_without_supported_claim() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="ssh_exec",
        tool_risk="high",
        phase="repair",
        action="journalctl -u ssh --no-pager | tail -100",
        expected_effect="Collect recent SSH logs from the remote host.",
        rollback="No rollback required.",
        verification="Inspect the remote log output.",
        approval_available=True,
    )

    assert decision.allowed is True
    assert decision.requires_approval is True


def test_diagnosis_task_blocks_compound_shell_when_any_segment_is_mutating() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="ssh_exec",
        tool_risk="high",
        phase="repair",
        action="cat /var/log/dpkg.log && rm -f /tmp/example",
        expected_effect="Inspect logs and then remove a file.",
        rollback="Restore the removed file if needed.",
        verification="Inspect the remote output.",
        approval_available=True,
    )

    assert decision.allowed is False
    assert "supported claim" in decision.reason.lower()


def test_diagnosis_task_still_blocks_high_risk_file_write_without_supported_claim() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="file_write",
        tool_risk="high",
        phase="execute",
        action="Write file temp/fix.py",
        expected_effect="Apply a code change.",
        rollback="Revert file contents.",
        verification="Run the verifier.",
        approval_available=False,
    )

    assert decision.allowed is False
    assert "supported claim" in decision.reason.lower()


def test_diagnosis_task_still_blocks_high_risk_file_patch_without_supported_claim() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="file_patch",
        tool_risk="high",
        phase="execute",
        action="Patch file temp/fix.py",
        expected_effect="Apply a local code change.",
        rollback="Revert file contents.",
        verification="Run the verifier.",
        approval_available=False,
    )

    assert decision.allowed is False
    assert "supported claim" in decision.reason.lower()
