from __future__ import annotations

import asyncio

from smallctl.harness.approvals import ApprovalService
from smallctl.state import ClaimRecord, LoopState
from smallctl.ui.approval import ApprovePromptScreen


def test_shell_approval_request_emits_proof_bundle() -> None:
    events: list[object] = []

    class _FakeHarness:
        allow_interactive_shell_approval = True
        shell_approval_session_default = False
        event_handler = object()

        async def _emit(self, handler: object, event: object) -> None:
            del handler
            events.append(event)

    async def _run() -> bool:
        harness = _FakeHarness()
        service = ApprovalService(harness)
        approval_task = asyncio.create_task(
            service.request_shell_approval(
                command="pytest -q",
                cwd="/tmp",
                timeout_sec=12,
                proof_bundle={
                    "phase": "execute",
                    "tool_risk": "high",
                    "task_classification": "diagnosis_remediation",
                    "action": "Run pytest -q",
                    "expected_effect": "Verify the failing tests after the repair.",
                    "rollback": "Stop the command and revert any in-progress edits.",
                    "verification": "Inspect the test output for pass/fail status.",
                    "supported_claim_ids": ["C1"],
                    "supporting_evidence_ids": ["E1"],
                },
            )
        )
        await asyncio.sleep(0)
        assert events
        approval_event = events[0]
        data = getattr(approval_event, "data", {})
        assert data.get("proof_bundle", {}).get("phase") == "execute"
        assert data.get("proof_bundle", {}).get("tool_risk") == "high"
        assert data.get("proof_bundle", {}).get("task_classification") == "diagnosis_remediation"
        assert data.get("proof_bundle", {}).get("supported_claim_ids") == ["C1"]
        assert data.get("proof_bundle", {}).get("supporting_evidence_ids") == ["E1"]
        approval_id = str(data.get("approval_id") or "")
        service.resolve_shell_approval(approval_id, True)
        return await approval_task

    assert asyncio.run(_run()) is True


def test_approval_screen_renders_proof_bundle_details() -> None:
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.claim_records.append(
        ClaimRecord(
            claim_id="C1",
            kind="causal",
            statement="The build fails because the test fixture is missing.",
            supporting_evidence_ids=["E1"],
            status="confirmed",
        )
    )
    screen = ApprovePromptScreen(
        approval_id="shell-123",
        command="pytest -q",
        cwd="/tmp",
        timeout_sec=12,
        proof_bundle={
            "phase": "execute",
            "tool_risk": "high",
            "task_classification": "diagnosis_remediation",
            "action": "Run pytest -q",
            "expected_effect": "Verify the failing tests after the repair.",
            "rollback": "Stop the command and revert any in-progress edits.",
            "verification": "Inspect the test output for pass/fail status.",
            "supported_claim_ids": ["C1"],
            "supporting_evidence_ids": ["E1"],
            "claims": [
                {
                    "claim_id": "C1",
                    "statement": "The build fails because the test fixture is missing.",
                    "confidence": 0.8,
                }
            ],
        },
    )

    body = screen._build_body()

    assert "Proof bundle:" in body
    assert "Phase: execute" in body
    assert "Risk: high" in body
    assert "Task: diagnosis_remediation" in body
    assert "Claims: C1" in body
    assert "Evidence: E1" in body
    assert "The build fails because the test fixture is missing." in body
    assert "Verification:" in body
