from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from smallctl.context.artifacts import ArtifactStore
from smallctl.context.policy import ContextPolicy
from smallctl.evidence import normalize_tool_result
from smallctl.graph.state import PendingToolCall
from smallctl.graph.tool_outcomes import _store_tool_execution_record
from smallctl.harness.tool_results import ToolResultService
from smallctl.models.tool_result import ToolEnvelope
from smallctl.risk_policy import evaluate_risk_policy
from smallctl.state import ArtifactRecord, LoopState
from smallctl.tools.artifact import artifact_read


def _make_harness(tmp_path: Path) -> SimpleNamespace:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-1"
    state.run_brief.original_task = "Inspect the workspace"
    state.artifacts = {}
    state.retrieval_cache = []
    state.scratchpad = {}
    return SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path, "run-1"),
        context_policy=ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4),
        summarizer_client=None,
        summarizer=None,
        _current_user_task=lambda: "Inspect the workspace",
        _runlog=lambda *args, **kwargs: None,
    )


def _prime_execution_record(
    harness: SimpleNamespace,
    *,
    operation_id: str,
    tool_name: str,
    tool_call_id: str,
    args: dict[str, object],
    result: ToolEnvelope,
) -> None:
    _store_tool_execution_record(
        harness,
        operation_id=operation_id,
        thread_id=harness.state.thread_id,
        step_count=harness.state.step_count,
        pending=PendingToolCall(tool_name=tool_name, args=args, tool_call_id=tool_call_id),
        result=result,
    )


def test_normalize_tool_result_builds_direct_observation() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0001",
        kind="file_read",
        source="README.md",
        created_at="2026-04-09T00:00:00+00:00",
        size_bytes=12,
        summary="README.md text (12 chars)",
        tool_name="file_read",
    )
    evidence = normalize_tool_result(
        tool_name="file_read",
        result=ToolEnvelope(success=True, output="hello", metadata={"path": "README.md"}),
        artifact=artifact,
        operation_id="op-0",
        phase="explore",
        evidence_context={"args": {"path": "README.md"}},
    )

    assert evidence.evidence_type == "direct_observation"
    assert evidence.artifact_id == "A0001"
    assert evidence.statement.startswith("file_read:")


def test_tool_execution_record_carries_provisional_evidence_context() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(tool_name="file_read", args={"path": "README.md"}, tool_call_id="call-1")
    result = ToolEnvelope(success=True, output="hello", metadata={"path": "README.md"})

    _store_tool_execution_record(
        harness,
        operation_id="op-1",
        thread_id="thread-1",
        step_count=3,
        pending=pending,
        result=result,
    )

    stored = state.tool_execution_records["op-1"]
    assert stored["evidence_context"]["operation_id"] == "op-1"
    assert stored["evidence_context"]["tool_name"] == "file_read"
    assert stored["evidence_context"]["args"] == {"path": "README.md"}


def test_file_read_records_evidence_and_preserves_raw_artifact(tmp_path: Path) -> None:
    async def _run() -> None:
        harness = _make_harness(tmp_path)
        service = ToolResultService(harness)
        result = ToolEnvelope(success=True, output="alpha\nbeta\n", metadata={"path": "README.md"})
        _prime_execution_record(
            harness,
            operation_id="op-1",
            tool_name="file_read",
            tool_call_id="call-1",
            args={"path": "README.md"},
            result=result,
        )

        message = await service.record_result(
            tool_name="file_read",
            tool_call_id="call-1",
            operation_id="op-1",
            result=result,
            arguments={"path": "README.md"},
        )

        artifact_id = str(message.metadata.get("artifact_id") or "")
        assert artifact_id
        artifact = harness.state.artifacts[artifact_id]
        assert Path(artifact.content_path).read_text(encoding="utf-8") == "alpha\nbeta\n"

        evidence = harness.state.reasoning_graph.evidence_records[-1]
        assert evidence.tool_name == "file_read"
        assert evidence.artifact_id == artifact_id
        assert evidence.negative is False
        assert evidence.evidence_type == "direct_observation"
        assert harness.state.tool_execution_records["op-1"]["evidence_id"] == evidence.evidence_id
        assert harness.state.tool_execution_records["op-1"]["evidence_record"]["artifact_id"] == artifact_id
        assert "README.md" in evidence.statement
        notepad = harness.state.scratchpad.get("_session_notepad", {})
        entries = notepad.get("entries", []) if isinstance(notepad, dict) else []
        assert any("read path: README.md" in str(entry) for entry in entries)

    asyncio.run(_run())


def test_shell_failure_records_negative_evidence(tmp_path: Path) -> None:
    async def _run() -> None:
        harness = _make_harness(tmp_path)
        service = ToolResultService(harness)
        result = ToolEnvelope(
            success=False,
            error="Permission denied",
            metadata={"command": "chmod +x ./script.sh"},
        )
        _prime_execution_record(
            harness,
            operation_id="op-2",
            tool_name="shell_exec",
            tool_call_id="call-2",
            args={"command": "chmod +x ./script.sh"},
            result=result,
        )

        message = await service.record_result(
            tool_name="shell_exec",
            tool_call_id="call-2",
            operation_id="op-2",
            result=result,
            arguments={"command": "chmod +x ./script.sh"},
        )

        artifact_id = str(message.metadata.get("artifact_id") or "")
        artifact = harness.state.artifacts[artifact_id]
        raw_json = json.loads(Path(artifact.content_path.replace(".txt", ".json")).read_text(encoding="utf-8"))
        assert raw_json["error"] == "Permission denied"

        evidence = harness.state.reasoning_graph.evidence_records[-1]
        assert evidence.tool_name == "shell_exec"
        assert evidence.negative is True
        assert evidence.evidence_type == "negative_evidence"
        assert "Permission denied" in evidence.statement
        invalidations = harness.state.scratchpad.get("_context_invalidations", [])
        assert any(str(item.get("reason") or "") == "verifier_failed" for item in invalidations if isinstance(item, dict))

    asyncio.run(_run())


def test_diagnosis_shell_evidence_auto_confirms_supported_claim(tmp_path: Path) -> None:
    async def _run() -> None:
        harness = _make_harness(tmp_path)
        harness.state.current_phase = "execute"
        harness.state.scratchpad["_task_classification"] = "diagnosis_remediation"
        service = ToolResultService(harness)
        result = ToolEnvelope(
            success=False,
            error="Permission denied",
            metadata={"command": "apt-cache search guacamole", "host": "192.168.1.63"},
        )
        _prime_execution_record(
            harness,
            operation_id="op-claim",
            tool_name="ssh_exec",
            tool_call_id="call-claim",
            args={"host": "192.168.1.63", "command": "apt-cache search guacamole"},
            result=result,
        )

        await service.record_result(
            tool_name="ssh_exec",
            tool_call_id="call-claim",
            operation_id="op-claim",
            result=result,
            arguments={"host": "192.168.1.63", "command": "apt-cache search guacamole"},
        )

        evidence = harness.state.reasoning_graph.evidence_records[-1]
        claim = harness.state.reasoning_graph.claim_records[-1]
        assert claim.status == "confirmed"
        assert claim.supporting_evidence_ids == [evidence.evidence_id]
        assert claim.claim_id in evidence.claim_ids
        assert harness.state.tool_execution_records["op-claim"]["claim_ids"] == [claim.claim_id]

        decision = evaluate_risk_policy(
            harness.state,
            tool_name="file_patch",
            tool_risk="high",
            phase="repair",
            action="Patch file temp/fix.py",
            expected_effect="Apply a repair after confirming the issue.",
            rollback="Revert the file.",
            verification="Rerun the verifier.",
            approval_available=False,
        )
        assert decision.allowed is True
        assert decision.proof_bundle["supported_claim_ids"] == [claim.claim_id]

    asyncio.run(_run())


def test_artifact_read_cache_hit_marks_replayed_evidence(tmp_path: Path) -> None:
    async def _run() -> None:
        harness = _make_harness(tmp_path)
        service = ToolResultService(harness)

        initial = ToolEnvelope(success=True, output="cached content", metadata={"path": "README.md"})
        _prime_execution_record(
            harness,
            operation_id="op-3",
            tool_name="file_read",
            tool_call_id="call-3",
            args={"path": "README.md"},
            result=initial,
        )
        initial_message = await service.record_result(
            tool_name="file_read",
            tool_call_id="call-3",
            operation_id="op-3",
            result=initial,
            arguments={"path": "README.md"},
        )
        cached_artifact_id = str(initial_message.metadata.get("artifact_id") or "")

        cached = ToolEnvelope(
            success=True,
            output={"status": "cached", "artifact_id": cached_artifact_id, "summary": "README.md text"},
            metadata={"cache_hit": True, "artifact_id": cached_artifact_id, "tool_name": "artifact_read"},
        )
        _prime_execution_record(
            harness,
            operation_id="op-4",
            tool_name="artifact_read",
            tool_call_id="call-4",
            args={"artifact_id": cached_artifact_id},
            result=cached,
        )
        message = await service.record_result(
            tool_name="artifact_read",
            tool_call_id="call-4",
            operation_id="op-4",
            result=cached,
            arguments={"artifact_id": cached_artifact_id},
        )

        assert message.metadata["artifact_id"] == cached_artifact_id
        evidence = harness.state.reasoning_graph.evidence_records[-1]
        assert evidence.tool_name == "artifact_read"
        assert evidence.replayed is True
        assert evidence.evidence_type == "replayed_or_cached"
        assert evidence.artifact_id == cached_artifact_id

    asyncio.run(_run())


def test_artifact_print_reuses_existing_artifact(tmp_path: Path) -> None:
    async def _run() -> None:
        harness = _make_harness(tmp_path)
        service = ToolResultService(harness)

        initial = ToolEnvelope(success=True, output="cached content", metadata={"path": "README.md"})
        _prime_execution_record(
            harness,
            operation_id="op-5",
            tool_name="file_read",
            tool_call_id="call-5",
            args={"path": "README.md"},
            result=initial,
        )
        initial_message = await service.record_result(
            tool_name="file_read",
            tool_call_id="call-5",
            operation_id="op-5",
            result=initial,
            arguments={"path": "README.md"},
        )
        cached_artifact_id = str(initial_message.metadata.get("artifact_id") or "")

        print_result = ToolEnvelope(
            success=True,
            output="",
            metadata={"artifact_id": cached_artifact_id, "tool_name": "artifact_print"},
        )
        _prime_execution_record(
            harness,
            operation_id="op-6",
            tool_name="artifact_print",
            tool_call_id="call-6",
            args={"artifact_id": cached_artifact_id},
            result=print_result,
        )
        message = await service.record_result(
            tool_name="artifact_print",
            tool_call_id="call-6",
            operation_id="op-6",
            result=print_result,
            arguments={"artifact_id": cached_artifact_id},
        )

        assert message.metadata["artifact_id"] == cached_artifact_id
        assert message.metadata["cache_hit"] is True
        evidence = harness.state.reasoning_graph.evidence_records[-1]
        assert evidence.tool_name == "artifact_print"
        assert evidence.replayed is True
        assert evidence.evidence_type == "replayed_or_cached"

    asyncio.run(_run())


def test_failed_artifact_read_does_not_create_self_referential_artifact(tmp_path: Path) -> None:
    async def _run() -> None:
        harness = _make_harness(tmp_path)
        service = ToolResultService(harness)

        shell_result = ToolEnvelope(
            success=True,
            output={"stdout": "nmap output", "stderr": "", "exit_code": 0},
            metadata={"command": "nmap -sn 192.168.1.0/24"},
        )
        _prime_execution_record(
            harness,
            operation_id="op-shell",
            tool_name="shell_exec",
            tool_call_id="call-shell",
            args={"command": "nmap -sn 192.168.1.0/24"},
            result=shell_result,
        )
        await service.record_result(
            tool_name="shell_exec",
            tool_call_id="call-shell",
            operation_id="op-shell",
            result=shell_result,
            arguments={"command": "nmap -sn 192.168.1.0/24"},
        )

        status_result = ToolEnvelope(
            success=True,
            output={"phase": "execute"},
            metadata={},
        )
        _prime_execution_record(
            harness,
            operation_id="op-status",
            tool_name="loop_status",
            tool_call_id="call-status",
            args={},
            result=status_result,
        )
        await service.record_result(
            tool_name="loop_status",
            tool_call_id="call-status",
            operation_id="op-status",
            result=status_result,
            arguments={},
        )

        missing_result = ToolEnvelope(
            success=False,
            output=None,
            error="Artifact A0003 not found in state.",
            metadata={"tool_name": "artifact_read"},
        )
        _prime_execution_record(
            harness,
            operation_id="op-missing",
            tool_name="artifact_read",
            tool_call_id="call-missing",
            args={"artifact_id": "A0003"},
            result=missing_result,
        )
        message = await service.record_result(
            tool_name="artifact_read",
            tool_call_id="call-missing",
            operation_id="op-missing",
            result=missing_result,
            arguments={"artifact_id": "A0003"},
        )

        assert message.content == "Artifact A0003 not found in state."
        assert message.metadata == {}
        assert sorted(harness.state.artifacts.keys()) == ["A0001", "A0002"]

        reread = artifact_read(harness.state, artifact_id="A0003")
        assert reread["success"] is False
        assert reread["error"] == "Artifact A0003 not found in state."

    asyncio.run(_run())
