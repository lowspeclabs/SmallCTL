from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from smallctl.models.conversation import ConversationMessage
from smallctl.state import ArtifactRecord, EvidenceRecord, LoopState
from smallctl.state_schema import ClaimRecord, DecisionRecord, ReasoningGraph


def _make_message(index: int, role: str = "user") -> ConversationMessage:
    return ConversationMessage(role=role, content=f"message {index}")


def _make_artifact(
    artifact_id: str,
    *,
    created_at: str | None = None,
    inline_content: str | None = None,
    session_id: str = "",
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        kind="tool_result",
        source="test",
        created_at=created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        size_bytes=0,
        summary="summary",
        inline_content=inline_content,
        content_path=None if inline_content else f"/tmp/{artifact_id}.txt",
        session_id=session_id,
    )


def test_transcript_messages_capped_on_append() -> None:
    state = LoopState(cwd="/tmp", transcript_message_limit=5)
    for index in range(10):
        state.append_message(_make_message(index, role="user" if index % 2 == 0 else "assistant"))

    assert len(state.transcript_messages) == 5
    assert state.transcript_messages[0].content == "message 5"
    assert state.transcript_messages[-1].content == "message 9"


def test_transcript_messages_capped_on_to_dict() -> None:
    state = LoopState(cwd="/tmp", transcript_message_limit=100)
    for index in range(10):
        state.transcript_messages.append(_make_message(index))

    state.transcript_message_limit = 5
    payload = state.to_dict()

    assert len(state.transcript_messages) == 5
    assert len(payload["transcript_messages"]) == 5
    assert payload["transcript_message_limit"] == 5


def test_reasoning_graph_records_capped() -> None:
    graph = ReasoningGraph(max_records_per_lane=5)
    for index in range(10):
        graph.evidence_records.append(EvidenceRecord(evidence_id=f"E{index}"))
        graph.decision_records.append(DecisionRecord(decision_id=f"D{index}"))
        graph.claim_records.append(ClaimRecord(claim_id=f"C{index}"))

    graph.trim_records()

    assert len(graph.evidence_records) == 5
    assert graph.evidence_records[0].evidence_id == "E5"
    assert graph.evidence_records[-1].evidence_id == "E9"
    assert len(graph.decision_records) == 5
    assert len(graph.claim_records) == 5
    assert graph.evidence_ids == [f"E{index}" for index in range(5, 10)]
    assert graph.decision_ids == [f"D{index}" for index in range(5, 10)]
    assert graph.claim_ids == [f"C{index}" for index in range(5, 10)]


def test_reasoning_graph_to_dict_uses_derived_ids() -> None:
    graph = ReasoningGraph(max_records_per_lane=3)
    for index in range(5):
        graph.evidence_records.append(EvidenceRecord(evidence_id=f"E{index}"))
    graph.trim_records()

    payload = graph.to_dict()

    assert payload["evidence_ids"] == ["E2", "E3", "E4"]
    assert payload["decision_ids"] == []


def test_loop_state_to_dict_caps_reasoning_graph() -> None:
    state = LoopState(cwd="/tmp", reasoning_graph_max_records_per_lane=3)
    for index in range(5):
        state.reasoning_graph.evidence_records.append(EvidenceRecord(evidence_id=f"E{index}"))

    payload = state.to_dict()

    assert len(state.reasoning_graph.evidence_records) == 3
    assert len(payload["reasoning_graph"]["evidence_records"]) == 3
    assert payload["reasoning_graph_max_records_per_lane"] == 3


def test_artifacts_capped_keeps_recent() -> None:
    state = LoopState(cwd="/tmp", artifact_limit=3)
    for index in range(5):
        created_at = f"2025-01-0{index + 1}T00:00:00+00:00"
        state.artifacts[f"A{index:04d}"] = _make_artifact(f"A{index:04d}", created_at=created_at)

    state.to_dict()

    assert len(state.artifacts) == 3
    assert "A0000" not in state.artifacts
    assert "A0001" not in state.artifacts
    assert "A0002" in state.artifacts
    assert "A0003" in state.artifacts
    assert "A0004" in state.artifacts


def test_artifacts_protects_referenced_ids() -> None:
    state = LoopState(cwd="/tmp", artifact_limit=2)
    for index in range(5):
        created_at = f"2025-01-0{index + 1}T00:00:00+00:00"
        state.artifacts[f"A{index:04d}"] = _make_artifact(f"A{index:04d}", created_at=created_at)
    state.tool_execution_records["op-1"] = {"artifact_id": "A0000"}

    state.to_dict()

    assert "A0000" in state.artifacts
    assert "A0003" in state.artifacts
    assert "A0004" in state.artifacts
    assert len(state.artifacts) == 3


def test_artifacts_protects_active_session_ids() -> None:
    from smallctl.state_schema import WriteSession

    state = LoopState(cwd="/tmp", artifact_limit=2)
    for index in range(5):
        created_at = f"2025-01-0{index + 1}T00:00:00+00:00"
        state.artifacts[f"A{index:04d}"] = _make_artifact(
            f"A{index:04d}", created_at=created_at, session_id="session-1"
        )
    state.active_write_sessions_by_path["/tmp/foo"] = WriteSession(write_session_id="session-1")

    state.to_dict()

    assert all(record.session_id == "session-1" for record in state.artifacts.values())
    assert len(state.artifacts) == 5


def test_artifacts_externalizes_inline_content_when_store_available() -> None:
    state = LoopState(cwd="/tmp", artifact_limit=2)
    store = MagicMock()
    store.persist_generated_text.return_value = _make_artifact(
        "A9999", inline_content=None, created_at="2025-01-10T00:00:00+00:00"
    )

    for index in range(5):
        created_at = f"2025-01-0{index + 1}T00:00:00+00:00"
        state.artifacts[f"A{index:04d}"] = _make_artifact(
            f"A{index:04d}", created_at=created_at, inline_content=f"content {index}"
        )

    state.to_dict(artifact_store=store)

    assert len(state.artifacts) == 2
    assert store.persist_generated_text.call_count == 3


def test_artifacts_keeps_inline_content_when_store_unavailable() -> None:
    state = LoopState(cwd="/tmp", artifact_limit=2)
    for index in range(5):
        created_at = f"2025-01-0{index + 1}T00:00:00+00:00"
        state.artifacts[f"A{index:04d}"] = _make_artifact(
            f"A{index:04d}", created_at=created_at, inline_content=f"content {index}"
        )

    state.to_dict()

    assert len(state.artifacts) == 5


def test_loop_state_round_trip_preserves_limits() -> None:
    state = LoopState(
        cwd="/tmp",
        transcript_message_limit=42,
        reasoning_graph_max_records_per_lane=43,
        artifact_limit=44,
    )

    restored = LoopState.from_dict(state.to_dict())

    assert restored.transcript_message_limit == 42
    assert restored.reasoning_graph_max_records_per_lane == 43
    assert restored.artifact_limit == 44


def test_loop_state_from_dict_uses_default_limits() -> None:
    restored = LoopState.from_dict({"cwd": "/tmp"})

    assert restored.transcript_message_limit == 5000
    assert restored.reasoning_graph_max_records_per_lane == 5000
    assert restored.artifact_limit == 5000
    assert restored.tool_execution_records_limit == 2000


def test_tool_execution_records_capped_keeps_recent() -> None:
    state = LoopState(cwd="/tmp", tool_execution_records_limit=3)
    for index in range(5):
        state.tool_execution_records[f"op-{index}"] = {
            "tool_name": "file_read",
            "step_count": index,
        }

    state.to_dict()

    assert len(state.tool_execution_records) == 3
    assert "op-0" not in state.tool_execution_records
    assert "op-1" not in state.tool_execution_records
    assert "op-2" in state.tool_execution_records
    assert "op-3" in state.tool_execution_records
    assert "op-4" in state.tool_execution_records


def test_tool_execution_records_protects_plan_records() -> None:
    from smallctl.state_schema import ExecutionPlan

    state = LoopState(cwd="/tmp", tool_execution_records_limit=2)
    plan = ExecutionPlan(plan_id="plan-1", goal="test")
    state.active_plan = plan
    for index in range(5):
        state.tool_execution_records[f"op-{index}"] = {
            "tool_name": "file_read",
            "step_count": index,
            "plan_id": "plan-1" if index == 0 else "plan-2",
        }

    state.to_dict()

    assert "op-0" in state.tool_execution_records
    assert "op-1" not in state.tool_execution_records
    assert len(state.tool_execution_records) <= 3


def test_tool_execution_records_protects_evidence_records() -> None:
    from smallctl.state_schema import EvidenceRecord

    state = LoopState(cwd="/tmp", tool_execution_records_limit=2)
    for index in range(5):
        state.tool_execution_records[f"op-{index}"] = {
            "tool_name": "file_read",
            "step_count": index,
            "evidence_id": "E0" if index == 0 else "",
        }
    state.reasoning_graph.evidence_records.append(EvidenceRecord(evidence_id="E0"))

    state.to_dict()

    assert "op-0" in state.tool_execution_records
    assert "op-1" not in state.tool_execution_records
    assert len(state.tool_execution_records) <= 3


def test_tool_execution_records_protects_step_evidence_operations() -> None:
    from smallctl.state_schema import StepEvidenceArtifact

    state = LoopState(cwd="/tmp", tool_execution_records_limit=2)
    for index in range(5):
        state.tool_execution_records[f"op-{index}"] = {
            "tool_name": "file_read",
            "step_count": index,
        }
    state.step_evidence["step-1"] = StepEvidenceArtifact(
        step_id="step-1", tool_operation_ids=["op-0"]
    )

    state.to_dict()

    assert "op-0" in state.tool_execution_records
    assert "op-1" not in state.tool_execution_records
    assert len(state.tool_execution_records) <= 3


def test_tool_execution_records_protects_artifact_records() -> None:
    state = LoopState(cwd="/tmp", tool_execution_records_limit=2)
    for index in range(5):
        state.tool_execution_records[f"op-{index}"] = {
            "tool_name": "file_read",
            "step_count": index,
            "artifact_id": "A0000" if index == 0 else "",
        }
    state.artifacts["A0000"] = _make_artifact("A0000")
    state.tool_execution_records["op-ref"] = {"artifact_id": "A0000"}

    state.to_dict()

    assert "op-0" in state.tool_execution_records
    assert "op-1" not in state.tool_execution_records
    assert "op-ref" in state.tool_execution_records
    assert len(state.tool_execution_records) <= 4


def test_loop_state_round_trip_preserves_tool_execution_records_limit() -> None:
    state = LoopState(cwd="/tmp", tool_execution_records_limit=42)

    restored = LoopState.from_dict(state.to_dict())

    assert restored.tool_execution_records_limit == 42


def test_persist_checkpoint_passes_artifact_store_to_to_dict() -> None:
    from unittest.mock import MagicMock
    from smallctl.harness.core_facade import _persist_checkpoint

    store = MagicMock()
    state = MagicMock()
    state.to_dict.return_value = {"thread_id": "t1"}

    harness = MagicMock()
    harness.state = state
    harness.artifact_store = store
    harness.checkpoint_path = "/tmp/checkpoint.json"
    harness.checkpoint_on_exit = True
    harness.log = MagicMock()
    harness.run_logger = None

    _persist_checkpoint(harness, {"status": "completed"})

    state.to_dict.assert_called_once_with(artifact_store=store)


def test_autosave_chat_session_passes_artifact_store_to_to_dict() -> None:
    from unittest.mock import MagicMock
    from smallctl.harness import runtime_facade

    store = MagicMock()
    state = MagicMock()
    state.to_dict.return_value = {"thread_id": "t1"}
    state.thread_id = "t1"
    state.cwd = "/tmp"
    state.step_count = 1
    state.task_received_at = "2026-01-01T00:00:00+00:00"
    state.recent_messages = []

    harness = MagicMock()
    harness.state = state
    harness.artifact_store = store
    harness.client = MagicMock(model="m1")
    harness.conversation_id = "c1"
    harness.log = MagicMock()

    scheduled: list[tuple[Any, ...]] = []
    original_schedule = runtime_facade._schedule_background_persistence

    def _fake_schedule(self: Any, func: Any, *args: Any) -> None:
        scheduled.append((func, args))

    try:
        runtime_facade._schedule_background_persistence = _fake_schedule
        runtime_facade._autosave_chat_session_state(harness)
    finally:
        runtime_facade._schedule_background_persistence = original_schedule

    state.to_dict.assert_called_once_with(artifact_store=store)
    assert len(scheduled) == 1
    func, args = scheduled[0]
    assert func is runtime_facade._persist_chat_session_state_from_runtime_state_sync
    assert isinstance(args[2], dict)
