from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness.tool_result_artifact_updates import _maybe_emit_artifact_read_eof_overread_nudge
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState


def _make_service(*, model_name: str = "qwen3.5:4b") -> SimpleNamespace:
    state = LoopState(cwd=".")
    state.scratchpad["_model_name"] = model_name
    harness = SimpleNamespace(
        state=state,
        client=SimpleNamespace(model=model_name),
        _runlog=lambda *args, **kwargs: None,
    )
    return SimpleNamespace(harness=harness)


def test_eof_overread_emits_corrective_nudge_once() -> None:
    service = _make_service()
    artifact = ArtifactRecord(
        artifact_id="A0002",
        kind="web_fetch",
        source="https://example.com/article",
        created_at="2026-04-27T00:00:00+00:00",
        size_bytes=120,
        summary="article excerpt",
        tool_name="web_fetch",
    )
    result = ToolEnvelope(
        success=True,
        output="[EOF: Start line 51 is past the end of the artifact. The artifact only has 6 lines. Stop reading and synthesize the results.]",
        metadata={
            "artifact_id": "A0002",
            "eof_overread": True,
            "requested_start_line": 51,
            "artifact_total_lines": 6,
        },
    )

    _maybe_emit_artifact_read_eof_overread_nudge(service, result=result, artifact=artifact)
    _maybe_emit_artifact_read_eof_overread_nudge(service, result=result, artifact=artifact)

    assert len(service.harness.state.recent_messages) == 1
    message = service.harness.state.recent_messages[-1]
    assert message.metadata["recovery_kind"] == "artifact_read_eof_overread"
    assert "hallucination signal" in message.content
    assert "Do not call `artifact_read` again past EOF" in message.content


def test_eof_overread_nudge_omits_small_model_sentence_for_larger_model() -> None:
    service = _make_service(model_name="gpt-5.2")
    artifact = ArtifactRecord(
        artifact_id="A0002",
        kind="web_fetch",
        source="https://example.com/article",
        created_at="2026-04-27T00:00:00+00:00",
        size_bytes=120,
        summary="article excerpt",
        tool_name="web_fetch",
    )
    result = ToolEnvelope(
        success=True,
        output="[EOF: Start line 51 is past the end of the artifact. The artifact only has 6 lines. Stop reading and synthesize the results.]",
        metadata={
            "artifact_id": "A0002",
            "eof_overread": True,
            "requested_start_line": 51,
            "artifact_total_lines": 6,
        },
    )

    _maybe_emit_artifact_read_eof_overread_nudge(service, result=result, artifact=artifact)

    message = service.harness.state.recent_messages[-1]
    assert "hallucination signal for the current small model" not in message.content
