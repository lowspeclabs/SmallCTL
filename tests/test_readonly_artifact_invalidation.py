"""Regression tests for read-only artifact invalidation (Bug 19)."""
from __future__ import annotations

from smallctl.context.frame_invalidation_filtering import (
    artifact_invalidated,
    filter_invalidated_artifact_snippets,
)
from smallctl.state import ArtifactRecord, ArtifactSnippet, LoopState


def _read_only_file_read_artifact(path: str, artifact_id: str = "A1") -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        kind="tool_result",
        source=path,
        created_at="2026-01-01T00:00:00Z",
        size_bytes=100,
        summary=f"read {path}",
        keywords=["read"],
        tool_name="file_read",
        inline_content="content",
        metadata={"path": path},
    )


def test_read_only_artifact_not_invalidated_by_file_changed() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.scratchpad["_context_invalidations"] = [
        {
            "reason": "file_changed",
            "step": 1,
            "phase": "execute",
            "paths": ["/tmp/config.txt"],
            "created_at": "2026-01-01T00:00:01Z",
        }
    ]
    artifact = _read_only_file_read_artifact("/tmp/config.txt")

    assert artifact_invalidated(state=state, artifact=artifact, invalidations=state.scratchpad["_context_invalidations"]) is False


def test_filter_retains_read_only_artifact_after_file_changed() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.scratchpad["_context_invalidations"] = [
        {
            "reason": "file_changed",
            "step": 1,
            "phase": "execute",
            "paths": ["/tmp/config.txt"],
            "created_at": "2026-01-01T00:00:01Z",
        }
    ]
    state.artifacts = {"A1": _read_only_file_read_artifact("/tmp/config.txt")}
    snippets = [ArtifactSnippet(artifact_id="A1", text="content", score=1.0)]

    kept, dropped_ids = filter_invalidated_artifact_snippets(
        state=state,
        snippets=snippets,
    )

    assert [snippet.artifact_id for snippet in kept] == ["A1"]
    assert dropped_ids == []
