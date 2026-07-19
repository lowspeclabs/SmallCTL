"""Regression tests for retrieval bundle history recording (Bug 18)."""
from __future__ import annotations

from smallctl.context.policy import ContextPolicy
from smallctl.context.retrieval import LexicalRetriever
from smallctl.state import ArtifactRecord, EpisodicSummary, LoopState


def test_retrieve_bundle_records_artifact_and_summary_history() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "inspect alpha beta"
    state.working_memory.current_goal = "inspect alpha beta"
    state.artifacts = {
        "A1": ArtifactRecord(
            artifact_id="A1",
            kind="tool_result",
            source="/tmp/alpha.txt",
            created_at="2026-01-01T00:00:00Z",
            size_bytes=100,
            summary="alpha beta primary file",
            keywords=["alpha", "beta"],
            tool_name="file_read",
            inline_content="alpha beta selected content",
        ),
    }
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="S1",
            created_at="2026-01-01T00:00:00Z",
            decisions=["read alpha"],
            files_touched=["/tmp/alpha.txt"],
            notes=["alpha summary"],
        ),
    ]

    retriever = LexicalRetriever(policy=ContextPolicy(max_artifact_snippets=1))
    bundle = retriever.retrieve_bundle(
        state=state,
        query="inspect alpha beta",
        include_experiences=False,
    )

    assert [snippet.artifact_id for snippet in bundle.artifacts] == ["A1"]
    assert [summary.summary_id for summary in bundle.summaries] == ["S1"]

    artifact_history = state.scratchpad.get("_retrieved_artifact_history")
    summary_history = state.scratchpad.get("_retrieved_summary_history")
    assert isinstance(artifact_history, list)
    assert isinstance(summary_history, list)
    assert [entry["id"] for entry in artifact_history] == ["A1"]
    assert [entry["id"] for entry in summary_history] == ["S1"]
    assert all("retrieved_at_step" in entry for entry in artifact_history)
    assert all("retrieved_at_step" in entry for entry in summary_history)
