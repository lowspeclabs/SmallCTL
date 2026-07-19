from __future__ import annotations

from smallctl.context.retrieval_scoring import score_artifact
from smallctl.state import LoopState
from smallctl.state_schema import ArtifactRecord, RunBrief, WriteSession


def test_fully_read_target_path_artifact_gets_downgraded() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief = RunBrief(original_task="read AGENTS.md and create a container")
    state.scratchpad["_task_target_paths"] = ["/tmp/proxmox-manager/AGENTS.md"]

    artifact_fully_read = ArtifactRecord(
        artifact_id="A0001",
        kind="file_read",
        source="/tmp/proxmox-manager/AGENTS.md",
        created_at="2026-01-01T00:00:00+00:00",
        size_bytes=100,
        summary="agents guide",
        metadata={"path": "/tmp/proxmox-manager/AGENTS.md", "complete_file": True},
    )

    artifact_partial = ArtifactRecord(
        artifact_id="A0002",
        kind="file_read",
        source="/tmp/proxmox-manager/AGENTS.md",
        created_at="2026-01-01T00:00:00+00:00",
        size_bytes=100,
        summary="agents guide",
        metadata={"path": "/tmp/proxmox-manager/AGENTS.md"},
    )

    query = "agents.md"
    query_tokens = {"agents.md"}

    score_fully_read = score_artifact(
        artifact=artifact_fully_read,
        query=query,
        query_tokens=query_tokens,
        recency=1,
        state=state,
    )
    score_partial = score_artifact(
        artifact=artifact_partial,
        query=query,
        query_tokens=query_tokens,
        recency=1,
        state=state,
    )

    assert score_fully_read < score_partial


def test_write_target_path_not_downgraded() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief = RunBrief(original_task="edit app.py")
    state.scratchpad["_task_target_paths"] = ["/tmp/app.py"]
    state.write_session = WriteSession()
    state.write_session.write_target_path = "/tmp/app.py"

    artifact = ArtifactRecord(
        artifact_id="A0001",
        kind="file_read",
        source="/tmp/app.py",
        created_at="2026-01-01T00:00:00+00:00",
        size_bytes=100,
        summary="app.py full file",
        metadata={"path": "/tmp/app.py", "complete_file": True},
    )

    score = score_artifact(
        artifact=artifact,
        query="app.py",
        query_tokens={"app.py"},
        recency=1,
        state=state,
    )

    assert score > 0
