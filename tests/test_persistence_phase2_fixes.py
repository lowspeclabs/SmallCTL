from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from smallctl.context import ContextPolicy, ContextSummarizer
from smallctl.graph.checkpoint import FileCheckpointSaver
from smallctl.models.conversation import ConversationMessage
from smallctl.state import EpisodicSummary, LoopState


def _make_summarizer_state() -> LoopState:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    return state


def test_episodic_summary_ids_unique_and_monotonic_after_trims() -> None:
    state = _make_summarizer_state()
    state.episodic_summaries.append(
        EpisodicSummary(summary_id="task-0001-summary", created_at="2026-07-16T00:00:00+00:00")
    )
    summarizer = ContextSummarizer(ContextPolicy())
    produced: list[str] = []
    for index in range(15):
        state.recent_messages = [
            ConversationMessage(role="user", content=f"task message {index}"),
            ConversationMessage(role="assistant", content=f"work {index}"),
        ]
        result = summarizer.compact_recent_messages_with_status(state=state, keep_recent=1)
        assert result.summary is not None
        produced.append(result.summary.summary_id)
        state.episodic_summaries = state.episodic_summaries[-12:]
    assert len(state.episodic_summaries) == 12
    assert len(produced) == len(set(produced))
    assert produced == [f"S{index:04d}" for index in range(1, 16)]


def test_brief_and_bundle_ids_unique_and_monotonic_after_trims() -> None:
    state = _make_summarizer_state()
    summarizer = ContextSummarizer(ContextPolicy())
    brief_ids: list[str] = []
    bundle_ids: list[str] = []
    for index in range(15):
        bundle = summarizer.compact_to_turn_bundle(
            state=state,
            messages=[
                ConversationMessage(role="user", content=f"bundle ask {index}"),
                ConversationMessage(role="assistant", content=f"bundle work {index}"),
            ],
            step_range=(index, index + 1),
        )
        assert bundle is not None
        bundle_ids.append(bundle.bundle_id)
        state.turn_bundles.append(bundle)
        state.turn_bundles = state.turn_bundles[-4:]
        brief = summarizer.compact_turn_bundles_to_brief(
            state=state,
            bundles=[bundle],
            step_range=(index, index + 1),
        )
        assert brief is not None
        brief_ids.append(brief.brief_id)
        state.context_briefs.append(brief)
        while len(state.context_briefs) > 12:
            state.context_briefs.pop(0)
    assert len(state.context_briefs) == 12
    assert len(brief_ids) == len(set(brief_ids))
    assert brief_ids == [f"B{index:04d}" for index in range(1, 16)]
    assert len(bundle_ids) == len(set(bundle_ids))
    assert bundle_ids == [f"TB{index:04d}" for index in range(1, 16)]


def _write_checkpoint(
    saver: FileCheckpointSaver,
    *,
    thread_id: str = "thread-1",
    checkpoint_id: str,
    channel: str = "loop_state",
    value: Any = {"step": 1},
) -> None:
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }
    checkpoint = {
        "id": checkpoint_id,
        "channel_values": {channel: value},
        "channel_versions": {channel: 1},
        "ts": "2024-01-01T00:00:00+00:00",
    }
    metadata: dict[str, Any] = {"source": "loop", "step": 1, "parents": {}, "run_id": "run-1"}
    saver.put(config, checkpoint, metadata, {channel: 1})
    saver.put_writes(config, [(channel, value)], "task-1", "prepare_prompt")


def _checkpoint_config(thread_id: str, checkpoint_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }


def test_checkpoint_corrupt_primary_recovers_prior_checkpoints_from_backup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "checkpoints.json"
    saver = FileCheckpointSaver(path)
    _write_checkpoint(saver, checkpoint_id="chk-1")
    assert (tmp_path / "checkpoints.json.bak").exists()

    path.write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="smallctl.graph.checkpoint"):
        recovered = FileCheckpointSaver(path)

    assert not recovered.degraded
    messages = [record.getMessage() for record in caplog.records]
    assert any("checkpoint_load_failed" in message for message in messages)
    assert any("checkpoint_backup_recovery" in message for message in messages)
    assert recovered.get_tuple(_checkpoint_config("thread-1", "chk-1")) is not None

    _write_checkpoint(recovered, checkpoint_id="chk-2")
    assert json.loads(path.read_text(encoding="utf-8"))["storage"]
    assert recovered.get_tuple(_checkpoint_config("thread-1", "chk-2")) is not None


def test_checkpoint_degraded_store_refuses_flush_without_valid_backup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "checkpoints.json"
    path.write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="smallctl.graph.checkpoint"):
        saver = FileCheckpointSaver(path)

    assert saver.degraded
    assert any("checkpoint_unrecoverable" in record.getMessage() for record in caplog.records)
    corrupt_bytes = path.read_bytes()

    config = _checkpoint_config("thread-1", "chk-1")
    checkpoint = {
        "id": "chk-1",
        "channel_values": {"loop_state": {"step": 1}},
        "channel_versions": {"loop_state": 1},
        "ts": "2024-01-01T00:00:00+00:00",
    }
    metadata: dict[str, Any] = {"source": "loop", "step": 1, "parents": {}, "run_id": "run-1"}
    with pytest.raises(RuntimeError, match="degraded"):
        saver.put(config, checkpoint, metadata, {"loop_state": 1})
    with pytest.raises(RuntimeError, match="degraded"):
        saver.put_writes(config, [("loop_state", {"step": 1})], "task-1", "prepare_prompt")
    with pytest.raises(RuntimeError, match="degraded"):
        saver.delete_thread("thread-1")

    assert path.read_bytes() == corrupt_bytes
    assert not (tmp_path / "checkpoints.json.bak").exists()


def test_checkpoint_degraded_state_clears_after_valid_reload(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.json"
    path.write_text("garbage", encoding="utf-8")
    saver = FileCheckpointSaver(path)
    assert saver.degraded

    good_path = tmp_path / "good.json"
    good_saver = FileCheckpointSaver(good_path)
    _write_checkpoint(good_saver, checkpoint_id="chk-1")
    path.write_bytes(good_path.read_bytes())

    assert saver.get_tuple(_checkpoint_config("thread-1", "chk-1")) is not None
    assert not saver.degraded
    _write_checkpoint(saver, checkpoint_id="chk-2")
    assert saver.get_tuple(_checkpoint_config("thread-1", "chk-2")) is not None


def test_checkpoint_retention_prunes_oldest_per_thread_and_keeps_latest(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.json"
    saver = FileCheckpointSaver(path, max_checkpoints_per_thread=3)
    for index in range(5):
        _write_checkpoint(saver, thread_id="thread-1", checkpoint_id=f"chk-{index:04d}")
    for index in range(2):
        _write_checkpoint(saver, thread_id="thread-2", checkpoint_id=f"chk-{index:04d}")

    assert sorted(saver.storage["thread-1"][""]) == ["chk-0002", "chk-0003", "chk-0004"]
    assert sorted(saver.storage["thread-2"][""]) == ["chk-0000", "chk-0001"]
    thread_one_writes = {key[2] for key in saver.writes if key[0] == "thread-1"}
    assert thread_one_writes == {"chk-0002", "chk-0003", "chk-0004"}

    reloaded = FileCheckpointSaver(path)
    assert sorted(reloaded.storage["thread-1"][""]) == ["chk-0002", "chk-0003", "chk-0004"]
    assert reloaded.get_tuple(_checkpoint_config("thread-1", "chk-0004")) is not None
    assert reloaded.get_tuple(_checkpoint_config("thread-1", "chk-0001")) is None
