"""Regression tests for the PERSISTENCE/REDACTION gap-closure group.

Covers:
- H18: ExperienceStore.scrub_sensitive_notes holds the sidecar write lock for
  the whole read-modify-write so concurrent upserts are not lost.
- M10: FileCheckpointSaver structurally validates the checkpoint database at
  load time (invalid payloads fall back to .bak / degraded mode) and retention
  pruning relies on LangGraph's lexically monotonic uuid6 checkpoint ids.
- C2: quoted assignment values are fully redacted, and ExperienceStore redacts
  note text at the persistence boundary (including the memory_cli promote
  path).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.base.id import uuid6

from smallctl.graph.checkpoint import FileCheckpointSaver
from smallctl.memory_cli import memory_cli
from smallctl.memory_store import ExperienceStore
from smallctl.redaction import redact_sensitive_text
from smallctl.state import ExperienceMemory


def _make_memory(memory_id: str, *, notes: str = "note") -> ExperienceMemory:
    return ExperienceMemory(
        memory_id=memory_id,
        tier="warm",
        tool_name="shell_exec",
        intent="general_task",
        outcome="success",
        notes=notes,
    )


def _write_checkpoint(
    saver: FileCheckpointSaver,
    *,
    thread_id: str = "thread-1",
    checkpoint_id: str,
    channel: str = "loop_state",
    value: Any = {"step": 1},
) -> None:
    config = _checkpoint_config(thread_id, checkpoint_id)
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


# ---------------------------------------------------------------------------
# H18: scrub read-modify-write holds the write lock for the whole transaction.
# ---------------------------------------------------------------------------


def test_memory_scrub_write_serializes_with_concurrent_upsert(tmp_path, monkeypatch) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store = ExperienceStore(path)
    secret = "sk-proj-scrub-secret-123"
    # Seed the plaintext-secret record directly, bypassing write-time
    # redaction, so the scrub has something to change.
    path.write_text(
        json.dumps(
            {
                "memory_id": "mem-secret",
                "tier": "warm",
                "intent": "general_task",
                "tool_name": "shell_exec",
                "outcome": "success",
                "notes": f"OPENAI_API_KEY={secret}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    original_read_payloads = ExperienceStore._read_payloads
    entered = threading.Event()
    release = threading.Event()

    def _blocking_read_payloads(self, *, force: bool = False):
        payloads = original_read_payloads(self, force=force)
        # Park exactly once, after the scrub's read, to force a concurrent
        # upsert into the gap between the scrub's read and its write. With the
        # lock held for the whole transaction the upsert must wait its turn.
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=30)
        return payloads

    monkeypatch.setattr(ExperienceStore, "_read_payloads", _blocking_read_payloads)

    errors: list[BaseException] = []
    summary: dict[str, int] = {}

    def _scrub() -> None:
        try:
            summary.update(store.scrub_sensitive_notes(write=True))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _upsert() -> None:
        try:
            assert store.upsert(_make_memory("mem-concurrent")) is not None
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    scrub_thread = threading.Thread(target=_scrub)
    scrub_thread.start()
    assert entered.wait(timeout=30)

    upsert_thread = threading.Thread(target=_upsert)
    upsert_thread.start()
    time.sleep(0.2)
    release.set()
    scrub_thread.join(timeout=30)
    upsert_thread.join(timeout=30)

    assert not scrub_thread.is_alive()
    assert not upsert_thread.is_alive()
    assert not errors
    assert summary == {"records": 1, "changed": 1, "written": 1}

    records = {m.memory_id: m for m in ExperienceStore(path).list()}
    assert set(records) == {"mem-secret", "mem-concurrent"}
    assert secret not in records["mem-secret"].notes
    assert secret not in path.read_text(encoding="utf-8")


def test_memory_scrub_dry_run_reports_without_writing(tmp_path) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store = ExperienceStore(path)

    # Seed a plaintext-secret record directly, bypassing write-time redaction,
    # to simulate a pre-fix store on disk.
    path.write_text(
        json.dumps(
            {
                "memory_id": "mem-legacy",
                "tier": "warm",
                "intent": "general_task",
                "tool_name": "shell_exec",
                "outcome": "success",
                "notes": "SSH_PASSWORD=hunter2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = store.scrub_sensitive_notes(write=False)
    assert summary == {"records": 1, "changed": 0, "written": 0}
    assert "hunter2" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# M10: structural validation of the checkpoint database payload.
# ---------------------------------------------------------------------------

_TYPED = {"type": "msgpack", "data": "aGVsbG8="}

_INVALID_PRIMARY_PAYLOADS = (
    pytest.param({}, id="bare-empty-object"),
    pytest.param({"storage": {}, "writes": [], "blobs": []}, id="storage-not-a-list"),
    pytest.param(
        {"storage": [{"thread_id": "thread-1"}], "writes": [], "blobs": []},
        id="storage-record-missing-fields",
    ),
    pytest.param(
        {
            "storage": [
                {
                    "thread_id": "thread-1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "chk-1",
                    "checkpoint": {"type": "msgpack", "data": "!!!not-base64!!!"},
                    "metadata": _TYPED,
                    "parent_checkpoint_id": None,
                }
            ],
            "writes": [],
            "blobs": [],
        },
        id="storage-record-bad-typed-blob",
    ),
    pytest.param(
        {
            "storage": [],
            "writes": [
                {
                    "thread_id": "thread-1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "chk-1",
                    "task_id": "task-1",
                    "idx": "zero",
                    "channel": "loop_state",
                    "value": _TYPED,
                    "task_path": "prepare_prompt",
                }
            ],
            "blobs": [],
        },
        id="writes-record-idx-not-int",
    ),
    pytest.param(
        {
            "storage": [],
            "writes": [],
            "blobs": [
                {
                    "thread_id": "thread-1",
                    "checkpoint_ns": "",
                    "channel": "loop_state",
                    "version": {"kind": "bogus", "value": 1},
                    "value": _TYPED,
                }
            ],
        },
        id="blobs-record-bad-version",
    ),
)


@pytest.mark.parametrize("invalid_payload", _INVALID_PRIMARY_PAYLOADS)
def test_checkpoint_structurally_invalid_primary_recovers_from_backup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, invalid_payload: Any
) -> None:
    path = tmp_path / "checkpoints.json"
    saver = FileCheckpointSaver(path)
    _write_checkpoint(saver, checkpoint_id="chk-1")
    assert (tmp_path / "checkpoints.json.bak").exists()

    invalid_text = json.dumps(invalid_payload)
    path.write_text(invalid_text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="smallctl.graph.checkpoint"):
        recovered = FileCheckpointSaver(path)

    assert not recovered.degraded
    messages = [record.getMessage() for record in caplog.records]
    assert any("checkpoint_validation_failed" in message for message in messages)
    assert any("checkpoint_backup_recovery" in message for message in messages)
    assert recovered.get_tuple(_checkpoint_config("thread-1", "chk-1")) is not None
    # The invalid primary is preserved on disk; nothing silently overwrites it.
    assert path.read_text(encoding="utf-8") == invalid_text

    _write_checkpoint(recovered, checkpoint_id="chk-2")
    assert recovered.get_tuple(_checkpoint_config("thread-1", "chk-2")) is not None


def test_checkpoint_structurally_invalid_primary_without_backup_goes_degraded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "checkpoints.json"
    path.write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="smallctl.graph.checkpoint"):
        saver = FileCheckpointSaver(path)

    assert saver.degraded
    messages = [record.getMessage() for record in caplog.records]
    assert any("checkpoint_validation_failed" in message for message in messages)
    assert any("checkpoint_unrecoverable" in message for message in messages)

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
    assert path.read_text(encoding="utf-8") == "{}"


def test_checkpoint_legitimate_empty_database_loads_cleanly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A fresh saver always flushes all three sections, so this empty-but-
    # well-formed payload (not a bare {}) is the legitimate empty state.
    path = tmp_path / "checkpoints.json"
    path.write_text(
        json.dumps({"storage": [], "writes": [], "blobs": []}), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="smallctl.graph.checkpoint"):
        saver = FileCheckpointSaver(path)

    assert not saver.degraded
    assert not caplog.records
    assert saver.get_tuple(_checkpoint_config("thread-1", "chk-1")) is None

    _write_checkpoint(saver, checkpoint_id="chk-1")
    assert saver.get_tuple(_checkpoint_config("thread-1", "chk-1")) is not None


def test_langgraph_checkpoint_ids_are_lexically_monotonic() -> None:
    # Retention pruning and latest-checkpoint selection order checkpoint ids
    # lexically; that only prunes the genuinely oldest checkpoint because
    # LangGraph generates time-ordered uuid6 ids. Lock the assumption in.
    ids = [str(uuid6(clock_seq=step)) for step in range(16)]
    assert ids == sorted(ids)


def test_checkpoint_retention_prunes_genuinely_oldest_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.json"
    saver = FileCheckpointSaver(path, max_checkpoints_per_thread=2)
    ids = [str(uuid6(clock_seq=step)) for step in range(4)]
    for checkpoint_id in ids:
        _write_checkpoint(saver, checkpoint_id=checkpoint_id)

    assert sorted(saver.storage["thread-1"][""]) == ids[2:]
    assert {key[2] for key in saver.writes if key[0] == "thread-1"} == set(ids[2:])
    assert saver.get_tuple(_checkpoint_config("thread-1", ids[0])) is None
    assert saver.get_tuple(_checkpoint_config("thread-1", ids[1])) is None
    assert saver.get_tuple(_checkpoint_config("thread-1", ids[-1])) is not None

    reloaded = FileCheckpointSaver(path)
    assert sorted(reloaded.storage["thread-1"][""]) == ids[2:]


# ---------------------------------------------------------------------------
# C2a: quoted assignment values are fully consumed and redacted.
# ---------------------------------------------------------------------------

_C2_QUOTED_SAMPLES = (
    ('AWS_SECRET_ACCESS_KEY="abc def, xyz"', ("abc def, xyz", "abc", "xyz")),
    ("AWS_SECRET_ACCESS_KEY='abc def, xyz'", ("abc def, xyz", "abc", "xyz")),
    ('OPENAI_API_KEY="sk-proj abc,123"', ("sk-proj abc,123", "abc,123")),
    ('export GH_TOKEN="ghp_secret value, with spaces"', ("ghp_secret value, with spaces", "spaces")),
    ('MY_SECRET_KEY="escaped \\" quote secret"', ("quote secret",)),
)


@pytest.mark.parametrize("text,fragments", _C2_QUOTED_SAMPLES)
def test_c2_quoted_assignment_values_are_fully_redacted(text: str, fragments: tuple[str, ...]) -> None:
    redacted = redact_sensitive_text(text)
    for fragment in fragments:
        assert fragment not in redacted
    assert "REDACTED" in redacted


def test_c2_quoted_assignment_redaction_is_idempotent() -> None:
    text = 'AWS_SECRET_ACCESS_KEY="abc def, xyz"'
    once = redact_sensitive_text(text)
    twice = redact_sensitive_text(once)
    assert once == twice


# ---------------------------------------------------------------------------
# C2b: redaction at the ExperienceStore persistence boundary.
# ---------------------------------------------------------------------------


def test_memory_upsert_redacts_secrets_in_notes_at_persistence_boundary(tmp_path) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store = ExperienceStore(path)
    api_key = "sk-proj-persist-boundary-456"
    bearer = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    memory = _make_memory(
        "mem-secrets",
        notes=f"debug with OPENAI_API_KEY={api_key} and Authorization: Bearer {bearer}",
    )

    # upsert/delete return contracts are unchanged by write-time redaction.
    assert store.upsert(memory) is memory

    stored_text = path.read_text(encoding="utf-8")
    assert api_key not in stored_text
    assert bearer not in stored_text
    assert "[REDACTED]" in stored_text

    loaded = store.get("mem-secrets")
    assert loaded is not None
    assert api_key not in loaded.notes
    assert bearer not in loaded.notes

    assert store.delete("mem-secrets") is True
    assert store.get("mem-secrets") is None


def test_memory_scrub_leaves_plaintext_passwords_unchanged(tmp_path) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    path.write_text(
        json.dumps(
            {
                "memory_id": "mem-legacy",
                "tier": "warm",
                "intent": "general_task",
                "tool_name": "shell_exec",
                "outcome": "success",
                "notes": 'SSH to root@192.0.2.10 with password "Temp@Pass" succeeded.',
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = ExperienceStore(path)
    summary = store.scrub_sensitive_notes(write=True)
    assert summary == {"records": 1, "changed": 0, "written": 0}
    rewritten = path.read_text(encoding="utf-8")
    assert "Temp@Pass" in rewritten


def test_memory_cli_add_and_promote_persist_redacted_notes(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    api_key = "sk-proj-cli-promote-789"

    exit_code = memory_cli(
        ["add", "--tier", "warm", "--intent", "general_task", "--note", f"OPENAI_API_KEY={api_key}"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "added"
    memory_id = payload["memory_ids"][0]

    warm_path = tmp_path / ".smallctl" / "memory" / "warm-experiences.jsonl"
    cold_path = tmp_path / ".smallctl" / "memory" / "cold-experiences.jsonl"
    assert api_key not in warm_path.read_text(encoding="utf-8")

    exit_code = memory_cli(["promote", "--memory-id", memory_id])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "promoted"

    cold_text = cold_path.read_text(encoding="utf-8")
    assert api_key not in cold_text
    assert "[REDACTED]" in cold_text
    assert memory_id not in warm_path.read_text(encoding="utf-8")
