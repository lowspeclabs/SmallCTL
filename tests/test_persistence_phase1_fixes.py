from __future__ import annotations

import json
import logging
import os
import threading

import pytest

import smallctl.chat_sessions as chat_sessions
import smallctl.memory_store as memory_store
from smallctl.chat_sessions import (
    _sanitize_filename,
    load_chat_session_state,
    persist_chat_session_state,
    session_state_path,
)
from smallctl.memory_store import ExperienceStore
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


def test_memory_write_failure_keeps_original_file_and_signals_caller(tmp_path, monkeypatch) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store = ExperienceStore(path)
    assert store.upsert(_make_memory("mem-original")) is not None
    original_bytes = path.read_bytes()

    def _raising_named_tempfile(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(memory_store.tempfile, "NamedTemporaryFile", _raising_named_tempfile)

    assert store.upsert(_make_memory("mem-new")) is None
    assert store.write_all([_make_memory("mem-new")]) is False
    assert store.delete("mem-original") is False
    assert store.scrub_sensitive_notes(write=True)["written"] == 0
    assert path.read_bytes() == original_bytes
    assert [m.memory_id for m in ExperienceStore(path).list()] == ["mem-original"]


def test_memory_replace_failure_leaves_original_file_intact(tmp_path, monkeypatch) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store = ExperienceStore(path)
    assert store.upsert(_make_memory("mem-original")) is not None
    original_bytes = path.read_bytes()

    def _raising_replace(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(memory_store.os, "replace", _raising_replace)

    assert store.upsert(_make_memory("mem-new")) is None
    assert path.read_bytes() == original_bytes
    leftover_temps = [p for p in tmp_path.iterdir() if p.name.startswith(f".{path.name}.")]
    assert leftover_temps == []


def test_memory_interleaved_writers_do_not_lose_records(tmp_path) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store_a = ExperienceStore(path)
    store_b = ExperienceStore(path)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _writer(store: ExperienceStore, prefix: str) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(25):
                assert store.upsert(_make_memory(f"{prefix}-{i}")) is not None
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_writer, args=(store_a, "mem-a")),
        threading.Thread(target=_writer, args=(store_b, "mem-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors
    stored_ids = {m.memory_id for m in ExperienceStore(path).list()}
    expected = {f"mem-a-{i}" for i in range(25)} | {f"mem-b-{i}" for i in range(25)}
    assert stored_ids == expected


def test_memory_unparseable_lines_tolerated_and_external_writes_visible(tmp_path, caplog) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    good = json.dumps({"memory_id": "mem-good", "tier": "warm", "intent": "general_task"})
    path.write_text(good + "\nnot-json\n\n", encoding="utf-8")
    store = ExperienceStore(path)

    with caplog.at_level(logging.WARNING, logger="smallctl.memory"):
        records = store.list()

    assert [m.memory_id for m in records] == ["mem-good"]
    assert any("unparseable" in record.getMessage() for record in caplog.records)

    other = ExperienceStore(path)
    assert other.upsert(_make_memory("mem-external")) is not None
    os.utime(path, None)
    assert {m.memory_id for m in store.list()} == {"mem-good", "mem-external"}


def test_chat_session_truncated_primary_recovers_from_backup(tmp_path, caplog) -> None:
    state = {"recent_messages": [{"role": "user", "content": "hello"}], "thread_id": "thread-bak"}
    persist_chat_session_state(cwd=tmp_path, thread_id="thread-bak", state_payload=state)
    path = session_state_path(tmp_path, "thread-bak")
    path.write_text('{"thread_id": "thread-bak", "state": {"recent_mess', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="smallctl.chat_sessions"):
        loaded = load_chat_session_state(cwd=tmp_path, thread_id="thread-bak")

    assert loaded == state
    assert any("chat_session_backup_recovery" in record.getMessage() for record in caplog.records)


def test_chat_session_colliding_thread_ids_use_distinct_files(tmp_path) -> None:
    path_slash = session_state_path(tmp_path, "a/b")
    path_underscore = session_state_path(tmp_path, "a_b")
    assert path_slash != path_underscore
    assert session_state_path(tmp_path, "..") != session_state_path(tmp_path, "/")
    assert session_state_path(tmp_path, "..").name != "session.json"

    persist_chat_session_state(
        cwd=tmp_path, thread_id="a/b", state_payload={"marker": "slash", "recent_messages": []}
    )
    persist_chat_session_state(
        cwd=tmp_path, thread_id="a_b", state_payload={"marker": "underscore", "recent_messages": []}
    )

    assert load_chat_session_state(cwd=tmp_path, thread_id="a/b")["marker"] == "slash"
    assert load_chat_session_state(cwd=tmp_path, thread_id="a_b")["marker"] == "underscore"


def test_chat_session_legacy_unhashed_filename_migrates(tmp_path) -> None:
    thread_id = "legacy-thread"
    legacy_dir = tmp_path / ".smallctl" / "chat_states"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / f"{_sanitize_filename(thread_id)}.json"
    state = {"recent_messages": [{"role": "user", "content": "legacy question"}]}
    legacy_path.write_text(
        json.dumps({"thread_id": thread_id, "saved_at": "2026-07-16T00:00:00+00:00", "state": state}),
        encoding="utf-8",
    )
    hashed_path = session_state_path(tmp_path, thread_id)
    assert legacy_path != hashed_path
    assert not hashed_path.exists()

    loaded = load_chat_session_state(cwd=tmp_path, thread_id=thread_id)

    assert loaded == state
    assert hashed_path.exists()
    assert not legacy_path.exists()
    assert load_chat_session_state(cwd=tmp_path, thread_id=thread_id) == state


def test_chat_session_corrupt_primary_preserved_on_subsequent_save(tmp_path) -> None:
    persist_chat_session_state(
        cwd=tmp_path, thread_id="thread-corrupt", state_payload={"marker": "v1", "recent_messages": []}
    )
    path = session_state_path(tmp_path, "thread-corrupt")
    corrupt_bytes = b'{"thread_id": "thread-corrupt", "state": {"mark'
    path.write_bytes(corrupt_bytes)

    persist_chat_session_state(
        cwd=tmp_path, thread_id="thread-corrupt", state_payload={"marker": "v2", "recent_messages": []}
    )

    corrupt_path = path.with_name(f"{path.name}.corrupt")
    assert corrupt_path.exists()
    assert corrupt_path.read_bytes() == corrupt_bytes
    assert load_chat_session_state(cwd=tmp_path, thread_id="thread-corrupt")["marker"] == "v2"


def test_chat_session_save_refused_when_corrupt_primary_cannot_be_preserved(tmp_path, monkeypatch) -> None:
    persist_chat_session_state(
        cwd=tmp_path, thread_id="thread-refuse", state_payload={"marker": "v1", "recent_messages": []}
    )
    path = session_state_path(tmp_path, "thread-refuse")
    corrupt_bytes = b'{"thread_id": "thread-refuse", "state": {"mark'
    path.write_bytes(corrupt_bytes)

    original_replace = os.replace

    def _guarded_replace(src, dst):
        if str(dst).endswith(".corrupt"):
            raise OSError("simulated preserve failure")
        return original_replace(src, dst)

    monkeypatch.setattr(chat_sessions.os, "replace", _guarded_replace)

    with pytest.raises(OSError):
        persist_chat_session_state(
            cwd=tmp_path, thread_id="thread-refuse", state_payload={"marker": "v2", "recent_messages": []}
        )

    assert path.read_bytes() == corrupt_bytes
