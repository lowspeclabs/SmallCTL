from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness import Harness
from smallctl.harness.tool_result_artifact_updates import _remember_session_ssh_target
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState
from smallctl.tools.memory import append_session_notepad_entry, log_note


def test_log_note_appends_and_dedupes() -> None:
    async def _run() -> None:
        state = LoopState(cwd="/tmp")
        first = await log_note(state=state, content="CWD is /tmp", tag="env")
        second = await log_note(state=state, content="CWD is /tmp", tag="env")

        assert first["success"] is True
        assert second["success"] is True
        assert second["metadata"]["duplicate"] is True
        payload = state.scratchpad["_session_notepad"]
        assert payload["entries"] == ["[env] CWD is /tmp"]

    import asyncio

    asyncio.run(_run())


def test_task_boundary_reset_preserves_session_notepad() -> None:
    state = LoopState(cwd="/tmp")
    append_session_notepad_entry(state, content="CWD is /tmp", tag="env")
    state.working_memory.known_facts = ["file_read: README exists"]
    state.recent_messages = []

    dummy_harness = SimpleNamespace(
        state=state,
        _initial_phase="explore",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
    )

    Harness._reset_task_boundary_state(
        dummy_harness,
        reason="run_task",
        new_task="next task",
        previous_task="old task",
    )

    payload = state.scratchpad.get("_session_notepad")
    assert isinstance(payload, dict)
    assert payload.get("entries") == ["[env] CWD is /tmp"]
    assert state.working_memory.known_facts == []


def test_task_boundary_reset_preserves_session_ssh_targets() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root"}
    }

    dummy_harness = SimpleNamespace(
        state=state,
        _initial_phase="explore",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
    )

    Harness._reset_task_boundary_state(
        dummy_harness,
        reason="run_task",
        new_task="next task",
        previous_task="old task",
    )

    payload = state.scratchpad.get("_session_ssh_targets")
    assert payload == {"192.168.1.63": {"host": "192.168.1.63", "user": "root"}}


def test_remembered_ssh_target_is_confirmed_when_transport_reaches_remote_host() -> None:
    state = LoopState(cwd="/tmp")
    service = SimpleNamespace(
        harness=SimpleNamespace(
            state=state,
        )
    )

    _remember_session_ssh_target(
        service,
        result=ToolEnvelope(
            success=False,
            error="Remote SSH command exited with code 1",
            metadata={
                "failure_kind": "remote_command",
                "ssh_transport_succeeded": True,
            },
        ),
        arguments={
            "host": "192.168.1.63",
            "user": "root",
        },
    )

    payload = state.scratchpad.get("_session_ssh_targets")
    assert payload == {
        "192.168.1.63": {
            "host": "192.168.1.63",
            "user": "root",
            "confirmed": True,
        }
    }


def test_remembered_ssh_target_tracks_validated_tool_and_path() -> None:
    state = LoopState(cwd="/tmp")
    service = SimpleNamespace(
        harness=SimpleNamespace(
            state=state,
        )
    )

    _remember_session_ssh_target(
        service,
        tool_name="ssh_file_read",
        result=ToolEnvelope(
            success=True,
            output={
                "path": "/etc/nginx/sites-enabled/default",
                "host": "192.168.1.63",
                "user": "root",
            },
            metadata={},
        ),
        arguments={
            "host": "192.168.1.63",
            "user": "root",
            "path": "/etc/nginx/sites-enabled/default",
        },
    )

    payload = state.scratchpad.get("_session_ssh_targets")
    assert payload["192.168.1.63"]["confirmed"] is True
    assert payload["192.168.1.63"]["validated_tools"] == ["ssh_file_read"]
    assert payload["192.168.1.63"]["last_success_tool"] == "ssh_file_read"
    assert payload["192.168.1.63"]["last_validated_path"] == "/etc/nginx/sites-enabled/default"
    assert payload["192.168.1.63"]["success_count"] == 1


def test_task_boundary_reset_preserves_session_artifacts() -> None:
    state = LoopState(cwd="/tmp")
    state.artifacts["A0004"] = ArtifactRecord(
        artifact_id="A0004",
        kind="shell_exec",
        source="nmap -sn 192.168.1.0/24",
        created_at="2026-04-21T12:17:42+00:00",
        size_bytes=128,
        summary="nmap host discovery transcript",
        tool_name="shell_exec",
        inline_content="nmap output",
    )

    dummy_harness = SimpleNamespace(
        state=state,
        _initial_phase="explore",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
    )

    Harness._reset_task_boundary_state(
        dummy_harness,
        reason="run_task",
        new_task="which of those are windows hosts?",
        previous_task="use nmap to list all hosts",
    )

    assert "A0004" in state.artifacts
    assert state.artifacts["A0004"].summary == "nmap host discovery transcript"
