from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.harness.tool_result_artifact_updates import _apply_file_mutation_updates
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.fs_patch_flow import handle_file_patch
from smallctl.write_session_fsm import new_write_session


class _FakeHarness:
    def __init__(self, state):
        self.state = state
        self.runlog_events = []

    def _runlog(self, event, *args, **kwargs):
        self.runlog_events.append(event)


class _FakeService:
    def __init__(self, state):
        self.harness = _FakeHarness(state)


def _coding_state(tmp_path):
    state = LoopState(cwd=str(tmp_path))
    state.challenge_progress.task_category = "coding"
    return state


class TestStagedMutationAccounting:
    def test_staged_only_file_write_is_not_recorded_as_code_change(self, tmp_path):
        state = _coding_state(tmp_path)
        service = _FakeService(state)
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws1__agents__stage.md"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("# agents\n", encoding="utf-8")
        result = ToolEnvelope(
            success=True,
            output="written",
            metadata={
                "path": str(tmp_path / "agents.md"),
                "staging_path": str(stage),
                "staged_only": True,
                "write_session_finalized": False,
                "write_session_id": "ws1",
            },
        )

        _apply_file_mutation_updates(
            service,
            tool_name="file_write",
            result=result,
            arguments={"path": str(tmp_path / "agents.md")},
            artifact=None,
        )

        assert state.challenge_progress.code_change_count == 0
        assert state.challenge_progress.last_code_change_paths == []

    def test_finalized_false_without_staged_flag_is_not_recorded(self, tmp_path):
        state = _coding_state(tmp_path)
        service = _FakeService(state)
        result = ToolEnvelope(
            success=True,
            output="written",
            metadata={
                "path": "./temp/example.py",
                "write_session_finalized": False,
            },
        )

        _apply_file_mutation_updates(
            service,
            tool_name="file_write",
            result=result,
            arguments={"path": "./temp/example.py"},
            artifact=None,
        )

        assert state.challenge_progress.code_change_count == 0
        assert state.challenge_progress.last_code_change_paths == []

    def test_real_file_write_is_still_recorded_as_code_change(self, tmp_path):
        target = tmp_path / "temp" / "example.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n", encoding="utf-8")
        state = _coding_state(tmp_path)
        service = _FakeService(state)
        result = ToolEnvelope(
            success=True,
            output="written",
            metadata={"path": "./temp/example.py", "changed": True},
        )

        _apply_file_mutation_updates(
            service,
            tool_name="file_write",
            result=result,
            arguments={"path": "./temp/example.py"},
            artifact=None,
        )

        assert state.challenge_progress.code_change_count == 1
        assert state.challenge_progress.last_code_change_paths == ["./temp/example.py"]

    def test_staged_write_invalidates_staging_path_read_cache(self, tmp_path):
        state = _coding_state(tmp_path)
        service = _FakeService(state)
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws1__example__stage.py"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("x = 1\n", encoding="utf-8")
        cache = state.scratchpad.setdefault("file_read_cache", {})
        cache[f"{stage.resolve()}|0:10"] = "artifact-old"
        result = ToolEnvelope(
            success=True,
            output="written",
            metadata={
                "path": str(tmp_path / "temp" / "example.py"),
                "staging_path": str(stage),
                "staged_only": True,
                "write_session_finalized": False,
            },
        )

        _apply_file_mutation_updates(
            service,
            tool_name="file_write",
            result=result,
            arguments={"path": str(tmp_path / "temp" / "example.py")},
            artifact=None,
        )

        assert not cache


class TestStagedPatchAccounting:
    def test_staged_patch_records_staging_path_not_target(self, tmp_path):
        target = tmp_path / "temp" / "example.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n", encoding="utf-8")
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws_patch__example__stage.py"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("x = 1\n", encoding="utf-8")

        state = LoopState(cwd=str(tmp_path))
        session = new_write_session(
            session_id="ws_patch",
            target_path=str(target),
            intent="patch_existing",
        )
        session.write_staging_path = str(stage)
        state.write_session = session
        state.active_write_sessions_by_path[str(target)] = session

        result = asyncio.run(
            handle_file_patch(
                path=str(target),
                target_text="x = 1",
                replacement_text="x = 2",
                cwd=str(tmp_path),
                state=state,
            )
        )

        assert result["success"] is True
        assert result["metadata"]["staged_only"] is True
        resolved_stage = str(stage.resolve())
        resolved_target = str(target.resolve())
        assert resolved_stage in state.files_changed_this_cycle
        assert resolved_target not in state.files_changed_this_cycle
        assert target.read_text(encoding="utf-8") == "x = 1\n"
        assert stage.read_text(encoding="utf-8") == "x = 2\n"
