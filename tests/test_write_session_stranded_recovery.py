from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.graph.write_session_outcomes import (
    _find_stranded_write_session_record,
    _handle_write_session_outcome,
    maybe_finalize_stranded_write_session,
    maybe_replay_stranded_write_session_record,
)
from smallctl.graph.write_session_recovery import (
    _invalidate_write_session_stage_artifacts,
    _register_write_session_stage_artifact,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.artifact import artifact_read
from smallctl.tools.control import (
    _write_session_resume_action,
    finalize_write_session,
    task_complete,
)
from smallctl.tools.fs import file_write
from smallctl.write_session_fsm import new_write_session, record_write_session_event


class _FakeHarness:
    def __init__(self, session=None, cwd=None):
        self.state = SimpleNamespace(
            write_session=session,
            cwd=cwd or "/tmp/workspace",
            scratchpad={},
            recent_messages=[],
            touch=lambda: None,
            current_phase="explore",
            append_message=lambda msg: self.state.recent_messages.append(msg),
            tool_execution_records={},
            step_count=0,
        )
        self.config = SimpleNamespace(
            enable_write_intent_recovery=True,
            write_recovery_allow_raw_text_targets=True,
            enable_assistant_code_write_recovery=True,
            write_recovery_min_confidence="high",
            enforce_write_recovery_readback=True,
            failed_local_patch_limit=3,
        )
        self.log = SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )
        self._cancel_requested = False

    def _runlog(self, *args, **kwargs):
        pass

    def _emit(self, *args, **kwargs):
        pass


def _store_tool_record(harness, tool_name, args, success=True, metadata=None):
    op_id = f"op_{len(harness.state.tool_execution_records)}"
    harness.state.tool_execution_records[op_id] = {
        "operation_id": op_id,
        "tool_name": tool_name,
        "args": dict(args),
        "tool_call_id": f"tc_{op_id}",
        "result": ToolEnvelope(
            success=success,
            output="done",
            metadata=dict(metadata or {}),
        ).to_dict(),
        "step_count": harness.state.step_count,
    }
    return op_id


class TestWriteSessionSyntaxFailureRepairSource:
    def test_failed_replace_file_stage_is_preserved_for_local_repair(self, tmp_path):
        target = tmp_path / "temp" / "broken.py"
        state = LoopState(cwd=str(tmp_path))
        session = new_write_session(
            session_id="ws_broken",
            target_path=str(target),
            intent="replace_file",
        )
        state.write_session = session
        harness = SimpleNamespace(
            state=state,
            config=SimpleNamespace(failed_local_patch_limit=3),
            log=SimpleNamespace(error=lambda *args, **kwargs: None),
            _runlog=lambda *args, **kwargs: None,
        )
        broken_content = "def example():\n    return (\n"

        result = asyncio.run(
            file_write(
                path=str(target),
                content=broken_content,
                cwd=str(tmp_path),
                state=state,
                write_session_id="ws_broken",
                section_name="body",
            )
        )
        assert result["success"] is True

        record = ToolExecutionRecord(
            operation_id="op-broken",
            tool_name="file_write",
            args={"path": str(target), "content": broken_content},
            tool_call_id="tc-broken",
            result=ToolEnvelope(
                success=True,
                output=result["output"],
                metadata=result["metadata"],
            ),
        )

        asyncio.run(_handle_write_session_outcome(harness, record))

        stage = Path(session.write_staging_path)
        assert stage.read_text(encoding="utf-8") == broken_content
        assert session.status == "local_repair"
        assert session.write_session_mode == "local_repair"
        assert session.write_next_section == "body"
        assert any(
            message.metadata.get("recovery_kind") == "syntax_error"
            for message in state.recent_messages
        )


class TestStrandedWriteSessionRecordReplay:
    def test_replay_finds_stranded_final_chunk(self, tmp_path):
        session = new_write_session(
            session_id="ws_replay",
            target_path="./test.py",
            intent="replace_file",
        )
        session.write_sections_completed = ["imports"]
        session.write_current_section = "imports"
        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        _store_tool_record(
            harness,
            "file_write",
            {"path": "./test.py"},
            metadata={
                "write_session_id": "ws_replay",
                "write_session_final_chunk": True,
                "write_session_finalized": False,
            },
        )

        record = _find_stranded_write_session_record(harness, session)
        assert record is not None
        assert record.tool_name == "file_write"
        assert record.result.metadata["write_session_final_chunk"] is True

    def test_replay_skips_already_finalized(self, tmp_path):
        session = new_write_session(
            session_id="ws_done",
            target_path="./test.py",
            intent="replace_file",
        )
        session.write_sections_completed = ["imports"]
        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        _store_tool_record(
            harness,
            "file_write",
            {"path": "./test.py"},
            metadata={
                "write_session_id": "ws_done",
                "write_session_final_chunk": True,
            },
        )
        record_write_session_event(harness.state, event="finalize_succeeded", session=session)

        record = _find_stranded_write_session_record(harness, session)
        assert record is None

    def test_maybe_replay_populates_last_tool_results(self, tmp_path):
        session = new_write_session(
            session_id="ws_replay2",
            target_path="./test.py",
            intent="replace_file",
        )
        session.write_sections_completed = ["imports"]
        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        _store_tool_record(
            harness,
            "file_write",
            {"path": "./test.py"},
            metadata={
                "write_session_id": "ws_replay2",
                "write_session_final_chunk": True,
            },
        )

        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id="t1",
            run_mode="loop",
        )
        assert len(graph_state.last_tool_results) == 0
        result = maybe_replay_stranded_write_session_record(harness, graph_state)
        assert result is True
        assert len(graph_state.last_tool_results) == 1
        assert graph_state.last_tool_results[0].replayed is True

    def test_maybe_replay_triggers_circuit_breaker_after_repeated_same_operation(self, tmp_path):
        session = new_write_session(
            session_id="ws_replay_loop",
            target_path="./test.py",
            intent="replace_file",
        )
        session.write_sections_completed = ["imports"]
        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        _store_tool_record(
            harness,
            "file_write",
            {"path": "./test.py"},
            metadata={
                "write_session_id": "ws_replay_loop",
                "write_session_final_chunk": True,
            },
        )

        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id="t1",
            run_mode="loop",
        )

        first = maybe_replay_stranded_write_session_record(harness, graph_state)
        second = maybe_replay_stranded_write_session_record(harness, graph_state)
        third = maybe_replay_stranded_write_session_record(harness, graph_state)

        assert first is True
        assert second is True
        assert third is False
        assert len(graph_state.last_tool_results) == 2
        assert any(
            message.metadata.get("recovery_kind") == "write_session_stranded_replay_circuit_breaker"
            for message in harness.state.recent_messages
        )


class TestStrandedWriteSessionDirectFinalize:
    def test_finalize_stranded_python_file(self, tmp_path):
        target = tmp_path / "test.py"
        target.write_text("import os\n", encoding="utf-8")
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws_finalize__test__stage.py"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("import os\nimport sys\n", encoding="utf-8")

        session = new_write_session(
            session_id="ws_finalize",
            target_path=str(target),
            intent="replace_file",
        )
        session.write_sections_completed = ["imports"]
        session.write_current_section = "imports"
        session.write_staging_path = str(stage)

        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id="t1",
            run_mode="loop",
        )

        async def _run():
            return await maybe_finalize_stranded_write_session(harness, graph_state)

        result = asyncio.run(_run())
        assert result is True
        assert session.status == "complete"
        assert target.read_text(encoding="utf-8") == "import os\nimport sys\n"

    def test_finalize_skips_when_last_tool_results_has_replay(self, tmp_path):
        session = new_write_session(
            session_id="ws_skip",
            target_path="./test.py",
            intent="replace_file",
        )
        session.write_sections_completed = ["imports"]
        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id="t1",
            run_mode="loop",
        )
        graph_state.last_tool_results.append(
            ToolExecutionRecord(
                operation_id="op1",
                tool_name="file_write",
                args={},
                tool_call_id="tc1",
                result=ToolEnvelope(
                    success=True,
                    metadata={
                        "write_session_id": "ws_skip",
                        "write_session_final_chunk": True,
                    },
                ),
            )
        )
        result = asyncio.run(maybe_finalize_stranded_write_session(harness, graph_state))
        assert result is False


class TestWriteSessionStageArtifactAliases:
    def test_register_stage_artifact_exposes_expected_aliases(self, tmp_path):
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws_alias__circuit_breaker__stage.py"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("print('ok')\n", encoding="utf-8")

        session = new_write_session(
            session_id="ws_alias",
            target_path="./temp/circuit_breaker.py",
            intent="replace_file",
        )
        session.write_staging_path = str(stage)

        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        harness.state.artifacts = {}

        artifact_id = _register_write_session_stage_artifact(harness, session)

        assert artifact_id == "ws_alias__stage"
        assert "ws_alias__stage" in harness.state.artifacts
        assert "ws_alias__circuit_breaker__stage.py" in harness.state.artifacts

    def test_promoted_stage_artifact_is_marked_stale_with_authoritative_hint(self, tmp_path):
        target = tmp_path / "temp" / "circuit_breaker.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("print('promoted')\n", encoding="utf-8")
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws_stale__circuit_breaker__stage.py"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("print('stale')\n", encoding="utf-8")

        session = new_write_session(
            session_id="ws_stale",
            target_path=str(target),
            intent="replace_file",
        )
        session.write_staging_path = str(stage)

        harness = _FakeHarness(session=session, cwd=str(tmp_path))
        harness.state.artifacts = {}
        artifact_id = _register_write_session_stage_artifact(harness, session)

        _invalidate_write_session_stage_artifacts(harness, session, target_path=str(target))

        assert artifact_id == "ws_stale__stage"
        record = harness.state.artifacts["ws_stale__stage"]
        assert record.metadata["stale"] is True
        assert record.metadata["model_visible"] is False
        assert record.metadata["artifact_stale_reason"] == "write_session_promoted"
        assert record.metadata["authoritative_path"] == str(target)

        read = artifact_read(harness.state, artifact_id="ws_stale__stage")
        assert read["success"] is True
        assert "WARNING: This artifact is stale" in read["output"]
        assert f"file_read(path='{target}')" in read["output"]


class TestResumeActionAndTaskComplete:
    def test_resume_action_suggests_finalize_when_ready(self):
        state = LoopState(cwd="/tmp")
        session = new_write_session(
            session_id="ws_resume",
            target_path="./app.py",
            intent="replace_file",
        )
        session.write_sections_completed = ["imports", "core"]
        state.write_session = session

        action = _write_session_resume_action(state, None)
        assert action is not None
        assert action["tool_name"] == "finalize_write_session"

    def test_resume_action_suggests_file_write_when_not_ready(self):
        state = LoopState(cwd="/tmp")
        session = new_write_session(
            session_id="ws_resume2",
            target_path="./app.py",
            intent="replace_file",
        )
        session.write_sections_completed = ["imports"]
        session.write_next_section = "core"
        state.write_session = session

        action = _write_session_resume_action(state, None)
        assert action is not None
        assert action["tool_name"] == "file_write"

    def test_task_complete_auto_finalizes_stranded_session(self, tmp_path):
        target = tmp_path / "app.py"
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws_auto__app__stage.py"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("x = 1\n", encoding="utf-8")

        state = LoopState(cwd=str(tmp_path))
        session = new_write_session(
            session_id="ws_auto",
            target_path=str(target),
            intent="replace_file",
        )
        session.write_sections_completed = ["body"]
        session.write_staging_path = str(stage)
        state.write_session = session

        harness = _FakeHarness(session=session, cwd=str(tmp_path))

        result = asyncio.run(task_complete("done", state, harness))
        assert result["success"] is True
        assert result["output"]["status"] == "complete"
        assert session.status == "complete"
        assert target.read_text(encoding="utf-8") == "x = 1\n"

    def test_finalize_write_session_tool(self, tmp_path):
        target = tmp_path / "tool.py"
        stage = tmp_path / ".smallctl" / "write_sessions" / "ws_tool__tool__stage.py"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text("y = 2\n", encoding="utf-8")

        state = LoopState(cwd=str(tmp_path))
        session = new_write_session(
            session_id="ws_tool",
            target_path=str(target),
            intent="replace_file",
        )
        session.write_sections_completed = ["body"]
        session.write_staging_path = str(stage)
        state.write_session = session

        harness = _FakeHarness(session=session, cwd=str(tmp_path))

        result = asyncio.run(finalize_write_session(state, harness))
        assert result["success"] is True
        assert result["output"]["status"] == "finalized"
        assert target.read_text(encoding="utf-8") == "y = 2\n"

    def test_finalize_write_session_reports_completed_target(self, tmp_path):
        target = tmp_path / "tool.py"
        target.write_text("y = 2\n", encoding="utf-8")

        state = LoopState(cwd=str(tmp_path))
        session = new_write_session(
            session_id="ws_done",
            target_path=str(target),
            intent="replace_file",
        )
        session.status = "complete"
        state.write_session = session

        harness = _FakeHarness(session=session, cwd=str(tmp_path))

        result = asyncio.run(finalize_write_session(state, harness))
        assert result["success"] is True
        assert result["output"]["status"] == "already_finalized"
        assert str(target) in result["output"]["message"]

    def test_finalize_write_session_no_session(self):
        state = LoopState(cwd="/tmp")
        harness = _FakeHarness(session=None, cwd="/tmp")
        result = asyncio.run(finalize_write_session(state, harness))
        assert result["success"] is False
        assert "No active write session" in result["error"]
