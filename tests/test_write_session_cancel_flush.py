from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.lifecycle_nodes import prepare_loop_step
from smallctl.graph.state import GraphRunState
from smallctl.harness.runtime_facade import _run_teardown
from smallctl.state import LoopState
from smallctl.write_session_fsm import new_write_session


class _FakeHarness:
    def __init__(self, state):
        self.state = state
        self._cancel_requested = False
        self.runlog_events = []

    def _runlog(self, event, message, **kwargs):
        self.runlog_events.append((event, dict(kwargs)))

    async def _emit(self, handler, event):
        return None


def _make_session(tmp_path, *, session_id="ws_cancel", content="x = 1\n", ready=True):
    target = tmp_path / "app.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / f"{session_id}__app__stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text(content, encoding="utf-8")
    session = new_write_session(
        session_id=session_id,
        target_path=str(target),
        intent="replace_file",
    )
    session.write_staging_path = str(stage)
    session.write_sections_completed = ["body"]
    if not ready:
        session.write_next_section = "core"
    return target, stage, session


class TestCancelFlush:
    def test_cancel_promotes_complete_write_session(self, tmp_path):
        state = LoopState(cwd=str(tmp_path))
        target, _stage, session = _make_session(tmp_path)
        state.write_session = session
        state.active_write_sessions_by_path[str(target)] = session
        harness = _FakeHarness(state)
        harness._cancel_requested = True
        graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
        deps = GraphRuntimeDeps(harness=harness, event_handler=None)

        asyncio.run(prepare_loop_step(graph_state, deps))

        assert graph_state.final_result == {"status": "cancelled", "reason": "cancel_requested"}
        assert session.status == "complete"
        assert target.read_text(encoding="utf-8") == "x = 1\n"
        assert any(event == "write_session_finalized_on_cancel" for event, _ in harness.runlog_events)

    def test_cancel_archives_incomplete_write_session(self, tmp_path):
        state = LoopState(cwd=str(tmp_path))
        target, stage, session = _make_session(tmp_path, ready=False)
        state.write_session = session
        state.active_write_sessions_by_path[str(target)] = session
        harness = _FakeHarness(state)
        harness._cancel_requested = True
        graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
        deps = GraphRuntimeDeps(harness=harness, event_handler=None)

        asyncio.run(prepare_loop_step(graph_state, deps))

        assert graph_state.final_result == {"status": "cancelled", "reason": "cancel_requested"}
        assert session.status != "complete"
        assert not target.exists()
        assert state.active_write_sessions_by_path == {}
        archived = state.scratchpad.get("_archived_write_sessions") or []
        assert archived and archived[-1]["write_session_id"] == "ws_cancel"
        assert archived[-1]["write_staging_path"] == str(stage)
        abandoned = [kw for event, kw in harness.runlog_events if event == "write_session_abandoned_on_cancel"]
        assert abandoned
        assert abandoned[-1]["staging_path"] == str(stage)
        assert abandoned[-1]["target_path"] == str(target)
        assert any(
            getattr(record, "metadata", {}).get("write_session_id") == "ws_cancel"
            for record in state.artifacts.values()
        )

    def test_cancel_without_write_session_is_noop(self, tmp_path):
        state = LoopState(cwd=str(tmp_path))
        harness = _FakeHarness(state)
        harness._cancel_requested = True
        graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
        deps = GraphRuntimeDeps(harness=harness, event_handler=None)

        asyncio.run(prepare_loop_step(graph_state, deps))

        assert graph_state.final_result == {"status": "cancelled", "reason": "cancel_requested"}
        assert not state.scratchpad.get("_archived_write_sessions")


class _TeardownHarness:
    def __init__(self, state):
        self.state = state
        self.conversation_id = "conv-teardown"
        self._pending_task_shutdown_reason = ""
        self.log = logging.getLogger("smallctl.test.teardown")
        self.event_handler = None
        self.runlog_events = []
        self.approvals = SimpleNamespace(
            reject_pending_shell_approvals=lambda: None,
            reject_pending_sudo_password_prompts=lambda: None,
        )

    def _runlog(self, event, message, **kwargs):
        self.runlog_events.append((event, dict(kwargs)))


class TestTeardownFlush:
    def test_teardown_promotes_complete_write_session(self, tmp_path):
        state = LoopState(cwd=str(tmp_path))
        target, _stage, session = _make_session(tmp_path, session_id="ws_td")
        state.write_session = session
        state.active_write_sessions_by_path[str(target)] = session
        harness = _TeardownHarness(state)

        asyncio.run(_run_teardown(harness))

        assert session.status == "complete"
        assert target.read_text(encoding="utf-8") == "x = 1\n"
        assert any(event == "write_session_finalized_on_teardown" for event, _ in harness.runlog_events)

    def test_teardown_archives_incomplete_write_session(self, tmp_path):
        state = LoopState(cwd=str(tmp_path))
        target, stage, session = _make_session(tmp_path, session_id="ws_td2", ready=False)
        state.write_session = session
        state.active_write_sessions_by_path[str(target)] = session
        harness = _TeardownHarness(state)

        asyncio.run(_run_teardown(harness))

        assert session.status != "complete"
        assert not target.exists()
        assert state.active_write_sessions_by_path == {}
        archived = state.scratchpad.get("_archived_write_sessions") or []
        assert archived and archived[-1]["write_session_id"] == "ws_td2"
        assert archived[-1]["reason"] == "teardown_abandoned"
        abandoned = [kw for event, kw in harness.runlog_events if event == "write_session_abandoned_on_teardown"]
        assert abandoned
        assert abandoned[-1]["staging_path"] == str(stage)

    def test_teardown_without_write_session_still_completes(self, tmp_path):
        state = LoopState(cwd=str(tmp_path))
        harness = _TeardownHarness(state)

        asyncio.run(_run_teardown(harness))

        assert not state.scratchpad.get("_archived_write_sessions")
