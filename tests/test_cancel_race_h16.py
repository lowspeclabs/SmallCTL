from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest

from smallctl.harness import Harness
from smallctl.harness.runtime_facade import (
    HarnessRunAlreadyActiveError,
    cancel,
    resume_task_with_events,
    run_auto_with_events,
    run_task_with_events,
)
from smallctl.ui.app_actions import SmallctlAppActionsMixin
from smallctl.ui.harness_bridge import HarnessBridge, HarnessRunBusyError


class _BlockingHarness:
    def __init__(self, *, cleanup_sec: float = 0.0) -> None:
        self.release = threading.Event()
        self.cleanup_sec = cleanup_sec
        self.cleanup_done = threading.Event()
        self.started: list[str] = []
        self.finished: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.cancel_sources: list[str] = []

    async def run_auto_with_events(self, task: str, event_handler) -> dict[str, object]:
        self.started.append(task)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.005)
            return {"status": "ok", "task": task}
        except asyncio.CancelledError:
            if self.cleanup_sec > 0:
                steps = max(1, int(self.cleanup_sec / 0.005))
                for _ in range(steps):
                    await asyncio.sleep(0.005)
                self.cleanup_done.set()
            raise
        finally:
            self.concurrent -= 1
            self.finished.append(task)

    async def resume_task_with_events(self, human_input: str, event_handler) -> dict[str, object]:
        return {"status": "resumed", "choice": human_input}

    def cancel(self, source: str = "manual") -> None:
        self.cancel_sources.append(source)

    async def teardown(self) -> None:
        await asyncio.sleep(0)


def test_harness_cancel_does_not_start_teardown_during_active_run() -> None:
    dispatch_task = SimpleNamespace(done=lambda: False, cancel=lambda: None)
    harness = SimpleNamespace(
        _cancel_requested=False,
        _cancel_source="",
        _active_dispatch_task=dispatch_task,
        approvals=SimpleNamespace(
            reject_pending_shell_approvals=lambda: None,
            reject_pending_sudo_password_prompts=lambda: None,
        ),
        note_task_shutdown=lambda reason: None,
        log=logging.getLogger(__name__),
    )

    cancel(harness, source="ui_stop_button")

    assert harness._cancel_requested is True
    assert harness._cancel_source == "ui_stop_button"


def _make_bridge(harness: _BlockingHarness) -> HarnessBridge:
    return HarnessBridge(harness=harness, post_ui_event=lambda event: None)


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    loops = max(1, int(timeout / 0.005))
    for _ in range(loops):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def test_bridge_refuses_second_run_while_first_in_flight() -> None:
    harness = _BlockingHarness()
    bridge = _make_bridge(harness)

    async def _run() -> None:
        first = asyncio.create_task(bridge.run_auto("first"))
        await _wait_until(lambda: bool(harness.started))
        assert bridge.is_run_active()

        with pytest.raises(HarnessRunBusyError):
            await bridge.run_auto("second")
        with pytest.raises(HarnessRunBusyError):
            await bridge.resume("yes")

        harness.release.set()
        assert (await first)["status"] == "ok"
        assert not bridge.is_run_active()

        third = await bridge.run_auto("third")
        assert third["status"] == "ok"
        await bridge.shutdown()

    asyncio.run(_run())

    assert harness.started == ["first", "third"]
    assert harness.max_concurrent == 1


def test_bridge_cancel_abort_then_immediate_resubmit_has_no_overlap() -> None:
    harness = _BlockingHarness(cleanup_sec=0.25)
    bridge = _make_bridge(harness)

    async def _run() -> None:
        first = asyncio.create_task(bridge.run_auto("first"))
        await _wait_until(lambda: bool(harness.started))

        bridge.cancel(source="ui_stop_button")
        bridge.abort()
        with pytest.raises(HarnessRunBusyError):
            await bridge.run_auto("second")

        assert await bridge.wait_for_idle(timeout=5.0) is True
        assert harness.cleanup_done.is_set()
        with pytest.raises(asyncio.CancelledError):
            await first

        harness.release.set()
        result = await bridge.run_auto("second")
        assert result == {"status": "ok", "task": "second"}
        await bridge.shutdown()

    asyncio.run(_run())

    assert harness.cancel_sources == ["ui_stop_button"]
    assert harness.started == ["first", "second"]
    assert harness.finished == ["first", "second"]
    assert harness.max_concurrent == 1


def test_action_cancel_task_invokes_abort_and_waits_for_idle() -> None:
    abort_calls: list[int] = []
    cancel_sources: list[str] = []
    wait_timeouts: list[float] = []

    class _BridgeStub:
        def cancel(self, source: str = "ui_stop_button") -> None:
            cancel_sources.append(source)

        def abort(self) -> None:
            abort_calls.append(1)

        async def wait_for_idle(self, timeout: float = 5.0) -> bool:
            wait_timeouts.append(timeout)
            return True

    class _Actions(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = None
            self._harness_bridge = _BridgeStub()
            self.active_task: asyncio.Task[None] | None = None
            self._app_logger = logging.getLogger("test.cancel_race_h16.actions")
            self._lines: list[str] = []

        def _dismiss_active_approval_prompt(self) -> None:
            return None

        def _get_console(self) -> None:
            return None

        async def _append_system_line(self, text: str, *, force: bool = False, kind: str | None = None) -> None:
            self._lines.append(text)

        def _refresh_status(self, step_override: int | str | None = None, *, snapshot: dict[str, object] | None = None) -> None:
            return None

    async def _run() -> None:
        actions = _Actions()

        async def _block() -> None:
            await asyncio.Event().wait()

        actions.active_task = asyncio.create_task(_block())
        task = actions.active_task
        await actions.action_cancel_task()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())

    assert cancel_sources == ["ui_stop_button"]
    assert abort_calls == [1]
    assert wait_timeouts == [5.0]


def _facade_harness_stub() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(task_received_at="", touch=lambda: None),
        get_pending_interrupt=lambda: None,
        event_handler=None,
        _cancel_requested=True,
    )


def test_facade_run_guard_refuses_overlap_and_resets_stale_cancel_flag(monkeypatch) -> None:
    release = threading.Event()
    started: list[str] = []
    flags_at_start: list[bool] = []

    class _StubRuntime:
        async def run(self, task: str) -> dict[str, object]:
            started.append(task)
            flags_at_start.append(bool(getattr(harness, "_cancel_requested", False)))
            try:
                while not release.is_set():
                    await asyncio.sleep(0.005)
                return {"status": "completed", "task": task}
            finally:
                pass

    monkeypatch.setattr(
        "smallctl.graph.runtime.AutoGraphRuntime.from_harness",
        lambda harness, event_handler=None: _StubRuntime(),
    )
    harness = _facade_harness_stub()

    async def _run() -> None:
        first = asyncio.create_task(run_auto_with_events(harness, "first"))
        await _wait_until(lambda: bool(started))

        with pytest.raises(HarnessRunAlreadyActiveError):
            await run_auto_with_events(harness, "second")

        release.set()
        assert (await first)["status"] == "completed"

        harness._cancel_requested = True
        third = await run_auto_with_events(harness, "third")
        assert third["status"] == "completed"

    asyncio.run(_run())

    assert started == ["first", "third"]
    assert flags_at_start == [False, False]
    assert harness._run_guard_in_flight is False


def test_facade_run_task_guard_releases_after_failure(monkeypatch) -> None:
    class _FailingRuntime:
        async def run(self, task: str) -> dict[str, object]:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        lambda harness, event_handler=None: _FailingRuntime(),
    )
    harness = _facade_harness_stub()
    harness.state.active_plan = None
    harness.state.draft_plan = None
    harness.state.planning_mode_enabled = False
    harness.config = SimpleNamespace(staged_execution_enabled=False)

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            await run_task_with_events(harness, "first")
        with pytest.raises(RuntimeError, match="boom"):
            await run_task_with_events(harness, "second")

    asyncio.run(_run())

    assert harness._run_guard_in_flight is False


def test_resume_task_with_events_resets_stale_cancel_flag(monkeypatch) -> None:
    flags_at_resume: list[bool] = []

    class _StubRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            flags_at_resume.append(bool(getattr(harness, "_cancel_requested", False)))
            return {"status": "resumed", "choice": human_input}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        lambda harness, event_handler=None: _StubRuntime(),
    )
    harness = SimpleNamespace(
        state=SimpleNamespace(),
        get_pending_interrupt=lambda: None,
        event_handler=None,
        _cancel_requested=True,
    )

    result = asyncio.run(resume_task_with_events(harness, "yes"))

    assert result == {"status": "resumed", "choice": "yes"}
    assert flags_at_resume == [False]


def test_core_facade_reset_cancel_requested_clears_stale_flag() -> None:
    harness = SimpleNamespace(_cancel_requested=True, _cancel_source="ui_stop_button")

    Harness._reset_cancel_requested(harness)

    assert harness._cancel_requested is False
    assert harness._cancel_source == ""
