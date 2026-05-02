from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from textual.css.query import NoMatches

from smallctl.models.events import UIEvent, UIEventType
from smallctl.ui.app_actions import SmallctlAppActionsMixin
from smallctl.ui.app_approvals import handle_approval_prompt, handle_sudo_password_prompt
from smallctl.ui.approval import ShellApprovalDecision
from smallctl.ui.app_flow import SmallctlAppFlowMixin
from smallctl.ui.model_selector import ModelSelectButton
from smallctl.ui.statusbar import StatusBar


def test_get_console_handles_unmounted_view_without_name_error() -> None:
    seen: list[type[object]] = []

    class _Flow(SmallctlAppFlowMixin):
        def query_one(self, widget_type: type[object]) -> object:
            seen.append(widget_type)
            raise NoMatches("no console mounted")

    flow = _Flow()

    assert flow._get_console() is None
    assert seen
    assert seen[0].__name__ == "ConsolePane"


def test_on_harness_event_uses_lazy_harness_event_import_for_cross_thread_posting() -> None:
    posted: list[object] = []

    class _Flow(SmallctlAppFlowMixin):
        is_running = True

        def __init__(self) -> None:
            self._loop_thread_id = -1

        def post_message(self, message: object) -> None:
            posted.append(message)

        async def _handle_harness_event(self, event: UIEvent) -> None:
            raise AssertionError(f"should not handle directly: {event}")

    flow = _Flow()

    asyncio.run(
        flow.on_harness_event(
            UIEvent(event_type=UIEventType.SYSTEM, content="hello")
        )
    )

    assert len(posted) == 1
    assert posted[0].__class__.__name__ == "HarnessEvent"


def test_on_harness_event_queues_same_thread_work_before_rendering() -> None:
    seen: list[str] = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._pending_harness_events = []
            self._pending_status_event = None
            self._ui_event_drain_task = None

        async def _handle_harness_event(self, event: UIEvent) -> None:
            seen.append(event.content)

    flow = _Flow()

    async def _run() -> None:
        await flow.on_harness_event(UIEvent(event_type=UIEventType.SYSTEM, content="queued"))
        assert seen == []
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert seen == ["queued"]


def test_handle_slash_command_routes_planning_mode_through_bridge() -> None:
    refresh_calls: list[dict[str, object]] = []
    events: list[UIEvent] = []
    lines: list[str] = []

    class _Bridge:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        async def set_planning_mode(self, enabled: bool) -> dict[str, object]:
            self.calls.append(enabled)
            return {"model": "bridge-model", "phase": "explore", "mode": "planning", "step": 0}

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = object()
            self._harness_bridge = _Bridge()
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = None

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            refresh_calls.append(dict(snapshot or {}))

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            lines.append(text)

        async def on_harness_event(self, event: UIEvent) -> None:
            events.append(event)

    flow = _Flow()

    result = asyncio.run(flow._handle_slash_command("/plan-mode"))

    assert result is True
    assert flow._harness_bridge.calls == [True]
    assert refresh_calls == [{"model": "bridge-model", "phase": "explore", "mode": "planning", "step": 0}]
    assert lines == ["Planning mode enabled."]
    assert len(events) == 1
    assert events[0].event_type == UIEventType.ALERT
    assert events[0].content == "Planning mode enabled."


def test_maybe_restore_harness_state_prefers_bridge_payload() -> None:
    class _Bridge:
        async def restore_graph_state(self, thread_id: str | None = None) -> dict[str, object]:
            assert thread_id == "thread-restore"
            return {
                "restored": True,
                "thread_id": "thread-restore",
                "has_pending_interrupt": True,
                "interrupt": {"kind": "plan_execute_approval"},
                "snapshot": {"model": "bridge-model", "phase": "explore", "step": 0},
            }

    class _ExplodingHarness:
        def restore_graph_state(self, thread_id: str | None = None) -> bool:
            raise AssertionError("restore should go through the bridge")

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.fresh_run = False
            self.restore_graph_state_on_startup = True
            self.restore_thread_id = "thread-restore"
            self.harness = _ExplodingHarness()
            self._harness_bridge = _Bridge()
            self._latest_status_snapshot = None

    flow = _Flow()

    result = asyncio.run(flow._maybe_restore_harness_state())

    assert result == {
        "status": "restored",
        "thread_id": "thread-restore",
        "has_pending_interrupt": True,
        "interrupt": {"kind": "plan_execute_approval"},
    }
    assert flow._latest_status_snapshot == {"model": "bridge-model", "phase": "explore", "step": 0}


def test_refresh_status_updates_model_button_and_status_details() -> None:
    class _Task:
        def done(self) -> bool:
            return False

    class _ModelButton:
        def __init__(self) -> None:
            self.model = ""
            self.busy = False

        def set_model(self, model: str) -> None:
            self.model = model

        def set_busy(self, busy: bool) -> None:
            self.busy = busy

    class _Status:
        def __init__(self) -> None:
            self.kwargs = {}

        def set_state(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = None
            self.harness_kwargs = {"model": "wrench-9b", "phase": "execute"}
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = _Task()
            self.model_button = _ModelButton()
            self.status = _Status()

        def query_one(self, widget_type: type[object]) -> object:
            if widget_type is ModelSelectButton:
                return self.model_button
            if widget_type is StatusBar:
                return self.status
            raise NoMatches("unexpected query")

    flow = _Flow()

    flow._refresh_status()

    assert flow.model_button.model == "wrench-9b"
    assert flow.model_button.busy is True
    assert flow.status.kwargs["model"] == "wrench-9b"
    assert flow.status.kwargs["phase"] == "execute"
    assert flow.status.kwargs["activity"] == "thinking..."


def test_refresh_status_prefers_cached_snapshot_over_live_harness_reads() -> None:
    class _ExplodingHarness:
        @property
        def state(self) -> object:
            raise AssertionError("refresh path should not read harness state")

    class _ModelButton:
        def __init__(self) -> None:
            self.model = ""
            self.busy = False

        def set_model(self, model: str) -> None:
            self.model = model

        def set_busy(self, busy: bool) -> None:
            self.busy = busy

    class _Status:
        def __init__(self) -> None:
            self.kwargs = {}

        def set_state(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = _ExplodingHarness()
            self.harness_kwargs = {"model": "wrench-9b", "phase": "execute"}
            self._status_activity = ""
            self._api_error_count = 0
            self._latest_status_snapshot = {
                "model": "snapshot-7b",
                "phase": "verify",
                "step": 4,
                "mode": "planning",
                "activity": "cached activity",
            }
            self.active_task = None
            self.model_button = _ModelButton()
            self.status = _Status()

        def query_one(self, widget_type: type[object]) -> object:
            if widget_type is ModelSelectButton:
                return self.model_button
            if widget_type is StatusBar:
                return self.status
            raise NoMatches("unexpected query")

    flow = _Flow()

    flow._refresh_status()

    assert flow.model_button.model == "snapshot-7b"
    assert flow.status.kwargs["phase"] == "verify"
    assert flow.status.kwargs["step"] == 4
    assert flow.status.kwargs["activity"] == "cached activity"


def test_refresh_status_without_cached_snapshot_uses_defaults_not_live_harness() -> None:
    class _ExplodingHarness:
        @property
        def state(self) -> object:
            raise AssertionError("refresh path should not read harness state")

    class _Task:
        def done(self) -> bool:
            return False

    class _ModelButton:
        def __init__(self) -> None:
            self.model = ""
            self.busy = False

        def set_model(self, model: str) -> None:
            self.model = model

        def set_busy(self, busy: bool) -> None:
            self.busy = busy

    class _Status:
        def __init__(self) -> None:
            self.kwargs = {}

        def set_state(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = _ExplodingHarness()
            self.harness_kwargs = {"model": "fallback-3b", "phase": "execute", "contract_flow_ui": True}
            self._status_activity = ""
            self._api_error_count = 2
            self._latest_status_snapshot = None
            self.active_task = _Task()
            self.model_button = _ModelButton()
            self.status = _Status()

        def query_one(self, widget_type: type[object]) -> object:
            if widget_type is ModelSelectButton:
                return self.model_button
            if widget_type is StatusBar:
                return self.status
            raise NoMatches("unexpected query")

    flow = _Flow()

    flow._refresh_status()

    assert flow.model_button.model == "fallback-3b"
    assert flow.model_button.busy is True
    assert flow.status.kwargs["phase"] == "execute"
    assert flow.status.kwargs["activity"] == "thinking..."
    assert flow.status.kwargs["contract_flow_ui"] is True
    assert flow.status.kwargs["api_errors"] == 2


def test_status_events_are_coalesced_latest_wins() -> None:
    calls: list[dict[str, object]] = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._latest_status_snapshot = None
            self._status_refresh_pending = False
            self._pending_user_echo = None
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = None

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            calls.append(dict(snapshot or {}))

    flow = _Flow()

    async def _run() -> None:
        await flow._handle_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "first", "phase": "plan", "step": 1}},
            )
        )
        await flow._handle_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "second", "phase": "execute", "step": 2}},
            )
        )
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert calls == [{"model": "second", "phase": "execute", "step": 2}]


def test_set_shell_approval_session_default_prefers_bridge() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.values: list[bool] = []

        def set_shell_approval_session_default(self, enabled: bool) -> None:
            self.values.append(enabled)

    class _Harness:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def set_shell_approval_session_default(self, enabled: bool) -> None:
            self.calls.append(enabled)

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._shell_approval_session_default = False
            self.harness = _Harness()
            self._harness_bridge = _Bridge()

    flow = _Flow()

    flow._set_shell_approval_session_default(True)

    assert flow._shell_approval_session_default is True
    assert flow._harness_bridge.values == [True]
    assert flow.harness.calls == []


def test_render_restored_chat_uses_serialized_messages_without_harness_state_reads() -> None:
    rendered: list[tuple[UIEventType, str]] = []

    class _Console:
        async def clear_bubbles(self) -> None:
            return None

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content))

    class _ExplodingHarness:
        @property
        def state(self) -> object:
            raise AssertionError("serialized restore messages should avoid live harness reads")

    class _Flow(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = _ExplodingHarness()
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()

    asyncio.run(
        flow._render_restored_chat(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "system", "content": "note"},
            ]
        )
    )

    assert rendered == [
        (UIEventType.USER, "hello"),
        (UIEventType.ASSISTANT, "hi"),
        (UIEventType.SYSTEM, "note"),
    ]


def test_on_harness_event_coalesces_queued_status_events_latest_wins() -> None:
    calls: list[dict[str, object]] = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._latest_status_snapshot = None
            self._status_refresh_pending = False
            self._pending_harness_events = []
            self._pending_status_event = None
            self._ui_event_drain_task = None
            self._pending_user_echo = None
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = None

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            calls.append(dict(snapshot or {}))

    flow = _Flow()

    async def _run() -> None:
        await flow.on_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "first", "phase": "plan", "step": 1}},
            )
        )
        await flow.on_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "second", "phase": "execute", "step": 2}},
            )
        )
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert calls == [{"model": "second", "phase": "execute", "step": 2}]


def test_run_harness_task_backfills_terminal_result_into_assistant_transcript() -> None:
    rendered: list[tuple[UIEventType, str, dict[str, object]]] = []
    system_lines: list[str] = []

    class _Console:
        def __init__(self) -> None:
            self._assistant_text = ""

        def get_active_assistant_text(self) -> str:
            return self._assistant_text

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content, dict(event.data)))
            if event.event_type == UIEventType.ASSISTANT:
                self._assistant_text += str(event.content)

    class _Bridge:
        async def run_auto(self, task: str) -> dict[str, object]:
            assert task == "weather"
            return {
                "status": "completed",
                "message": {"status": "complete", "message": "Jacksonville is 85F today."},
                "assistant": "",
            }

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = object()
            self._harness_bridge = _Bridge()
            self._status_activity = ""
            self._api_error_count = 0
            self._pending_user_echo = None
            self._show_system_messages = True
            self._show_tool_calls = True
            self.active_task = None
            self.console = _Console()
            self._app_logger = logging.getLogger("test.ui_flow_regressions.terminal_result")

        def _get_console(self) -> _Console:
            return self.console

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            return None

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            system_lines.append(text)

    flow = _Flow()

    asyncio.run(flow._run_harness_task("weather"))

    assistant_events = [item for item in rendered if item[0] == UIEventType.ASSISTANT]
    assert assistant_events == [
        (
            UIEventType.ASSISTANT,
            "Jacksonville is 85F today.",
            {"promoted_from": "terminal_result"},
        )
    ]
    assert system_lines == [
        'RESULT {"status": "completed", "message": {"status": "complete", "message": "Jacksonville is 85F today."}, "assistant": ""}'
    ]


def test_handle_approval_prompt_prefers_bridge_resolution_after_thread_split() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.resolutions: list[tuple[str, bool]] = []

        def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
            self.resolutions.append((approval_id, approved))

    class _Harness:
        def __init__(self) -> None:
            self.state = SimpleNamespace(cwd="/repo")

        def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
            raise AssertionError("bridge should own shell approval resolution")

    class _App:
        def __init__(self) -> None:
            self.harness = _Harness()
            self._harness_bridge = _Bridge()
            self._active_approval_prompt = None
            self._status_activity = ""
            self._shell_approval_session_default = False
            self._app_logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

        async def push_screen(self, prompt, callback) -> None:
            self._active_approval_prompt = prompt
            callback(ShellApprovalDecision(True, True))

        def _refresh_status(self) -> None:
            return None

        def _set_shell_approval_session_default(self, enabled: bool) -> None:
            self._shell_approval_session_default = bool(enabled)

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            return None

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

    app = _App()

    asyncio.run(
        handle_approval_prompt(
            app,
            UIEvent(
                event_type=UIEventType.ALERT,
                data={
                    "ui_kind": "approve_prompt",
                    "approval_id": "shell-123",
                    "command": "pytest",
                    "cwd": "/repo",
                    "timeout_sec": 30,
                    "proof_bundle": {"phase": "verify"},
                },
            ),
        )
    )

    assert app._harness_bridge.resolutions == [("shell-123", True)]
    assert app._shell_approval_session_default is True
    assert app._status_activity == "running shell..."
    assert app._active_approval_prompt is None


def test_handle_sudo_password_prompt_prefers_bridge_resolution_after_thread_split() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.resolutions: list[tuple[str, str | None]] = []

        def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
            self.resolutions.append((prompt_id, password))

    class _Harness:
        def __init__(self) -> None:
            self.state = SimpleNamespace(cwd="/repo")

        def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
            raise AssertionError("bridge should own sudo prompt resolution")

    class _App:
        def __init__(self) -> None:
            self.harness = _Harness()
            self._harness_bridge = _Bridge()
            self._active_approval_prompt = None
            self._status_activity = ""
            self._app_logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

        async def push_screen(self, prompt, callback) -> None:
            self._active_approval_prompt = prompt
            callback("secret")

        def _refresh_status(self) -> None:
            return None

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

    app = _App()

    asyncio.run(
        handle_sudo_password_prompt(
            app,
            UIEvent(
                event_type=UIEventType.ALERT,
                data={
                    "ui_kind": "sudo_password_prompt",
                    "prompt_id": "sudo-123",
                    "command": "sudo ls",
                    "prompt_text": "Password please",
                },
            ),
        )
    )

    assert app._harness_bridge.resolutions == [("sudo-123", "secret")]
    assert app._status_activity == "running shell..."
    assert app._active_approval_prompt is None
