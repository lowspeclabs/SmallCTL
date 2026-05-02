from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from textual.app import App, ComposeResult, ScreenStackError, SystemCommand
from textual.css.query import NoMatches
from textual.containers import Container, Vertical, Horizontal
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button

from ..harness import Harness
from ..logging_utils import RunLogger
from ..logging_utils import log_kv
from ..models.events import UIEvent, UIEventType
from .console import ConsolePane
from .app_approvals import handle_approval_prompt
from .app_approvals import handle_sudo_password_prompt
from .app_approvals import maybe_handle_plan_approval_result
from .app_actions import SmallctlAppActionsMixin
from .input import InputPane
from .chat_selector import ChatSelectButton
from .model_selector import ModelSelectButton
from .statusbar import StatusBar
from .app_flow import SmallctlAppFlowMixin
from .harness_bridge import HarnessBridge


class HarnessEvent(Message):
    def __init__(self, event: UIEvent) -> None:
        super().__init__()
        self.event = event


class SmallctlApp(SmallctlAppActionsMixin, SmallctlAppFlowMixin, App[None]):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("ctrl+l", "clear_console", "Clear"),
        ("ctrl+k", "cancel_task", "Cancel Task"),
        ("ctrl+shift+c", "copy_last_system_message", "Copy System"),
        ("ctrl+y", "copy_last_system_message", "Copy System"),
        ("ctrl+alt+n", "new_conversation", "Harness: New Conversation"),
        ("ctrl+alt+s", "toggle_system_messages", "View: System Messages"),
        ("ctrl+alt+t", "toggle_tool_calls", "View: Tool Calls"),
        ("pageup", "scroll_page_up", "Scroll Up"),
        ("pagedown", "scroll_page_down", "Scroll Down"),
        ("ctrl+up", "scroll_up", "Scroll Up"),
        ("ctrl+down", "scroll_down", "Scroll Down"),
        ("ctrl+c", "interrupt_or_quit", "Interrupt/Quit"),
    ]

    def __init__(self, harness_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self._app_logger = logging.getLogger("smallctl.ui")
        self.harness_kwargs = dict(harness_kwargs)
        self.fresh_run = bool(self.harness_kwargs.pop("fresh_run", False))
        self.restore_graph_state_on_startup = bool(
            self.harness_kwargs.pop("restore_graph_state_on_startup", False)
        )
        self.restore_thread_id = self.harness_kwargs.pop("restore_thread_id", None)
        self.initial_task = self.harness_kwargs.pop("task", None)
        self.run_logger: RunLogger | None = harness_kwargs.get("run_logger")
        self.harness: Harness | None = None
        self.active_task: asyncio.Task[None] | None = None
        self.task_history: list[str] = []
        self.history_index = -1
        self._show_system_messages = False
        self._show_tool_calls = True
        self._status_activity = ""
        self._api_error_count = 0
        self._latest_status_snapshot: dict[str, Any] | None = None
        self._status_refresh_pending = False
        self._pending_harness_events: list[UIEvent] = []
        self._pending_status_event: UIEvent | None = None
        self._ui_event_drain_task: asyncio.Task[None] | None = None
        self._harness_bridge: HarnessBridge | None = None
        self._pending_user_echo: str | None = None
        self._active_approval_prompt: Screen | None = None
        self.closed_by_ctrl_c = False
        self._shell_approval_session_default = bool(
            self.harness_kwargs.pop("shell_approval_session_default", False)
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Container(id="console-wrap"):
                yield ConsolePane(id="console")
            with Horizontal(id="status-row"):
                yield ModelSelectButton("n/a", id="model-button")
                yield ChatSelectButton(id="chat-button")
                yield StatusBar(model="n/a", phase="explore", step=0, id="status", show_model=False)
                yield Button("Stop (Ctrl+K)", id="stop-button", variant="error")
            yield InputPane("", id="input")

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand(
            "Harness: New Conversation",
            "Reset harness state and start a fresh conversation",
            self.action_new_conversation,
        )
        yield SystemCommand(
            "View: System Messages",
            "Toggle rendering of system-level messages",
            self.action_toggle_system_messages,
        )
        yield SystemCommand(
            "View: Tool Calls",
            "Toggle rendering of tool call and result events",
            self.action_toggle_tool_calls,
        )

    async def on_mount(self) -> None:
        self._create_harness()
        restore_status = await self._maybe_restore_harness_state()
        if self.run_logger:
            self.run_logger.set_listener(self._on_run_log_row)
        log_kv(self._app_logger, logging.INFO, "ui_mounted")
        self._capture_status_snapshot_from_harness()
        self._refresh_status()
        if restore_status is not None:
            asyncio.create_task(
                self._append_system_line(
                    self._format_restore_status(restore_status),
                    force=True,
                )
            )
        if self.initial_task:
            self.task_history.append(self.initial_task)
            self.history_index = len(self.task_history)
            self._record_chat_session_prompt(self.initial_task)
            self.active_task = asyncio.create_task(self._run_harness_task(self.initial_task))
        self.query_one(InputPane).focus()

    async def on_unmount(self) -> None:
        self._dismiss_active_approval_prompt()
        bridge = self._harness_bridge
        if bridge is not None:
            try:
                await bridge.shutdown()
            except Exception as exc:
                self._app_logger.warning(f"Error during harness bridge shutdown on unmount: {exc}")
            finally:
                self._harness_bridge = None
        elif self.harness:
            try:
                await self.harness.teardown()
            except Exception as exc:
                self._app_logger.warning(f"Error during harness teardown on unmount: {exc}")
        if self.run_logger:
            self.run_logger.set_listener(None)
