from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from textual.app import App, ComposeResult, ScreenStackError, SystemCommand
from textual.css.query import NoMatches
from textual.containers import Container, Vertical, Horizontal
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Static

from ..harness import Harness, HarnessConfig
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
    TITLE = "SmallctlApp"
    BINDINGS = [
        ("ctrl+l", "clear_console", "Clear"),
        ("ctrl+k", "cancel_task", "Cancel Task"),
        ("ctrl+shift+c", "copy_last_system_message", "Copy System"),
        ("ctrl+y", "copy_last_system_message", "Copy System"),
        ("ctrl+alt+n", "new_conversation", "Harness: New Conversation"),
        ("ctrl+alt+s", "toggle_system_messages", "View: System Messages"),
        ("ctrl+alt+t", "toggle_tool_calls", "View: Tool Calls"),
        ("ctrl+alt+b", "toggle_model_bar_layout", "View: Model Bar"),
        ("pageup", "scroll_page_up", "Scroll Up"),
        ("pagedown", "scroll_page_down", "Scroll Down"),
        ("ctrl+up", "scroll_up", "Scroll Up"),
        ("ctrl+down", "scroll_down", "Scroll Down"),
        ("ctrl+c", "interrupt_or_quit", "Interrupt/Quit"),
    ]

    def __init__(self, harness_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self._app_logger = logging.getLogger("smallctl.ui")
        raw = dict(harness_kwargs)
        self.restore_graph_state_on_startup = bool(raw.pop("restore_graph_state_on_startup", False))
        self.restore_thread_id = raw.pop("restore_thread_id", None)
        show_system_messages = bool(raw.pop("show_system_messages", False))
        self.initial_task = raw.pop("task", None)
        self.run_logger: RunLogger | None = raw.get("run_logger")
        self.harness_config = HarnessConfig(**raw)
        self.fresh_run = self.harness_config.fresh_run
        self.harness: Harness | None = None
        self.active_task: asyncio.Task[None] | None = None
        self.task_history: list[str] = []
        self.history_index = -1
        self._show_system_messages = show_system_messages
        self._show_tool_calls = True
        self._status_activity = ""
        self._api_error_count = 0
        self._latest_status_snapshot: dict[str, Any] | None = None
        self._status_refresh_pending = False
        self._recovery_banner = ""
        self._pending_harness_events: list[UIEvent] = []
        self._pending_status_event: UIEvent | None = None
        self._ui_event_drain_task: asyncio.Task[None] | None = None
        self._ui_transcript_persist_handle: asyncio.TimerHandle | None = None
        self._ui_transcript_debounce_seconds = 0.25
        self._harness_bridge: HarnessBridge | None = None
        self._pending_user_echo: str | None = None
        self._active_approval_prompt: Screen | None = None
        self.closed_by_ctrl_c = False
        self._task_start_time: float | None = None
        self._activity_timer: Any | None = None
        self._shell_approval_session_default = bool(self.harness_config.shell_approval_session_default)
        self._model_bar_layout = "bottom"
        self._verbose = bool(self.harness_config.verbose)
        self._goal_bar_expanded = False
        self._goal_bar_goal = "No active goal"
        self._goal_bar_tasks: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Horizontal(id="main-row"):
                with Container(id="console-wrap"):
                    yield Button("v Goal: No active goal", id="goal-bar-toggle")
                    yield Static("", id="goal-bar-details", classes="hidden")
                    yield ConsolePane(id="console", verbose=self._verbose)
                with Vertical(id="model-sidebar", classes="hidden"):
                    yield ModelSelectButton("n/a", id="model-button-sidebar")
                    yield ChatSelectButton(id="chat-button-sidebar")
                    yield Button("bar: right", id="model-bar-toggle-sidebar")
                    yield StatusBar(model="n/a", phase="explore", step=0, id="status-sidebar", vertical=True)
                    yield Button("Stop (Ctrl+K)", id="stop-button-sidebar")
            with Horizontal(id="status-row"):
                yield ModelSelectButton("n/a", id="model-button")
                yield ChatSelectButton(id="chat-button")
                yield StatusBar(model="n/a", phase="explore", step=0, id="status", show_model=False)
                yield Button("bar: bottom", id="model-bar-toggle")
                yield Button("Stop (Ctrl+K)", id="stop-button")
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
        yield SystemCommand(
            "View: Model Bar",
            "Toggle the model bar between bottom and right layouts",
            self.action_toggle_model_bar_layout,
        )

    async def on_mount(self) -> None:
        self._apply_model_bar_layout()
        await self._create_harness()
        if self.harness is None:
            # Harness init failed; error already surfaced by _create_harness.
            # Keep UI alive so user can read the message and exit cleanly.
            self._refresh_status(step_override="error")
            self.query_one(InputPane).focus()
            return
        restore_status = await self._maybe_restore_harness_state()
        if self.run_logger:
            self.run_logger.set_listener(self._on_run_log_row)
        log_kv(self._app_logger, logging.INFO, "ui_mounted")
        # Fix 1: Lifecycle telemetry — waiting for task
        log_kv(self._app_logger, logging.INFO, "waiting_for_task")
        self._capture_status_snapshot_from_harness()
        self._refresh_status()
        if restore_status is not None:
            restored_messages = restore_status.get("ui_transcript") if isinstance(restore_status, dict) else None
            if isinstance(restored_messages, list):
                self._ui_transcript = [dict(item) for item in restored_messages if isinstance(item, dict)]
            if not isinstance(restored_messages, list):
                restored_messages = restore_status.get("recent_messages") if isinstance(restore_status, dict) else None
            if isinstance(restored_messages, list):
                await self._render_restored_chat(messages=restored_messages)
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
        else:
            # Fix 4: Startup confirmation when no pre-loaded task
            if restore_status is None:
                asyncio.create_task(
                    self._append_system_line(
                        "Ready. Type a message to begin.",
                        force=True,
                        kind="system",
                    )
                )
            # Fix 3: FAMA disabled warning in TUI
            fama_config = getattr(
                getattr(self.harness, "state", None), "scratchpad", {}
            ).get("_fama_config", {})
            if fama_config.get("enabled") is False and not fama_config.get("_tui_warning_shown"):
                asyncio.create_task(
                    self._append_system_line(
                        "Warning: FAMA (failure-aware mitigation) is disabled. "
                        "Retry loops and tool misuse will not be auto-detected.",
                        force=True,
                    )
                )
                fama_config["_tui_warning_shown"] = True
        self.query_one(InputPane).focus()

    async def on_unmount(self) -> None:
        self._flush_ui_transcript_persist()
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
