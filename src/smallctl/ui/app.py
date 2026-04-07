from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Iterable

from textual import events
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
from .approval import ApprovePromptScreen
from .approval import PlanApprovalScreen
from .approval import ShellApprovalDecision
from .approval import PlanApprovalDecision
from .approval import SudoPasswordPromptScreen
from .console import ConsolePane
from .display import (
    format_restore_status,
    should_render_run_log_row,
    should_render_event,
    compute_activity_for_event,
    should_promote_tool_args_to_assistant,
    check_duplicate_promotion,
    StatusState,
)
from .input import InputPane
from .statusbar import StatusBar


class HarnessEvent(Message):
    def __init__(self, event: UIEvent) -> None:
        super().__init__()
        self.event = event


class SmallctlApp(App[None]):
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
                yield StatusBar(model="n/a", phase="explore", step=0, id="status")
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

    def on_mount(self) -> None:
        self._create_harness()
        restore_status = self._maybe_restore_harness_state()
        if self.run_logger:
            self.run_logger.set_listener(self._on_run_log_row)
        log_kv(self._app_logger, logging.INFO, "ui_mounted")
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
            self.active_task = asyncio.create_task(self._run_harness_task(self.initial_task))
        self.query_one(InputPane).focus()

    async def on_unmount(self) -> None:
        self._dismiss_active_approval_prompt()
        if self.harness:
             # Make sure we kill all subprocesses before the app fully closes
             try:
                 await self.harness.teardown()
             except Exception as exc:
                 self._app_logger.warning(f"Error during harness teardown on unmount: {exc}")
        if self.run_logger:
            self.run_logger.set_listener(None)

    async def on_input_pane_submitted(self, event: InputPane.Submitted) -> None:
        task = event.value.strip()
        input_widget = self.query_one(InputPane)
        input_widget.text = ""
        if not task:
            return
        if task.startswith("/"):
            handled = await self._handle_slash_command(task)
            if handled:
                return
        if self.active_task and not self.active_task.done():
            console = self._get_console()
            if console is not None:
                await self._append_system_line("Task already running.")
            return
        self.task_history.append(task)
        self.history_index = len(self.task_history)
        console = self._get_console()
        if console is not None:
            self._pending_user_echo = task
            await console.append_event(
                UIEvent(event_type=UIEventType.USER, content=task)
            )
            await console.begin_assistant_turn()
        else:
            self._pending_user_echo = None
        self._set_activity("thinking...")
        self._refresh_status(step_override="running")
        self.active_task = asyncio.create_task(self._run_harness_task(task))
        log_kv(self._app_logger, logging.INFO, "ui_task_started", task=task)

    def on_text_selected(self, event: events.TextSelected) -> None:
        selected_text = self._get_selected_screen_text()
        if not selected_text:
            return
        self._copy_selection_to_clipboard(selected_text)

    async def _handle_slash_command(self, task: str) -> bool:
        harness = self.harness
        command = task.strip().lower()
        if harness is None:
            return False
        if command == "/plan-mode":
            harness.state.planning_mode_enabled = True
            harness.state.planner_resume_target_mode = "loop"
            harness.state.touch()
            self._set_activity("planning mode active")
            self._refresh_status()
            await self._append_system_line("Planning mode enabled.", force=True)
            await self.on_harness_event(
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content="Planning mode enabled.",
                    data={"status_activity": "planning mode active"},
                )
            )
            return True
        if command == "/exit-plan-mode":
            pending = harness.get_pending_interrupt() or {}
            if str(pending.get("kind") or "") == "plan_execute_approval":
                await self._append_system_line(
                    "Planning mode is awaiting approval; finish that prompt before exiting.",
                    force=True,
                )
                return True
            harness.state.planning_mode_enabled = False
            harness.state.planner_requested_output_path = ""
            harness.state.planner_requested_output_format = ""
            harness.state.touch()
            self._set_activity("planning mode off")
            self._refresh_status()
            await self._append_system_line("Planning mode disabled.", force=True)
            await self.on_harness_event(
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content="Planning mode disabled.",
                    data={"status_activity": "planning mode off"},
                )
            )
            return True
        return False

    async def _run_harness_task(self, task: str) -> None:
        assert self.harness is not None
        self._set_activity("thinking...")
        self._refresh_status(step_override="running")
        try:
            result = await self.harness.run_auto_with_events(task, self.on_harness_event)
            result = await self._maybe_handle_plan_approval_result(result)
            log_kv(
                self._app_logger,
                logging.INFO,
                "ui_task_finished",
                status=result.get("status"),
            )
        except asyncio.CancelledError:
            console = self._get_console()
            if console is not None:
                await self._append_system_line("Task cancelled.")
            self._set_activity("")
            self._refresh_status(step_override="cancelled")
            self._pending_user_echo = None
            return
        except Exception as exc:
            console = self._get_console()
            if console is not None:
                await self._append_system_line(f"Task failed: {exc}")
            self._set_activity("")
            self._refresh_status(step_override="error")
            self._pending_user_echo = None
            return
        console = self._get_console()
        if console is not None:
            await self._append_system_line(f"RESULT {json.dumps(result, ensure_ascii=True)}")
        self._set_activity("")
        self._refresh_status()
        self._pending_user_echo = None

    async def _maybe_handle_plan_approval_result(self, result: dict[str, Any]) -> dict[str, Any]:
        harness = self.harness
        if harness is None:
            return result
        if str(result.get("status") or "") != "needs_human":
            return result
        interrupt = result.get("interrupt")
        if not isinstance(interrupt, dict):
            return result
        if str(interrupt.get("kind") or "") != "plan_execute_approval":
            return result

        decision = await self._prompt_for_plan_approval(interrupt)
        if decision is None:
            return result
        return await harness.resume_task_with_events(decision.choice, self.on_harness_event)

    async def on_harness_event(self, event: UIEvent | HarnessEvent) -> None:
        if isinstance(event, HarnessEvent):
            await self._handle_harness_event(event.event)
            return

        # Thread-safety: If called from a non-UI thread, queue a message for the main
        # UI thread instead of blocking on synchronous rendering.
        app_thread_id = getattr(self, "_loop_thread_id", None) or getattr(self, "_thread_id", None)
        if app_thread_id is not None and app_thread_id != threading.get_ident():
            if self.is_running:
                self.post_message(HarnessEvent(event))
            return

        await self._handle_harness_event(event)

    async def _handle_harness_event(self, event: UIEvent) -> None:

        if event.event_type == UIEventType.USER and self._pending_user_echo is not None:
            if str(event.content).strip() == self._pending_user_echo:
                self._pending_user_echo = None
                return

        self._update_activity_for_event(event)

        if event.data.get("ui_kind") == "approve_prompt":
            await self._handle_approval_prompt(event)
            return
        if event.data.get("ui_kind") == "sudo_password_prompt":
            await self._handle_sudo_password_prompt(event)
            return

        if event.event_type == UIEventType.ERROR:
            if event.data.get("is_api_error"):
                self._api_error_count += 1
                self._refresh_status()

        tool_name = str(event.data.get("tool_name") or event.content or "").strip()

        # Keep terminal control-tool noise out of the transcript by default.
        # The final answer is already promoted into the assistant bubble, so
        # the explicit task_complete call is only useful when system messages
        # are enabled.
        suppress_task_complete = tool_name == "task_complete" and not self._show_system_messages

        if suppress_task_complete and event.event_type == UIEventType.TOOL_RESULT:
            if self.harness is not None:
                self._refresh_status()
            return

        # Special handling: Promote tool arguments to assistant text if appropriate
        # before deciding whether to render the originating tool call.
        #
        # This lets task_complete surface the final answer even when the
        # underlying control tool call is hidden from the default transcript.
        if event.event_type == UIEventType.TOOL_CALL:
            name = event.content
            args = event.data.get("args") or {}
            promote_text = should_promote_tool_args_to_assistant(name, args)

            if promote_text:
                # Check if we already have a bubble group for this turn with substantial text.
                # If the model already gave a full report, promoting the task_complete summary
                # results in the "double summary" the user reported.
                console = self._get_console()
                skip_promotion = False
                if name == "task_complete" and console:
                    active_text = console.get_active_assistant_text().lower()
                    skip_promotion = check_duplicate_promotion(promote_text.lower(), active_text)

                if not skip_promotion:
                    # Create a synthetic assistant event to show the text in a bubble
                    await self.on_harness_event(
                        UIEvent(
                            event_type=UIEventType.ASSISTANT,
                            content=promote_text,
                            data={"promoted_from": name},
                        )
                    )

            if suppress_task_complete:
                if self.harness is not None:
                    self._refresh_status()
                return

        if not self._should_render_event(event):
            if self.harness is not None:
                self._refresh_status()
            return

        console = self._get_console()
        if console is None:
            return
        await console.append_event(event)
        if self.harness is not None:
            self._refresh_status()

    async def _handle_approval_prompt(self, event: UIEvent) -> None:
        harness = self.harness
        if harness is None:
            return
        approval_id = str(event.data.get("approval_id") or "").strip()
        command = str(event.data.get("command") or "").strip()
        cwd = str(event.data.get("cwd") or "").strip()
        timeout_raw = event.data.get("timeout_sec", 30)
        try:
            timeout_sec = int(timeout_raw)
        except (TypeError, ValueError):
            timeout_sec = 30

        prompt = ApprovePromptScreen(
            approval_id=approval_id or "pending",
            command=command or "(empty command)",
            cwd=cwd or harness.state.cwd,
            timeout_sec=max(1, timeout_sec),
        )
        self._active_approval_prompt = prompt
        self._refresh_status()

        approved = False
        remember_session = False
        runlog = getattr(harness, "_runlog", None)
        prompt_timeout_sec = max(1, timeout_sec)
        if callable(runlog):
            runlog(
                "shell_approval_prompt",
                "awaiting shell approval",
                approval_id=approval_id or "pending",
                command=command or "(empty command)",
                cwd=cwd or harness.state.cwd,
                timeout_sec=prompt_timeout_sec,
            )
        try:
            loop = asyncio.get_running_loop()
            decision_future: asyncio.Future[ShellApprovalDecision | bool] = loop.create_future()

            def _resolve_decision(decision: ShellApprovalDecision | bool | None) -> None:
                if decision_future.done():
                    return
                if isinstance(decision, ShellApprovalDecision):
                    decision_future.set_result(decision)
                    return
                decision_future.set_result(bool(decision))

            await self.push_screen(prompt, callback=_resolve_decision)
            decision = await decision_future
            approved = bool(decision)
            if isinstance(decision, ShellApprovalDecision):
                remember_session = decision.remember_session
            else:
                remember_session = bool(getattr(decision, "remember_session", False))
        except Exception as exc:
            self._app_logger.warning("Approval prompt failed: %s", exc)
            if callable(runlog):
                runlog(
                    "shell_approval_error",
                    "shell approval prompt failed",
                    approval_id=approval_id or "pending",
                    error=str(exc),
                )
        finally:
            self._active_approval_prompt = None
            if approved and remember_session:
                self._set_shell_approval_session_default(True)
            if approval_id:
                try:
                    harness.resolve_shell_approval(approval_id, approved)
                except Exception as exc:
                    self._app_logger.warning(
                        "Failed to resolve shell approval %s: %s",
                        approval_id,
                        exc,
                    )
            if callable(runlog):
                runlog(
                    "shell_approval_decision",
                    "shell approval resolved",
                    approval_id=approval_id or "pending",
                    approved=approved,
                    remember_session=remember_session,
                    command=command or "(empty command)",
                    cwd=cwd or harness.state.cwd,
                    timeout_sec=prompt_timeout_sec,
                )
            if approved and remember_session:
                await self._append_system_line(
                    "Shell commands will auto-approve for this session.",
                    force=True,
                )
            self._set_activity("running shell..." if approved else "thinking...")
            self._refresh_status()

    async def _prompt_for_plan_approval(self, interrupt: dict[str, Any]) -> PlanApprovalDecision | None:
        harness = self.harness
        if harness is None:
            return None
        prompt = PlanApprovalScreen(
            question=str(interrupt.get("question") or "Plan ready. Execute it now?").strip(),
            plan_id=str(interrupt.get("plan_id") or "").strip(),
            response_mode=str(interrupt.get("response_mode") or "yes/no/revise").strip(),
        )
        self._active_approval_prompt = prompt
        self._refresh_status()

        try:
            loop = asyncio.get_running_loop()
            decision_future: asyncio.Future[PlanApprovalDecision | str | None] = loop.create_future()

            def _resolve_decision(decision: PlanApprovalDecision | str | None) -> None:
                if decision_future.done():
                    return
                if isinstance(decision, PlanApprovalDecision):
                    decision_future.set_result(decision)
                    return
                if decision is None:
                    decision_future.set_result(None)
                    return
                decision_future.set_result(PlanApprovalDecision(str(decision)))

            await self.push_screen(prompt, callback=_resolve_decision)
            decision = await decision_future
            if isinstance(decision, PlanApprovalDecision):
                return decision
            if isinstance(decision, str) and decision.strip():
                return PlanApprovalDecision(decision.strip())
        except Exception as exc:
            self._app_logger.warning("Plan approval prompt failed: %s", exc)
        finally:
            self._active_approval_prompt = None
        return None

    async def _handle_sudo_password_prompt(self, event: UIEvent) -> None:
        harness = self.harness
        if harness is None:
            return
        prompt_id = str(event.data.get("prompt_id") or "").strip()
        command = str(event.data.get("command") or "").strip()
        prompt_text = str(event.data.get("prompt_text") or "").strip()

        prompt = SudoPasswordPromptScreen(
            prompt_id=prompt_id or "pending",
            command=command or "(empty command)",
            prompt_text=prompt_text or "Enter the sudo password to continue this command.",
        )
        self._active_approval_prompt = prompt
        self._refresh_status()

        password: str | None = None
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "sudo_password_prompt",
                "awaiting sudo password",
                prompt_id=prompt_id or "pending",
                command=command or "(empty command)",
            )
        try:
            loop = asyncio.get_running_loop()
            decision_future: asyncio.Future[str | None] = loop.create_future()

            def _resolve_decision(decision: str | None) -> None:
                if decision_future.done():
                    return
                decision_future.set_result(decision)

            await self.push_screen(prompt, callback=_resolve_decision)
            password = await decision_future
        except Exception as exc:
            self._app_logger.warning("Sudo password prompt failed: %s", exc)
            if callable(runlog):
                runlog(
                    "sudo_password_error",
                    "sudo password prompt failed",
                    prompt_id=prompt_id or "pending",
                    error=str(exc),
                )
        finally:
            self._active_approval_prompt = None
            if prompt_id:
                try:
                    harness.resolve_sudo_password(prompt_id, password)
                except Exception as exc:
                    self._app_logger.warning(
                        "Failed to resolve sudo password prompt %s: %s",
                        prompt_id,
                        exc,
                    )
            if callable(runlog):
                runlog(
                    "sudo_password_resolved",
                    "sudo password prompt resolved",
                    prompt_id=prompt_id or "pending",
                    provided=bool(password),
                    command=command or "(empty command)",
                )
            self._set_activity("running shell..." if password is not None else "thinking...")
            self._refresh_status()

    async def action_clear_console(self) -> None:
        console = self.query_one(ConsolePane)
        await console.clear_bubbles()

    async def action_cancel_task(self) -> None:
        self._dismiss_active_approval_prompt()
        if self.harness is not None:
            self.harness.cancel()
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
            log_kv(self._app_logger, logging.INFO, "ui_task_cancelled")
            console = self._get_console()
            if console is not None:
                await self._append_system_line("Active task cancelled.")
            self._refresh_status(step_override="cancelled")
            return
        console = self._get_console()
        if console is not None:
            await self._append_system_line("No active task.")

    def _dismiss_active_approval_prompt(self) -> None:
        prompt = self._active_approval_prompt
        if prompt is None:
            return
        self._active_approval_prompt = None
        try:
            prompt.dismiss(False)
        except Exception:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-button":
            await self.action_cancel_task()

    async def action_interrupt_or_quit(self) -> None:
        if self.active_task and not self.active_task.done():
            await self.action_cancel_task()
            return
        self.closed_by_ctrl_c = True
        self.exit()

    async def action_copy_last_system_message(self) -> None:
        console = self._get_console()
        if console is None:
            return
        text = console.get_last_system_message()
        if not text:
            await self._append_system_line("No system message available to copy.")
            return
        try:
            self.copy_to_clipboard(text)
            await self._append_system_line("Copied latest system message to clipboard.")
        except Exception as exc:
            await self._append_system_line(f"Copy failed: {exc}")

    def _get_selected_screen_text(self) -> str:
        try:
            selected_text = self.screen.get_selected_text()
        except Exception:
            return ""
        return str(selected_text or "").strip()

    def _copy_selection_to_clipboard(self, text: str) -> bool:
        selected_text = str(text or "")
        if not selected_text.strip():
            return False
        try:
            self.copy_to_clipboard(selected_text)
        except Exception as exc:
            self.notify(
                f"Copy failed: {exc}",
                title="Clipboard",
                severity="error",
                timeout=2.5,
                markup=False,
            )
            return False
        count = len(selected_text)
        noun = "character" if count == 1 else "characters"
        self.notify(
            f"Copied {count} {noun} to clipboard.",
            title="Selection Copied",
            timeout=1.5,
            markup=False,
        )
        return True

    async def action_new_conversation(self) -> None:
        if self.active_task and not self.active_task.done():
            await self._append_system_line("Cannot start a new conversation while task is running.", force=True)
            return
        self._create_harness()
        self._refresh_status(step_override=0)
        await self._append_system_line("Started a new conversation state.", force=True)

    async def action_toggle_system_messages(self) -> None:
        self._show_system_messages = not self._show_system_messages
        await self._append_system_line(
            f"System messages {'ON' if self._show_system_messages else 'OFF'}.",
            force=True,
        )

    async def action_toggle_tool_calls(self) -> None:
        self._show_tool_calls = not self._show_tool_calls
        await self._append_system_line(
            f"Tool calls {'ON' if self._show_tool_calls else 'OFF'}.",
            force=True,
        )

    def _scroll_console(self, delta: int) -> None:
        console = self._get_console()
        if console is None:
            return
        console.scroll_relative(y=delta, animate=False)

    def action_scroll_up(self) -> None:
        self._scroll_console(-3)

    def action_scroll_down(self) -> None:
        self._scroll_console(3)

    def action_scroll_page_up(self) -> None:
        console = self._get_console()
        if console is None:
            return
        step = max(1, console.size.height - 2)
        self._scroll_console(-step)

    def action_scroll_page_down(self) -> None:
        console = self._get_console()
        if console is None:
            return
        step = max(1, console.size.height - 2)
        self._scroll_console(step)

    def history_prev(self) -> str:
        if not self.task_history:
            return ""
        self.history_index = max(0, self.history_index - 1)
        return self.task_history[self.history_index]

    def history_next(self) -> str:
        if not self.task_history:
            return ""
        self.history_index = min(len(self.task_history), self.history_index + 1)
        if self.history_index == len(self.task_history):
            return ""
        return self.task_history[self.history_index]

    def _on_run_log_row(self, row: dict[str, Any]) -> None:
        if not self._should_render_run_log_row(row):
            return
        if not self._show_system_messages:
            return
        if not self._show_tool_calls:
            if row.get("channel") == "tools":
                return
            if row.get("event") in {"harness_tool_dispatch", "harness_tool_result"}:
                return
        from .display import format_run_log_row
        text = format_run_log_row(row)
        emit = lambda: asyncio.create_task(
            self.on_harness_event(UIEvent(event_type= UIEventType.SYSTEM, content=text))
        )
        app_thread_id = getattr(self, "_loop_thread_id", None) or getattr(self, "_thread_id", None)
        if app_thread_id == threading.get_ident():
            emit()
            return
        try:
            self.call_from_thread(emit)
        except RuntimeError:
            # Fallback for environments where current thread already owns the loop.
            emit()

    @staticmethod
    def _should_render_run_log_row(row: dict[str, Any]) -> bool:
        return should_render_run_log_row(row)

    def _refresh_status(self, step_override: int | str | None = None) -> None:
        state = StatusState.from_harness(
            self.harness,
            self.harness_kwargs,
            activity=self._status_activity,
            api_errors=self._api_error_count,
            active_task=self.active_task,
        )
        if step_override is not None:
            state.step = step_override
        try:
            self.query_one(StatusBar).set_state(
                model=state.model,
                phase=state.phase,
                step=state.step,
                mode=state.mode,
                plan=state.plan,
                active_step=state.active_step,
                activity=state.activity,
                contract_flow_ui=state.contract_flow_ui,
                contract_phase=state.contract_phase,
                acceptance_progress=state.acceptance_progress,
                latest_verdict=state.latest_verdict,
                token_usage=state.token_usage,
                token_total=state.token_total,
                token_limit=state.token_limit,
                api_errors=state.api_errors,
            )
        except (NoMatches, ScreenStackError):
            return

    def _get_console(self) -> ConsolePane | None:
        try:
            return self.query_one(ConsolePane)
        except (NoMatches, ScreenStackError):
            return None

    def _create_harness(self) -> None:
        self.harness = Harness(**self.harness_kwargs)
        interactive_setter = getattr(self.harness, "set_interactive_shell_approval", None)
        if callable(interactive_setter):
            interactive_setter(True)
        session_setter = getattr(self.harness, "set_shell_approval_session_default", None)
        if callable(session_setter):
            session_setter(self._shell_approval_session_default)

    def _maybe_restore_harness_state(self) -> dict[str, Any] | None:
        if self.fresh_run or not self.restore_graph_state_on_startup or self.harness is None:
            return None
        restored = self.harness.restore_graph_state(thread_id=self.restore_thread_id)
        if not restored:
            return {
                "status": "not_found",
                "thread_id": self.restore_thread_id,
            }
        payload: dict[str, Any] = {
            "status": "restored",
            "thread_id": self.harness.state.thread_id,
            "has_pending_interrupt": self.harness.has_pending_interrupt(),
        }
        interrupt = self.harness.get_pending_interrupt()
        if interrupt is not None:
            payload["interrupt"] = interrupt
        return payload

    @staticmethod
    def _format_restore_status(status: dict[str, Any]) -> str:
        return format_restore_status(status)

    async def _append_system_line(self, text: str, *, force: bool = False) -> None:
        if not force and not self._show_system_messages:
            return
        console = self._get_console()
        if console is not None:
            await console.append_line(text)

    def _set_activity(self, text: str | None) -> None:
        self._status_activity = str(text or "").strip()

    def _set_shell_approval_session_default(self, enabled: bool) -> None:
        self._shell_approval_session_default = bool(enabled)
        harness = self.harness
        if harness is not None:
            setter = getattr(harness, "set_shell_approval_session_default", None)
            if callable(setter):
                setter(self._shell_approval_session_default)

    def _update_activity_for_event(self, event: UIEvent) -> None:
        activity = compute_activity_for_event(
            event,
            active_task_done=None if self.active_task is None else self.active_task.done(),
        )
        if activity is not None:
            self._set_activity(activity)
        # Handle the case where status_activity is in data but activity is None
        if "status_activity" in event.data and activity is None:
            self._set_activity(str(event.data.get("status_activity") or "").strip())

    def _should_render_event(self, event: UIEvent) -> bool:
        return should_render_event(
            event,
            show_system_messages=self._show_system_messages,
            show_tool_calls=self._show_tool_calls,
        )
