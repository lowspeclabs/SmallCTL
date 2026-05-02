from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from textual import events
from textual.app import ScreenStackError
from textual.css.query import NoMatches

from ..harness import Harness
from ..logging_utils import RunLogger
from ..logging_utils import log_kv
from ..models.events import UIEvent, UIEventType, UIStatusSnapshot
from ..chat_sessions import record_chat_session_prompt
from .app_approvals import handle_approval_prompt
from .app_approvals import handle_sudo_password_prompt
from .app_approvals import maybe_handle_plan_approval_result
from .console import ConsolePane
from .display import (
    StatusState,
    check_duplicate_promotion,
    compute_activity_for_event,
    format_restore_status,
    should_promote_tool_args_to_assistant,
    should_render_event,
    should_render_run_log_row,
)
from .input import InputPane
from .chat_selector import ChatSelectButton
from .harness_bridge import HarnessBridge
from .model_selector import ModelSelectButton
from .statusbar import StatusBar


def _harness_event_message_type() -> type[Any]:
    # Imported lazily to avoid a module cycle with app.py, which mixes this class in.
    from .app import HarnessEvent

    return HarnessEvent


def _extract_terminal_result_text(result: dict[str, Any]) -> str:
    if not isinstance(result, dict) or not result:
        return ""

    message = result.get("message")
    if isinstance(message, dict):
        for key in ("message", "output", "text", "question"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(message, str) and message.strip():
        return message.strip()

    assistant = str(result.get("assistant") or "").strip()
    if assistant:
        return assistant

    reason = str(result.get("reason") or "").strip()
    return reason


class SmallctlAppFlowMixin:
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
        self._record_chat_session_prompt(task)
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
            bridge = getattr(self, "_harness_bridge", None)
            if bridge is not None:
                snapshot = await bridge.set_planning_mode(True)
            else:
                setter = getattr(harness, "set_planning_mode", None)
                if callable(setter):
                    snapshot = setter(True)
                else:
                    harness.state.planning_mode_enabled = True
                    harness.state.planner_resume_target_mode = "loop"
                    harness.state.touch()
                    snapshot = self._capture_status_snapshot_from_harness()
            self._set_activity("planning mode active")
            self._refresh_status(snapshot=snapshot)
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
            bridge = getattr(self, "_harness_bridge", None)
            if bridge is not None:
                snapshot = await bridge.set_planning_mode(False)
            else:
                setter = getattr(harness, "set_planning_mode", None)
                if callable(setter):
                    snapshot = setter(False)
                else:
                    harness.state.planning_mode_enabled = False
                    harness.state.planner_requested_output_path = ""
                    harness.state.planner_requested_output_format = ""
                    harness.state.touch()
                    snapshot = self._capture_status_snapshot_from_harness()
            self._set_activity("planning mode off")
            self._refresh_status(snapshot=snapshot)
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
            bridge = getattr(self, "_harness_bridge", None)
            if bridge is not None:
                result = await bridge.run_auto(task)
            else:
                result = await self.harness.run_auto_with_events(task, self.on_harness_event)
            result = await maybe_handle_plan_approval_result(self, result)
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
            await self._maybe_render_terminal_result(result, console=console)
            await self._append_system_line(f"RESULT {json.dumps(result, ensure_ascii=True)}")
        self._set_activity("")
        self._refresh_status()
        self._pending_user_echo = None

    async def on_harness_event(self, event: UIEvent | "HarnessEvent") -> None:
        harness_event_type = _harness_event_message_type()
        if isinstance(event, harness_event_type):
            self._enqueue_harness_event(event.event)
            return

        app_thread_id = getattr(self, "_loop_thread_id", None) or getattr(self, "_thread_id", None)
        if app_thread_id is not None and app_thread_id != threading.get_ident():
            if self.is_running:
                self.post_message(harness_event_type(event))
            return

        self._enqueue_harness_event(event)

    def _enqueue_harness_event(self, event: UIEvent) -> None:
        if event.event_type == UIEventType.STATUS:
            self._pending_status_event = event
        else:
            pending = getattr(self, "_pending_harness_events", None)
            if pending is None:
                pending = []
                self._pending_harness_events = pending
            pending.append(event)
        self._schedule_harness_event_drain()

    def _schedule_harness_event_drain(self) -> None:
        task = getattr(self, "_ui_event_drain_task", None)
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._ui_event_drain_task = loop.create_task(self._drain_harness_events())

    async def _drain_harness_events(self) -> None:
        try:
            while True:
                event: UIEvent | None = None
                pending = getattr(self, "_pending_harness_events", None) or []
                if pending:
                    event = pending.pop(0)
                else:
                    status_event = getattr(self, "_pending_status_event", None)
                    if status_event is not None:
                        event = status_event
                        self._pending_status_event = None
                if event is None:
                    return
                await self._handle_harness_event(event)
        finally:
            self._ui_event_drain_task = None
            pending = getattr(self, "_pending_harness_events", None) or []
            if pending or getattr(self, "_pending_status_event", None) is not None:
                self._schedule_harness_event_drain()

    async def _handle_harness_event(self, event: UIEvent) -> None:
        if event.event_type == UIEventType.STATUS:
            snapshot_payload = event.data.get("snapshot")
            if isinstance(snapshot_payload, dict):
                self._queue_status_refresh(snapshot_payload)
            return

        if event.event_type == UIEventType.USER and self._pending_user_echo is not None:
            if str(event.content).strip() == self._pending_user_echo:
                self._pending_user_echo = None
                return

        self._update_activity_for_event(event)

        if event.data.get("ui_kind") == "approve_prompt":
            await handle_approval_prompt(self, event)
            return
        if event.data.get("ui_kind") == "sudo_password_prompt":
            await handle_sudo_password_prompt(self, event)
            return

        tool_name = str(event.data.get("tool_name") or event.content or "").strip()
        suppress_task_complete = tool_name == "task_complete" and not self._show_system_messages

        if suppress_task_complete and event.event_type == UIEventType.TOOL_RESULT:
            return

        if event.event_type == UIEventType.TOOL_CALL:
            name = event.content
            args = event.data.get("args") or {}
            promote_text = should_promote_tool_args_to_assistant(name, args)

            if promote_text:
                console = self._get_console()
                skip_promotion = False
                if name == "task_complete" and console:
                    active_text = console.get_active_assistant_text().lower()
                    skip_promotion = check_duplicate_promotion(promote_text.lower(), active_text)

                if not skip_promotion:
                    await self.on_harness_event(
                        UIEvent(
                            event_type=UIEventType.ASSISTANT,
                            content=promote_text,
                            data={"promoted_from": name},
                        )
                    )

            if suppress_task_complete:
                return

        if not self._should_render_event(event):
            return

        console = self._get_console()
        if console is None:
            return
        await console.append_event(event)

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
            self.on_harness_event(UIEvent(event_type=UIEventType.SYSTEM, content=text))
        )
        app_thread_id = getattr(self, "_loop_thread_id", None) or getattr(self, "_thread_id", None)
        if app_thread_id == threading.get_ident():
            emit()
            return
        try:
            self.call_from_thread(emit)
        except RuntimeError:
            emit()

    @staticmethod
    def _should_render_run_log_row(row: dict[str, Any]) -> bool:
        return should_render_run_log_row(row)

    def _queue_status_refresh(self, snapshot: dict[str, Any]) -> None:
        self._latest_status_snapshot = dict(snapshot)
        if self._status_refresh_pending:
            return
        self._status_refresh_pending = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._flush_status_refresh()
            return
        loop.call_soon(self._flush_status_refresh)

    def _flush_status_refresh(self) -> None:
        self._status_refresh_pending = False
        snapshot = self._latest_status_snapshot
        if snapshot is None:
            return
        self._refresh_status(snapshot=snapshot)

    def _default_status_snapshot(self) -> dict[str, Any]:
        return UIStatusSnapshot(
            model=str(self.harness_kwargs.get("model", "n/a") or "n/a"),
            phase=str(self.harness_kwargs.get("phase", "explore") or "explore"),
            contract_flow_ui=bool(self.harness_kwargs.get("contract_flow_ui", False)),
            api_errors=max(0, int(self._api_error_count)),
        ).to_dict()

    def _capture_status_snapshot_from_harness(self) -> dict[str, Any]:
        activity = self._status_activity
        if not activity and self.active_task is not None and not self.active_task.done():
            activity = "thinking..."
        builder = getattr(self.harness, "build_status_snapshot", None) if self.harness is not None else None
        if callable(builder):
            snapshot = builder(activity=activity, api_errors=self._api_error_count)
        else:
            snapshot = UIStatusSnapshot.from_harness(
                self.harness,
                self.harness_kwargs,
                activity=activity,
                api_errors=self._api_error_count,
            ).to_dict()
        self._latest_status_snapshot = dict(snapshot)
        return dict(snapshot)

    def _status_snapshot_for_render(
        self,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if snapshot is not None:
            payload = dict(snapshot)
        elif getattr(self, "_latest_status_snapshot", None) is not None:
            payload = dict(getattr(self, "_latest_status_snapshot"))
        else:
            payload = self._default_status_snapshot()

        if not str(payload.get("activity", "") or "").strip():
            activity = self._status_activity
            if not activity and self.active_task is not None and not self.active_task.done():
                activity = "thinking..."
            payload["activity"] = activity

        payload["api_errors"] = max(0, int(self._api_error_count))
        return payload

    def _refresh_status(
        self,
        step_override: int | str | None = None,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        state = StatusState.from_snapshot(self._status_snapshot_for_render(snapshot))
        if step_override is not None:
            state.step = step_override
        self._latest_status_snapshot = state.__dict__.copy()
        self._status_activity = state.activity
        self._api_error_count = state.api_errors
        try:
            model_button = self.query_one(ModelSelectButton)
            model_button.set_model(state.model)
            model_button.set_busy(self.active_task is not None and not self.active_task.done())
            try:
                chat_button = self.query_one(ChatSelectButton)
                chat_button.set_busy(self.active_task is not None and not self.active_task.done())
            except NoMatches:
                pass
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
        self._harness_bridge = HarnessBridge(
            harness=self.harness,
            post_ui_event=self._post_harness_bridge_event,
        )
        self._harness_bridge.start()

    def _post_harness_bridge_event(self, event: UIEvent) -> None:
        if not getattr(self, "is_running", False):
            return
        self.post_message(_harness_event_message_type()(event))

    def _record_chat_session_prompt(self, task: str) -> None:
        harness = self.harness
        if harness is None:
            return
        state = getattr(harness, "state", None)
        if state is None:
            return
        client = getattr(harness, "client", None)
        try:
            record_chat_session_prompt(
                cwd=getattr(state, "cwd", "."),
                thread_id=getattr(state, "thread_id", "") or getattr(harness, "conversation_id", ""),
                message=task,
                model=str(getattr(client, "model", "") or ""),
                created_at=str(getattr(state, "created_at", "") or ""),
            )
        except Exception:
            self._app_logger.debug("Unable to record chat session prompt", exc_info=True)

    async def _switch_model(self, model: str) -> None:
        model_name = str(model or "").strip()
        if not model_name:
            return
        self.harness_kwargs["model"] = model_name
        harness = self.harness
        if harness is not None:
            bridge = getattr(self, "_harness_bridge", None)
            if bridge is not None:
                result = await bridge.switch_model(
                    model_name,
                    activity=self._status_activity,
                    api_errors=self._api_error_count,
                )
                self.harness_kwargs["provider_profile"] = str(
                    result.get("provider_profile")
                    or getattr(harness, "provider_profile", None)
                    or self.harness_kwargs.get("provider_profile", "generic")
                )
                self.harness_kwargs["context_limit"] = None
                snapshot = result.get("snapshot")
                if isinstance(snapshot, dict):
                    self._refresh_status(snapshot=snapshot)
                else:
                    self._capture_status_snapshot_from_harness()
                    self._refresh_status()
                return
            switcher = getattr(harness, "switch_model", None)
            if callable(switcher):
                switcher(model_name)
                self.harness_kwargs["provider_profile"] = getattr(
                    harness,
                    "provider_profile",
                    self.harness_kwargs.get("provider_profile", "generic"),
                )
                self.harness_kwargs["context_limit"] = None
            else:
                old_state = harness.state
                self._create_harness()
                if self.harness is not None:
                    self.harness.state = old_state
                    self.harness.state.scratchpad["_model_name"] = model_name
                    self.harness.state.scratchpad["_model_is_small"] = self.harness._is_small_model_name(model_name)
        self._capture_status_snapshot_from_harness()
        self._refresh_status()

    async def _maybe_restore_harness_state(self) -> dict[str, Any] | None:
        if self.fresh_run or not self.restore_graph_state_on_startup or self.harness is None:
            return None
        bridge = getattr(self, "_harness_bridge", None)
        if bridge is not None:
            result = await bridge.restore_graph_state(thread_id=self.restore_thread_id)
            restored = bool(result.get("restored"))
            if not restored:
                return {
                    "status": "not_found",
                    "thread_id": self.restore_thread_id,
                }
            snapshot = result.get("snapshot")
            if isinstance(snapshot, dict):
                self._latest_status_snapshot = dict(snapshot)
            payload: dict[str, Any] = {
                "status": "restored",
                "thread_id": str(result.get("thread_id") or self.restore_thread_id or ""),
                "has_pending_interrupt": bool(result.get("has_pending_interrupt")),
            }
            interrupt = result.get("interrupt")
            if isinstance(interrupt, dict):
                payload["interrupt"] = interrupt
            return payload

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

    async def _maybe_render_terminal_result(
        self,
        result: dict[str, Any],
        *,
        console: ConsolePane | None = None,
    ) -> None:
        status = str(result.get("status") or "").strip().lower()
        if status not in {"completed", "chat_completed"}:
            return
        final_text = _extract_terminal_result_text(result)
        if not final_text:
            return
        console = console or self._get_console()
        if console is None:
            return
        active_text = console.get_active_assistant_text().strip()
        if active_text and check_duplicate_promotion(final_text.lower(), active_text.lower()):
            return
        await console.append_event(
            UIEvent(
                event_type=UIEventType.ASSISTANT,
                content=final_text,
                data={"promoted_from": "terminal_result"},
            )
        )

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
            bridge = getattr(self, "_harness_bridge", None)
            if bridge is not None:
                bridge.set_shell_approval_session_default(self._shell_approval_session_default)
                return
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
        if "status_activity" in event.data and activity is None:
            self._set_activity(str(event.data.get("status_activity") or "").strip())

    def _should_render_event(self, event: UIEvent) -> bool:
        return should_render_event(
            event,
            show_system_messages=self._show_system_messages,
            show_tool_calls=self._show_tool_calls,
        )
