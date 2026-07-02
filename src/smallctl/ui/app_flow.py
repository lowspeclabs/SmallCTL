from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .app import HarnessEvent

from textual import events
from textual.app import ScreenStackError
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from ..harness import Harness
from ..logging_utils import RunLogger
from ..logging_utils import log_kv
from ..models.events import UIEvent, UIEventType, UIStatusSnapshot
from ..chat_sessions import (
    load_chat_session_state,
    load_chat_session_summaries,
    load_chat_session_ui_transcript,
    persist_chat_session_ui_transcript,
    record_chat_session_prompt,
)
from ..client.chunk_parser import sanitize_assistant_content_for_history
from .app_approvals import handle_approval_prompt
from .app_approvals import handle_sudo_password_prompt
from .app_approvals import maybe_handle_plan_approval_result
from .app_flow_commands import handle_slash_command
from .app_flow_input import handle_input_pane_submitted
from .app_flow_utils import _extract_terminal_result_text, _harness_event_message_type
from .console import ConsolePane
from .display import (
    StatusState,
    _build_backend_rca_strip,
    _CRITICAL_EVENTS,
    check_duplicate_promotion,
    compute_activity_for_event,
    format_recovery_banner,
    format_restore_status,
    should_promote_tool_args_to_assistant,
    should_render_event,
    should_render_run_log_row,
)
from .input import InputPane
from .chat_selector import ChatSelectButton
from .harness_bridge import HarnessBridge, _serialize_recent_messages
from .model_selector import ModelSelectButton
from .statusbar import StatusBar


class SmallctlAppFlowMixin:
    async def on_input_pane_submitted(self, event: InputPane.Submitted) -> None:
        await handle_input_pane_submitted(self, event)

    def on_text_selected(self, event: events.TextSelected) -> None:
        selected_text = self._get_selected_screen_text()
        if not selected_text:
            return
        self._copy_selection_to_clipboard(selected_text)

    async def _handle_slash_command(self, task: str) -> bool:
        return await handle_slash_command(self, task)

    async def _run_harness_task(self, task: str) -> None:
        assert self.harness is not None
        self._set_activity("[thinking...]")
        self._refresh_status(step_override="running")
        self._task_start_time = time.monotonic()
        self._activity_timer = self.set_interval(1.0, self._tick_activity_timer)
        step_override = None
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
            console = self._get_console()
            if console is not None:
                await self._maybe_render_terminal_result(result, console=console)
                status = str(result.get("status") or "done")
                await self._append_system_line(
                    f"Task {status}. Type a new message or press Ctrl+C to exit."
                )
                detail = _terminal_status_detail(result, self.harness)
                if detail:
                    await self._append_system_line(detail)
                warning = str(result.get("unverified_change_warning") or "").strip()
                if warning:
                    await self._append_system_line(warning, force=True, kind="warning")
        except asyncio.CancelledError:
            console = self._get_console()
            if console is not None:
                await self._append_system_line("Task cancelled.", force=True, kind="cancel")
                rca = _build_backend_rca_strip(self.harness)
                if rca:
                    await self._append_system_line(rca, force=True, kind="cancel")
            step_override = "cancelled"
        except Exception as exc:
            console = self._get_console()
            if console is not None:
                await self._append_system_line(f"Task failed: {exc}", force=True)
            step_override = "error"
        finally:
            if self._activity_timer is not None:
                self._activity_timer.stop()
                self._activity_timer = None
            self._task_start_time = None
            self.active_task = None
            self._pending_user_echo = None
            self._set_activity("")
            self._refresh_status(step_override=step_override)

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
        self._update_recovery_banner_from_event(event)

        if event.data.get("ui_kind") == "approve_prompt":
            await handle_approval_prompt(self, event)
            return
        if event.data.get("ui_kind") == "sudo_password_prompt":
            await handle_sudo_password_prompt(self, event)
            return

        if event.data.get("ui_kind") == "subtask_checklist":
            self._update_goal_bar(event.content)
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
                    active_text = console.get_active_assistant_text().strip()
                    skip_promotion = bool(active_text) and (
                        suppress_task_complete
                        or check_duplicate_promotion(
                            promote_text.lower(),
                            active_text.lower(),
                        )
                    )

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

        event = self._sanitize_visible_text_event(event)
        if event is None:
            return
        if not self._should_render_event(event):
            return

        console = self._get_console()
        if console is None:
            return
        self._record_ui_transcript_event(event)
        await console.append_event(event)

    def _update_goal_bar(self, content: str) -> None:
        goal, tasks = self._parse_goal_bar_content(content)
        self._goal_bar_goal = goal or "No active goal"
        self._goal_bar_tasks = tasks
        self._refresh_goal_bar()

    def _refresh_goal_bar(self) -> None:
        try:
            toggle = self.query_one("#goal-bar-toggle", Button)
            details = self.query_one("#goal-bar-details", Static)
        except (NoMatches, ScreenStackError, TypeError):
            return
        expanded = bool(getattr(self, "_goal_bar_expanded", False))
        goal = str(getattr(self, "_goal_bar_goal", "") or "No active goal").strip()
        tasks = list(getattr(self, "_goal_bar_tasks", []) or [])
        short_goal = goal if len(goal) <= 140 else goal[:139].rstrip() + "~"
        toggle.label = f"{'^' if expanded else 'v'} Goal: {short_goal}"
        details.set_class(not expanded, "hidden")
        if not expanded:
            details.update("")
            return
        detail_lines = [f"[bold #93c5fd]Goal[/] {goal}"]
        if tasks:
            detail_lines.append("[bold #bfdbfe]Tasks[/]")
            detail_lines.extend(f"[#94a3b8]{task}[/]" for task in tasks)
        else:
            detail_lines.append("[#94a3b8]No active tasks[/]")
        details.update("\n".join(detail_lines))

    @staticmethod
    def _parse_goal_bar_content(content: str) -> tuple[str, list[str]]:
        lines = [line.strip() for line in str(content or "").splitlines() if line.strip()]
        if not lines:
            return "", []
        first = lines[0]
        if first.startswith("Goal Objective: "):
            first = first[16:].strip()
        tasks = []
        for line in lines[1:]:
            task = line.strip()
            if task:
                tasks.append(task if len(task) <= 180 else task[:179].rstrip() + "~")
        return first, tasks

    def _on_run_log_row(self, row: dict[str, Any]) -> None:
        self._update_recovery_banner_from_run_log_row(row)
        if not self._should_render_run_log_row(row):
            return
        event = str(row.get("event") or "")
        if not self._show_system_messages:
            # Critical backend state changes should always be visible
            if event not in _CRITICAL_EVENTS:
                return
        if not self._show_tool_calls:
            if row.get("channel") == "tools":
                return
            if event in {"harness_tool_dispatch", "harness_tool_result"}:
                return
        from .display import format_run_log_row

        text = format_run_log_row(row)
        event_data: dict[str, Any] = {}
        if event in _CRITICAL_EVENTS:
            event_data["ui_kind"] = event
            event_data["event_payload"] = dict(row.get("data") or {})
        emit = lambda: asyncio.create_task(
            self.on_harness_event(UIEvent(event_type=UIEventType.SYSTEM, content=text, data=event_data))
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
            model=str(self.harness_config.model or "n/a"),
            phase=str(self.harness_config.phase or "explore"),
            contract_flow_ui=bool(self.harness_config.contract_flow_ui),
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
                self.harness_config,
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
        payload["recovery_banner"] = str(getattr(self, "_recovery_banner", "") or "")
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
            is_busy = self.active_task is not None and not self.active_task.done()
            query = getattr(self, "query", None)
            if callable(query):
                model_buttons = list(query(ModelSelectButton))
                chat_buttons = list(query(ChatSelectButton))
                status_bars = list(query(StatusBar))
            else:
                model_buttons = [self.query_one(ModelSelectButton)]
                try:
                    chat_buttons = [self.query_one(ChatSelectButton)]
                except NoMatches:
                    chat_buttons = []
                status_bars = [self.query_one(StatusBar)]
            for model_button in model_buttons:
                model_button.set_model(state.model)
                model_button.set_busy(is_busy)
            for chat_button in chat_buttons:
                chat_button.set_busy(is_busy)
            for status_bar in status_bars:
                status_bar.set_state(
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
                    context_window=state.context_window,
                    api_errors=state.api_errors,
                    fama_off=getattr(state, "fama_off", False),
                    fama_mitigation=getattr(state, "fama_mitigation", ""),
                    recovery_banner=getattr(state, "recovery_banner", ""),
                )
        except (NoMatches, ScreenStackError):
            return

    def _update_recovery_banner_from_event(self, event: UIEvent) -> None:
        event_name = str(event.data.get("event") or event.data.get("ui_kind") or "").strip()
        banner = format_recovery_banner(event_name, event.data)
        if not banner:
            return
        self._recovery_banner = banner
        if self.harness is not None and isinstance(getattr(self.harness.state, "scratchpad", None), dict):
            self.harness.state.scratchpad["_ui_recovery_banner"] = banner
        self._refresh_status()

    def _update_recovery_banner_from_run_log_row(self, row: dict[str, Any]) -> None:
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        banner = format_recovery_banner(str(row.get("event") or "").strip(), data)
        if not banner:
            return
        self._recovery_banner = banner
        if self.harness is not None and isinstance(getattr(self.harness.state, "scratchpad", None), dict):
            self.harness.state.scratchpad["_ui_recovery_banner"] = banner
        self._refresh_status()

    def _get_console(self) -> ConsolePane | None:
        try:
            return self.query_one(ConsolePane)
        except (NoMatches, ScreenStackError):
            return None

    async def _shutdown_harness(self) -> None:
        """Defensively shut down any existing bridge/harness."""
        from .app_lifecycle import shutdown_harness
        old_bridge = getattr(self, "_harness_bridge", None)
        await shutdown_harness(old_bridge, self.harness if old_bridge is None else None)
        if old_bridge is not None:
            self._harness_bridge = None
        else:
            self.harness = None

    async def _create_harness(self) -> None:
        await self._shutdown_harness()

        try:
            self.harness = Harness(self.harness_config)
        except Exception as exc:
            self._app_logger.exception("harness_init_failed")
            await self._append_system_line(
                f"Failed to initialize harness: {exc}", force=True
            )
            self._refresh_status(step_override="error")
            self.harness = None
            self._harness_bridge = None
            return
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
        await asyncio.to_thread(self._harness_bridge.start)

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
        self.harness_config.model = model_name
        harness = self.harness
        if harness is not None:
            bridge = getattr(self, "_harness_bridge", None)
            if bridge is not None:
                result = await bridge.switch_model(
                    model_name,
                    activity=self._status_activity,
                    api_errors=self._api_error_count,
                )
                self.harness_config.provider_profile = str(
                    result.get("provider_profile")
                    or getattr(harness, "provider_profile", None)
                    or self.harness_config.provider_profile
                    or "generic"
                )
                self.harness_config.context_limit = None
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
                self.harness_config.provider_profile = getattr(
                    harness, "provider_profile", self.harness_config.provider_profile or "generic"
                )
                self.harness_config.context_limit = None
            else:
                await self._shutdown_harness()
                old_state = harness.state
                await self._create_harness()
                if self.harness is not None:
                    self.harness.state = old_state
                    self.harness.state.scratchpad["_model_name"] = model_name
                    self.harness.state.scratchpad["_model_is_small"] = self.harness._is_small_model_name(model_name)
        self._capture_status_snapshot_from_harness()
        self._refresh_status()

    async def _maybe_restore_harness_state(self) -> dict[str, Any] | None:
        if self.fresh_run or not self.restore_graph_state_on_startup or self.harness is None:
            return None
        cwd = str(getattr(getattr(self.harness, "state", None), "cwd", ".") or ".")
        requested_thread_id = str(self.restore_thread_id or "").strip()
        bridge = getattr(self, "_harness_bridge", None)
        if bridge is not None:
            fallback_ui_transcript: list[dict[str, Any]] = []
            result = await bridge.restore_graph_state(thread_id=self.restore_thread_id)
            restored = bool(result.get("restored"))
            if not restored:
                fallback = self._resolve_saved_chat_state(cwd=cwd, thread_id=requested_thread_id)
                if fallback is None:
                    return {
                        "status": "not_found",
                        "thread_id": self.restore_thread_id,
                    }
                fallback_thread_id, state_payload = fallback
                fallback_ui_transcript = load_chat_session_ui_transcript(
                    cwd=cwd,
                    thread_id=fallback_thread_id,
                )
                result = await bridge.replace_state_from_payload(state_payload)
                restored = True
                self.restore_thread_id = fallback_thread_id
            snapshot = result.get("snapshot")
            if isinstance(snapshot, dict):
                self._latest_status_snapshot = dict(snapshot)
            payload: dict[str, Any] = {
                "status": "restored",
                "thread_id": str(result.get("thread_id") or self.restore_thread_id or ""),
                "has_pending_interrupt": bool(result.get("has_pending_interrupt")),
            }
            restored_thread_id = str(payload.get("thread_id") or "").strip()
            checkpoint_ui_transcript = (
                []
                if fallback_ui_transcript or not restored_thread_id
                else load_chat_session_ui_transcript(cwd=cwd, thread_id=restored_thread_id)
            )
            restored_messages = result.get("recent_messages")
            if isinstance(restored_messages, list):
                payload["recent_messages"] = restored_messages
            ui_transcript = fallback_ui_transcript or checkpoint_ui_transcript
            if ui_transcript:
                payload["ui_transcript"] = ui_transcript
            interrupt = result.get("interrupt")
            if isinstance(interrupt, dict):
                payload["interrupt"] = interrupt
            return payload

        restored = self.harness.restore_graph_state(thread_id=self.restore_thread_id)
        fallback_ui_transcript: list[dict[str, Any]] = []
        if not restored:
            fallback = self._resolve_saved_chat_state(cwd=cwd, thread_id=requested_thread_id)
            if fallback is None:
                return {
                    "status": "not_found",
                    "thread_id": self.restore_thread_id,
                }
            fallback_thread_id, state_payload = fallback
            fallback_ui_transcript = load_chat_session_ui_transcript(
                cwd=cwd,
                thread_id=fallback_thread_id,
            )
            from ..state import LoopState

            self.harness.state = LoopState.from_dict(state_payload)
            if isinstance(getattr(self.harness.state, "scratchpad", None), dict):
                self.harness.state.scratchpad["_session_restored"] = True
                self.harness.state.scratchpad["_resume_contract"] = {
                    "kind": "chat_session_resume",
                    "thread_id": str(getattr(self.harness.state, "thread_id", "") or fallback_thread_id),
                }
            self.harness._sync_run_logger_session_id()
        payload: dict[str, Any] = {
            "status": "restored",
            "thread_id": self.harness.state.thread_id,
            "has_pending_interrupt": self.harness.has_pending_interrupt(),
        }
        payload["recent_messages"] = _serialize_recent_messages(self.harness.state)
        restored_thread_id = str(payload.get("thread_id") or "").strip()
        checkpoint_ui_transcript = (
            []
            if fallback_ui_transcript or not restored_thread_id
            else load_chat_session_ui_transcript(cwd=cwd, thread_id=restored_thread_id)
        )
        ui_transcript = fallback_ui_transcript or checkpoint_ui_transcript
        if ui_transcript:
            payload["ui_transcript"] = ui_transcript
        interrupt = self.harness.get_pending_interrupt()
        if interrupt is not None:
            payload["interrupt"] = interrupt
        return payload

    @staticmethod
    def _resolve_saved_chat_state(
        *,
        cwd: str,
        thread_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        candidate_ids: list[str] = []
        if thread_id:
            candidate_ids.append(thread_id)
        else:
            summaries = load_chat_session_summaries(cwd=cwd)
            candidate_ids.extend(summary.thread_id for summary in summaries)
        seen: set[str] = set()
        for candidate_id in candidate_ids:
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            state_payload = load_chat_session_state(cwd=cwd, thread_id=candidate_id)
            if isinstance(state_payload, dict):
                return candidate_id, state_payload
        return None

    @staticmethod
    def _format_restore_status(status: dict[str, Any]) -> str:
        return format_restore_status(status)

    def _sanitize_visible_text_event(self, event: UIEvent) -> UIEvent | None:
        if event.event_type not in {UIEventType.ASSISTANT, UIEventType.THINKING}:
            return event
        cleaned, extracted = sanitize_assistant_content_for_history(
            str(event.content or ""),
            strip_result=False,
        )
        if event.event_type == UIEventType.ASSISTANT:
            if not cleaned and extracted:
                return None
            if re.fullmatch(
                r"\s*(?:thought|thinking|analysis|reasoning)\s*",
                cleaned,
                flags=re.IGNORECASE,
            ):
                return None
            return UIEvent(
                event_type=event.event_type,
                content=cleaned,
                data=dict(event.data),
                timestamp=event.timestamp,
            )
        if not str(event.content or "").strip():
            return None
        return event

    def _is_loop_halt_placeholder(self, event: UIEvent) -> bool:
        if event.event_type != UIEventType.ASSISTANT:
            return False
        if event.data.get("kind") != "replace":
            return False
        return str(event.content or "").strip() == "[Previous assistant output was halted because it entered a repetition loop.]"

    def _is_model_output_loop_event(self, event: UIEvent) -> bool:
        data = event.data if isinstance(event.data, dict) else {}
        return (
            str(data.get("ui_kind") or "") == "model_output_degenerate_loop_exhausted"
            or str(data.get("event") or "") == "model_output_degenerate_loop_exhausted"
        )

    def _prune_trailing_assistant_transcript_events(self) -> bool:
        transcript = getattr(self, "_ui_transcript", None)
        if not isinstance(transcript, list):
            return False
        barrier_types = {
            UIEventType.USER.value,
            UIEventType.TOOL_CALL.value,
            UIEventType.TOOL_RESULT.value,
        }
        remove_indexes: list[int] = []
        for index in range(len(transcript) - 1, -1, -1):
            item = transcript[index]
            if not isinstance(item, dict):
                continue
            event_type = str(item.get("event_type") or "")
            if event_type in barrier_types:
                break
            if event_type == UIEventType.ASSISTANT.value:
                remove_indexes.append(index)
        for index in remove_indexes:
            del transcript[index]
        return bool(remove_indexes)

    def _record_ui_transcript_event(self, event: UIEvent) -> None:
        if event.event_type == UIEventType.STATUS:
            return
        if event.event_type == UIEventType.SHELL_STREAM:
            return
        pruned_for_loop = False
        if self._is_model_output_loop_event(event):
            pruned_for_loop = self._prune_trailing_assistant_transcript_events()
        if self._is_loop_halt_placeholder(event):
            if pruned_for_loop:
                self._schedule_ui_transcript_persist()
            return
        event = self._sanitize_visible_text_event(event)
        if event is None:
            return
        transcript = getattr(self, "_ui_transcript", None)
        if not isinstance(transcript, list):
            transcript = []
            self._ui_transcript = transcript
        event_dict = event.to_dict()
        if self._coalesce_stream_transcript_event(transcript, event_dict):
            self._schedule_ui_transcript_persist()
            return
        fingerprint = self._ui_transcript_fingerprint(event_dict)
        recent_fingerprints = getattr(self, "_ui_transcript_recent_fingerprints", None)
        if not isinstance(recent_fingerprints, list):
            recent_fingerprints = []
            self._ui_transcript_recent_fingerprints = recent_fingerprints
        if fingerprint in recent_fingerprints:
            return
        recent_fingerprints.append(fingerprint)
        del recent_fingerprints[:-80]
        transcript.append(event_dict)
        del transcript[:-500]
        self._schedule_ui_transcript_persist()

    @staticmethod
    def _coalesce_stream_transcript_event(
        transcript: list[dict[str, Any]],
        event: dict[str, Any],
    ) -> bool:
        event_type = str(event.get("event_type") or "")
        if event_type not in {UIEventType.ASSISTANT.value, UIEventType.THINKING.value}:
            return False
        if not transcript:
            return False
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if data.get("kind") or data.get("promoted_from"):
            return False
        previous = transcript[-1]
        if str(previous.get("event_type") or "") != event_type:
            return False
        previous_data = previous.get("data") if isinstance(previous.get("data"), dict) else {}
        for key in ("kind", "promoted_from", "ui_kind", "tool_call_id", "trace_id", "task_id"):
            if (previous_data.get(key) or "") != (data.get(key) or ""):
                return False
        previous["content"] = str(previous.get("content") or "") + str(event.get("content") or "")
        if event.get("timestamp"):
            previous["timestamp"] = event["timestamp"]
        return True

    @staticmethod
    def _ui_transcript_fingerprint(event: dict[str, Any]) -> str:
        data = event.get("data") if isinstance(event, dict) else {}
        payload = data.get("event_payload") if isinstance(data, dict) else None
        key = {
            "event_type": event.get("event_type"),
            "content": event.get("content"),
            "ui_kind": data.get("ui_kind") if isinstance(data, dict) else "",
            "tool_call_id": data.get("tool_call_id") if isinstance(data, dict) else "",
            "payload": payload if isinstance(payload, dict) else {},
        }
        raw = json.dumps(key, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _schedule_ui_transcript_persist(self) -> None:
        delay = getattr(self, "_ui_transcript_debounce_seconds", None)
        if delay is None:
            self._persist_ui_transcript()
            return
        try:
            delay_seconds = max(0.0, float(delay))
        except (TypeError, ValueError):
            delay_seconds = 0.25
        handle = getattr(self, "_ui_transcript_persist_handle", None)
        if handle is not None and not handle.cancelled():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._persist_ui_transcript()
            return
        self._ui_transcript_persist_handle = loop.call_later(
            delay_seconds,
            self._flush_ui_transcript_persist,
        )

    def _flush_ui_transcript_persist(self) -> None:
        handle = getattr(self, "_ui_transcript_persist_handle", None)
        if handle is not None and not handle.cancelled():
            handle.cancel()
        self._ui_transcript_persist_handle = None
        self._persist_ui_transcript()

    def _persist_ui_transcript(self) -> None:
        harness = getattr(self, "harness", None)
        if harness is None:
            return
        state = getattr(harness, "state", None)
        cwd = str(getattr(state, "cwd", "") or "").strip()
        thread_id = str(getattr(state, "thread_id", "") or "").strip()
        transcript = getattr(self, "_ui_transcript", None)
        if not cwd or not thread_id or not isinstance(transcript, list):
            return
        persist_chat_session_ui_transcript(
            cwd=cwd,
            thread_id=thread_id,
            ui_transcript=transcript,
        )

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
        if status == "chat_completed" and active_text:
            return
        if active_text and check_duplicate_promotion(final_text.lower(), active_text.lower()):
            return
        event = UIEvent(
            event_type=UIEventType.ASSISTANT,
            content=final_text,
            data={"promoted_from": "terminal_result"},
        )
        self._record_ui_transcript_event(event)
        await console.append_event(event)

    async def _append_system_line(self, text: str, *, force: bool = False, kind: str = "system") -> None:
        if not force and not self._show_system_messages:
            return
        console = self._get_console()
        if console is not None:
            self._record_ui_transcript_event(
                UIEvent(event_type=UIEventType.SYSTEM, content=text, data={"kind": kind})
            )
            await console.append_line(text, kind=kind)
    def _set_activity(self, text: str | None) -> None:
        self._status_activity = str(text or "").strip()

    def _tick_activity_timer(self) -> None:
        if self._task_start_time is None:
            return
        elapsed = time.monotonic() - self._task_start_time
        self._set_activity(f"[thinking {elapsed:.1f}s]")
        self._refresh_status()
        console = self._get_console()
        if console is not None:
            import asyncio
            asyncio.create_task(console.update_thinking_indicator())

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
            show_system_messages=self._show_system_messages or self._verbose,
            show_tool_calls=self._show_tool_calls,
        )


def _terminal_status_detail(result: dict[str, Any], harness: Any) -> str:
    status = str(result.get("status") or "").strip().lower()
    if status != "stopped":
        return ""
    reason = str(result.get("reason") or "").strip().lower()
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if isinstance(scratchpad, dict) and isinstance(scratchpad.get("_install_source_invalid_blocker"), dict):
        return "Blocked: installer source invalid or unavailable. Research a current official install path or get approval for an alternate/manual path."
    if reason == "no_tool_calls":
        return "Stopped: no valid next tool call."
    return f"Stopped: {reason}." if reason else "Stopped."
