from __future__ import annotations

import asyncio
import logging

from textual.widgets import Button

from ..chat_sessions import load_chat_session_state, load_chat_session_summaries
from ..logging_utils import log_kv
from ..models.events import UIEvent, UIEventType
from .chat_selector import ChatMenuScreen, ChatSessionSelectScreen
from .console import ConsolePane
from .model_selector import ModelSelectScreen


class SmallctlAppActionsMixin:
    async def action_clear_console(self) -> None:
        console = self.query_one(ConsolePane)
        await console.clear_bubbles()

    async def action_cancel_task(self) -> None:
        self._dismiss_active_approval_prompt()
        bridge = getattr(self, "_harness_bridge", None)
        if bridge is not None:
            bridge.cancel()
        elif self.harness is not None:
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
            return
        if event.button.id == "model-button":
            await self.action_open_model_selector()
            return
        if event.button.id == "chat-button":
            await self.action_open_chat_selector()

    async def action_open_model_selector(self) -> None:
        if self.active_task and not self.active_task.done():
            message = "Model can be changed after the active task finishes."
            self.notify(message, title="Model Selector", timeout=2.5, markup=False)
            await self._append_system_line(message, force=True)
            return

        harness = self.harness
        client = getattr(harness, "client", None)
        base_url = str(
            getattr(client, "base_url", None) or self.harness_kwargs.get("endpoint") or ""
        ).strip()
        current_model = str(
            getattr(client, "model", None) or self.harness_kwargs.get("model") or ""
        ).strip()
        api_key = getattr(client, "api_key", None)
        if api_key is None:
            api_key = self.harness_kwargs.get("api_key")
        provider_profile = str(
            getattr(client, "provider_profile", None)
            or getattr(harness, "provider_profile", None)
            or self.harness_kwargs.get("provider_profile")
            or "generic"
        ).strip()

        if not base_url:
            await self._append_system_line("Cannot list models: no provider endpoint is configured.", force=True)
            return

        screen = ModelSelectScreen(
            base_url=base_url,
            api_key=api_key,
            provider_profile=provider_profile,
            current_model=current_model,
        )
        await self._push_model_selector(screen, current_model=current_model)

    async def action_open_chat_selector(self) -> None:
        if self.active_task and not self.active_task.done():
            message = "Chat can be changed after the active task finishes."
            self.notify(message, title="Chat", timeout=2.5, markup=False)
            await self._append_system_line(message, force=True)
            return

        await self.push_screen(
            ChatMenuScreen(),
            callback=lambda choice: asyncio.create_task(self._handle_chat_menu_result(choice)),
        )

    async def _handle_chat_menu_result(self, choice: str | None) -> None:
        if choice == "new":
            await self.action_new_conversation()
            return
        if choice != "resume":
            return

        harness = self.harness
        if harness is None:
            await self._append_system_line("Cannot resume chat: harness is not ready.", force=True)
            return

        checkpointer = getattr(harness, "_graph_checkpointer", None)
        if checkpointer is None:
            try:
                from ..graph.runtime_payloads import get_runtime_checkpointer

                checkpointer = get_runtime_checkpointer(harness)
            except Exception:
                checkpointer = None

        sessions = load_chat_session_summaries(
            cwd=getattr(harness.state, "cwd", "."),
            checkpointer=checkpointer,
        )
        await self.push_screen(
            ChatSessionSelectScreen(sessions=sessions),
            callback=lambda thread_id: asyncio.create_task(self._handle_chat_session_selection(thread_id)),
        )

    async def _handle_chat_session_selection(self, thread_id: str | None) -> None:
        selected_thread_id = str(thread_id or "").strip()
        if not selected_thread_id:
            return
        await self._resume_chat_session(selected_thread_id)

    async def _resume_chat_session(self, thread_id: str) -> None:
        if self.active_task and not self.active_task.done():
            await self._append_system_line("Cannot resume a chat while task is running.", force=True)
            return
        old_harness = self.harness
        old_bridge = getattr(self, "_harness_bridge", None)
        old_checkpointer = getattr(old_harness, "_graph_checkpointer", None) if old_harness is not None else None
        if old_bridge is not None:
            await old_bridge.shutdown()
            self._harness_bridge = None
        elif old_harness is not None:
            await old_harness.teardown()
        self.restore_thread_id = thread_id
        self._create_harness()
        if old_checkpointer is not None and self.harness is not None:
            setattr(self.harness, "_graph_checkpointer", old_checkpointer)
        bridge = getattr(self, "_harness_bridge", None)
        restore_result: dict[str, object] | None = None
        if bridge is not None:
            restore_result = await bridge.restore_graph_state(thread_id=thread_id)
            restored = bool(restore_result.get("restored"))
        else:
            restored = bool(self.harness and self.harness.restore_graph_state(thread_id=thread_id))
        if not restored and self.harness is not None:
            state_payload = load_chat_session_state(
                cwd=getattr(self.harness.state, "cwd", "."),
                thread_id=thread_id,
            )
            if isinstance(state_payload, dict):
                if bridge is not None:
                    restore_result = await bridge.replace_state_from_payload(state_payload)
                else:
                    from ..state import LoopState

                    self.harness.state = LoopState.from_dict(state_payload)
                    self.harness._sync_run_logger_session_id()
                restored = True
        snapshot = restore_result.get("snapshot") if isinstance(restore_result, dict) else None
        restored_messages = restore_result.get("recent_messages") if isinstance(restore_result, dict) else None
        if isinstance(snapshot, dict):
            self._latest_status_snapshot = dict(snapshot)
            self._refresh_status(step_override=0, snapshot=snapshot)
        else:
            self._capture_status_snapshot_from_harness()
            self._refresh_status(step_override=0)
        if not restored:
            await self._append_system_line(f"Could not restore chat session {thread_id}.", force=True)
            return
        await self._render_restored_chat(messages=restored_messages if isinstance(restored_messages, list) else None)
        await self._append_system_line(f"Restored chat session {thread_id}.", force=True)

    async def _render_restored_chat(self, messages: list[dict[str, str]] | None = None) -> None:
        console = self._get_console()
        harness = self.harness
        if console is None:
            return
        await console.clear_bubbles()
        if messages is None:
            if harness is None:
                return
            source_messages = [
                {
                    "role": str(getattr(message, "role", "") or "").strip().lower(),
                    "content": str(getattr(message, "content", "") or "").strip(),
                }
                for message in (getattr(harness.state, "recent_messages", []) or [])
            ]
        else:
            source_messages = list(messages)
        for message in source_messages:
            role = str(message.get("role") or "").strip().lower()
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                await console.append_event(UIEvent(UIEventType.USER, content))
            elif role == "assistant":
                await console.append_event(UIEvent(UIEventType.ASSISTANT, content))
            elif role == "system":
                await console.append_event(UIEvent(UIEventType.SYSTEM, content))

    async def _push_model_selector(
        self,
        screen: ModelSelectScreen,
        *,
        current_model: str,
    ) -> None:
        def _resolve(selection: str | None) -> None:
            asyncio.create_task(
                self._handle_model_selection_result(selection, current_model=current_model)
            )

        await self.push_screen(screen, callback=_resolve)

    async def _handle_model_selection_result(
        self,
        selected: str | None,
        *,
        current_model: str,
    ) -> None:
        if not selected or selected == current_model:
            return
        await self._switch_model(selected)
        await self._append_system_line(f"Model switched to {selected}.", force=True)

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
        old_harness = self.harness
        old_bridge = getattr(self, "_harness_bridge", None)
        old_checkpointer = getattr(old_harness, "_graph_checkpointer", None) if old_harness is not None else None
        if old_bridge is not None:
            await old_bridge.shutdown()
            self._harness_bridge = None
        elif old_harness is not None:
            await old_harness.teardown()
        self._create_harness()
        if old_checkpointer is not None and self.harness is not None:
            setattr(self.harness, "_graph_checkpointer", old_checkpointer)
        self._capture_status_snapshot_from_harness()
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
