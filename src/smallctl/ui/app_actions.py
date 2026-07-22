from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ..chat_sessions import load_chat_session_state, load_chat_session_summaries, load_chat_session_ui_transcript
from ..logging_utils import log_kv
from ..models.events import UIEvent, UIEventType
from ..provider_profiles import supported_provider_profiles
from .chat_selector import ChatMenuScreen, ChatSessionSelectScreen
from .console import ConsolePane
from .approval import ApprovePromptScreen, ShellApprovalDecision
from .display import format_tool_call_for_display
from .model_selector import ModelSelectScreen, ProviderSelectScreen
from .statusbar import StatusBar


def _restored_speaker_data(metadata: dict[str, Any]) -> dict[str, Any]:
    speaker = str(metadata.get("speaker") or "").strip().lower()
    return {"speaker": speaker} if speaker else {}


def _restored_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    restored: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            function = {}
        name = str(function.get("name") or item.get("name") or "").strip()
        if not name:
            continue
        args = _parse_restored_tool_args(function.get("arguments", item.get("arguments")))
        tool_call_id = str(item.get("id") or item.get("tool_call_id") or "").strip()
        restored.append(
            {
                "name": name,
                "data": {
                    "tool_name": name,
                    "tool_call_id": tool_call_id,
                    "args": args,
                    "display_text": format_tool_call_for_display(name, args),
                },
            }
        )
    return restored


def _tool_call_only_summary(tool_calls: list[dict[str, Any]]) -> str:
    names = [str(call.get("name") or "").strip() for call in tool_calls]
    names = [name for name in names if name]
    if not names:
        return "Calling tool..."
    if len(names) == 1:
        return f"Calling {names[0]}..."
    return "Calling " + ", ".join(names[:3]) + ("..." if len(names) > 3 else "...")


def _parse_restored_tool_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class SmallctlAppActionsMixin:
    async def action_clear_console(self) -> None:
        console = self.query_one(ConsolePane)
        await console.clear_bubbles()

    async def action_cancel_task(self) -> None:
        self._dismiss_active_approval_prompt()
        bridge = getattr(self, "_harness_bridge", None)
        if bridge is not None:
            bridge.cancel(source="ui_stop_button")
            bridge.abort()
        elif self.harness is not None:
            self.harness.cancel(source="ui_stop_button")
        active_task = self.active_task
        if active_task is not None and not active_task.done():
            current = asyncio.current_task()
            is_cancelling = bool(
                callable(getattr(active_task, "cancelling", None))
                and active_task.cancelling()
            )
            if active_task is current or is_cancelling:
                log_kv(
                    self._app_logger,
                    logging.WARNING,
                    "ui_cancel_skipped_self_or_cancelling",
                    task_id=id(active_task),
                    task_name=getattr(active_task, "get_name", lambda: "?")(),
                )
            else:
                await asyncio.sleep(0.05)
                if active_task is not self.active_task or active_task.done():
                    active_task = self.active_task
                if active_task is not None and not active_task.done():
                    try:
                        active_task.cancel()
                        log_kv(self._app_logger, logging.INFO, "ui_task_cancelled")
                    except RecursionError:
                        log_kv(
                            self._app_logger,
                            logging.ERROR,
                            "ui_cancel_recursion_error",
                            task_id=id(active_task),
                            task_name=getattr(active_task, "get_name", lambda: "?")(),
                        )
            if bridge is not None:
                terminated = await bridge.wait_for_idle(timeout=5.0)
                if not terminated:
                    log_kv(
                        self._app_logger,
                        logging.WARNING,
                        "ui_cancel_run_still_in_flight",
                    )
            self._refresh_status(step_override="cancelled")
            return
        console = self._get_console()
        if console is not None:
            await self._append_system_line("No active task.", kind="cancel")

    def _dismiss_active_approval_prompt(self) -> None:
        prompt = self._active_approval_prompt
        if prompt is None:
            return
        self._active_approval_prompt = None
        try:
            if isinstance(prompt, ApprovePromptScreen):
                prompt.dismiss(ShellApprovalDecision(False, False, cancelled=True))
            else:
                prompt.dismiss(False)
        except Exception:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in {"stop-button", "stop-button-sidebar"}:
            await self.action_cancel_task()
            return
        if event.button.id in {"model-button", "model-button-sidebar"}:
            await self.action_open_model_selector()
            return
        if event.button.id in {"chat-button", "chat-button-sidebar"}:
            await self.action_open_chat_selector()
            return
        if event.button.id in {"model-bar-toggle", "model-bar-toggle-sidebar"}:
            self.action_toggle_model_bar_layout()
            return
        if event.button.id == "goal-bar-toggle":
            self.action_toggle_goal_bar()

    def action_toggle_goal_bar(self) -> None:
        self._goal_bar_expanded = not bool(getattr(self, "_goal_bar_expanded", False))
        self._refresh_goal_bar()

    def action_toggle_model_bar_layout(self) -> None:
        self._model_bar_layout = "right" if self._model_bar_layout == "bottom" else "bottom"
        self._apply_model_bar_layout()

    def _apply_model_bar_layout(self) -> None:
        right = self._model_bar_layout == "right"
        for selector in ("#status-row", "#model-sidebar"):
            try:
                widget = self.query_one(selector)
            except Exception:
                continue
            hidden = (selector == "#status-row" and right) or (selector == "#model-sidebar" and not right)
            widget.set_class(hidden, "hidden")
        for selector, label in (
            ("#model-bar-toggle", "bar: bottom"),
            ("#model-bar-toggle-sidebar", "bar: right"),
        ):
            try:
                self.query_one(selector, Button).label = label
            except Exception:
                pass
        try:
            for bar in self.query(StatusBar):
                bar._refresh_display()
                bar.refresh(layout=True)
        except Exception:
            pass

    async def action_open_model_selector(self) -> None:
        if self.active_task and not self.active_task.done():
            message = "Model can be changed after the active task finishes."
            self.notify(message, title="Model Selector", timeout=2.5, markup=False)
            await self._append_system_line(message, force=True)
            return

        harness = self.harness
        client = getattr(harness, "client", None)
        base_url = str(
            getattr(client, "base_url", None) or self.harness_config.endpoint or ""
        ).strip()
        current_model = str(
            getattr(client, "model", None) or self.harness_config.model or ""
        ).strip()
        api_key = getattr(client, "api_key", None) or self.harness_config.api_key
        provider_profile = str(
            getattr(client, "provider_profile", None)
            or getattr(harness, "provider_profile", None)
            or self.harness_config.provider_profile
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
        if choice == "provider":
            await self.action_open_provider_selector()
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
        await self._create_harness()
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
                    if isinstance(getattr(self.harness.state, "scratchpad", None), dict):
                        self.harness.state.scratchpad["_session_restored"] = True
                        self.harness.state.scratchpad["_resume_contract"] = {
                            "kind": "chat_session_resume",
                            "thread_id": str(getattr(self.harness.state, "thread_id", "") or thread_id),
                        }
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
        if self.harness is not None:
            ui_transcript = load_chat_session_ui_transcript(
                cwd=getattr(self.harness.state, "cwd", "."),
                thread_id=thread_id,
            )
            self._ui_transcript = [dict(item) for item in ui_transcript if isinstance(item, dict)]
        await self._render_restored_chat(messages=restored_messages if isinstance(restored_messages, list) else None)
        await self._append_system_line(f"Restored chat session {thread_id}.", force=True)

    async def _render_restored_chat(self, messages: list[dict[str, Any]] | None = None) -> None:
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
                    "content": "" if getattr(message, "content", None) is None else str(getattr(message, "content")),
                    "name": str(getattr(message, "name", "") or ""),
                    "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
                    "tool_calls": getattr(message, "tool_calls", []) or [],
                    "metadata": getattr(message, "metadata", {}) or {},
                }
                for message in (
                    getattr(harness.state, "transcript_messages", None)
                    or getattr(harness.state, "recent_messages", [])
                    or []
                )
            ]
        else:
            source_messages = list(messages)
        for message in source_messages:
            is_ui_transcript_event = "event_type" in message
            role = str(message.get("role") or message.get("event_type") or "").strip().lower()
            content = str(message.get("content") or "")
            show_tool_calls = bool(getattr(self, "_show_tool_calls", True))
            show_system_messages = bool(
                getattr(self, "_show_system_messages", True) or getattr(self, "_verbose", False)
            )
            if not show_tool_calls and role in {"tool", "tool_call", "tool_result", "shell_stream"}:
                continue
            metadata = message.get("metadata")
            if not isinstance(metadata, dict):
                data = message.get("data")
                metadata = data if isinstance(data, dict) else {}
            if metadata.get("hidden_from_ui") or metadata.get("ui_hidden"):
                continue
            speaker_data = _restored_speaker_data(metadata)
            if role == "user":
                if content.strip():
                    await console.append_event(UIEvent(UIEventType.USER, content.strip()))
            elif role == "assistant":
                tool_calls = _restored_tool_calls(message.get("tool_calls"))
                if content.strip():
                    await console.append_event(
                        UIEvent(UIEventType.ASSISTANT, content, data=speaker_data)
                    )
                elif tool_calls and show_tool_calls:
                    await console.append_event(
                        UIEvent(
                            UIEventType.ASSISTANT,
                            _tool_call_only_summary(tool_calls),
                            data={**speaker_data, "synthetic_tool_call_summary": True},
                        )
                    )
                if show_tool_calls:
                    for tool_call in tool_calls:
                        await console.append_event(
                            UIEvent(
                                UIEventType.TOOL_CALL,
                                tool_call["name"],
                                data={**speaker_data, **tool_call["data"]},
                            )
                        )
            elif role in {"thinking", "reasoning"}:
                if content.strip():
                    await console.append_event(
                        UIEvent(UIEventType.THINKING, content, data=speaker_data)
                    )
            elif role == "tool":
                if content.strip():
                    tool_name = str(message.get("name") or metadata.get("tool_name") or "").strip()
                    tool_call_id = str(message.get("tool_call_id") or "").strip()
                    await console.append_event(
                        UIEvent(
                            UIEventType.TOOL_RESULT,
                            content,
                            data={
                                **speaker_data,
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                **metadata,
                            },
                        )
                    )
            elif role == "tool_call":
                await console.append_event(
                    UIEvent(UIEventType.TOOL_CALL, content, data={**speaker_data, **metadata})
                )
            elif role == "tool_result":
                if content.strip():
                    await console.append_event(
                        UIEvent(UIEventType.TOOL_RESULT, content, data={**speaker_data, **metadata})
                    )
            elif role == "system":
                is_recovery_system = bool(
                    metadata.get("is_recovery_nudge")
                    or metadata.get("recovery_kind")
                    or metadata.get("ui_kind")
                    or metadata.get("event")
                )
                if not show_system_messages and not is_recovery_system:
                    continue
                if content.strip() and (is_ui_transcript_event or is_recovery_system):
                    await console.append_event(
                        UIEvent(
                            UIEventType.SYSTEM,
                            content,
                            data={**speaker_data, **metadata},
                        )
                    )

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

    async def action_open_provider_selector(self) -> None:
        if self.active_task and not self.active_task.done():
            message = "Provider can be changed after the active task finishes."
            self.notify(message, title="Provider Selector", timeout=2.5, markup=False)
            await self._append_system_line(message, force=True)
            return

        harness = self.harness
        client = getattr(harness, "client", None)
        current_profile = str(
            getattr(client, "provider_profile", None)
            or getattr(harness, "provider_profile", None)
            or self.harness_config.provider_profile
            or "auto"
        ).strip()
        current_endpoint = str(
            getattr(client, "base_url", None) or self.harness_config.endpoint or ""
        ).strip()
        current_api_key = getattr(client, "api_key", None) or self.harness_config.api_key

        profiles = supported_provider_profiles()
        screen = ProviderSelectScreen(
            profiles=profiles,
            current_profile=current_profile,
            current_endpoint=current_endpoint,
            current_api_key=current_api_key,
        )
        await self.push_screen(
            screen,
            callback=lambda choice: asyncio.create_task(
                self._handle_provider_selection_result(choice)
            ),
        )

    async def _handle_provider_selection_result(self, choice: dict[str, str] | None) -> None:
        if not choice:
            return
        profile = str(choice.get("profile") or "").strip()
        endpoint = str(choice.get("endpoint") or "").strip()
        api_key = str(choice.get("api_key") or "").strip() or None
        if not profile:
            return

        self.harness_config.provider_profile = profile
        if endpoint:
            self.harness_config.endpoint = endpoint
        if api_key is not None:
            self.harness_config.api_key = api_key

        harness = self.harness
        if harness is not None:
            old_state = harness.state
            await self._shutdown_harness()
            await self._create_harness()
            if self.harness is not None:
                self.harness.state = old_state
                self.harness.state.scratchpad["_provider_profile"] = profile
                if endpoint:
                    self.harness.state.scratchpad["_provider_endpoint"] = endpoint
                self.harness._sync_run_logger_session_id()
        await self._append_system_line(
            f"Provider switched to {profile}" + (f" at {endpoint}" if endpoint else "."),
            force=True,
        )
        self._capture_status_snapshot_from_harness()
        self._refresh_status()
        await self.action_open_model_selector()

    async def action_interrupt_or_quit(self) -> None:
        if self.active_task and not self.active_task.done():
            await self.action_cancel_task()
            return

        def _on_result(quit: bool | None) -> None:
            if quit:
                self.closed_by_ctrl_c = True
                self.exit()

        await self.push_screen(QuitConfirmScreen(), callback=_on_result)

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
        await self._create_harness()
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
        active_task = getattr(self, "active_task", None)
        if active_task is not None and not active_task.done():
            return
        if getattr(self, "harness", None) is None:
            return
        await self._render_restored_chat()

    def _scroll_console(self, delta: int) -> None:
        console = self._get_console()
        if console is None:
            return
        started = time.perf_counter()
        try:
            console.scroll_relative(y=delta, animate=False)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if elapsed_ms > 5:
                log_kv(
                    self._app_logger,
                    logging.DEBUG,
                    "ui_scroll_event",
                    delta=delta,
                    scroll_y=getattr(console, "scroll_y", None),
                    max_scroll_y=getattr(console, "max_scroll_y", None),
                    elapsed_ms=elapsed_ms,
                    bubble_widget_count=getattr(console, "_bubble_stack_widget_count", lambda: 0)(),
                )
            else:
                log_kv(
                    self._app_logger,
                    logging.DEBUG,
                    "ui_scroll_event",
                    delta=delta,
                    scroll_y=getattr(console, "scroll_y", None),
                    max_scroll_y=getattr(console, "max_scroll_y", None),
                    elapsed_ms=elapsed_ms,
                )

    def action_scroll_up(self) -> None:
        log_kv(self._app_logger, logging.DEBUG, "ui_scroll_key", key="ctrl+up", direction="up")
        self._scroll_console(-3)

    def action_scroll_down(self) -> None:
        log_kv(self._app_logger, logging.DEBUG, "ui_scroll_key", key="ctrl+down", direction="down")
        self._scroll_console(3)

    def action_scroll_page_up(self) -> None:
        log_kv(self._app_logger, logging.DEBUG, "ui_scroll_key", key="pageup", direction="page_up")
        console = self._get_console()
        if console is None:
            return
        step = max(1, console.size.height - 2)
        self._scroll_console(-step)

    def action_scroll_page_down(self) -> None:
        log_kv(self._app_logger, logging.DEBUG, "ui_scroll_key", key="pagedown", direction="page_down")
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


class QuitConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="quit-confirm-overlay"):
            with Container(id="quit-confirm-dialog"):
                with Vertical(id="quit-confirm-content"):
                    yield Static("Quit smallctl?", id="quit-confirm-title")
                    with Horizontal(id="quit-confirm-buttons"):
                        yield Button("Quit", id="quit-confirm-yes", variant="error")
                        yield Button("Cancel", id="quit-confirm-no", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#quit-confirm-no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-confirm-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
