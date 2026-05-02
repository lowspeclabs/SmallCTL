from __future__ import annotations

from datetime import datetime
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListItem, ListView, Static

from ..chat_sessions import ChatSessionSummary, format_relative_age


class ChatSelectButton(Button):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("chat v", compact=True, flat=True, **kwargs)

    def set_busy(self, busy: bool) -> None:
        self.set_class(bool(busy), "chat-button-busy")


class ChatMenuScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="chat-menu-shell"):
            with Vertical(id="chat-menu"):
                yield Static("Chat", id="chat-menu-title")
                yield Static("Choose how to continue.", id="chat-menu-message")
                yield Button("Resume Previous Chat", id="chat-menu-resume", variant="primary", compact=True)
                yield Button("Start New Chat", id="chat-menu-new", variant="success", compact=True)
                yield Button("Cancel", id="chat-menu-cancel", variant="error", compact=True)

    def on_mount(self) -> None:
        self.query_one("#chat-menu-resume", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-menu-resume":
            self.dismiss("resume")
            event.stop()
            return
        if event.button.id == "chat-menu-new":
            self.dismiss("new")
            event.stop()
            return
        if event.button.id == "chat-menu-cancel":
            self.dismiss(None)
            event.stop()


class ChatSessionSelectScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "submit", "Resume"),
    ]

    def __init__(
        self,
        *,
        sessions: list[ChatSessionSummary],
        now: datetime | None = None,
    ) -> None:
        super().__init__()
        self.sessions = list(sessions)
        self.now = now

    def compose(self) -> ComposeResult:
        with Container(id="chat-session-shell"):
            with Vertical(id="chat-session"):
                yield Static("Resume Previous Chat", id="chat-session-title")
                yield Static(self._message_text(), id="chat-session-message")
                yield ListView(id="chat-session-list")
                with Horizontal(id="chat-session-buttons"):
                    yield Button("Resume", id="chat-session-confirm", variant="success", compact=True)
                    yield Button("Cancel", id="chat-session-cancel", variant="error", compact=True)

    async def on_mount(self) -> None:
        list_view = self.query_one("#chat-session-list", ListView)
        await self._render_sessions()
        if self.sessions:
            list_view.focus()
        self.query_one("#chat-session-confirm", Button).disabled = not bool(self.sessions)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        selected = self.query_one("#chat-session-list", ListView).highlighted_child
        thread_id = str(getattr(selected, "thread_id", "") or "").strip()
        if thread_id:
            self.dismiss(thread_id)

    async def _render_sessions(self) -> None:
        list_view = self.query_one("#chat-session-list", ListView)
        await list_view.clear()
        for session in self.sessions:
            item = ListItem(Label(self._format_session_label(session), markup=False))
            setattr(item, "thread_id", session.thread_id)
            await list_view.append(item)
        list_view.index = 0 if self.sessions else None

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        thread_id = str(getattr(event.item, "thread_id", "") or "").strip()
        if thread_id:
            self.dismiss(thread_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-session-confirm":
            self.action_submit()
            event.stop()
            return
        if event.button.id == "chat-session-cancel":
            self.dismiss(None)
            event.stop()

    def _message_text(self) -> str:
        if self.sessions:
            return "Select a saved chat session to resume."
        return "No saved chat sessions found."

    def _format_session_label(self, session: ChatSessionSummary) -> str:
        started_at = session.created_at or session.updated_at
        age = format_relative_age(started_at, now=self.now)
        preview = session.first_user_message or "(no first message)"
        if len(preview) > 72:
            preview = preview[:69].rstrip() + "..."
        parts = [preview, f"started {age}"]
        if session.model:
            parts.append(session.model)
        return " | ".join(parts)
