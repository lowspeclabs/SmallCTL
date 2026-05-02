from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class InputPane(TextArea):
    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    BINDINGS = [
        ("enter", "submit", "Submit"),
        ("ctrl+shift+v", "paste", "Paste"),
        ("shift+insert", "paste", "Paste"),
        ("pageup", "scroll_page_up", "Scroll Up"),
        ("pagedown", "scroll_page_down", "Scroll Down"),
        ("ctrl+up", "scroll_up", "Scroll Up"),
        ("ctrl+down", "scroll_down", "Scroll Down"),
        ("up", "history_prev", "Prev History"),
        ("down", "history_next", "Next History"),
        ("escape", "cancel_pending", "Cancel Input"),
    ]

    def action_submit(self) -> None:
        value = self.text
        self.post_message(self.Submitted(value))

    def on_key(self, event: events.Key) -> None:
        # TextArea can consume Enter in some terminals; force submit on plain Enter.
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.action_submit()

    def action_cancel_pending(self) -> None:
        self.text = ""

    def action_history_prev(self) -> None:
        if hasattr(self.app, "history_prev"):
            self.text = self.app.history_prev()
            self.cursor_location = self._end_location()

    def action_history_next(self) -> None:
        if hasattr(self.app, "history_next"):
            self.text = self.app.history_next()
            self.cursor_location = self._end_location()

    def action_scroll_up(self) -> None:
        if hasattr(self.app, "_scroll_console"):
            self.app._scroll_console(-3)

    def action_scroll_down(self) -> None:
        if hasattr(self.app, "_scroll_console"):
            self.app._scroll_console(3)

    def action_scroll_page_up(self) -> None:
        if hasattr(self.app, "action_scroll_page_up"):
            self.app.action_scroll_page_up()

    def action_scroll_page_down(self) -> None:
        if hasattr(self.app, "action_scroll_page_down"):
            self.app.action_scroll_page_down()

    def _end_location(self) -> tuple[int, int]:
        lines = self.text.split("\n")
        if not lines:
            return (0, 0)
        return (len(lines) - 1, len(lines[-1]))
