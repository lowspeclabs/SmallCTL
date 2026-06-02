from __future__ import annotations

import asyncio

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class InputPane(TextArea):
    class Submitted(Message):
        def __init__(self, value: str, display_value: str = "") -> None:
            super().__init__()
            self.value = value
            self.display_value = display_value or value

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._raw_text_override: str | None = None

    def action_submit(self) -> None:
        value = self._raw_text_override if self._raw_text_override is not None else self.text
        display_value = self.text
        self._raw_text_override = None
        self.post_message(self.Submitted(value, display_value=display_value))

    def action_paste(self) -> None:
        if self.read_only:
            return
        self._paste_text(str(self.app.clipboard or ""))

    def _paste_text(self, value: str) -> None:
        if not value:
            return
        raw_source = self._raw_text_override if self._raw_text_override is not None else self.text
        raw_text = _replace_text_range(raw_source, value, *self.selection)
        self._raw_text_override = raw_text
        self.text = _format_paste_preview(raw_text)
        self.cursor_location = self._end_location()

    def on_key(self, event: events.Key) -> None:
        # TextArea can consume Enter in some terminals; force submit on plain Enter.
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.action_submit()
            return
        self._raw_text_override = None

    def action_cancel_pending(self) -> None:
        self.text = ""
        self._raw_text_override = None

    def action_history_prev(self) -> None:
        if hasattr(self.app, "history_prev"):
            self._raw_text_override = None
            self.text = self.app.history_prev()
            self.cursor_location = self._end_location()

    def action_history_next(self) -> None:
        if hasattr(self.app, "history_next"):
            self._raw_text_override = None
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

    def action_copy(self) -> None:
        app = self.app
        if (
            hasattr(app, "active_task")
            and getattr(app, "active_task", None) is not None
            and not app.active_task.done()
        ):
            asyncio.create_task(app.action_interrupt_or_quit())
        else:
            super().action_copy()

    def action_delete_to_end_of_line_or_delete_line(self) -> None:
        app = self.app
        if (
            hasattr(app, "active_task")
            and getattr(app, "active_task", None) is not None
            and not app.active_task.done()
        ):
            asyncio.create_task(app.action_cancel_task())
        else:
            super().action_delete_to_end_of_line_or_delete_line()

    def _end_location(self) -> tuple[int, int]:
        lines = self.text.split("\n")
        if not lines:
            return (0, 0)
        return (len(lines) - 1, len(lines[-1]))


def _format_paste_preview(text: str) -> str:
    line_count = text.count("\n") + 1
    if line_count > 3:
        return f"[pasted ~{line_count} lines]"
    if len(text) > 120:
        return f"[pasted ~{len(text)} chars]"
    return text


def _replace_text_range(text: str, insert: str, start: tuple[int, int], end: tuple[int, int]) -> str:
    lines = text.split("\n")
    if not lines:
        lines = [""]
    start_row, start_col = start
    end_row, end_col = end
    start_row = max(0, min(start_row, len(lines) - 1))
    end_row = max(0, min(end_row, len(lines) - 1))
    if (end_row, end_col) < (start_row, start_col):
        start_row, start_col, end_row, end_col = end_row, end_col, start_row, start_col
    start_col = max(0, min(start_col, len(lines[start_row])))
    end_col = max(0, min(end_col, len(lines[end_row])))
    before = lines[start_row][:start_col]
    after = lines[end_row][end_col:]
    replacement = f"{before}{insert}{after}".split("\n")
    return "\n".join(lines[:start_row] + replacement + lines[end_row + 1 :])
