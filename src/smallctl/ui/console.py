from __future__ import annotations
from typing import Any

from textual.containers import Vertical, VerticalScroll

from ..models.events import UIEvent, UIEventType
from .bubbles import AssistantTurnWidget, BubbleWidget


class ConsolePane(VerticalScroll):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_assistant_turn: AssistantTurnWidget | None = None
        self._last_system_message: str = ""

    async def on_mount(self) -> None:
        await self.mount(Vertical(id="bubble-stack"))

    async def append_line(self, line: str) -> None:
        await self._add_bubble("system", line)
        self._last_system_message = line
        self._schedule_autoscroll()

    async def clear_bubbles(self) -> None:
        stack = self.query_one("#bubble-stack", Vertical)
        await stack.remove_children()
        self._active_assistant_turn = None

    def has_active_assistant_text(self) -> bool:
        if self._active_assistant_turn is None:
            return False
        return self._active_assistant_turn.has_assistant_text()

    def get_active_assistant_text(self) -> str:
        if self._active_assistant_turn is None:
            return ""
        return self._active_assistant_turn.get_assistant_text()

    async def begin_assistant_turn(self) -> None:
        """Create the assistant turn container up front.

        This lets the UI show that the model is responding before the first
        assistant token arrives, which is useful when the model spends a while
        emitting thinking output first.
        """
        await self._ensure_assistant_turn()
        self._schedule_autoscroll()

    async def append_event(self, event: UIEvent) -> None:
        if event.event_type == UIEventType.ASSISTANT:
            if event.data.get("kind") == "print":
                await self._append_full_printout(
                    event.content, 
                    artifact_id=event.data.get("artifact_id")
                )
            elif event.data.get("kind") == "replace":
                await self._replace_assistant(event.content)
            else:
                await self._append_assistant(event.content)
            return
        if event.event_type == UIEventType.THINKING:
            await self._append_thinking(event.content)
            return
        if event.event_type == UIEventType.SHELL_STREAM:
            await self._append_shell_stream(event.content)
            return
        if event.event_type == UIEventType.TOOL_CALL:
            await self._append_tool_call(
                str(event.data.get("display_text") or event.content),
                tool_name=event.content,
                tool_call_id=_coerce_str(event.data.get("tool_call_id")),
            )
            return
        if event.event_type == UIEventType.TOOL_RESULT:
            nested = await self._append_tool_result(
                str(event.data.get("display_text") or event.content),
                tool_name=_coerce_str(event.data.get("tool_name")),
                tool_call_id=_coerce_str(event.data.get("tool_call_id")),
                data=event.data,
            )
            if nested:
                self._schedule_autoscroll()
                return

        kind_map = {
            UIEventType.USER: "user",
            UIEventType.TOOL_RESULT: "tool_result",
            UIEventType.ERROR: "error",
            UIEventType.ANSIBLE: "ansible",
            UIEventType.SYSTEM: "system",
            UIEventType.ALERT: "alert",
        }
        kind = kind_map.get(event.event_type, "system")
        if event.event_type == UIEventType.USER:
            self._active_assistant_turn = None
        text = str(event.data.get("display_text") or event.content)
        await self._add_bubble(kind, text)
        if kind == "system":
            self._last_system_message = event.content
        self._schedule_autoscroll()

    async def _append_assistant(self, text: str) -> None:
        if not text:
            return
        turn = await self._ensure_assistant_turn()
        await turn.append_assistant_text(text)
        self._schedule_autoscroll()

    async def _replace_assistant(self, text: str) -> None:
        """Overwrite the streamed assistant text with the cleaned version.
        This is called after _extract_inline_tool_calls strips tool JSON."""
        if self._active_assistant_turn is not None:
            self._active_assistant_turn.replace_assistant_text(text)
            self._schedule_autoscroll()

    async def _append_full_printout(self, text: str, *, artifact_id: str | None) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.add_full_printout(text, artifact_id=artifact_id)
        self._schedule_autoscroll()

    async def _append_thinking(self, text: str) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.append_thinking_text(text)
        self._schedule_autoscroll()

    async def _append_shell_stream(self, text: str) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.append_shell_stream(text)
        self._schedule_autoscroll()

    async def _append_tool_call(
        self,
        text: str,
        *,
        tool_name: str,
        tool_call_id: str | None,
    ) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.add_tool_call(text, tool_name=tool_name, tool_call_id=tool_call_id)
        self._schedule_autoscroll()

    async def _append_tool_result(
        self,
        text: str,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
        data: dict[str, Any] | None = None,
    ) -> bool:
        if self._active_assistant_turn is None:
            return False
        nested = await self._active_assistant_turn.add_tool_result(
            text,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            data=data,
        )
        return nested

    async def _add_bubble(self, kind: str, text: str) -> BubbleWidget:
        bubble = BubbleWidget(kind=kind, text=text)
        stack = self.query_one("#bubble-stack", Vertical)
        await stack.mount(bubble)
        return bubble

    async def _ensure_assistant_turn(self) -> AssistantTurnWidget:
        if self._active_assistant_turn is None:
            turn = AssistantTurnWidget()
            stack = self.query_one("#bubble-stack", Vertical)
            await stack.mount(turn)
            self._active_assistant_turn = turn
        return self._active_assistant_turn

    def _schedule_autoscroll(self) -> None:
        """
        Smart autoscroll: only scroll to end if we are already at or near the bottom.
        This prevents UI jumping while a user is manually scrolling up to read.
        """
        # If we are within 2 lines of the bottom, consider it 'at bottom' for autoscroll.
        # We use a small threshold (e.g. 20 pixels or 2 lines) to account for layout jitter.
        THRESHOLD = 2
        if self.scroll_y >= self.max_scroll_y - THRESHOLD:
            self.call_after_refresh(self.scroll_end, animate=False)

    def get_last_system_message(self) -> str:
        return self._last_system_message


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
