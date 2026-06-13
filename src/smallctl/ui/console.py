from __future__ import annotations
from typing import Any

from textual.containers import Vertical, VerticalScroll

from ..models.events import UIEvent, UIEventType
from .bubbles import ArtifactBubbleWidget, AssistantTurnWidget, BubbleWidget, SystemInterruptWidget
from .display import _CRITICAL_EVENTS, format_test_time_scaling_event


class ConsolePane(VerticalScroll):
    def __init__(self, *args, verbose: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_assistant_turn: AssistantTurnWidget | None = None
        self._last_system_message: str = ""
        self._verbose = bool(verbose)

    def set_verbose(self, verbose: bool) -> None:
        self._verbose = bool(verbose)

    async def on_mount(self) -> None:
        await self.mount(Vertical(id="bubble-stack"))

    async def append_line(self, line: str, kind: str = "system") -> None:
        await self._add_bubble(kind, line)
        self._last_system_message = line

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

    async def append_event(self, event: UIEvent) -> None:
        speaker = _coerce_speaker(event.data.get("speaker"))
        if event.event_type == UIEventType.ASSISTANT:
            await self._ensure_assistant_turn(speaker=speaker)
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
            await self._ensure_assistant_turn(speaker=speaker)
            if event.data.get("kind") == "replace":
                await self._replace_thinking(event.content)
            else:
                await self._append_thinking(event.content)
            return
        if event.event_type == UIEventType.SHELL_STREAM:
            await self._ensure_assistant_turn(speaker=speaker)
            await self._append_shell_stream(
                event.content,
                tool_name=_coerce_str(event.data.get("tool_name")),
                tool_call_id=_coerce_str(event.data.get("tool_call_id")),
            )
            return
        if event.event_type == UIEventType.TOOL_CALL:
            await self._ensure_assistant_turn(speaker=speaker)
            await self._append_tool_call(
                str(event.data.get("display_text") or event.content),
                tool_name=event.content,
                tool_call_id=_coerce_str(event.data.get("tool_call_id")),
            )
            return
        if event.event_type == UIEventType.TOOL_RESULT:
            if self._active_assistant_turn is not None:
                if speaker:
                    self._active_assistant_turn.set_speaker(speaker)
                nested = await self._append_tool_result(
                    str(event.data.get("display_text") or event.content),
                    tool_name=_coerce_str(event.data.get("tool_name")),
                    tool_call_id=_coerce_str(event.data.get("tool_call_id")),
                    data=event.data,
                )
                if nested:
                    return
            self._active_assistant_turn = None
            await self._add_bubble("system", str(event.data.get("display_text") or event.content))
            return

        if event.event_type == UIEventType.SYSTEM and event.data.get("kind") == "test_time_scaling":
            await self._append_test_time_scaling_event(event)
            return

        if event.data.get("ui_kind") == "subtask_checklist":
            turn = await self._ensure_assistant_turn()
            title = event.data.get("checklist_title") or "Task Checklist"
            content = event.content or ""
            self.app._app_logger.debug(
                "ui_subtask_checklist: title=%r content_len=%d empty=%s",
                title, len(content), not content.strip(),
            )
            await turn.set_task_checklist(content, title=f"📋 {title}")
            return

        # P2.2: critical backend events render as compact inline interruptions
        # only when verbose mode is enabled (default off).
        if self._verbose and event.event_type == UIEventType.SYSTEM and event.data.get("ui_kind") in _CRITICAL_EVENTS:
            await self._append_critical_interrupt(event)
            return

        kind_map = {
            UIEventType.USER: "user",
            UIEventType.ERROR: "system",
            UIEventType.SYSTEM: "system",
            UIEventType.ALERT: "system",
        }
        kind = kind_map.get(event.event_type, "system")
        if event.event_type in {
            UIEventType.USER,
            UIEventType.ERROR,
            UIEventType.SYSTEM,
            UIEventType.METRICS,
            UIEventType.ALERT,
        }:
            self._active_assistant_turn = None
        text = str(event.data.get("display_text") or event.content)
        await self._add_bubble(kind, text)
        if kind == "system":
            self._last_system_message = event.content

    async def _append_assistant(self, text: str) -> None:
        if not text:
            return
        turn = await self._ensure_assistant_turn()
        await turn.append_assistant_text(text)

    async def _replace_assistant(self, text: str) -> None:
        """Overwrite the streamed assistant text with the cleaned version.
        This is called after _extract_inline_tool_calls strips tool JSON."""
        if self._active_assistant_turn is not None:
            self._active_assistant_turn.replace_assistant_text(text)

    async def _append_full_printout(self, text: str, *, artifact_id: str | None) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.add_full_printout(text, artifact_id=artifact_id)

    async def _append_thinking(self, text: str) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.append_thinking_text(text)

    async def _replace_thinking(self, text: str) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.replace_thinking_text(text)

    async def _append_shell_stream(
        self,
        text: str,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
    ) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.append_shell_stream(text, tool_name=tool_name, tool_call_id=tool_call_id)

    async def _append_tool_call(
        self,
        text: str,
        *,
        tool_name: str,
        tool_call_id: str | None,
    ) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.add_tool_call(text, tool_name=tool_name, tool_call_id=tool_call_id)

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

    async def _append_test_time_scaling_event(self, event: UIEvent) -> None:
        text = format_test_time_scaling_event(event)
        title = "Test-Time Scaling"
        phase = str(event.data.get("phase") or "").strip()
        if phase:
            title += f" ({phase})"
        panel = ArtifactBubbleWidget(title=title, text=text, collapsed=False)
        panel.add_class("assistant-detail-test-time-scaling")
        stack = self.query_one("#bubble-stack", Vertical)
        await stack.mount(panel)
        self._active_assistant_turn = None
        self._last_system_message = event.content

    async def _append_critical_interrupt(self, event: UIEvent) -> None:
        text = str(event.data.get("display_text") or event.content)
        interrupt = SystemInterruptWidget(text=text)
        if self._active_assistant_turn is not None:
            await self._active_assistant_turn.add_meta_widget(interrupt)
        else:
            stack = self.query_one("#bubble-stack", Vertical)
            await stack.mount(interrupt)
        self._last_system_message = event.content

    async def _ensure_assistant_turn(self, *, speaker: str | None = None) -> AssistantTurnWidget:
        if self._active_assistant_turn is None:
            turn = AssistantTurnWidget(speaker=speaker or "assistant")
            stack = self.query_one("#bubble-stack", Vertical)
            await stack.mount(turn)
            self._active_assistant_turn = turn
        elif speaker:
            self._active_assistant_turn.set_speaker(speaker)
        return self._active_assistant_turn

    async def update_thinking_indicator(self) -> None:
        if self._active_assistant_turn is not None:
            if self._active_assistant_turn._last_thinking_detail is not None:
                self._active_assistant_turn._last_thinking_detail.update_thinking_timer()
            if self._active_assistant_turn._current_tool_calls_container is not None:
                self._active_assistant_turn._current_tool_calls_container.update_timer()
            for detail in self._active_assistant_turn._tool_call_details:
                if not detail.has_result:
                    detail.update_timer()

    def get_last_system_message(self) -> str:
        return self._last_system_message


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_speaker(value: object) -> str:
    speaker = str(value or "assistant").strip().lower()
    return speaker or "assistant"
