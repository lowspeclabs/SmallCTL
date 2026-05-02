from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll

from smallctl.models.events import UIEvent, UIEventType
from smallctl.ui.bubbles import AssistantDetailWidget, TextBlockWidget, ToolCallDetailWidget, ToolCallsContainerWidget
from smallctl.ui.console import ConsolePane


class _ConsoleApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsolePane()


def test_thinking_after_visible_assistant_starts_a_new_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Visible summary up top."))
            await console.append_event(UIEvent(UIEventType.THINKING, "Plan the next step."))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "shell_exec",
                    data={"display_text": "shell_exec(command='pytest')", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Final user-facing status."))
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            turns = list(stack.children)
            assert len(turns) == 2

            first_turn, second_turn = turns
            assert isinstance(first_turn, type(console._active_assistant_turn))
            assert isinstance(second_turn, type(console._active_assistant_turn))

            assert first_turn.get_assistant_text() == "Visible summary up top."
            assert second_turn.get_assistant_text() == "Final user-facing status."

            second_body = second_turn.query_one(".assistant-turn-body", Vertical)
            second_main = second_turn.query_one(".assistant-turn-main", Vertical)
            second_meta = second_turn.query_one(".assistant-turn-meta", Vertical)

            assert list(second_body.children) == [second_meta, second_main]

            main_children = list(second_main.children)
            assert [type(child) for child in main_children] == [TextBlockWidget]
            assert [child.text for child in main_children] == ["Final user-facing status."]

            meta_children = list(second_meta.children)
            assert [type(child) for child in meta_children] == [
                AssistantDetailWidget,
                ToolCallsContainerWidget,
            ]
            assert meta_children[0].text == "Plan the next step."

    asyncio.run(_run())


def test_consecutive_tool_calls_group_between_thinking_statements() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.THINKING, "First thought."))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "shell_exec",
                    data={"display_text": "shell_exec(command='pwd')", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "shell_exec",
                    data={"display_text": "shell_exec(command='ls')", "tool_call_id": "tool-2"},
                )
            )
            await console.append_event(UIEvent(UIEventType.THINKING, "Second thought."))
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None

            meta = turn.query_one(".assistant-turn-meta", Vertical)
            meta_children = list(meta.children)

            assert [type(child) for child in meta_children] == [
                AssistantDetailWidget,
                ToolCallsContainerWidget,
                AssistantDetailWidget,
            ]
            assert meta_children[0].text == "First thought."
            assert meta_children[2].text == "Second thought."

            tool_group = meta_children[1]
            assert isinstance(tool_group, ToolCallsContainerWidget)
            tool_children = list(tool_group.query_one(".tool-calls-scroll-container", VerticalScroll).children)
            assert [type(child) for child in tool_children] == [
                ToolCallDetailWidget,
                ToolCallDetailWidget,
            ]
            assert tool_children[0].text == "shell_exec(command='pwd')"
            assert tool_children[1].text == "shell_exec(command='ls')"

    asyncio.run(_run())


def test_console_autoscroll_requests_are_coalesced() -> None:
    scheduled: list[object] = []
    scroll_calls: list[bool] = []

    class _FakeConsole:
        def __init__(self) -> None:
            self._autoscroll_scheduled = False
            self.scroll_y = 10
            self.max_scroll_y = 10

        def call_after_refresh(self, callback):
            scheduled.append(callback)

        def scroll_end(self, animate=False) -> None:
            scroll_calls.append(bool(animate))

    console = _FakeConsole()
    console._run_scheduled_autoscroll = ConsolePane._run_scheduled_autoscroll.__get__(console, _FakeConsole)
    console._schedule_autoscroll = ConsolePane._schedule_autoscroll.__get__(console, _FakeConsole)

    console._schedule_autoscroll()
    console._schedule_autoscroll()

    assert len(scheduled) == 1
    assert console._autoscroll_scheduled is True

    callback = scheduled[0]
    callback()

    assert console._autoscroll_scheduled is False
    assert scroll_calls == [False]


def test_assistant_turn_stays_compact_without_stylesheet() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.begin_assistant_turn()
            await console.append_event(UIEvent(UIEventType.THINKING, "Plan the next step."))
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Visible summary up top."))
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.region.height < 12

    asyncio.run(_run())


def test_assistant_text_drops_leading_blank_lines_after_thinking() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.THINKING, "Plan the next step."))
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "\n\nVisible summary up top."))
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.get_assistant_text() == "Visible summary up top."

    asyncio.run(_run())


def test_tool_call_after_meta_and_assistant_starts_a_new_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.THINKING, "Inspect the first result."))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "web_fetch",
                    data={"display_text": "web_fetch(result_id='r1')", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "First fetch is complete."))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "shell_exec",
                    data={"display_text": "shell_exec(command='ssh host')", "tool_call_id": "tool-2"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Second step is underway."))
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            turns = list(stack.children)
            assert len(turns) == 2

            first_turn, second_turn = turns
            assert first_turn.get_assistant_text() == "First fetch is complete."
            assert second_turn.get_assistant_text() == "Second step is underway."

            second_meta = second_turn.query_one(".assistant-turn-meta", Vertical)
            meta_children = list(second_meta.children)
            assert [type(child) for child in meta_children] == [ToolCallsContainerWidget]

            tool_group = meta_children[0]
            assert isinstance(tool_group, ToolCallsContainerWidget)
            tool_children = list(tool_group.query_one(".tool-calls-scroll-container", VerticalScroll).children)
            assert [child.text for child in tool_children] == ["shell_exec(command='ssh host')"]

    asyncio.run(_run())
