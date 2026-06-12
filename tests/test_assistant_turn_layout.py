from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Vertical

from smallctl.models.events import UIEvent, UIEventType
from smallctl.ui.bubbles import (
    AssistantDetailWidget,
    ArtifactBubbleWidget,
    BubbleWidget,
    TextBlockWidget,
    ToolCallDetailWidget,
    ToolCallsContainerWidget,
)
from smallctl.ui.console import ConsolePane


class _ConsoleApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsolePane()


def test_thinking_after_visible_assistant_stays_in_one_turn() -> None:
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
            assert len(turns) == 1

            turn = turns[0]
            assert isinstance(turn, type(console._active_assistant_turn))

            assert turn.get_assistant_text() == "Visible summary up top.\nFinal user-facing status."

            content = turn.query_one(".assistant-turn-content", Vertical)
            content_children = list(content.children)

            assert [type(child) for child in content_children] == [
                TextBlockWidget,
                AssistantDetailWidget,
                ToolCallsContainerWidget,
                TextBlockWidget,
            ]
            assert content_children[0].text == "Visible summary up top."
            assert content_children[1].text == "Plan the next step."
            assert content_children[3].text == "Final user-facing status."

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

            content = turn.query_one(".assistant-turn-content", Vertical)
            content_children = list(content.children)

            assert [type(child) for child in content_children] == [
                AssistantDetailWidget,
                ToolCallsContainerWidget,
                AssistantDetailWidget,
            ]
            assert content_children[0].text == "First thought."
            assert content_children[2].text == "Second thought."

            tool_group = content_children[1]
            assert isinstance(tool_group, ToolCallsContainerWidget)
            tool_children = list(tool_group.query_one(".tool-calls-container", Vertical).children)
            assert [type(child) for child in tool_children] == [
                ToolCallDetailWidget,
                ToolCallDetailWidget,
            ]
            assert tool_children[0].text == "shell_exec(command='pwd')"
            assert tool_children[1].text == "shell_exec(command='ls')"

    asyncio.run(_run())


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


def test_tool_call_after_meta_and_assistant_stays_in_one_turn() -> None:
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
            assert len(turns) == 1

            turn = turns[0]
            assert turn.get_assistant_text() == "First fetch is complete.\nSecond step is underway."

            content = turn.query_one(".assistant-turn-content", Vertical)
            content_children = list(content.children)
            assert [type(child) for child in content_children] == [
                AssistantDetailWidget,
                ToolCallsContainerWidget,
                TextBlockWidget,
                ToolCallsContainerWidget,
                TextBlockWidget,
            ]

            assert content_children[0].text == "Inspect the first result."
            tool_group_1 = content_children[1]
            assert isinstance(tool_group_1, ToolCallsContainerWidget)
            tool_children_1 = list(tool_group_1.query_one(".tool-calls-container", Vertical).children)
            assert [child.text for child in tool_children_1] == [
                "web_fetch(result_id='r1')",
            ]

            assert content_children[2].text == "First fetch is complete."

            tool_group_2 = content_children[3]
            assert isinstance(tool_group_2, ToolCallsContainerWidget)
            tool_children_2 = list(tool_group_2.query_one(".tool-calls-container", Vertical).children)
            assert [child.text for child in tool_children_2] == [
                "shell_exec(command='ssh host')",
            ]

            assert content_children[4].text == "Second step is underway."

    asyncio.run(_run())


def test_user_event_breaks_active_assistant_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "First response."))
            await console.append_event(UIEvent(UIEventType.USER, "Next request."))
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Second response."))
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
            assert [type(child) for child in children] == [
                type(console._active_assistant_turn),
                BubbleWidget,
                type(console._active_assistant_turn),
            ]
            assert children[0].get_assistant_text() == "First response."
            assert children[2].get_assistant_text() == "Second response."

    asyncio.run(_run())


def test_system_event_breaks_active_assistant_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Before system."))
            await console.append_event(UIEvent(UIEventType.SYSTEM, "System notice."))
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "After system."))
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
            assert [type(child) for child in children] == [
                type(console._active_assistant_turn),
                BubbleWidget,
                type(console._active_assistant_turn),
            ]
            assert children[0].get_assistant_text() == "Before system."
            assert children[2].get_assistant_text() == "After system."

    asyncio.run(_run())


def test_matched_tool_result_keeps_active_assistant_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "shell_exec",
                    data={"display_text": "shell_exec(command='pwd')", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "ok",
                    data={"tool_name": "shell_exec", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Done."))
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
            assert len(children) == 1
            assert children[0] is console._active_assistant_turn
            assert children[0].get_assistant_text() == "Done."

    asyncio.run(_run())


def test_shell_stream_nests_under_matching_tool_call() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "ssh_exec",
                    data={"display_text": "ssh_exec(command='journalctl -f')", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.SHELL_STREAM,
                    "line 1\n",
                    data={"tool_name": "ssh_exec", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.SHELL_STREAM,
                    "line 2\n",
                    data={"tool_name": "ssh_exec", "tool_call_id": "tool-1"},
                )
            )
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None

            content = turn.query_one(".assistant-turn-content", Vertical)
            content_children = list(content.children)
            assert [type(child) for child in content_children] == [ToolCallsContainerWidget]

            tool_group = content_children[0]
            assert isinstance(tool_group, ToolCallsContainerWidget)
            tool_children = list(tool_group.query_one(".tool-calls-container", Vertical).children)
            assert len(tool_children) == 1

            detail = tool_children[0]
            assert isinstance(detail, ToolCallDetailWidget)
            nested_results = detail._result_widgets
            assert len(nested_results) == 1
            assert isinstance(nested_results[0], ArtifactBubbleWidget)
            assert nested_results[0]._title_base == "Live Output"
            assert nested_results[0]._content_widget.text == "line 1\nline 2\n"

    asyncio.run(_run())


def test_unmatched_tool_result_breaks_active_assistant_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Before result."))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "orphan output",
                    data={"tool_name": "shell_exec", "tool_call_id": "missing"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "After result."))
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
            assert [type(child) for child in children] == [
                type(console._active_assistant_turn),
                BubbleWidget,
                type(console._active_assistant_turn),
            ]
            assert children[0].get_assistant_text() == "Before result."
            assert children[1].kind == "system"
            assert children[2].get_assistant_text() == "After result."

    asyncio.run(_run())
