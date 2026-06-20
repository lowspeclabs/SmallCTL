from __future__ import annotations

import asyncio
from pathlib import Path

from textual.color import Color
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical

from smallctl.models.events import UIEvent, UIEventType
from smallctl.ui import __file__ as ui_module_file
from smallctl.ui.bubbles import (
    AssistantDetailWidget,
    AssistantTurnWidget,
    BubbleWidget,
    LiveOutputBubbleWidget,
    TextBlockWidget,
    ToolCallDetailWidget,
    ToolCallsContainerWidget,
)
from smallctl.ui.console import ConsolePane


class _ConsoleApp(App[None]):
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        super().__init__()

    def compose(self) -> ComposeResult:
        yield ConsolePane(verbose=self._verbose)


def test_console_hides_scrollbar_but_remains_scrollable() -> None:
    class _ScrollableConsoleApp(App[None]):
        CSS = "#console { height: 6; } #bubble-stack { height: auto; }"

        def compose(self) -> ComposeResult:
            yield ConsolePane(id="console")

    async def _run() -> None:
        app = _ScrollableConsoleApp()
        async with app.run_test(size=(80, 12)) as pilot:
            console = app.query_one(ConsolePane)

            for index in range(40):
                await console.append_line(f"line {index}")
            await pilot.pause()

            assert console.styles.scrollbar_size_vertical == 0
            assert console.styles.scrollbar_size_horizontal == 0
            assert console.max_scroll_y > 0

            console.scroll_relative(y=3, animate=False)
            await pilot.pause()

            assert console.scroll_y > 0

    asyncio.run(_run())


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
            await console.flush_stream_buffers()
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
            await console.flush_stream_buffers()
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
            await console.flush_stream_buffers()
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
            await console.flush_stream_buffers()
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.get_assistant_text() == "Visible summary up top."

    asyncio.run(_run())


def test_complete_assistant_markdown_list_renders_as_markdown() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            markdown = "**Done**\n\n- first\n- second\n\n```python\nprint(\"hi\")\n```"
            await console.append_event(UIEvent(UIEventType.ASSISTANT, markdown))
            await console.flush_stream_buffers()
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.get_assistant_text() == markdown

            content = turn.query_one(".assistant-turn-content", Vertical)
            block = content.children[0]
            assert isinstance(block, TextBlockWidget)
            assert isinstance(block._rendered_content, RichMarkdown)

    asyncio.run(_run())


def test_incomplete_assistant_markdown_table_renders_plain_while_streaming() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            partial = "The Docker containers are:\n\n| CONTAINER ID | IMAGE | COMMAND | STATUS | NAMES |\n| :--- | :--- | :--- | :--- | :--- |\n| be793d0eb63a | ghcr.io/dagucloud/dagu:latest |"
            await console.append_event(UIEvent(UIEventType.ASSISTANT, partial))
            await console.flush_stream_buffers()
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.get_assistant_text() == partial

            content = turn.query_one(".assistant-turn-content", Vertical)
            block = content.children[0]
            assert isinstance(block, TextBlockWidget)
            assert isinstance(block._rendered_content, Text)

    asyncio.run(_run())


def test_complete_assistant_markdown_table_renders_as_markdown() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            table = "The Docker containers are:\n\n| CONTAINER ID | IMAGE | COMMAND | STATUS | NAMES |\n| :--- | :--- | :--- | :--- | :--- |\n| be793d0eb63a | ghcr.io/dagucloud/dagu:latest | /usr/local/bin/tini | Exited | dagu |\n| acac144a9b54 | pihole/pihole:latest | start.sh | Up healthy | pihole |"
            await console.append_event(UIEvent(UIEventType.ASSISTANT, table))
            await console.flush_stream_buffers()
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.get_assistant_text() == table

            content = turn.query_one(".assistant-turn-content", Vertical)
            block = content.children[0]
            assert isinstance(block, TextBlockWidget)
            assert isinstance(block._rendered_content, RichMarkdown)

    asyncio.run(_run())


def test_scheduled_stream_flush_defers_markdown_render_until_boundary() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            table = (
                "The Docker containers are:\n\n"
                "| CONTAINER ID | IMAGE | COMMAND | STATUS | NAMES |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                "| be793d0eb63a | ghcr.io/dagucloud/dagu:latest | /usr/local/bin/tini | Exited | dagu |"
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, table))
            await console.flush_stream_buffers(finalize_assistant=False)
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            content = turn.query_one(".assistant-turn-content", Vertical)
            block = content.children[0]
            assert isinstance(block, TextBlockWidget)
            assert isinstance(block._rendered_content, Text)

            await console.flush_stream_buffers()
            await pilot.pause()
            assert isinstance(block._rendered_content, RichMarkdown)

    asyncio.run(_run())


def test_assistant_replace_after_alert_cleans_previous_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp(verbose=True)
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            raw = "The following Docker containers are installed:\n\n| Container ID | Image |\n| :--- | :--- |\n| `abc` | `demo` |"
            await console.append_event(UIEvent(UIEventType.ASSISTANT, raw))
            await console.flush_stream_buffers()
            await console.append_event(
                UIEvent(
                    UIEventType.ALERT,
                    "Model output entered a repetition loop; halting this turn and requesting recovery.",
                    data={"ui_kind": "model_output_degenerate_loop_exhausted"},
                )
            )
            assert console._active_assistant_turn is None

            await console.append_event(
                UIEvent(
                    UIEventType.ASSISTANT,
                    "[Previous assistant output was halted because it entered a repetition loop.]",
                    data={"kind": "replace"},
                )
            )
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            turns = [child for child in stack.children if hasattr(child, "get_assistant_text")]
            assert len(turns) == 1
            assert turns[0].get_assistant_text() == ""
            assert "Container ID" not in turns[0].get_assistant_text()

    asyncio.run(_run())


def test_late_assistant_replace_after_tool_call_cleans_previous_text_block() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            raw = "Final answer.\n{\"name\": \"task_complete\", \"arguments\": {\"message\": \"done\"}}"
            await console.append_event(UIEvent(UIEventType.ASSISTANT, raw))
            await console.flush_stream_buffers()
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "task_complete",
                    data={"display_text": "task_complete(message=\"done\")", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.ASSISTANT,
                    "Final answer.",
                    data={"kind": "replace"},
                )
            )
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.get_assistant_text() == "Final answer."
            assert "task_complete" not in turn.get_assistant_text()

    asyncio.run(_run())


def test_empty_assistant_replace_hides_raw_tool_json_block() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            raw = "{\"name\": \"task_complete\", \"arguments\": {\"message\": \"done\"}}"
            await console.append_event(UIEvent(UIEventType.ASSISTANT, raw))
            await console.flush_stream_buffers()
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "task_complete",
                    data={"display_text": "task_complete(message=\"done\")", "tool_call_id": "tool-1"},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.ASSISTANT,
                    "",
                    data={"kind": "replace"},
                )
            )
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            assert turn.get_assistant_text() == ""
            content = turn.query_one(".assistant-turn-content", Vertical)
            text_blocks = [child for child in content.children if isinstance(child, TextBlockWidget)]
            assert len(text_blocks) == 1
            assert text_blocks[0].display is False

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
            await console.flush_stream_buffers()
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
            await console.flush_stream_buffers()
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
        app = _ConsoleApp(verbose=True)
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Before system."))
            await console.append_event(UIEvent(UIEventType.SYSTEM, "System notice."))
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "After system."))
            await console.flush_stream_buffers()
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


def test_system_event_suppressed_without_verbose_but_breaks_turn() -> None:
    async def _run() -> None:
        app = _ConsoleApp(verbose=False)
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Before system."))
            await console.append_event(UIEvent(UIEventType.SYSTEM, "System notice."))
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "After system."))
            await console.flush_stream_buffers()
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
            assert [type(child) for child in children] == [
                type(console._active_assistant_turn),
                type(console._active_assistant_turn),
            ]
            assert children[0].get_assistant_text() == "Before system."
            assert children[1].get_assistant_text() == "After system."

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
            await asyncio.sleep(0.2)
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
                    data={
                        "display_text": "ssh_exec(command='journalctl -f')",
                        "tool_call_id": "tool-1",
                        "args": {"command": "journalctl -f", "host": "192.168.1.161"},
                    },
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
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "SSH reached the remote host; remote command exited non-zero.",
                    data={
                        "tool_name": "ssh_exec",
                        "tool_call_id": "tool-1",
                        "success": True,
                    },
                )
            )
            await asyncio.sleep(0.2)
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
            assert detail.tool_name == "ssh_exec"
            assert "succeeded" in detail.title
            nested_results = detail._result_widgets
            assert len(nested_results) == 1
            assert isinstance(nested_results[0], LiveOutputBubbleWidget)
            assert nested_results[0]._title_base == "Live Output"
            assert nested_results[0].command == "journalctl -f"
            assert "journalctl -f" in nested_results[0].title
            assert "line 1\nline 2\n" in nested_results[0]._content_widget.text
            assert nested_results[0]._content_widget.styles.color == Color(22, 163, 74)

    asyncio.run(_run())


def test_shell_exec_failure_shows_red_command_and_failed_status() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "ssh_exec",
                    data={
                        "display_text": "ssh_exec(command='docker ps')",
                        "tool_call_id": "tool-1",
                        "args": {"command": "docker ps", "host": "192.168.1.161"},
                    },
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "Conflicting SSH targets provided via `target` and `host`.",
                    data={
                        "tool_name": "ssh_exec",
                        "tool_call_id": "tool-1",
                        "success": False,
                    },
                )
            )
            await asyncio.sleep(0.2)
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None

            content = turn.query_one(".assistant-turn-content", Vertical)
            tool_group = content.children[0]
            assert isinstance(tool_group, ToolCallsContainerWidget)
            detail = tool_group.query_one(".tool-calls-container", Vertical).children[0]
            assert isinstance(detail, ToolCallDetailWidget)
            assert detail.tool_name == "ssh_exec"
            assert "failed" in detail.title
            assert "#ef4444" in detail.title

            nested_results = detail._result_widgets
            assert len(nested_results) == 1
            bubble = nested_results[0]
            assert isinstance(bubble, LiveOutputBubbleWidget)
            assert bubble.command == "docker ps"
            assert "docker ps" in bubble.title
            assert "(failed)" in bubble.title
            assert "#ef4444" in bubble.title
            assert bubble._content_widget.styles.color == Color(239, 68, 68)
            assert "Conflicting SSH targets" in bubble._content_widget.text

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
                    data={"tool_name": "web_fetch", "tool_call_id": "missing"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "After result."))
            await console.flush_stream_buffers()
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


def test_unmatched_shell_tool_result_is_not_rendered_as_system() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Before result."))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "SSH reached the remote host; remote command exited non-zero.",
                    data={"tool_name": "ssh_exec", "tool_call_id": "missing"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "After result."))
            await console.flush_stream_buffers()
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
            assert len(children) == 1
            assert children[0].get_assistant_text() == "Before result.After result."

    asyncio.run(_run())


def test_shell_stream_before_ssh_tool_call_is_reparented_under_command() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.SHELL_STREAM, "early output\n"))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "ssh_exec",
                    data={
                        "display_text": "ssh_exec(command='ls /etc')",
                        "tool_call_id": "tool-early",
                        "args": {"command": "ls /etc", "target": "root@192.168.1.89"},
                    },
                )
            )
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            content = turn.query_one(".assistant-turn-content", Vertical)
            assert len(content.children) == 1
            tool_group = content.children[0]
            assert isinstance(tool_group, ToolCallsContainerWidget)
            detail = tool_group.query_one(".tool-calls-container", Vertical).children[0]
            assert isinstance(detail, ToolCallDetailWidget)
            assert len(detail._result_widgets) == 1
            bubble = detail._result_widgets[0]
            assert isinstance(bubble, LiveOutputBubbleWidget)
            assert bubble.command == "ls /etc"
            assert "command:" in bubble.title
            assert "early output" in bubble.text_content

    asyncio.run(_run())


def test_unmatched_file_read_result_is_not_rendered_as_system() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)

            await console.append_event(UIEvent(UIEventType.ASSISTANT, "Before result."))
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "raw file content that should not leak into chat",
                    data={"tool_name": "file_read", "tool_call_id": "missing"},
                )
            )
            await console.append_event(UIEvent(UIEventType.ASSISTANT, "After result."))
            await console.flush_stream_buffers()
            await pilot.pause()

            stack = console.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
            assert len(children) == 1
            assert children[0].get_assistant_text() == "Before result.After result."

    asyncio.run(_run())


def test_assistant_turn_has_dark_grey_left_border() -> None:
    """Assistant turns should have a dark grey left border, like user (white) and system (gold)."""

    class _StyledApp(App[None]):
        CSS_PATH = str(Path(ui_module_file).parent / "styles.tcss")

        def compose(self) -> ComposeResult:
            yield BubbleWidget(kind="user", text="user")
            yield BubbleWidget(kind="system", text="system")
            yield AssistantTurnWidget()

    async def _run() -> None:
        app = _StyledApp()
        async with app.run_test(size=(80, 24)) as pilot:
            user_bubble = app.query_one(".bubble-user", BubbleWidget)
            system_bubble = app.query_one(".bubble-system", BubbleWidget)
            assistant_turn = app.query_one(AssistantTurnWidget)
            assert user_bubble.styles.border_left == ("thick", Color(255, 255, 255))
            assert system_bubble.styles.border_left == ("thick", Color(234, 179, 8))
            assert assistant_turn.styles.border_left == ("thick", Color(75, 85, 99))

    asyncio.run(_run())
