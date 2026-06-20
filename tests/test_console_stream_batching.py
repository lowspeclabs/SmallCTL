from __future__ import annotations

import asyncio
import logging

from smallctl.models.events import UIEvent, UIEventType
from smallctl.ui.console import ConsolePane


class _RecordingConsole(ConsolePane):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, str | None, str | None]] = []
        self._active_assistant_turn = object()  # type: ignore[assignment]

    async def _ensure_assistant_turn(self, *, speaker: str | None = None):  # type: ignore[override]
        return self._active_assistant_turn

    async def _append_assistant(self, text: str) -> None:
        self.calls.append(("assistant", text, None, None))

    async def _replace_assistant(self, text: str, *, speaker: str | None = None) -> None:
        self.calls.append(("replace_assistant", text, None, None))

    async def _append_thinking(self, text: str) -> None:
        self.calls.append(("thinking", text, None, None))

    async def _append_shell_stream(
        self,
        text: str,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
    ) -> None:
        self.calls.append(("shell", text, tool_name, tool_call_id))

    async def _append_tool_call(
        self,
        text: str,
        *,
        tool_name: str,
        tool_call_id: str | None,
        args: dict[str, object] | None = None,
    ) -> None:
        self.calls.append(("tool_call", text, tool_name, tool_call_id))

    async def _add_bubble(self, kind: str, text: str):  # type: ignore[override]
        self.calls.append((f"bubble:{kind}", text, None, None))
        return None


def test_assistant_stream_chunks_are_coalesced_until_flush() -> None:
    async def _run() -> None:
        console = _RecordingConsole()
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "hel"))
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "lo"))

        assert console.calls == []

        await console.flush_stream_buffers()

        assert console.calls == [("assistant", "hello", None, None)]

    asyncio.run(_run())


def test_tool_call_flushes_pending_assistant_stream_before_boundary() -> None:
    async def _run() -> None:
        console = _RecordingConsole()
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "before "))
        await console.append_event(
            UIEvent(
                UIEventType.TOOL_CALL,
                "shell_exec",
                data={"display_text": "run command", "tool_call_id": "call-1"},
            )
        )

        assert console.calls == [
            ("assistant", "before ", None, None),
            ("tool_call", "run command", "shell_exec", "call-1"),
        ]

    asyncio.run(_run())


def test_shell_stream_chunks_are_coalesced_by_tool_identity() -> None:
    async def _run() -> None:
        console = _RecordingConsole()
        await console.append_event(
            UIEvent(
                UIEventType.SHELL_STREAM,
                "out-1",
                data={"tool_name": "shell_exec", "tool_call_id": "call-1"},
            )
        )
        await console.append_event(
            UIEvent(
                UIEventType.SHELL_STREAM,
                "out-2",
                data={"tool_name": "shell_exec", "tool_call_id": "call-1"},
            )
        )

        await console.flush_stream_buffers()

        assert console.calls == [("shell", "out-1out-2", "shell_exec", "call-1")]

    asyncio.run(_run())


def test_replace_flushes_pending_assistant_stream_first() -> None:
    async def _run() -> None:
        console = _RecordingConsole()
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "raw"))
        await console.append_event(
            UIEvent(UIEventType.ASSISTANT, "clean", data={"kind": "replace"})
        )

        assert console.calls == [
            ("assistant", "raw", None, None),
            ("replace_assistant", "clean", None, None),
        ]

    asyncio.run(_run())


def test_mixed_stream_groups_flush_in_arrival_order() -> None:
    async def _run() -> None:
        console = _RecordingConsole()
        await console.append_event(UIEvent(UIEventType.THINKING, "think"))
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "answer"))

        await console.flush_stream_buffers()

        assert console.calls == [
            ("thinking", "think", None, None),
            ("assistant", "answer", None, None),
        ]

    asyncio.run(_run())


def test_stream_flush_logs_timing_metrics(caplog) -> None:
    async def _run() -> None:
        caplog.set_level(logging.DEBUG, logger="smallctl.ui.console")
        console = _RecordingConsole()
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "hello"))
        await console.flush_stream_buffers()

    asyncio.run(_run())

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "ui_stream_flush" in message
    assert "flushed_chars" in message


def test_console_unmount_flushes_pending_stream() -> None:
    async def _run() -> None:
        console = _RecordingConsole()
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "pending"))

        await console.on_unmount()

        assert console.calls == [("assistant", "pending", None, None)]

    asyncio.run(_run())


def test_hidden_system_boundary_flushes_pending_assistant_stream() -> None:
    async def _run() -> None:
        console = _RecordingConsole()
        await console.append_event(UIEvent(UIEventType.ASSISTANT, "before"))
        await console.append_event(
            UIEvent(
                UIEventType.SYSTEM,
                "hidden recovery note",
                data={"ui_kind": "model_output_degenerate_loop_exhausted"},
            )
        )

        assert console.calls == [("assistant", "before", None, None)]
        assert console._stream_buffer_groups == []

        await console.append_event(UIEvent(UIEventType.ASSISTANT, "after"))
        await console.flush_stream_buffers()

        assert console.calls == [
            ("assistant", "before", None, None),
            ("assistant", "after", None, None),
        ]

    asyncio.run(_run())
