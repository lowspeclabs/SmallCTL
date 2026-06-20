from __future__ import annotations

import asyncio
from types import SimpleNamespace

from textual.app import App, ComposeResult
from textual.containers import Vertical

from smallctl.fama.capsules import render_fama_capsules
from smallctl.graph.display import format_tool_result_display
from smallctl.harness.core_facade import _emit
from smallctl.models.conversation import ConversationMessage
from smallctl.models.events import UIEvent, UIEventType
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.ansi_utils import detect_tui_application
from smallctl.tools.network_interactive_sessions import _SSH_INTERACTIVE_SESSIONS, ssh_session_start
from smallctl.tools.network_ssh_helpers import ssh_error_class
from smallctl.ui.bubbles import ToolCallDetailWidget, ToolCallsContainerWidget
from smallctl.ui.console import ConsolePane
from smallctl.ui.harness_bridge import _serialize_recent_messages


class _ConsoleApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsolePane()


def test_dialog_detector_handles_generic_installer_terminal_failure() -> None:
    output = "Error opening terminal: unknown.\n[i] Building dependency package...\nInstaller exited at static IP message."

    detected = detect_tui_application(output)

    assert detected is not None
    assert detected["kind"] == "interactive_installer"
    assert ssh_error_class(exit_code=1, stderr=output) == "interactive_installer_blocked"


def test_dialog_detector_handles_line_drawing_charset_output() -> None:
    output = "\x1b=Support installerqqqqOpen Source Softwareqqq Press Enter to continue"

    detected = detect_tui_application(output)

    assert detected is not None
    assert detected["tui_detected"] is True


def test_format_tool_result_display_names_interactive_installer_blocker() -> None:
    result = ToolEnvelope(
        success=False,
        error="Error opening terminal: unknown.",
        metadata={
            "ssh_transport_succeeded": True,
            "failure_mode": "interactive_installer_blocked",
            "next_required_action": "Use unattended configuration or one managed PTY session.",
        },
    )

    text = format_tool_result_display(tool_name="ssh_exec", result=result)

    assert "installer/dialog interaction blocked execution" in text
    assert "Recovery hint" in text


def test_format_tool_result_display_preserves_executed_404_script_error() -> None:
    result = ToolEnvelope(
        success=False,
        error=(
            "% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
            "100    14  100    14    0     0     97      0 --:--:-- --:--:-- --:--:--    97\n"
            "setup-webmin-repo.sh: 1: 404:: not found"
        ),
        output={
            "exit_code": 127,
            "stdout": "",
            "stderr": (
                "% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
                "100    14  100    14    0     0     97      0 --:--:-- --:--:-- --:--:--    97\n"
                "setup-webmin-repo.sh: 1: 404:: not found\n"
            ),
        },
        metadata={
            "ssh_transport_succeeded": True,
            "failure_mode": "remote_installer_download_error",
            "args": {
                "command": "curl -o setup-webmin-repo.sh https://raw.githubusercontent.com/webmin/webmin/master/setup-webmin-repo.sh && sh setup-webmin-repo.sh"
            },
        },
    )

    text = format_tool_result_display(tool_name="ssh_exec", result=result)

    assert "--- [ACTIONABLE FAILURE] ---" in text
    assert "Exit code: 127" in text
    assert "setup-webmin-repo.sh: 1: 404:: not found" in text


def test_restored_chat_skips_tool_scoped_only_system_nudges() -> None:
    state = LoopState()
    state.transcript_messages = [
        ConversationMessage(
            role="system",
            content="Guard tripped: cycling between artifact_read and ssh_session_read.",
            metadata={"tui_visibility": "tool_scoped_only"},
        ),
        ConversationMessage(role="assistant", content="Visible response."),
    ]

    serialized = _serialize_recent_messages(state)

    assert serialized == [{"role": "assistant", "content": "Visible response."}]


def test_interactive_guard_tool_result_nests_under_session_start() -> None:
    async def _run() -> None:
        app = _ConsoleApp()
        async with app.run_test(size=(120, 40)) as pilot:
            console = app.query_one(ConsolePane)
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "ssh_session_start",
                    data={"display_text": "ssh_session_start(command='installer')", "tool_call_id": "start-1"},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "started",
                    data={"tool_name": "ssh_session_start", "tool_call_id": "start-1", "success": True},
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_RESULT,
                    "guard",
                    data={
                        "tool_name": "ssh_session_start",
                        "success": False,
                        "ui_kind": "interactive_session_guard",
                        "display_text": "Interactive SSH session guard.",
                    },
                )
            )
            await pilot.pause()

            turn = console._active_assistant_turn
            assert turn is not None
            content = turn.query_one(".assistant-turn-content", Vertical)
            tool_group = next(child for child in content.children if isinstance(child, ToolCallsContainerWidget))
            tool_detail = next(
                child for child in tool_group.query_one(".tool-calls-container", Vertical).children
                if isinstance(child, ToolCallDetailWidget)
            )
            assert len(tool_detail._result_widgets) == 2

    asyncio.run(_run())


def test_fama_capsules_remote_interactive_override_when_config_disabled() -> None:
    state = LoopState()
    state.task_mode = "remote_execute"
    state.scratchpad["_fama_config"] = {"enabled": False, "mode": "lite", "capsule_token_budget": 180}
    state.recent_errors.append("ssh_exec failed: Error opening terminal: unknown")

    lines = render_fama_capsules(state, token_budget=180)

    assert any("dialog/TUI" in line for line in lines)


def test_emit_records_lightweight_ui_event_ledger() -> None:
    async def _run() -> None:
        events: list[UIEvent] = []
        state = LoopState()
        harness = SimpleNamespace(
            state=state,
            build_status_snapshot=lambda activity="": {"activity": activity},
            run_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )

        async def handler(event: UIEvent) -> None:
            events.append(event)

        await _emit(
            harness,
            handler,
            UIEvent(UIEventType.SYSTEM, "secret password=abc", data={"tool_name": "ssh_exec"}),
        )

        ledger = state.scratchpad.get("_ui_event_ledger")
        assert isinstance(ledger, list)
        assert ledger[0]["event_type"] == "system"
        assert "abc" not in ledger[0]["content"]
        assert len(events) == 2

    asyncio.run(_run())


def test_ssh_session_start_blocks_second_active_session_same_target() -> None:
    async def _run() -> None:
        _SSH_INTERACTIVE_SESSIONS.clear()
        try:
            _SSH_INTERACTIVE_SESSIONS["sess_existing"] = {
                "proc": SimpleNamespace(returncode=None),
                "host": "192.168.1.161",
                "user": "root",
                "command": "installer",
            }
            result = await ssh_session_start(
                host="192.168.1.161",
                user="root",
                command="another installer",
            )
        finally:
            _SSH_INTERACTIVE_SESSIONS.clear()

        assert result["success"] is False
        assert result["metadata"]["reason"] == "active_interactive_session_exists"
        assert result["metadata"]["active_session_id"] == "sess_existing"

    asyncio.run(_run())
