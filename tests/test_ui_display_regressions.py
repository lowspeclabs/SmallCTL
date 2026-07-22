from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.models.events import UIEvent, UIEventType
from smallctl.state import LoopState
from smallctl.tools import control
from smallctl.ui.app_flow import _terminal_status_detail
from smallctl.ui.bubbles import LiveOutputBubbleWidget, ToolCallDetailWidget
from smallctl.ui.bubbles import _inline_code_backticks_balanced, _style_text_lite
from smallctl.ui.console import ConsolePane
from smallctl.ui.display import (
    _build_backend_rca_strip,
    format_recovery_banner,
    format_run_log_row,
    should_render_event,
    should_render_run_log_row,
)


def test_tui_formats_phase_change_and_context_invalidation_events() -> None:
    phase_row = {
        "event": "context_invalidated",
        "data": {
            "reason": "phase_advanced",
            "details": {"from_phase": "execute", "to_phase": "repair"},
        },
    }
    invalidation_row = {
        "event": "context_invalidated",
        "data": {
            "reason": "verifier_failed",
            "invalidated_artifact_count": 2,
            "invalidated_observation_count": 1,
            "invalidated_summary_count": 0,
        },
    }
    subtask_row = {
        "event": "subtask_transition",
        "data": {"subtask_id": "S2", "title": "Escalate to bigger model", "old_status": "active", "new_status": "done"},
    }

    assert "Phase changed: execute -> repair" in format_run_log_row(phase_row)
    assert "Context invalidated: artifacts=2, observations=1, summaries=0" in format_run_log_row(invalidation_row)
    assert "Subtask update: S2 | Escalate to bigger model | active->done" in format_run_log_row(subtask_row)


def test_ssh_exec_tool_call_title_shows_only_target() -> None:
    widget = ToolCallDetailWidget.__new__(ToolCallDetailWidget)
    widget.kind = "tool_call"
    widget.tool_name = "ssh_exec"
    widget._args = {"host": "192.168.1.89", "target": "root@192.168.1.89", "command": "ls /etc"}
    widget._start_time = 0.0
    widget._done_time = 1.0
    widget._success = True

    title = widget._build_title("ssh_exec raw args")

    assert "ssh_exec" in title
    assert "root@192.168.1.89" in title
    assert "host" not in title


def test_ssh_exec_nested_output_title_and_success_color() -> None:
    widget = LiveOutputBubbleWidget.__new__(LiveOutputBubbleWidget)
    widget.command = "ls /etc"
    widget.tool_name = "ssh_exec"
    widget.success = True
    widget._content_widget = SimpleNamespace(styles=SimpleNamespace(color=None))

    title = widget._build_title()
    widget._set_content_color()

    assert "command:" in title
    assert '"ls /etc"' in title
    assert "Live Output" not in title
    assert widget._content_widget.styles.color == "#16a34a"


def test_task_complete_blocks_fake_escalation_completion() -> None:
    state = LoopState()
    state.run_brief.original_task = "Escalate to a bigger model for help with this bug"
    harness = SimpleNamespace(
        config=SimpleNamespace(escalation_enabled=False, escalation_expose_tool=False),
    )

    blocked = asyncio.run(control.task_complete("Escalated to a bigger model for help.", state=state, harness=harness))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "escalation_not_performed"


def test_ui_status_detail_and_rca_surface_install_source_blocker() -> None:
    state = LoopState()
    state.current_phase = "repair"
    state.scratchpad["_install_source_invalid_blocker"] = {
        "reason": "upstream_install_source_invalid",
        "host": "apt.fogproject.org",
    }
    state.scratchpad["_context_invalidations"] = [
        {
            "invalidated_artifact_count": 2,
            "invalidated_observation_count": 1,
            "invalidated_summary_count": 0,
        }
    ]
    harness = SimpleNamespace(state=state)

    detail = _terminal_status_detail({"status": "stopped", "reason": "no_tool_calls"}, harness)
    rca = _build_backend_rca_strip(harness)

    assert detail.startswith("Blocked: installer source invalid or unavailable")
    assert "Phase: repair" in rca
    assert "Blocked: invalid upstream install source (apt.fogproject.org)" in rca
    assert "Context invalidated: artifacts=2, observations=1, summaries=0" in rca


def test_backend_rca_surfaces_latest_execution_blocker() -> None:
    state = LoopState()
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "dnf install -y vikunja-server",
        "latest_blocker": {
            "salient_error": "Errors during downloading metadata for repository 'vikunja': repomd.xml 404",
        },
    }
    harness = SimpleNamespace(state=state)

    rca = _build_backend_rca_strip(harness)

    assert "Primary blocker" in rca
    assert "repomd.xml 404" in rca


def test_should_render_event_shows_critical_events_despite_system_messages_off() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "context_lane_dropped", "message": "Dropped"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True

    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "fama_health_warning", "message": "FAMA warning"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True

    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "terminal_control_failed", "message": "Control failed"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True


def test_should_render_event_hides_normal_system_when_off() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "info", "message": "Some info"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is False


def test_build_backend_rca_strip_shows_failing_verifier() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "pytest tests/",
    }
    rca = _build_backend_rca_strip(harness)
    assert "Last failing verifier: `pytest tests/`" in rca


def test_build_backend_rca_strip_shows_passing_verifier() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.last_verifier_verdict = {
        "verdict": "pass",
        "command": "pihole status",
    }
    rca = _build_backend_rca_strip(harness)
    assert "Last passing verifier: `pihole status`" in rca


def test_build_backend_rca_strip_shows_recent_failures() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.recent_errors = ["Error 1", "Error 2", "Error 3"]
    rca = _build_backend_rca_strip(harness)
    assert "Recent failures:" in rca
    assert "Error 1" in rca
    assert "Error 2" in rca
    assert "Error 3" in rca


def test_build_backend_rca_strip_sanitizes_shell_recent_failures() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.recent_errors = [
        "ssh_exec: bash: line 1: pip3: command not found",
        "Guard tripped: max_consecutive_errors (5) - Errors: ['ssh_exec: /usr/bin/python3: No module named pip']",
    ]

    rca = _build_backend_rca_strip(harness)

    assert "ssh_exec failed; see tool output" in rca
    assert "pip3: command not found" not in rca
    assert "No module named pip" not in rca


def test_build_backend_rca_strip_shows_fama_health() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.scratchpad["_fama"] = {"signals": [{"kind": "preflight_contradiction"}]}
    rca = _build_backend_rca_strip(harness)
    assert "FAMA signals: 1" in rca


def test_build_backend_rca_strip_hides_fama_when_not_requested() -> None:
    harness = SimpleNamespace(state=LoopState())
    harness.state.current_phase = "repair"
    harness.state.scratchpad["_fama"] = {"signals": [{"kind": "preflight_contradiction"}]}

    rca = _build_backend_rca_strip(harness, include_fama=False)

    assert "Phase: repair" in rca
    assert "FAMA" not in rca


def test_build_backend_rca_strip_shows_hidden_artifacts() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    harness.state.scratchpad["suppressed_truncated_artifact_ids"] = ["art-1", "art-2"]
    rca = _build_backend_rca_strip(harness)
    assert "Hidden artifacts:" in rca
    assert "art-1" in rca


def test_build_backend_rca_strip_empty_when_no_data() -> None:
    harness = SimpleNamespace(
        state=LoopState(),
    )
    rca = _build_backend_rca_strip(harness)
    assert rca == ""


def test_format_run_log_row_timeout_override() -> None:
    row = {
        "channel": "harness",
        "event": "timeout_override",
        "data": {"requested_timeout_sec": 600, "effective_timeout_sec": 300, "reason": "capped by harness limit"},
    }
    formatted = format_run_log_row(row)
    assert "timeout capped" in formatted
    assert "⏱️" in formatted


def test_format_run_log_row_fama_health_alarm() -> None:
    row = {
        "channel": "harness",
        "event": "fama_capsule_health_warning",
        "data": {"warning": "FAMA capsules are empty for 3 consecutive prompts"},
    }
    formatted = format_run_log_row(row)
    assert "FAMA capsules are empty" in formatted
    assert "🚨" in formatted


def test_format_run_log_row_patch_recovery_autoread() -> None:
    row = {
        "channel": "harness",
        "event": "file_patch_read_autocontinue",
        "data": {"target_path": "temp/example.py", "error_kind": "patch_target_not_found"},
    }

    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "Patch recovery" in formatted
    assert "temp/example.py" in formatted
    assert "patch_target_not_found" in formatted


def test_format_run_log_row_task_interrupted_reason() -> None:
    row = {
        "channel": "harness",
        "event": "task_interrupted",
        "data": {"result": {"reason": "cancel_requested"}},
    }

    assert should_render_run_log_row(row) is True
    assert format_run_log_row(row) == "[harness] Task interrupted: cancel_requested"


def test_format_run_log_row_fama_signal_and_mitigation_are_visible() -> None:
    signal_row = {
        "channel": "harness",
        "event": "fama_signal_detected",
        "data": {"kind": "bad_tool_args", "failure_class": "patch_target_not_found", "tool_name": "file_patch"},
    }
    route_row = {
        "channel": "harness",
        "event": "fama_signal_to_mitigation",
        "data": {"signal_kind": "bad_tool_args", "activated_mitigations": ["patch_target_not_found_capsule"]},
    }
    mitigation_row = {
        "channel": "harness",
        "event": "fama_mitigation_activated",
        "data": {"mitigation": "patch_target_not_found_capsule", "reason": "bad_tool_args"},
    }

    assert should_render_run_log_row(signal_row) is True
    assert "patch_target_not_found" in format_run_log_row(signal_row)
    assert should_render_run_log_row(route_row) is True
    assert "patch_target_not_found_capsule" in format_run_log_row(route_row)
    assert should_render_run_log_row(mitigation_row) is True
    assert "FAMA mitigation active" in format_run_log_row(mitigation_row)


def test_format_run_log_row_same_scope_iteration_visible() -> None:
    row = {
        "channel": "harness",
        "event": "same_scope_iteration_recorded",
        "data": {"turn_type": "ITERATION"},
    }

    assert should_render_run_log_row(row) is True
    assert format_run_log_row(row) == "[harness] Same-scope follow-up recorded: ITERATION"


def test_critical_backend_interrupt_suppressed_by_default() -> None:
    console = ConsolePane(verbose=False)
    calls: list[str] = []
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "context_invalidated", "display_text": "Context invalidated"},
        content="Context invalidated",
    )

    async def _record_critical_interrupt(event: UIEvent) -> None:
        calls.append("critical_interrupt")

    async def _record_add_bubble(kind: str, text: str) -> object:
        calls.append(f"bubble:{kind}")
        return None

    console._append_critical_interrupt = _record_critical_interrupt  # type: ignore[assignment]
    console._add_bubble = _record_add_bubble  # type: ignore[assignment]
    asyncio.run(console.append_event(event))
    assert "critical_interrupt" not in calls
    assert not any(c.startswith("bubble:") for c in calls)
    assert console.get_last_system_message() == "Context invalidated"


def test_suppressed_system_event_preserves_active_assistant_turn() -> None:
    console = ConsolePane(verbose=False)
    active_turn = object()
    console._active_assistant_turn = active_turn  # type: ignore[assignment]
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "info", "display_text": "Background status"},
        content="Background status",
    )

    asyncio.run(console.append_event(event))

    assert console._active_assistant_turn is active_turn
    assert console.get_last_system_message() == "Background status"


def test_suppressed_critical_event_preserves_active_assistant_turn() -> None:
    console = ConsolePane(verbose=False)
    active_turn = object()
    console._active_assistant_turn = active_turn  # type: ignore[assignment]
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "context_invalidated", "display_text": "Context invalidated"},
        content="Context invalidated",
    )

    asyncio.run(console.append_event(event))

    assert console._active_assistant_turn is active_turn
    assert console.get_last_system_message() == "Context invalidated"


def test_critical_backend_interrupt_rendered_when_verbose() -> None:
    console = ConsolePane(verbose=True)
    calls: list[str] = []
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={"ui_kind": "context_invalidated", "display_text": "Context invalidated"},
        content="Context invalidated",
    )

    async def _record_critical_interrupt(event: UIEvent) -> None:
        calls.append("critical_interrupt")

    async def _record_add_bubble(kind: str, text: str) -> object:
        calls.append(f"bubble:{kind}")
        return None

    console._append_critical_interrupt = _record_critical_interrupt  # type: ignore[assignment]
    console._add_bubble = _record_add_bubble  # type: ignore[assignment]
    asyncio.run(console.append_event(event))
    assert "critical_interrupt" in calls
    assert not any(c.startswith("bubble:") for c in calls)


def test_console_verbose_toggle() -> None:
    console = ConsolePane(verbose=False)
    assert console._verbose is False
    console.set_verbose(True)
    assert console._verbose is True
    console.set_verbose(False)
    assert console._verbose is False


def test_should_render_event_shows_ssh_host_key_recovery_required() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        data={
            "event": "ssh_host_key_recovery_required",
            "host": "192.168.1.161",
            "suggested_command": "ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts",
        },
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True


def test_format_run_log_row_ssh_host_key_recovery_required() -> None:
    row = {
        "channel": "harness",
        "event": "ssh_host_key_recovery_required",
        "data": {
            "host": "192.168.1.161",
            "suggested_command": "ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts",
        },
    }
    formatted = format_run_log_row(row)
    assert "SSH host key changed for 192.168.1.161" in formatted
    assert "ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts" in formatted


def test_format_recovery_banner_ssh_host_key_recovery_required() -> None:
    data = {
        "host": "192.168.1.161",
        "suggested_command": "ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts",
    }
    banner = format_recovery_banner("ssh_host_key_recovery_required", data)
    assert "192.168.1.161" in banner
    assert "ssh-keygen -R" in banner


def test_should_render_event_shows_partial_tool_call_cancelled_when_system_hidden() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        content="Model had started streaming `file_write` when cancelled.",
        data={"ui_kind": "partial_tool_call_cancelled", "tool_name": "file_write"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True


def test_format_run_log_row_partial_tool_call_cancelled() -> None:
    row = {
        "channel": "harness",
        "event": "partial_tool_call_cancelled",
        "data": {"tool_name": "file_write", "argument_chars": 4096},
    }
    assert "Partial tool call cancelled before dispatch: file_write (4096 argument chars received)" in format_run_log_row(row)


def test_format_run_log_row_file_patch_fresh_read_required() -> None:
    row = {
        "channel": "harness",
        "event": "file_patch_fresh_read_required",
        "data": {"target_path": "temp/example.py", "recovery_count": 2, "error_kind": "patch_target_not_found"},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "fresh read required" in formatted.lower()
    assert "temp/example.py" in formatted


def test_format_run_log_row_file_patch_blocked_pending_fresh_read() -> None:
    row = {
        "channel": "harness",
        "event": "file_patch_blocked_pending_fresh_read",
        "data": {"target_path": "temp/example.py"},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "blocked" in formatted.lower()
    assert "temp/example.py" in formatted


def test_format_run_log_row_file_patch_fresh_read_satisfied() -> None:
    row = {
        "channel": "harness",
        "event": "file_patch_fresh_read_satisfied",
        "data": {"target_path": "temp/example.py", "read_path": "temp/example.py"},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "fresh read satisfied" in formatted.lower()
    assert "temp/example.py" in formatted


def test_format_run_log_row_near_budget_verifier_scheduled() -> None:
    row = {
        "channel": "harness",
        "event": "near_budget_verifier_scheduled",
        "data": {"command": "python3 ./temp/example.py info"},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "verification reserved" in formatted.lower()
    assert "python3 ./temp/example.py info" in formatted


def test_format_run_log_row_repair_stall_recovery() -> None:
    row = {
        "channel": "harness",
        "event": "repair_stall_recovery",
        "data": {"tool_name": "file_patch", "failure_class": "patch_target_not_found"},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "repair stall" in formatted.lower()
    assert "file_patch" in formatted


def test_format_run_log_row_stagnation_recovery() -> None:
    row = {
        "channel": "harness",
        "event": "stagnation_recovery",
        "data": {},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "stagnation recovery" in formatted.lower()


def test_format_run_log_row_guard_trip_diagnosis() -> None:
    row = {
        "channel": "harness",
        "event": "guard_trip_diagnosis",
        "data": {"guard_error": "max_steps exceeded", "recent_error_count": 5},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "guard diagnosis" in formatted.lower()
    assert "max_steps exceeded" in formatted


def test_format_run_log_row_write_overwrite_guard_read_autocontinue() -> None:
    row = {
        "channel": "harness",
        "event": "write_overwrite_guard_read_autocontinue",
        "data": {"target_path": "temp/example.py", "session_id": "ws-1"},
    }
    assert should_render_run_log_row(row) is True
    formatted = format_run_log_row(row)
    assert "write recovery" in formatted.lower()
    assert "temp/example.py" in formatted
    assert "file_patch" in formatted
    assert "ast_patch" in formatted


def test_critical_events_rendered_when_system_messages_hidden() -> None:
    for ui_kind in (
        "file_patch_fresh_read_required",
        "file_patch_blocked_pending_fresh_read",
        "file_patch_fresh_read_satisfied",
        "near_budget_verifier_scheduled",
        "repair_stall_recovery",
        "stagnation_recovery",
        "guard_trip_diagnosis",
        "write_overwrite_guard_read_autocontinue",
    ):
        event = UIEvent(
            event_type=UIEventType.SYSTEM,
            data={"ui_kind": ui_kind, "message": "recovery event"},
        )
        assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True


def test_format_recovery_banner_fresh_read_before_patch() -> None:
    banner = format_recovery_banner("file_patch_fresh_read_required", {})
    assert "fresh file read" in banner.lower()
    assert "patch" in banner.lower()

    banner = format_recovery_banner("file_patch_blocked_pending_fresh_read", {})
    assert "fresh file read" in banner.lower()
    assert "patch" in banner.lower()


def test_format_recovery_banner_near_budget_verifier() -> None:
    banner = format_recovery_banner("near_budget_verifier_scheduled", {})
    assert "verification reserved" in banner.lower()


def test_should_render_event_shows_tool_dispatch_cancelled_when_system_hidden() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        content="Tool dispatch cancelled: shell_exec.",
        data={"ui_kind": "tool_dispatch_cancelled", "tool_name": "shell_exec"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True


def test_format_run_log_row_tool_dispatch_cancelled() -> None:
    row = {
        "channel": "harness",
        "event": "tool_dispatch_cancelled",
        "data": {"tool_name": "shell_exec", "elapsed_sec": 3.711},
    }
    assert "Tool dispatch cancelled: shell_exec after 3.711s" in format_run_log_row(row)


def test_backend_failure_error_event_renders_red_bubble_by_default() -> None:
    console = ConsolePane(verbose=False)
    calls: list[str] = []
    event = UIEvent(
        event_type=UIEventType.ERROR,
        content="Stream error: backend wedged",
    )

    async def _record_add_bubble(kind: str, text: str) -> object:
        calls.append(f"bubble:{kind}:{text}")
        return None

    console._add_bubble = _record_add_bubble  # type: ignore[assignment]
    asyncio.run(console.append_event(event))
    assert any(c.startswith("bubble:error:") for c in calls)
    assert "Stream error: backend wedged" in calls[0]


def test_guard_trip_diagnosis_recovery_banner() -> None:
    banner = format_recovery_banner(
        "guard_trip_diagnosis",
        {"guard_error": "max_steps exceeded", "recent_error_count": 5},
    )
    assert "guard tripped" in banner.lower()
    assert "max_steps exceeded" in banner
    assert "5" in banner


def test_text_block_lite_markdown_styles_headers_and_lists() -> None:
    rendered = _style_text_lite("# Title\n- item 1\n- item `code`\n\n**bold** and *italic*\n")
    assert rendered.plain == "# Title\n• item 1\n• item code\n\nbold and italic\n"
    assert any(
        span.style == "bold #ffffff" and rendered.plain[span.start : span.end] == "Title"
        for span in rendered.spans
    )
    assert any(
        "bold #9ca3af" in span.style and "•" in rendered.plain[span.start : span.end]
        for span in rendered.spans
    )
    assert any(
        span.style == "italic #93c5fd" and rendered.plain[span.start : span.end] == "code"
        for span in rendered.spans
    )
    assert any(
        span.style == "bold" and rendered.plain[span.start : span.end] == "bold"
        for span in rendered.spans
    )
    assert any(
        span.style == "italic" and rendered.plain[span.start : span.end] == "italic"
        for span in rendered.spans
    )


def test_text_block_lite_markdown_styles_numbered_lists_and_quotes() -> None:
    rendered = _style_text_lite("1. first\n2. second\n\n> quote")
    assert "1." in rendered.plain
    assert "2." in rendered.plain
    assert "> quote" in rendered.plain
    assert any(
        "bold #9ca3af" in span.style and "1." in rendered.plain[span.start : span.end]
        for span in rendered.spans
    )
    assert any(
        span.style == "#9ca3af" and rendered.plain[span.start : span.end] == ">"
        for span in rendered.spans
    )


def test_text_block_lite_markdown_fences_are_dim() -> None:
    rendered = _style_text_lite("```python\nx = 1\n```")
    assert rendered.plain == "```python\nx = 1\n```"
    assert all(
        span.style == "#9ca3af" or not span.style for span in rendered.spans
    )


def test_text_block_unclosed_inline_code_falls_back_to_plain() -> None:
    rendered = _style_text_lite("start `unclosed")
    assert rendered.plain == "start `unclosed"
    assert not rendered.spans


def test_text_block_lite_markdown_preserves_plain_text() -> None:
    rendered = _style_text_lite("plain text without markup")
    assert rendered.plain == "plain text without markup"
    assert not rendered.spans


def test_text_block_lite_markdown_handles_strikethrough() -> None:
    rendered = _style_text_lite("~~deleted~~\n")
    assert rendered.plain == "deleted\n"
    assert any(
        span.style == "strike" and rendered.plain[span.start : span.end] == "deleted"
        for span in rendered.spans
    )


def test_text_block_lite_markdown_skips_inline_on_incomplete_last_line() -> None:
    rendered = _style_text_lite("**bold** and `code`")
    assert rendered.plain == "**bold** and `code`"
    assert not rendered.spans


def test_text_block_lite_markdown_streaming_chunk_boundary_safe() -> None:
    # Simulate chunks that split a bullet + bold line. Block-level markers
    # (like bullets) are styled even on incomplete lines, but inline emphasis
    # and code are left plain until the line completes.
    chunks = ["*   ", "**CPU", ":** root", " PID `4117806`"]
    acc = ""
    for chunk in chunks:
        acc += chunk
    rendered = _style_text_lite(acc)
    assert rendered.plain == "•   **CPU:** root PID `4117806`"
    # Bullet marker styled, inline markers not styled yet.
    assert any("bold #9ca3af" in span.style for span in rendered.spans)
    assert not any(span.style == "bold" for span in rendered.spans)

    rendered = _style_text_lite(acc + "\n")
    assert "•   CPU:" in rendered.plain
    assert any(span.style == "bold" for span in rendered.spans)


def test_inline_code_backticks_balanced_ignores_fences_and_embedded_ticks() -> None:
    assert _inline_code_backticks_balanced("plain text") is True
    assert _inline_code_backticks_balanced("`code`") is True
    assert _inline_code_backticks_balanced("`` ` ``") is True
    assert _inline_code_backticks_balanced("`unclosed") is False
    assert _inline_code_backticks_balanced("```\ncode\n```") is True
    assert _inline_code_backticks_balanced("```\ncode\n```\n`inline`") is True
    assert _inline_code_backticks_balanced("```\n`nested`\n```") is True
    assert _inline_code_backticks_balanced("`a` ``b``") is True
