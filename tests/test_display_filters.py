from smallctl.models.events import UIEvent, UIEventType
from smallctl.ui.display import should_render_event


def test_alert_suppressed_when_system_messages_hidden() -> None:
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="Gathering planning facts...",
        data={"status_activity": "gathering facts..."},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=True) is False


def test_alert_visible_when_system_messages_shown() -> None:
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="Gathering planning facts...",
        data={"status_activity": "gathering facts..."},
    )
    assert should_render_event(event, show_system_messages=True, show_tool_calls=True) is True


def test_alert_with_interrupt_visible_even_when_system_hidden() -> None:
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="Plan ready. Execute it now?",
        data={"interrupt": {"kind": "plan_execute_approval"}},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=True) is True


def test_alert_with_ui_kind_approve_prompt_visible_when_system_hidden() -> None:
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="Approve shell command?",
        data={"ui_kind": "approve_prompt", "approval_id": "shell-123"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=True) is True


def test_alert_with_ui_kind_sudo_password_visible_when_system_hidden() -> None:
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="Sudo password required.",
        data={"ui_kind": "sudo_password_prompt", "prompt_id": "sudo-123"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=True) is True


def test_subtask_checklist_alert_visible_when_system_hidden() -> None:
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="[ ] task: SSH to remote server - needed",
        data={"ui_kind": "subtask_checklist"},
    )
    assert should_render_event(event, show_system_messages=False, show_tool_calls=False) is True


def test_system_event_suppressed_when_system_messages_hidden() -> None:
    event = UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled.")
    assert should_render_event(event, show_system_messages=False, show_tool_calls=True) is False


def test_status_event_always_suppressed() -> None:
    event = UIEvent(event_type=UIEventType.STATUS, data={"snapshot": {}})
    assert should_render_event(event, show_system_messages=True, show_tool_calls=True) is False


def test_tool_call_suppressed_when_tool_calls_hidden() -> None:
    event = UIEvent(event_type=UIEventType.TOOL_CALL, content="shell_exec")
    assert should_render_event(event, show_system_messages=True, show_tool_calls=False) is False


def test_tool_result_visible_when_tool_calls_shown() -> None:
    event = UIEvent(event_type=UIEventType.TOOL_RESULT, content="ok")
    assert should_render_event(event, show_system_messages=True, show_tool_calls=True) is True
