from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.progress_guard import (
    _is_ssh_exec_read_command,
    _ssh_exec_has_novel_remote_observation,
    _ssh_exec_read_is_new,
    _record_progress_read,
    _turn_has_actionable_progress,
)
from smallctl.state import LoopState


def test_is_ssh_exec_read_command_detects_cat_and_ls() -> None:
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": "cat /var/www/html/index.html"})) is True
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": "ls -la /var/www/html/"})) is True
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": "grep -r foo /etc/"})) is True
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": "apt list --upgradable 2>/dev/null | grep '/' | wc -l"})) is True
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": "sed -i 's/old/new/' /etc/nginx.conf"})) is False
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": "apt-get update -qq 2>&1 && apt list --upgradable 2>/dev/null"})) is False
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": "systemctl restart nginx"})) is False
    assert _is_ssh_exec_read_command(SimpleNamespace(args={"command": ""})) is False


def test_ssh_exec_read_is_new_detects_repeats() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_read_history"] = [
        {"tool_name": "ssh_exec", "command": "cat /var/www/html/index.html"},
    ]
    harness = SimpleNamespace(state=state)

    new_record = SimpleNamespace(args={"command": "cat /var/www/html/index.html"})
    assert _ssh_exec_read_is_new(harness, new_record) is False

    new_record2 = SimpleNamespace(args={"command": "cat /var/www/html/page.html"})
    assert _ssh_exec_read_is_new(harness, new_record2) is True


def test_turn_has_actionable_progress_counts_new_ssh_exec_read() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_read_history"] = []
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="ssh_exec",
                args={"command": "cat /var/www/html/index.html"},
                result=SimpleNamespace(success=True, metadata={}),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_turn_has_actionable_progress_ignores_repeated_ssh_exec_read() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_read_history"] = [
        {"tool_name": "ssh_exec", "command": "cat /var/www/html/index.html"},
    ]
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="ssh_exec",
                args={"command": "cat /var/www/html/index.html"},
                result=SimpleNamespace(success=True, metadata={}),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is False


def test_record_progress_read_records_ssh_exec_command() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(state=state)

    record = SimpleNamespace(
        tool_name="ssh_exec",
        args={"command": "cat /var/www/html/index.html"},
        result=SimpleNamespace(success=True),
    )
    _record_progress_read(harness, record)

    history = state.scratchpad["_progress_read_history"]
    assert len(history) == 1
    assert history[0]["tool_name"] == "ssh_exec"
    assert history[0]["command"] == "cat /var/www/html/index.html"


def test_turn_has_actionable_progress_counts_new_apt_list_pipeline() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_read_history"] = []
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="ssh_exec",
                args={"command": "apt list --upgradable 2>/dev/null | grep '/' | wc -l"},
                result=SimpleNamespace(success=True, metadata={}),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_turn_has_actionable_progress_counts_non_read_ssh_exec_as_progress() -> None:
    """Successful remote mutation commands (e.g. systemctl restart) represent real progress."""
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_read_history"] = []
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="ssh_exec",
                args={"command": "systemctl restart nginx"},
                result=SimpleNamespace(success=True, metadata={}),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_turn_has_actionable_progress_counts_apt_get_install_as_progress() -> None:
    """Package installation via ssh_exec should count as actionable progress."""
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_read_history"] = []
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="ssh_exec",
                args={"command": "apt-get install -y nginx"},
                result=SimpleNamespace(success=True, metadata={}),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_turn_has_actionable_progress_counts_new_ssh_error_class() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_ssh_observation_history"] = [
        {"host": "192.168.1.63", "failure_class": "connection_timeout", "paths": [], "auth_mode": "key", "reached_remote_host": False},
    ]
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="ssh_exec",
                args={"host": "192.168.1.63", "command": "whoami"},
                result=SimpleNamespace(
                    success=False,
                    metadata={"failure_kind": "transport", "ssh_error_class": "auth_permission_denied", "ssh_auth_mode": "key"},
                ),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_turn_has_actionable_progress_counts_new_remote_target_path() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_ssh_observation_history"] = [
        {"host": "192.168.1.63", "failure_class": "remote_exit_nonzero", "paths": ["/etc/nginx/nginx.conf"], "auth_mode": "password", "reached_remote_host": True},
    ]
    harness = SimpleNamespace(state=state)

    record = SimpleNamespace(
        tool_name="ssh_exec",
        args={"host": "192.168.1.63", "command": "cat /etc/nginx/sites-enabled/default"},
        result=SimpleNamespace(
            success=False,
            metadata={"failure_kind": "remote_command", "ssh_error_class": "remote_exit_nonzero", "ssh_transport_succeeded": True, "ssh_auth_mode": "password"},
        ),
    )

    assert _ssh_exec_has_novel_remote_observation(harness, record) is True


def test_turn_has_actionable_progress_counts_successful_auth_mode_change_for_repeated_read() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_progress_read_history"] = [
        {"tool_name": "ssh_exec", "command": "cat /var/www/html/index.html"},
    ]
    state.scratchpad["_progress_ssh_observation_history"] = [
        {"host": "192.168.1.63", "failure_class": "", "paths": ["/var/www/html/index.html"], "auth_mode": "key", "reached_remote_host": True},
    ]
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="ssh_exec",
                args={"host": "192.168.1.63", "command": "cat /var/www/html/index.html"},
                result=SimpleNamespace(
                    success=True,
                    metadata={"ssh_auth_mode": "password"},
                ),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True
