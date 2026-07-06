from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.progress_guard import (
    _artifact_read_is_continuation_page,
    _check_completion_confabulation,
    _tool_history_entry_is_mutation,
    _turn_has_actionable_progress,
    _record_is_mutation,
    clear_stale_confabulation_nudge,
)
from smallctl.state import LoopState


def test_turn_has_actionable_progress_counts_file_patch() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="file_patch",
                args={"path": "./temp/pong.py"},
                result=SimpleNamespace(
                    success=True,
                    metadata={"path": "./temp/pong.py", "changed": True},
                ),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_turn_has_actionable_progress_counts_file_write() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="file_write",
                args={"path": "./temp/pong.py"},
                result=SimpleNamespace(
                    success=True,
                    metadata={"path": "./temp/pong.py", "changed": True},
                ),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_artifact_read_continuation_page_counts_as_progress() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_artifact_read_coverage"] = {
        "pong_py": {
            "ranges": [{"start_line": 1, "end_line": 150}],
        }
    }
    harness = SimpleNamespace(state=state)

    record = SimpleNamespace(
        tool_name="artifact_read",
        args={"artifact_id": "pong_py", "start_line": 150, "end_line": 275},
        result=SimpleNamespace(success=True, metadata={}),
    )

    assert _artifact_read_is_continuation_page(harness, record) is True

    graph_state = SimpleNamespace(
        last_tool_results=[record],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_tool_history_entry_is_mutation_counts_file_write() -> None:
    assert _tool_history_entry_is_mutation(
        'file_write|{"path": "./temp/pong.py"}|success'
    ) is True


def test_tool_history_entry_is_mutation_counts_shell_mkdir() -> None:
    assert _tool_history_entry_is_mutation(
        'shell_exec|{"command": "mkdir -p ./temp"}|success'
    ) is True


def test_tool_history_entry_is_mutation_ignores_shell_read() -> None:
    assert _tool_history_entry_is_mutation(
        'shell_exec|{"command": "ls -la ./temp"}|success'
    ) is False


def test_tool_history_entry_is_mutation_ignores_failed_shell() -> None:
    assert _tool_history_entry_is_mutation(
        'shell_exec|{"command": "rm -rf ./temp"}|error:exit 1'
    ) is False


def test_check_completion_confabulation_skips_after_shell_mkdir() -> None:
    state = LoopState(cwd="/tmp")
    state.tool_history.append(
        'shell_exec|{"command": "mkdir -p ./tmp"}|success'
    )
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)

    graph_state = SimpleNamespace(
        last_assistant_text="I have already created the directory.",
        last_thinking_text="",
    )

    assert _check_completion_confabulation(harness, graph_state) is None
    assert state.scratchpad.get("_confabulation_nudged") is None


def test_record_is_mutation_counts_ssh_exec_backup_script() -> None:
    record = SimpleNamespace(
        tool_name="ssh_exec",
        args={"command": "/root/backup.sh", "host": "192.168.1.64", "user": "root"},
        result=SimpleNamespace(success=True),
    )
    assert _record_is_mutation(record) is True


def test_record_is_mutation_ignores_ssh_exec_read() -> None:
    record = SimpleNamespace(
        tool_name="ssh_exec",
        args={"command": "cat /root/backup.sh", "host": "192.168.1.64", "user": "root"},
        result=SimpleNamespace(success=True),
    )
    assert _record_is_mutation(record) is False


def test_clear_stale_confabulation_nudge_removes_message_after_mutation() -> None:
    from smallctl.models.conversation import ConversationMessage

    state = LoopState(cwd="/tmp")
    state.scratchpad["_confabulation_nudged"] = True
    nudge = ConversationMessage(
        role="user",
        content="GROUND TRUTH CHECK: No mutating operations...",
        metadata={"is_recovery_nudge": True, "recovery_kind": "completion_confabulation"},
    )
    other = ConversationMessage(role="user", content="Do the task.", metadata={})
    state.messages = [other, nudge]

    clear_stale_confabulation_nudge(state)

    assert state.scratchpad.get("_confabulation_nudged") is None
    assert nudge not in state.messages
    assert other in state.messages

