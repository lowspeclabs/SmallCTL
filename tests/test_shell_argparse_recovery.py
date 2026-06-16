from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.frame_compiler import _maybe_inject_argparse_subcommand_note
from smallctl.context.messages_compact_helpers import collapse_repeated_shell_failures
from smallctl.harness.task_classifier_support import task_is_local_coding_target
from smallctl.models.conversation import ConversationMessage
from smallctl.state_schema import WorkingMemory
from smallctl.tools.shell_support_argparse import (
    _build_argparse_unrecognized_args_hint,
    _detect_unbalanced_quotes,
    _extract_unrecognized_argparse_arguments,
)
from smallctl.tools.shell_support_constants import _ARGPARSE_UNRECOGNIZED_ARGS_PATTERN


def test_detect_unbalanced_quotes_finds_single_quote() -> None:
    msg = _detect_unbalanced_quotes("echo '")
    assert msg is not None
    assert "unmatched single quote" in msg


def test_detect_unbalanced_quotes_finds_double_quote() -> None:
    msg = _detect_unbalanced_quotes('echo "')
    assert msg is not None
    assert "unmatched double quote" in msg


def test_detect_unbalanced_quotes_allows_balanced() -> None:
    assert _detect_unbalanced_quotes("echo 'hello'") is None
    assert _detect_unbalanced_quotes('echo "hello"') is None
    assert _detect_unbalanced_quotes("echo `date`") is None


def test_task_is_local_coding_target_broadened_to_any_local_py() -> None:
    assert task_is_local_coding_target("python3 ./temp/vikunja-9b.py") is True
    assert task_is_local_coding_target("python3 ./scripts/my-tool.py") is True
    assert task_is_local_coding_target("python3 ./vikunja-9b.py --url X") is True


def test_task_is_local_coding_target_still_rejects_remote() -> None:
    task = "SSH to 192.168.1.10 and run ./temp/script.py"
    assert task_is_local_coding_target(task) is False


def test_extract_unrecognized_argparse_arguments_finds_flags() -> None:
    error = "error: unrecognized arguments: --url --token"
    result = _extract_unrecognized_argparse_arguments(error)
    assert result == ["--url", "--token"]


def test_extract_unrecognized_argparse_arguments_returns_empty_when_clean() -> None:
    error = "error: the following arguments are required: --url"
    result = _extract_unrecognized_argparse_arguments(error)
    assert result == []


def test_build_argparse_unrecognized_args_hint_includes_ordering_guidance() -> None:
    hint = _build_argparse_unrecognized_args_hint("script.py", ["--url", "--token"])
    assert hint is not None
    assert "place global flags" in hint.lower()
    assert "BEFORE the subcommand" in hint


def test_build_argparse_unrecognized_args_hint_returns_none_for_empty() -> None:
    assert _build_argparse_unrecognized_args_hint("script.py", []) is None


def test_argparse_unrecognized_pattern_matches_variants() -> None:
    assert _ARGPARSE_UNRECOGNIZED_ARGS_PATTERN.search("unrecognized arguments: --foo")
    assert _ARGPARSE_UNRECOGNIZED_ARGS_PATTERN.search("error: unrecognized arguments: --foo --bar")
    assert not _ARGPARSE_UNRECOGNIZED_ARGS_PATTERN.search("required arguments: --foo")


def test_collapse_repeated_shell_failures_preserves_under_threshold() -> None:
    messages = [
        ConversationMessage(role="tool", name="shell_exec", content="failed: error"),
        ConversationMessage(role="tool", name="shell_exec", content="failed: error"),
    ]
    result = collapse_repeated_shell_failures(messages)
    assert len(result) == 2


def test_collapse_repeated_shell_failures_collapses_identical_failures() -> None:
    messages = [
        ConversationMessage(role="tool", name="shell_exec", content="failed: same error"),
        ConversationMessage(role="tool", name="shell_exec", content="failed: same error"),
        ConversationMessage(role="tool", name="shell_exec", content="failed: same error"),
        ConversationMessage(role="tool", name="shell_exec", content="failed: same error"),
    ]
    result = collapse_repeated_shell_failures(messages)
    assert len(result) == 3
    assert "collapsed to save tokens" in result[0].content
    assert result[1].content == "failed: same error"
    assert result[2].content == "failed: same error"


def test_collapse_repeated_shell_failures_ignores_non_failures() -> None:
    messages = [
        ConversationMessage(role="tool", name="shell_exec", content="success: done"),
        ConversationMessage(role="tool", name="shell_exec", content="success: done"),
        ConversationMessage(role="tool", name="shell_exec", content="success: done"),
    ]
    result = collapse_repeated_shell_failures(messages)
    assert len(result) == 3


def test_failure_event_suggests_pip_install_on_module_not_found() -> None:
    from smallctl.graph.tool_outcomes import _failure_event_from_record
    from smallctl.graph.state import ToolExecutionRecord
    from smallctl.models.tool_result import ToolEnvelope

    record = ToolExecutionRecord(
        operation_id="op1",
        tool_name="shell_exec",
        tool_call_id="tc1",
        args={"command": "python3 script.py"},
        result=ToolEnvelope(
            success=False,
            error="ModuleNotFoundError: No module named 'requests'",
            output={"stderr": "ModuleNotFoundError: No module named 'requests'"},
        ),
    )
    harness = SimpleNamespace(state=SimpleNamespace())
    event = _failure_event_from_record(harness, record, subtask_id="")
    assert event.suggested_next_action == "pip install requests"


def test_failure_event_generic_suggestion_when_no_module_error() -> None:
    from smallctl.graph.tool_outcomes import _failure_event_from_record
    from smallctl.graph.state import ToolExecutionRecord
    from smallctl.models.tool_result import ToolEnvelope

    record = ToolExecutionRecord(
        operation_id="op1",
        tool_name="shell_exec",
        tool_call_id="tc1",
        args={"command": "ls"},
        result=ToolEnvelope(
            success=False,
            error="Permission denied",
            output={"stderr": "Permission denied"},
        ),
    )
    harness = SimpleNamespace(state=SimpleNamespace())
    event = _failure_event_from_record(harness, record, subtask_id="")
    assert "next smallest different action" in event.suggested_next_action


def test_maybe_inject_argparse_subcommand_note_detects_subparser() -> None:
    wm = WorkingMemory()
    state = SimpleNamespace(
        working_memory=wm,
        artifacts={
            "a1": {
                "kind": "file_read",
                "source": "./script.py",
                "inline_content": "parser = argparse.ArgumentParser()\nsubparsers = parser.add_subparsers()\nsubparsers.add_parser('list')\nparser.add_argument('--url')",
            }
        },
    )
    _maybe_inject_argparse_subcommand_note(state)
    assert len(wm.known_facts) == 1
    assert "CLI subcommand ordering" in wm.known_facts[0]


def test_maybe_inject_argparse_skips_non_py_files() -> None:
    wm = WorkingMemory()
    state = SimpleNamespace(
        working_memory=wm,
        artifacts={
            "a1": {
                "kind": "file_read",
                "source": "./script.sh",
                "inline_content": "parser = argparse.ArgumentParser()\nsubparsers.add_parser('list')",
            }
        },
    )
    _maybe_inject_argparse_subcommand_note(state)
    assert len(wm.known_facts) == 0


def test_maybe_inject_argparse_skips_when_already_present() -> None:
    wm = WorkingMemory()
    wm.known_facts.append(
        "CLI subcommand ordering: this script uses argparse subcommands. Place global flags (e.g., --url, --token) BEFORE the subcommand, not after."
    )
    state = SimpleNamespace(
        working_memory=wm,
        artifacts={
            "a1": {
                "kind": "file_read",
                "source": "./script.py",
                "inline_content": "subparsers.add_parser('list')",
            }
        },
    )
    _maybe_inject_argparse_subcommand_note(state)
    assert len(wm.known_facts) == 1
