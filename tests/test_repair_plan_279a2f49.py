from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.harness.task_classifier_support import task_is_local_coding_target
from smallctl.tools.shell_support_argparse import (
    _build_argparse_unrecognized_args_hint,
    _extract_unrecognized_argparse_arguments,
)
from smallctl.tools.shell_support_constants import _ARGPARSE_UNRECOGNIZED_ARGS_PATTERN
from smallctl.context.messages_compact_helpers import collapse_repeated_shell_failures
from smallctl.models.conversation import ConversationMessage


# ─── P1.3: Local coding target regex ─────────────────────────────────────────

def test_task_is_local_coding_target_broadened_to_any_local_py() -> None:
    assert task_is_local_coding_target("python3 ./temp/vikunja-9b.py") is True
    assert task_is_local_coding_target("python3 ./scripts/my-tool.py") is True
    assert task_is_local_coding_target("python3 ./vikunja-9b.py --url X") is True


def test_task_is_local_coding_target_still_rejects_remote() -> None:
    task = "SSH to 192.168.1.10 and run ./temp/script.py"
    assert task_is_local_coding_target(task) is False


# ─── P2.5: Argparse unrecognized args recovery ───────────────────────────────

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


# ─── P2.6: Repeated shell failure compaction ─────────────────────────────────

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
    # Should collapse first 2, keep last 2
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


# ─── P2.8: Pip install suggestion ────────────────────────────────────────────

def test_failure_event_suggests_pip_install_on_module_not_found() -> None:
    from smallctl.graph.tool_outcomes import _failure_event_from_record
    from smallctl.models.tool_result import ToolEnvelope
    from smallctl.graph.state import ToolExecutionRecord

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
    from smallctl.models.tool_result import ToolEnvelope
    from smallctl.graph.state import ToolExecutionRecord

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


# ─── P3.9: Tool-plan fallback for repeated shell failures ────────────────────

def test_detect_tool_plan_hard_route_triggers_on_repeated_shell_failures() -> None:
    from smallctl.fama.detectors import detect_tool_plan_hard_route

    state = SimpleNamespace(
        scratchpad={
            "_repeated_failure_observations": [
                {
                    "key": "shell_exec::argparse_error",
                    "tool_name": "shell_exec",
                    "domain": "",
                    "pattern": "argparse_error",
                    "count": 3,
                    "last_step": 5,
                    "first_step": 1,
                }
            ]
        },
        step_count=5,
    )
    assert detect_tool_plan_hard_route(state) is True
    assert state.scratchpad.get("_fama_force_tool_plan_next_turn") is True


def test_detect_tool_plan_hard_route_ignores_repeated_non_shell_failures() -> None:
    from smallctl.fama.detectors import detect_tool_plan_hard_route

    state = SimpleNamespace(
        scratchpad={
            "_repeated_failure_observations": [
                {
                    "key": "file_read::not_found",
                    "tool_name": "file_read",
                    "domain": "",
                    "pattern": "not_found",
                    "count": 3,
                    "last_step": 5,
                    "first_step": 1,
                }
            ]
        },
        step_count=5,
    )
    assert detect_tool_plan_hard_route(state) is False


# ─── P1.1: FAMA loop-mode guard ──────────────────────────────────────────────

def _apply_fama_loop_mode_guard(config: Any) -> None:
    """Mirror of the guard logic from initialization.py for testing."""
    if (
        str(config.run_mode or "").strip().lower() == "loop"
        and not config.fama_enabled
        and not config.fama_disabled
    ):
        config.fama_enabled = True


def test_fama_disabled_flag_blocks_loop_mode_guard() -> None:
    from smallctl.harness.config import HarnessConfig

    config = HarnessConfig(
        endpoint="http://localhost:11434",
        model="test",
        run_mode="loop",
        fama_enabled=False,
        fama_disabled=True,
    )
    _apply_fama_loop_mode_guard(config)
    assert config.fama_enabled is False  # Should stay disabled


def test_loop_mode_guard_auto_enables_fama() -> None:
    from smallctl.harness.config import HarnessConfig

    config = HarnessConfig(
        endpoint="http://localhost:11434",
        model="test",
        run_mode="loop",
        fama_enabled=False,
        fama_disabled=False,
    )
    _apply_fama_loop_mode_guard(config)
    assert config.fama_enabled is True  # Guard should flip it


def test_loop_mode_guard_respects_non_loop_mode() -> None:
    from smallctl.harness.config import HarnessConfig

    config = HarnessConfig(
        endpoint="http://localhost:11434",
        model="test",
        run_mode="chat",
        fama_enabled=False,
        fama_disabled=False,
    )
    _apply_fama_loop_mode_guard(config)
    assert config.fama_enabled is False  # Should NOT flip in chat mode


# ─── P2.7: Subcommand heuristic injection ────────────────────────────────────

def test_maybe_inject_argparse_subcommand_note_detects_subparser() -> None:
    from smallctl.context.frame_compiler import _maybe_inject_argparse_subcommand_note
    from smallctl.state_schema import WorkingMemory

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
    from smallctl.context.frame_compiler import _maybe_inject_argparse_subcommand_note
    from smallctl.state_schema import WorkingMemory

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
    from smallctl.context.frame_compiler import _maybe_inject_argparse_subcommand_note
    from smallctl.state_schema import WorkingMemory

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
    assert len(wm.known_facts) == 1  # Should not duplicate
