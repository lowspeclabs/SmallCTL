from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph import tool_call_parser_support, tool_loop_guards
from smallctl.graph.state import PendingToolCall
from smallctl.graph.tool_call_parser import (
    _record_tool_attempt as _dispatch_record_tool_attempt,
)
from smallctl.graph.tool_inline_parsing import _extract_inline_tool_calls
from smallctl.graph.tool_loop_guard_constants import _STRICT_LOOP_GUARD_IDENTICAL_LIMIT
from smallctl.graph.tool_loop_guards import _detect_repeated_tool_loop
from smallctl.graph.shell_outcomes import (
    _tool_call_fingerprint as _shell_outcomes_fingerprint,
)
from smallctl.repeat_loop_policy import strict_identical_limit


# ---------------------------------------------------------------------------
# C1 — Tool-call argument "repair" silently corrupts valid JSON
# ---------------------------------------------------------------------------


def _pending_from_raw(raw_arguments: str, tool_name: str = "shell_exec") -> PendingToolCall:
    pending = PendingToolCall.from_payload(
        {"function": {"name": tool_name, "arguments": raw_arguments}}
    )
    assert pending is not None
    return pending


def test_c1_valid_json_regex_quantifier_passes_through_byte_identical() -> None:
    raw = '{"command": "grep -E \'[0-9]{1,}\' f"}'
    pending = _pending_from_raw(raw)
    assert pending.args == {"command": "grep -E '[0-9]{1,}' f"}
    assert "arguments_repaired" not in pending.parser_metadata


def test_c1_valid_json_comma_bracket_inside_string_passes_through() -> None:
    raw = '{"command": "echo \'a,]b\'"}'
    pending = _pending_from_raw(raw)
    assert pending.args == {"command": "echo 'a,]b'"}
    assert "arguments_repaired" not in pending.parser_metadata


def test_c1_file_write_content_with_comma_bracket_unchanged() -> None:
    raw = '{"path": "rules.txt", "content": "a,]b and {1,} stay"}'
    pending = _pending_from_raw(raw, tool_name="file_write")
    assert pending.args == {"path": "rules.txt", "content": "a,]b and {1,} stay"}
    assert "arguments_repaired" not in pending.parser_metadata


def test_c1_trailing_comma_outside_strings_still_repaired() -> None:
    # ``true`` makes ast.literal_eval fail too, so the repair path must run.
    pending = _pending_from_raw('{"command": "ls -la", "background": true,}')
    assert pending.args == {"command": "ls -la", "background": True}
    assert pending.parser_metadata.get("arguments_repaired") is True


def test_c1_trailing_comma_repair_never_touches_string_literals() -> None:
    raw = '{"command": "echo \',]\'", "ok": true, "tags": ["a", "b",],}'
    pending = _pending_from_raw(raw)
    assert pending.args == {"command": "echo ',]'", "ok": True, "tags": ["a", "b"]}
    assert pending.parser_metadata.get("arguments_repaired") is True


def test_c1_trailing_comma_after_string_ending_in_comma() -> None:
    pending = _pending_from_raw('{"command": "ls,", "flag": true,}')
    assert pending.args == {"command": "ls,", "flag": True}
    assert pending.parser_metadata.get("arguments_repaired") is True


# ---------------------------------------------------------------------------
# H1 — Loop-guard fingerprint mismatch disables fast identical-call guards
# ---------------------------------------------------------------------------


def test_h1_parser_support_reexports_canonical_loop_guard_implementations() -> None:
    assert tool_call_parser_support._record_tool_attempt is tool_loop_guards._record_tool_attempt
    assert (
        tool_call_parser_support._clear_tool_attempt_history
        is tool_loop_guards._clear_tool_attempt_history
    )
    assert (
        tool_call_parser_support._tool_call_fingerprint is tool_loop_guards._tool_call_fingerprint
    )
    assert tool_call_parser_support._normalize_tool_args is tool_loop_guards._normalize_tool_args
    assert tool_call_parser_support._normalize_json_like is tool_loop_guards._normalize_json_like
    # tool_execution_nodes / lifecycle_nodes import via tool_call_parser.
    assert _dispatch_record_tool_attempt is tool_loop_guards._record_tool_attempt
    assert _shell_outcomes_fingerprint is tool_loop_guards._tool_call_fingerprint


def test_h1_fingerprint_parity_for_same_inputs(tmp_path) -> None:
    args = {"path": "src/app.py", "start_line": 1, "end_line": 20}
    support_fingerprint = tool_call_parser_support._tool_call_fingerprint(
        "file_read", args, cwd=str(tmp_path)
    )
    guards_fingerprint = tool_loop_guards._tool_call_fingerprint(
        "file_read", args, cwd=str(tmp_path)
    )
    assert support_fingerprint == guards_fingerprint


def test_h1_relative_path_file_tool_detection_trips_at_strict_limit(tmp_path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("print('ok')\n", encoding="utf-8")
    harness = SimpleNamespace(state=SimpleNamespace(scratchpad={}, cwd=str(tmp_path)))
    args = {"path": "src/app.py", "start_line": 1, "end_line": 20}
    pending = PendingToolCall(tool_name="file_read", args=args)

    identical_limit = strict_identical_limit("file_read", _STRICT_LOOP_GUARD_IDENTICAL_LIMIT)
    for _ in range(identical_limit - 2):
        _dispatch_record_tool_attempt(harness, pending)

    # One below the strict identical-call limit: no guard trip yet.
    assert _detect_repeated_tool_loop(harness, pending) is None

    _dispatch_record_tool_attempt(harness, pending)
    guard_error = _detect_repeated_tool_loop(harness, pending)
    assert guard_error is not None
    assert "repeated tool call loop" in guard_error

    history = harness.state.scratchpad["_tool_attempt_history"]
    assert history[-1]["fingerprint"] == tool_loop_guards._tool_call_fingerprint(
        "file_read", args, cwd=str(tmp_path)
    )


# ---------------------------------------------------------------------------
# H11 — Inline JSON brace scanner not string-aware
# ---------------------------------------------------------------------------


def test_h11_unbalanced_brace_inside_command_string_recovers_call() -> None:
    text = (
        '{"name":"shell_exec","arguments":{"command":"printf \'%s\\n\' \'{\' >> out.txt"}}'
    )
    cleaned, calls = _extract_inline_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "shell_exec"
    assert calls[0].args["command"] == "printf '%s\n' '{' >> out.txt"
    assert cleaned.strip() == ""


def test_h11_escaped_quotes_and_braces_inside_content_recover_call() -> None:
    text = (
        '{"tool_name":"file_write","arguments":{"path":"out.txt",'
        '"content":"she said \\"{\\" and }{{"}}'
    )
    cleaned, calls = _extract_inline_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "file_write"
    assert calls[0].args["path"] == "out.txt"
    assert calls[0].args["content"] == 'she said "{" and }{{'
    assert cleaned.strip() == ""


def test_h11_truncated_inline_json_still_recovered() -> None:
    text = '{"name":"shell_exec","arguments":{"command":"echo hi"'
    _cleaned, calls = _extract_inline_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "shell_exec"
    assert calls[0].args["command"] == "echo hi"
