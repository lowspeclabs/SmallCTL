from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness.conversation_logging import record_assistant_message
from smallctl.state import LoopState


def _make_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _refresh_task_handoff_action_options=lambda _text: None,
    )


def test_record_assistant_message_strips_think_tags_and_preserves_thinking() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    record_assistant_message(
        harness,
        assistant_text="Hello.<think>I should read the file.</think>",
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.role == "assistant"
    assert msg.content == "Hello."
    assert msg.metadata.get("thinking_text") == "I should read the file."


def test_record_assistant_message_strips_channel_markers() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    record_assistant_message(
        harness,
        assistant_text="I'll call file_read.<|channel>thought: read the file<channel|>",
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.content == "I'll call file_read."
    assert "thought: read the file" in msg.metadata.get("thinking_text", "")


def test_record_assistant_message_strips_channel_markers_both_sides() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    record_assistant_message(
        harness,
        assistant_text="Ready.<|channel|>inner thought</|channel|>",
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.content == "Ready."
    assert "inner thought" in msg.metadata.get("thinking_text", "")


def test_record_assistant_message_strips_reasoning_tags() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    record_assistant_message(
        harness,
        assistant_text="Answer:<reasoning>Step 1, step 2.</reasoning>",
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.content == "Answer:"
    assert msg.metadata.get("thinking_text") == "Step 1, step 2."


def test_record_assistant_message_leaves_clean_text_unchanged() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    record_assistant_message(
        harness,
        assistant_text="This is a clean response.",
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.content == "This is a clean response."
    assert "thinking_text" not in msg.metadata


def test_record_assistant_message_does_not_strip_thinking_from_tool_calls() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "task_complete",
                "arguments": '{"message": "<think>not thinking</think>"}',
            },
        }
    ]
    record_assistant_message(
        harness,
        assistant_text="Done.",
        tool_calls=tool_calls,
    )
    msg = state.recent_messages[-1]
    assert msg.content == "Done."
    assert msg.tool_calls == tool_calls


def test_record_assistant_message_collapses_large_file_code_block() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    html_dump = "```html\n<!DOCTYPE html>\n<html>\n<body>\n" + "\n".join(f"<p>line {i}</p>" for i in range(500)) + "\n</body>\n</html>\n```"
    record_assistant_message(
        harness,
        assistant_text=html_dump,
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.role == "assistant"
    assert "<p>line 0</p>" not in msg.content
    assert "block omitted" in msg.content
    assert "html" in msg.content


def test_record_assistant_message_preserves_small_code_block() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    code = "```python\nprint('hello')\n```"
    record_assistant_message(
        harness,
        assistant_text=code,
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.content == code


def test_record_assistant_message_preserves_large_block_with_explanation() -> None:
    state = LoopState(cwd="/tmp")
    harness = _make_harness(state)
    explanation = "Here is the helper function you requested:\n"
    code = "```python\n" + "\n".join(f"x = {i}" for i in range(60)) + "\n```"
    text = explanation + code
    record_assistant_message(
        harness,
        assistant_text=text,
        tool_calls=[],
    )
    msg = state.recent_messages[-1]
    assert msg.content == text
