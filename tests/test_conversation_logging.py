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
