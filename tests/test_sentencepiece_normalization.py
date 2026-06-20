from __future__ import annotations

from smallctl.client.chunk_parser import (
    extract_content_fragments,
    normalize_sentencepiece_whitespace,
    sanitize_assistant_content_for_history,
)
from smallctl.client.stream_collectors import collect_stream, collect_timeline


def test_normalize_sentencepiece_whitespace_converts_marker() -> None:
    raw = "\u2581The\u2581listof\u2581Docker\u2581containers"
    assert normalize_sentencepiece_whitespace(raw) == " The listof Docker containers"


def test_normalize_sentencepiece_whitespace_no_marker_unchanged() -> None:
    text = "The list of Docker containers"
    assert normalize_sentencepiece_whitespace(text) == text


def test_extract_content_fragments_normalizes_whitespace() -> None:
    content = "\u2581Hello\u2581world"
    fragments = extract_content_fragments(content)
    assert fragments == [("assistant", " Hello world")]


def test_sanitize_assistant_content_normalizes_whitespace() -> None:
    raw = "\u2581The\u2581quick\u2581brown\u2581fox"
    assistant, thinking = sanitize_assistant_content_for_history(raw)
    assert assistant == "The quick brown fox"
    assert thinking == ""


def test_sanitize_assistant_content_normalizes_thinking_whitespace() -> None:
    raw = "<think>\u2581Deep\u2581thoughts</think>\u2581Answer"
    assistant, thinking = sanitize_assistant_content_for_history(raw)
    assert assistant == "Answer"
    assert thinking == "Deep thoughts"


def test_collect_stream_normalizes_content_and_tool_args() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "\u2581Hello\u2581world",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "shell\u2581exec",
                                        "arguments": '{"command": "echo\u2581hi"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        }
    ]
    result = collect_stream(chunks, reasoning_mode="off")
    assert result.assistant_text == " Hello world"
    assert result.tool_calls[0]["function"]["name"] == "shell exec"
    assert result.tool_calls[0]["function"]["arguments"] == '{"command": "echo hi"}'


def test_collect_timeline_normalizes_content() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "\u2581Hello\u2581world",
                        }
                    }
                ]
            },
        }
    ]
    timeline = collect_timeline(chunks, reasoning_mode="off")
    assert len(timeline) == 1
    assert timeline[0].kind == "assistant"
    assert timeline[0].content == " Hello world"
