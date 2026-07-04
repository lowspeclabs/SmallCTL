import pytest

from smallctl.graph.state import PendingToolCall


def test_repair_gemma_swallowed_replace_strategy_empty() -> None:
    raw = '{"content": "abc", "path": "./temp/gemma-4-12b-mini-kanban.html\\",replace_strategy:"}'
    pending = PendingToolCall.from_payload(
        {"function": {"name": "file_write", "arguments": raw}}
    )
    assert pending is not None
    assert pending.args["path"] == "./temp/gemma-4-12b-mini-kanban.html"
    assert "replace_strategy" not in pending.args
    repairs = pending.parser_metadata.get("gemma_swallowed_key_repairs")
    assert repairs == [
        {
            "original_key": "path",
            "swallowed_key": "replace_strategy",
            "swallowed_value": "",
        }
    ]


def test_repair_gemma_swallowed_section_name_empty() -> None:
    raw = '{"content": "abc", "path": "./temp/app.py\\",section_name:"}'
    pending = PendingToolCall.from_payload(
        {"function": {"name": "file_write", "arguments": raw}}
    )
    assert pending is not None
    assert pending.args["path"] == "./temp/app.py"
    assert "section_name" not in pending.args
    repairs = pending.parser_metadata.get("gemma_swallowed_key_repairs")
    assert repairs == [
        {
            "original_key": "path",
            "swallowed_key": "section_name",
            "swallowed_value": "",
        }
    ]


def test_no_repair_for_clean_arguments() -> None:
    raw = '{"path": "./temp/app.py", "content": "abc"}'
    pending = PendingToolCall.from_payload(
        {"function": {"name": "file_write", "arguments": raw}}
    )
    assert pending is not None
    assert pending.args["path"] == "./temp/app.py"
    assert pending.args["content"] == "abc"
    assert "gemma_swallowed_key_repairs" not in pending.parser_metadata
