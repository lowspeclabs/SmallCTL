from __future__ import annotations

from smallctl.graph.display import format_tool_result_display
from smallctl.models.tool_result import ToolEnvelope


def test_format_tool_result_display_preserves_actionable_tail_for_long_errors() -> None:
    prefix = "Patch-existing write sessions need an explicit first-chunk choice. " * 8
    tail = (
        "If earlier chunks are not fully visible in local context, call "
        "`file_read(path='./temp/logwatch.py')` first; during an active write session that reads from the staged copy. "
        "Do not assume earlier chunks were lost or rewrite the whole staged file from memory."
    )
    result = ToolEnvelope(
        success=False,
        error=prefix + tail,
        metadata={},
    )

    rendered = format_tool_result_display(tool_name="file_write", result=result)

    assert "... error truncated" not in rendered
    assert "file_read(path='./temp/logwatch.py')" in rendered
    assert "Do not assume earlier chunks were lost" in rendered
    assert "\n...\n" in rendered


def test_format_tool_result_display_keeps_short_errors_unchanged() -> None:
    result = ToolEnvelope(
        success=False,
        error="Short failure.",
        metadata={},
    )

    rendered = format_tool_result_display(tool_name="file_write", result=result)

    assert rendered == "Short failure."


def test_format_tool_result_display_preserves_word_boundaries_in_tail_guidance() -> None:
    prefix = "Patch-existing write sessions need an explicit first-chunk choice. " * 8
    tail = (
        "Choose `replace_strategy='overwrite'` only after rereading the staged copy, "
        "and do not broad-rewrite from memory."
    )
    result = ToolEnvelope(
        success=False,
        error=prefix + tail,
        metadata={},
    )

    rendered = format_tool_result_display(tool_name="file_write", result=result)

    assert "replace_strategy='overwrite'" in rendered
    assert "broad-rewrite from memory" in rendered
