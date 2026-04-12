from __future__ import annotations

from smallctl.graph.display import format_dir_list_display
from smallctl.models.tool_result import ToolEnvelope


def test_dir_list_display_shows_more_than_old_eight_item_preview() -> None:
    result = ToolEnvelope(
        success=True,
        output=[f"item-{index:02d}" for index in range(1, 21)],
        metadata={"path": "./temp", "count": 20},
    )

    rendered = format_dir_list_display(result=result)

    assert "./temp (20 items)" in rendered
    assert "item-09" in rendered
    assert "item-20" in rendered
    assert "... 12 more items" not in rendered
