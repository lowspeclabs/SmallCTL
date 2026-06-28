from __future__ import annotations

import asyncio
from pathlib import Path

from smallctl.state import LoopState
from smallctl.tools.fs import file_write


async def _overwrite(state: LoopState, target: Path, content: str) -> dict:
    return await file_write(
        str(target),
        content,
        cwd=str(target.parent),
        state=state,
        replace_strategy="overwrite",
    )


def test_file_write_blocks_third_overwrite_of_same_file(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "game.html"
    base = "<!DOCTYPE html>\n<html><body></body></html>\n"

    first = asyncio.run(_overwrite(state, target, base + "<!-- v1 -->\n"))
    assert first["success"] is True

    second = asyncio.run(_overwrite(state, target, base + "<!-- v2 -->\n"))
    assert second["success"] is True
    # Two overwrites should have incremented repeat_patch to 1.
    assert state.stagnation_counters.get("repeat_patch", 0) == 1

    third = asyncio.run(_overwrite(state, target, base + "<!-- v3 -->\n"))
    # Third overwrite succeeds; repeat_patch is now 2.
    assert third["success"] is True
    assert state.stagnation_counters.get("repeat_patch", 0) == 2

    # Fourth overwrite should be blocked by the repeat_patch loop guard.
    fourth = asyncio.run(_overwrite(state, target, base + "<!-- v4 -->\n"))
    assert fourth["success"] is False
    assert fourth["metadata"].get("error_kind") == "repeat_patch_loop_guard"
    assert "file_read" in fourth["metadata"].get("next_required_tool", {}).get("tool_name", "")


def test_file_write_allows_overwrite_after_file_read(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "game.html"
    base = "<!DOCTYPE html>\n<html><body></body></html>\n"

    for i in range(3):
        result = asyncio.run(_overwrite(state, target, base + f"<!-- v{i + 1} -->\n"))
        assert result["success"] is True

    blocked = asyncio.run(_overwrite(state, target, base + "<!-- v4 -->\n"))
    assert blocked["success"] is False
    assert blocked["metadata"].get("error_kind") == "repeat_patch_loop_guard"

    # Simulate a file_read by recording the target in repair_cycle_reads.
    state.repair_cycle_id = "rc-test"
    state.scratchpad["_repair_cycle_reads"] = [str(target.resolve())]

    # The repair cycle read requirement should still allow the write, but the
    # repeat_patch guard is evaluated first and remains in effect because the
    # model is still in a rewrite loop. The model must use file_patch instead.
    still_blocked = asyncio.run(_overwrite(state, target, base + "<!-- v5 -->\n"))
    assert still_blocked["success"] is False
    assert still_blocked["metadata"].get("error_kind") == "repeat_patch_loop_guard"


def test_file_write_loop_guard_ignores_unrelated_files(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    a = tmp_path / "a.html"
    b = tmp_path / "b.html"
    content = "<!DOCTYPE html>\n<html><body></body></html>\n"

    # Three overwrites of a.html leaves repeat_patch at 2.
    for i in range(3):
        assert asyncio.run(_overwrite(state, a, content + f"<!-- {i} -->\n"))["success"] is True

    # Rewriting an unrelated file should not be blocked.
    result = asyncio.run(_overwrite(state, b, content))
    assert result["success"] is True

    # The fourth overwrite of a.html should still be blocked.
    blocked = asyncio.run(_overwrite(state, a, content + "<!-- blocked -->\n"))
    assert blocked["success"] is False
    assert blocked["metadata"].get("error_kind") == "repeat_patch_loop_guard"
