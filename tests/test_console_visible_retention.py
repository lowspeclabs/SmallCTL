from __future__ import annotations

from smallctl.ui.console import _select_visible_retention_removals


class _Placeholder:
    classes = "bubble-retention-placeholder"


def test_visible_retention_removes_oldest_excess_entries() -> None:
    children = ["a", "b", "c", "d"]

    assert _select_visible_retention_removals(children, active=None, limit=2) == ["a", "b"]


def test_visible_retention_preserves_active_assistant_turn() -> None:
    active = object()
    children = ["old", active, "middle", "new"]

    assert _select_visible_retention_removals(children, active=active, limit=2) == ["old", "middle"]


def test_visible_retention_noops_when_under_limit_or_disabled() -> None:
    children = ["a", "b"]

    assert _select_visible_retention_removals(children, active=None, limit=2) == []
    assert _select_visible_retention_removals(children, active=None, limit=0) == []


def test_visible_retention_preserves_placeholder() -> None:
    placeholder = _Placeholder()
    children = [placeholder, "old", "new"]

    assert _select_visible_retention_removals(children, active=None, limit=1) == ["old", "new"]
