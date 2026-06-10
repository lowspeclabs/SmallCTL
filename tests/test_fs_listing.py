from __future__ import annotations

from pathlib import Path

from smallctl.tools.fs_listing import _resolve


def test_resolve_does_not_double_encode_cwd() -> None:
    """Paths that already encode the CWD as relative segments should not be doubled."""
    cwd = "/home/stephen/Scripts/Harness-Redo"
    # Model emitted path without leading slash but including CWD
    path = "home/stephen/Scripts/Harness-Redo/temp/vikunja-9b.py"
    result = _resolve(path, cwd)
    expected = Path(cwd) / "temp" / "vikunja-9b.py"
    assert result == expected.resolve()


def test_resolve_normal_relative_path() -> None:
    """Normal relative paths should still resolve correctly."""
    cwd = "/home/stephen/Scripts/Harness-Redo"
    path = "temp/vikunja-9b.py"
    result = _resolve(path, cwd)
    expected = Path(cwd) / "temp" / "vikunja-9b.py"
    assert result == expected.resolve()


def test_resolve_absolute_path_unchanged() -> None:
    """Absolute paths should be resolved directly."""
    path = "/etc/nginx/nginx.conf"
    result = _resolve(path)
    assert result == Path("/etc/nginx/nginx.conf").resolve()
