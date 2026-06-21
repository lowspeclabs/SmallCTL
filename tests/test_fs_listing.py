from __future__ import annotations

from pathlib import Path

import pytest

from smallctl.tools.fs_listing import WorkspaceContainmentError, _resolve


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


def test_resolve_expands_tilde_to_home() -> None:
    """Paths with ~ should expand to the user's home directory."""
    import os
    home = os.path.expanduser("~")
    path = "~/.ssh/authorized_keys"
    result = _resolve(path)
    expected = Path(home) / ".ssh" / "authorized_keys"
    assert result == expected.resolve()


def test_resolve_tilde_with_cwd() -> None:
    """~ should expand even when cwd is provided."""
    import os
    home = os.path.expanduser("~")
    cwd = "/tmp"
    path = "~/.bashrc"
    result = _resolve(path, cwd)
    expected = Path(home) / ".bashrc"
    assert result == expected.resolve()


def test_resolve_tilde_in_dir_list() -> None:
    """~/.ssh should list the real ~/.ssh directory."""
    import asyncio
    import os
    from smallctl.tools.fs_listing import dir_list
    home = os.path.expanduser("~")
    result = asyncio.run(dir_list("~/.ssh"))
    assert result["success"] is True
    assert result["metadata"]["path"] == str(Path(home) / ".ssh")


def test_resolve_tilde_in_file_read_missing() -> None:
    """file_read with ~ on a nonexistent path should mention ~ in the error."""
    import asyncio
    from smallctl.tools.fs_listing import file_read
    result = asyncio.run(file_read("~/nonexistent_file_12345.txt"))
    assert result["success"] is False
    assert "~" in result["error"]


def test_file_read_blocks_likely_secret_file(tmp_path) -> None:
    import asyncio
    from smallctl.tools.fs_listing import file_read

    env_file = tmp_path / ".env"
    env_file.write_text("API_TOKEN=secret-value\n", encoding="utf-8")

    result = asyncio.run(file_read(str(env_file)))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "sensitive_file_read_blocked"
    assert "secret-value" not in result["error"]


def test_resolve_enforces_workspace_containment_for_writes(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_workspace.txt"
    with pytest.raises(WorkspaceContainmentError) as exc_info:
        _resolve(str(outside), str(tmp_path), operation="file_write")
    assert exc_info.value.metadata["error_kind"] == "workspace_path_traversal"
    assert exc_info.value.metadata["operation"] == "file_write"


def test_resolve_allows_out_of_workspace_when_approved(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_workspace.txt"
    result = _resolve(
        str(outside),
        str(tmp_path),
        operation="file_write",
        approved_out_of_workspace=True,
    )
    assert result == outside.resolve()


def test_resolve_blocks_sensitive_out_of_workspace(tmp_path: Path) -> None:
    import os

    home = os.path.expanduser("~")
    with pytest.raises(WorkspaceContainmentError) as exc_info:
        _resolve("~/.ssh/authorized_keys", str(tmp_path), operation="file_write")
    assert exc_info.value.metadata["error_kind"] == "sensitive_location_unapproved"
    assert exc_info.value.metadata["path"] == str(Path(home) / ".ssh" / "authorized_keys")


def test_resolve_does_not_enforce_containment_for_reads(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_workspace.txt"
    result = _resolve(str(outside), str(tmp_path))
    assert result == outside.resolve()
