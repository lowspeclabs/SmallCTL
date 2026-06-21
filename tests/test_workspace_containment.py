from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from smallctl.state import LoopState
from smallctl.tools.fs import file_write
from smallctl.tools.fs_listing import WorkspaceContainmentError, _guard_workspace_containment
from smallctl.tools.fs_mutations import file_append, file_delete


@pytest.fixture
def state(tmp_path: Path) -> LoopState:
    return LoopState(cwd=str(tmp_path))


def test_file_write_rejects_outside_workspace(tmp_path: Path, state: LoopState) -> None:
    target = tmp_path.parent / "outside_write.txt"
    result = asyncio.run(
        file_write(str(target), "secret content", cwd=str(tmp_path), state=state)
    )
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "workspace_path_traversal"
    assert not target.exists()


def test_file_append_rejects_outside_workspace(tmp_path: Path, state: LoopState) -> None:
    target = tmp_path.parent / "outside_append.txt"
    result = asyncio.run(
        file_append(str(target), "extra content", cwd=str(tmp_path), state=state)
    )
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "workspace_path_traversal"
    assert not target.exists()


def test_file_delete_rejects_outside_workspace(tmp_path: Path, state: LoopState) -> None:
    target = tmp_path.parent / "outside_delete.txt"
    target.write_text("existing", encoding="utf-8")
    result = asyncio.run(file_delete(str(target), cwd=str(tmp_path), state=state))
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "workspace_path_traversal"
    assert target.exists()


def test_guard_workspace_containment_uses_resolve_for_enforcement(
    tmp_path: Path,
) -> None:
    result = _guard_workspace_containment(
        str(tmp_path.parent / "outside.txt"),
        str(tmp_path),
        operation="file_write",
    )
    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "workspace_path_traversal"
    assert result["metadata"]["operation"] == "file_write"


def test_guard_workspace_containment_allows_approved_out_of_workspace(
    tmp_path: Path,
) -> None:
    result = _guard_workspace_containment(
        str(tmp_path.parent / "outside.txt"),
        str(tmp_path),
        operation="file_write",
        approved_out_of_workspace=True,
    )
    assert result is None


def test_guard_workspace_containment_flags_sensitive_location(
    tmp_path: Path,
) -> None:
    result = _guard_workspace_containment(
        "~/.ssh/authorized_keys",
        str(tmp_path),
        operation="file_write",
    )
    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "sensitive_location_unapproved"


def test_guard_workspace_containment_allows_in_workspace_paths(
    tmp_path: Path,
) -> None:
    result = _guard_workspace_containment(
        "safe.txt",
        str(tmp_path),
        operation="file_write",
    )
    assert result is None


def test_workspace_containment_error_exposes_metadata(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceContainmentError) as exc_info:
        from smallctl.tools.fs_listing import _resolve

        _resolve(str(tmp_path.parent / "x"), str(tmp_path), operation="file_delete")
    assert exc_info.value.metadata["operation"] == "file_delete"
    assert exc_info.value.metadata["error_kind"] == "workspace_path_traversal"
