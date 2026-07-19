from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from smallctl.state import LoopState
from smallctl.tools.fs import file_write
from smallctl.tools.fs_listing import _looks_like_sensitive_env_path


def _make_state(tmp_path: Path, task: str) -> LoopState:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = task
    return state


@pytest.mark.parametrize(
    "name,expected",
    [
        (".env", True),
        (".env.local", True),
        (".env.production", True),
        (".envrc", True),
        (".env.example", False),
        (".env.sample", False),
        (".env.template", False),
        (".env.dist", False),
        ("config.txt", False),
        (".bashrc", False),
    ],
)
def test_looks_like_sensitive_env_path(name: str, expected: bool) -> None:
    assert _looks_like_sensitive_env_path(f"/tmp/{name}") is expected


def test_file_write_blocks_env_overwrite_when_not_requested(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ORIGINAL=value\n", encoding="utf-8")
    state = _make_state(
        tmp_path,
        "read AGENTS.md and load the proxmox cli, api info token=abc host=1.2.3.4",
    )

    result = asyncio.run(
        file_write(
            path=str(env_file),
            content="NEW=value\n",
            cwd=str(tmp_path),
            state=state,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "sensitive_env_file_overwrite_blocked"
    assert env_file.read_text(encoding="utf-8") == "ORIGINAL=value\n"


def test_file_write_blocks_env_local_overwrite_when_not_requested(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("ORIGINAL=value\n", encoding="utf-8")
    state = _make_state(tmp_path, "load the project and run the tests")

    result = asyncio.run(
        file_write(
            path=str(env_file),
            content="NEW=value\n",
            cwd=str(tmp_path),
            state=state,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "sensitive_env_file_overwrite_blocked"


def test_file_write_allows_env_overwrite_when_explicitly_requested(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OLD=value\n", encoding="utf-8")
    state = _make_state(
        tmp_path,
        "read AGENTS.md and update the .env with token=abc, then run the cli",
    )

    result = asyncio.run(
        file_write(
            path=str(env_file),
            content="TOKEN=abc\n",
            cwd=str(tmp_path),
            state=state,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert "TOKEN=abc" in env_file.read_text(encoding="utf-8")


def test_file_write_allows_env_creation_when_explicitly_requested(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    state = _make_state(tmp_path, "create a .env from .env.example with the provided credentials")

    result = asyncio.run(
        file_write(
            path=str(env_file),
            content="TOKEN=abc\n",
            cwd=str(tmp_path),
            state=state,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert env_file.read_text(encoding="utf-8") == "TOKEN=abc\n"


def test_file_write_allows_env_example_template_overwrite(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    template.write_text("# placeholder\n", encoding="utf-8")
    state = _make_state(tmp_path, "document the required environment variables in .env.example")

    result = asyncio.run(
        file_write(
            path=str(template),
            content="TOKEN=replace-me\n",
            cwd=str(tmp_path),
            state=state,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True


def test_file_write_blocks_env_overwrite_without_state(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ORIGINAL=value\n", encoding="utf-8")

    result = asyncio.run(
        file_write(
            path=str(env_file),
            content="NEW=value\n",
            cwd=str(tmp_path),
            state=None,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "sensitive_env_file_overwrite_blocked"
