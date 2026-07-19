from __future__ import annotations

import asyncio

from smallctl.tools.git_tools import git_status, git_diff, read_log


def test_git_status_on_non_git_dir(tmp_path) -> None:
    result = asyncio.run(git_status(path=str(tmp_path), short=True))
    assert result["success"] is False
    assert "git status failed" in result["error"] or "not a git" in result["error"].lower()


def test_git_status_on_git_repo(tmp_path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    result = asyncio.run(git_status(path=str(tmp_path), short=True))
    assert result["success"] is True
    assert result["output"]["clean"] is True
    assert result["output"]["dirty"] is False


def test_git_status_detects_dirty(tmp_path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello")
    result = asyncio.run(git_status(path=str(tmp_path), short=True))
    assert result["success"] is True
    assert result["output"]["dirty"] is True
    assert "a.txt" in result["output"]["output"]


def test_git_diff_on_non_git_dir(tmp_path) -> None:
    result = asyncio.run(git_diff(path=str(tmp_path)))
    assert result["success"] is False


def test_git_diff_clean_repo(tmp_path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    result = asyncio.run(git_diff(path=str(tmp_path)))
    assert result["success"] is True
    assert result["output"]["has_changes"] is False


def test_git_diff_with_changes(tmp_path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    result = asyncio.run(git_diff(path=str(tmp_path), cached=True))
    assert result["success"] is True
    assert result["output"]["has_changes"] is True
    assert "hello" in result["output"]["output"]


def test_git_diff_with_target_file(tmp_path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello modified")
    result = asyncio.run(git_diff(path=str(tmp_path), target="a.txt"))
    assert result["success"] is True
    assert "modified" in result["output"]["output"]
    assert "world" not in result["output"]["output"]


def test_read_log_missing_file(tmp_path) -> None:
    result = asyncio.run(read_log(path=str(tmp_path / "missing.log")))
    assert result["success"] is False
    assert "does not exist" in result["error"]


def test_read_log_directory_rejected(tmp_path) -> None:
    result = asyncio.run(read_log(path=str(tmp_path)))
    assert result["success"] is False
    assert "not a file" in result["error"]


def test_read_log_tail_default(tmp_path) -> None:
    log = tmp_path / "app.log"
    log.write_text("\n".join(f"line {i}" for i in range(1, 201)))
    result = asyncio.run(read_log(path=str(log)))
    assert result["success"] is True
    assert result["output"].startswith("line 101\n")
    assert "line 200" in result["output"]
    assert result["metadata"]["total_lines"] == 200
    assert result["metadata"]["start_line"] == 101


def test_read_log_bounded_lines(tmp_path) -> None:
    log = tmp_path / "app.log"
    log.write_text("\n".join(f"line {i}" for i in range(1, 51)))
    result = asyncio.run(read_log(path=str(log), lines=10))
    assert result["success"] is True
    assert result["output"].startswith("line 41\n")
    assert result["output"].rstrip("\n").endswith("line 50")
    assert "line 40\n" not in result["output"]
    assert result["metadata"]["start_line"] == 41


def test_read_log_with_offset(tmp_path) -> None:
    log = tmp_path / "app.log"
    log.write_text("\n".join(f"line {i}" for i in range(1, 51)))
    result = asyncio.run(read_log(path=str(log), lines=5, offset=10))
    assert result["success"] is True
    assert result["output"].startswith("line 11\n")
    assert result["output"].rstrip("\n").endswith("line 15")
    assert "line 10\n" not in result["output"]
    assert "line 16\n" not in result["output"]
    assert result["metadata"]["start_line"] == 11
    assert result["metadata"]["end_line"] == 15


def test_git_status_resolves_path_against_cwd(monkeypatch, tmp_path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello")

    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    monkeypatch.chdir(outside)

    result = asyncio.run(git_status(path=".", cwd=str(tmp_path), short=True))
    assert result["success"] is True
    assert result["output"]["dirty"] is True
    assert "a.txt" in result["output"]["output"]


def test_git_diff_resolves_path_against_cwd(monkeypatch, tmp_path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello modified")

    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    monkeypatch.chdir(outside)

    result = asyncio.run(git_diff(path=".", cwd=str(tmp_path)))
    assert result["success"] is True
    assert result["output"]["has_changes"] is True
    assert "modified" in result["output"]["output"]


def test_read_log_resolves_path_against_cwd(monkeypatch, tmp_path) -> None:
    log = tmp_path / "subdir" / "app.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"line {i}" for i in range(1, 11)))

    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    monkeypatch.chdir(outside)

    result = asyncio.run(read_log(path="subdir/app.log", cwd=str(tmp_path), lines=5))
    assert result["success"] is True
    assert result["output"].startswith("line 6\n")
    assert result["metadata"]["total_lines"] == 10
