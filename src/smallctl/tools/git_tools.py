from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .common import fail, ok


def _run_git(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return 127, "", "git command not found"
    except subprocess.TimeoutExpired:
        return 124, "", "git command timed out"
    except Exception as exc:
        return 1, "", f"git error: {exc}"


async def git_status(
    path: str = ".",
    short: bool = True,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.exists():
        return fail(f"Path does not exist: {target}")
    args = ["-C", str(target), "status"]
    if short:
        args.append("--short")
    returncode, stdout, stderr = _run_git(*args)
    if returncode != 0:
        return fail(f"git status failed: {stderr or stdout}")
    clean = not stdout.strip()
    return ok(
        {
            "clean": clean,
            "dirty": not clean,
            "output": stdout.strip(),
        },
        metadata={"path": str(target), "short": short},
    )


async def git_diff(
    path: str = ".",
    cached: bool = False,
    target: str | None = None,
) -> dict[str, Any]:
    target_dir = Path(path).resolve()
    if not target_dir.exists():
        return fail(f"Path does not exist: {target_dir}")
    args = ["-C", str(target_dir), "diff"]
    if cached:
        args.append("--cached")
    if target:
        args.extend(["--", target])
    returncode, stdout, stderr = _run_git(*args)
    if returncode not in (0, 1):
        return fail(f"git diff failed: {stderr or stdout}")
    has_changes = bool(stdout.strip())
    return ok(
        {
            "has_changes": has_changes,
            "output": stdout.strip(),
        },
        metadata={"path": str(target_dir), "cached": cached, "target": target},
    )


async def read_log(
    path: str,
    lines: int = 100,
    offset: int | None = None,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.exists():
        return fail(f"Path does not exist: {target}")
    if not target.is_file():
        return fail(f"Path is not a file: {target}")
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            all_lines = handle.readlines()
    except Exception as exc:
        return fail(f"Failed to read log file: {exc}")

    total = len(all_lines)
    start = max(0, offset) if offset is not None else max(0, total - lines)
    end = min(total, start + lines)
    selected = all_lines[start:end]
    return ok(
        "".join(selected),
        metadata={
            "path": str(target),
            "total_lines": total,
            "start_line": start + 1,
            "end_line": end,
            "requested_lines": lines,
        },
    )
