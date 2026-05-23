"""
aho/git_utils.py
----------------
Git handling utilities for the researcher module.

This module extracts git operations from researcher.py to keep the
main research flow as a clear pipeline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# Repo root for git operations
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repo root. Returns CompletedProcess."""
    return subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=check,
    )


def git_short_hash() -> str:
    """Return the current HEAD short hash, or 'no-commit' if not yet committed."""
    try:
        result = _git("rev-parse", "--short", "HEAD")
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "no-commit"


def git_commit_config(config_path: Path, strategy_id: str, score: float) -> str:
    """
    Stage harness_config.json and commit it.
    Returns the new short hash.
    """
    msg = f"aho: keep {strategy_id} (score={score:.4f})"
    _git("add", str(config_path))
    _git("commit", "-m", msg)
    return git_short_hash()


def git_discard_config(config_path: Path) -> None:
    """
    Revert harness_config.json to the last committed state.
    Equivalent to autoresearch's `git reset --hard HEAD` but scoped to one file.
    """
    try:
        _git("checkout", "--", str(config_path))
    except subprocess.CalledProcessError:
        pass  # no prior commit yet — file stays as-is


def git_is_available() -> bool:
    """Check if git is available and we're in a git repository."""
    try:
        _git("rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def git_has_uncommitted_changes(config_path: Path) -> bool:
    """Check if the config file has uncommitted changes."""
    try:
        result = _git("status", "--porcelain", str(config_path), check=False)
        return bool(result.stdout.strip())
    except Exception:
        return False
