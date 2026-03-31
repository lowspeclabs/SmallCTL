from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .logging_utils import log_kv


def run_cleanup(root: str = ".") -> dict[str, int]:
    logger = logging.getLogger("smallctl.cleanup")
    base = Path(root).resolve()
    removed_dirs = 0
    removed_files = 0

    for directory in base.rglob("*"):
        if directory.is_dir() and directory.name in {"__pycache__", ".pytest_cache"}:
            removed_files += sum(1 for p in directory.rglob("*") if p.is_file())
            shutil.rmtree(directory, ignore_errors=True)
            removed_dirs += 1

    for file_path in base.rglob("*.pyc"):
        if file_path.exists():
            file_path.unlink(missing_ok=True)
            removed_files += 1
    for file_path in base.rglob("*.pyo"):
        if file_path.exists():
            file_path.unlink(missing_ok=True)
            removed_files += 1

    log_kv(
        logger,
        logging.INFO,
        "cleanup_complete",
        root=str(base),
        removed_dirs=removed_dirs,
        removed_files=removed_files,
    )
    return {"removed_dirs": removed_dirs, "removed_files": removed_files}
