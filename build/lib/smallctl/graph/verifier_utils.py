from __future__ import annotations

import re
from typing import Any


def _command_looks_like_verifier(command: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(command or "").strip().lower())
    if not normalized:
        return False
    verifier_markers = (
        "pytest",
        "unittest",
        " npm test",
        "npm run test",
        "pnpm test",
        "yarn test",
        "go test",
        "cargo test",
        "cargo clippy",
        "ruff",
        "mypy",
        "eslint",
        "vitest",
        "jest",
        "py_compile",
    )
    padded = f" {normalized}"
    return any(marker in padded for marker in verifier_markers)


def _verifier_output_text(record: Any) -> str:
    result = getattr(record, "result", None)
    chunks: list[str] = []
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        for key in ("stderr", "stdout"):
            value = str(output.get(key) or "").strip()
            if value:
                chunks.append(value)
    elif output is not None:
        value = str(output or "").strip()
        if value:
            chunks.append(value)
    metadata = getattr(result, "metadata", {}) if result is not None else {}
    if isinstance(metadata, dict):
        meta_output = metadata.get("output")
        if isinstance(meta_output, dict):
            for key in ("stderr", "stdout"):
                value = str(meta_output.get(key) or "").strip()
                if value:
                    chunks.append(value)
    error = str(getattr(result, "error", "") or "").strip()
    if error:
        chunks.append(error)
    return "\n".join(chunks)


def _summarize_verifier_failure(text: str) -> list[str]:
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    summary: list[str] = []
    patterns = (
        "ERROR:",
        "FAIL:",
        "AttributeError:",
        "AssertionError:",
        "SyntaxError:",
        "TypeError:",
        "NameError:",
        "FAILED",
    )
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("File ") and len(summary) < 6:
            summary.append(stripped)
            continue
        if any(marker in stripped for marker in patterns):
            summary.append(stripped)
        if len(summary) >= 8:
            break
    if summary:
        return summary
    for line in lines:
        stripped = line.strip()
        if stripped:
            summary.append(stripped)
        if len(summary) >= 4:
            break
    return summary
