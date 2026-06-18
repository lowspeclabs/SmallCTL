from __future__ import annotations

from typing import Any


def ok(output: Any = None, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": True,
        "output": output,
        "error": None,
        "metadata": metadata or {},
    }


def fail(error: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": False,
        "output": None,
        "error": error,
        "metadata": metadata or {},
    }


def needs_human(question: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": False,
        "status": "needs_human",
        "output": None,
        "error": f"Human input required: {question}",
        "metadata": {
            **(metadata or {}),
            "question": question,
        },
    }

