from __future__ import annotations

from typing import Any


def phase_verifier_is_weak(command: str) -> bool:
    normalized = " ".join(str(command or "").strip().lower().split())
    if not normalized:
        return True
    weak_patterns = (
        " -m py_compile ",
        " py_compile ",
        " pip install ",
        " python -m pip install ",
        " python3 -m pip install ",
        " python -m venv ",
        " python3 -m venv ",
        " rm -rf ",
        " ls ",
        " grep ",
    )
    padded = f" {normalized} "
    return any(pattern in padded for pattern in weak_patterns)


def verifier_quality(command: str) -> dict[str, Any]:
    normalized = " ".join(str(command or "").strip().lower().split())
    if not normalized:
        return {"score": 0, "label": "none"}
    padded = f" {normalized} "
    if any(pattern in padded for pattern in (" pip install ", " -m pip install ", " -m venv ", " rm -rf ", " ls ", " grep ")):
        return {"score": 0, "label": "setup_or_probe"}
    if " -m py_compile " in padded or " py_compile " in padded:
        return {"score": 1, "label": "syntax"}
    if any(pattern in padded for pattern in (" pytest ", " -m pytest ", " unittest ", " -m unittest ")):
        return {"score": 3, "label": "behavioral"}
    if " -c " in padded:
        if "assert " in normalized or ".handle_" in normalized or ".spawn" in normalized or ".move" in normalized:
            return {"score": 3, "label": "behavioral"}
        if "import " in normalized:
            return {"score": 2, "label": "import"}
    if " --smoke" in padded or " smoke" in padded:
        return {"score": 3, "label": "behavioral"}
    if " selenium " in padded or " playwright " in padded:
        return {"score": 5, "label": "e2e"}
    if any(token in normalized for token in ("curl ", "http://", "https://")):
        return {"score": 4, "label": "integration"}
    return {"score": 2, "label": "execution"}


def phase_verifier_is_inconclusive(
    verifier: dict[str, Any],
    *,
    command: str,
    failure_mode: str,
    notes: str,
) -> bool:
    haystack = "\n".join(
        str(part or "")
        for part in [
            command,
            failure_mode,
            notes,
            verifier.get("key_stdout"),
            verifier.get("key_stderr"),
        ]
    ).lower()
    return any(token in haystack for token in ("timed out", "timeout", "infinite_loop", "infinite loop"))


def verifier_notes_text(verifier: dict[str, Any]) -> str:
    acceptance_delta = verifier.get("acceptance_delta") if isinstance(verifier, dict) else None
    if not isinstance(acceptance_delta, dict):
        return ""
    notes = acceptance_delta.get("notes")
    if isinstance(notes, list):
        return "\n".join(str(note or "") for note in notes)
    return str(notes or "")
