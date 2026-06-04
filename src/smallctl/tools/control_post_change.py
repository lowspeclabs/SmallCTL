from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from ..state import LoopState

_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError:\s+No module named ['\"](?P<module>[^'\"]+)['\"]")


def post_change_verification_block(state: LoopState) -> dict[str, Any] | None:
    progress = getattr(state, "challenge_progress", None)
    if progress is None:
        return None
    if int(getattr(progress, "code_change_count", 0) or 0) <= 0:
        return None
    if bool(getattr(progress, "verified_after_last_change", False)):
        return None
    paths = [str(path or "").strip() for path in getattr(progress, "last_code_change_paths", []) or [] if str(path or "").strip()]
    html_paths = [path for path in paths if path.lower().endswith((".html", ".htm"))]
    if html_paths:
        target = html_paths[-1]
        command = (
            f"test -s {shlex.quote(target)} && python3 - <<'PY'\n"
            "from pathlib import Path\n"
            f"s = Path({target!r}).read_text()\n"
            "assert '<!DOCTYPE html' in s\n"
            "assert '<canvas' in s\n"
            "assert s.count('<script') == s.count('</script>')\n"
            "assert '<small_model_thought>' not in s\n"
            "print('OK')\n"
            "PY"
        )
    else:
        target = paths[-1] if paths else "the changed artifact"
        command = focused_verifier_command_for_path(target, state=state) or "run the smallest focused verifier for the changed artifact"
    dependency_block = missing_dependency_block(state)
    notes = [
        f"Verify `{target}` after the latest change before task_complete/final success.",
        "task_complete, final_verify, and success are blocked while verified_after_last_change=false.",
    ]
    if dependency_block:
        notes = [*dependency_block["notes"], *notes]
    required_command = dependency_block["command"] if dependency_block else command
    return {
        "reason": dependency_block["reason"] if dependency_block else "post_change_verification_required",
        "verified_after_last_change": False,
        "last_code_change_paths": paths,
        "next_required_action": {
            "tool_name": "shell_exec",
            "required_arguments": {"command": required_command} if required_command else {},
            "notes": notes,
        },
    }


def focused_verifier_command_for_path(path: str, *, state: LoopState) -> str:
    target = str(path or "").strip()
    if not target.lower().endswith(".py"):
        return ""
    python = python_for_path(target, state=state)
    return f"{shlex.quote(python)} -m py_compile {shlex.quote(target)}"


def python_for_path(path: str, *, state: LoopState) -> str:
    cwd = Path(str(getattr(state, "cwd", "") or "."))
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    venv_python = candidate.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def missing_dependency_block(state: LoopState) -> dict[str, Any] | None:
    verifier = getattr(state, "last_verifier_verdict", None)
    if not isinstance(verifier, dict):
        return None
    if str(verifier.get("verdict") or "").strip().lower() == "pass":
        return None
    text = "\n".join(
        str(verifier.get(key) or "")
        for key in ("key_stderr", "key_stdout")
        if str(verifier.get(key) or "").strip()
    )
    match = _MISSING_MODULE_RE.search(text)
    if not match:
        return None
    module = match.group("module").strip()
    paths = [str(path or "").strip() for path in getattr(state.challenge_progress, "last_code_change_paths", []) or [] if str(path or "").strip()]
    python = python_for_path(paths[-1], state=state) if paths else "python3"
    return {
        "reason": "missing_runtime_dependency",
        "module": module,
        "command": f"{shlex.quote(python)} -m pip install {shlex.quote(module)}",
        "notes": [
            f"The latest verifier is blocked by missing Python module `{module}`.",
            "Install the dependency in the verifier interpreter, then rerun the focused verifier instead of retrying task_complete.",
        ],
    }
