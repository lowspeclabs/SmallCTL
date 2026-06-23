from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any


_TRACEBACK_FILE_RE = re.compile(r'File ["\'](?P<path>[^"\']+)["\'], line (?P<line>\d+)', re.IGNORECASE)
_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError:\s+No module named ['\"](?P<module>[^'\"]+)['\"]")
_IMPORT_ERROR_RE = re.compile(r"ImportError:\s+(?P<message>.+)")
_EXCEPTION_RE = re.compile(r"(?P<kind>[A-Z][A-Za-z_]*(?:Error|Exception)):\s*(?P<message>[^\n]+)")
_RUNTIME_ERROR_HINT_RE = re.compile(
    r"\b(?:traceback|error:|exception|"
    r"does not run|doesn't run|crashes|failed when|here is the error)\b",
    re.IGNORECASE,
)
_DIFF_HUNK_RE = re.compile(r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@", re.MULTILINE)


def maybe_record_reported_runtime_error(state: Any, text: str) -> dict[str, Any] | None:
    """Record a user-reported runtime error in scratchpad.

    This intentionally uses scratchpad instead of LoopState schema fields so the
    guard can be introduced without state migration risk.
    """
    report = parse_reported_runtime_error(text)
    if report is None:
        return None

    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None

    previous = scratchpad.get("_reported_runtime_error")
    repeated_count = 1
    if isinstance(previous, dict):
        prev_signature = str(previous.get("signature") or "")
        if prev_signature and prev_signature == report["signature"]:
            repeated_count = int(previous.get("repeated_count", 1) or 1) + 1

    report["status"] = "open"
    report["repeated_count"] = repeated_count
    report["verified_after_report"] = False
    scratchpad["_reported_runtime_error"] = report
    scratchpad["_reported_runtime_error_prior_fix_classes"] = _runtime_error_prior_fix_classes(scratchpad, report)
    try:
        state.active_intent = "reported_runtime_error_repair"
    except Exception:
        pass
    touch = getattr(state, "touch", None)
    if callable(touch):
        touch()
    return report


def parse_reported_runtime_error(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw or _RUNTIME_ERROR_HINT_RE.search(raw) is None:
        return None

    kind = "RuntimeError"
    message = ""
    module = ""
    missing_module = _MISSING_MODULE_RE.search(raw)
    if missing_module:
        kind = "ModuleNotFoundError"
        module = missing_module.group("module").strip()
        message = f"No module named {module!r}"
    else:
        import_error = _IMPORT_ERROR_RE.search(raw)
        generic = _EXCEPTION_RE.search(raw)
        if import_error:
            kind = "ImportError"
            message = import_error.group("message").strip()
        elif generic:
            kind = generic.group("kind").strip()
            message = generic.group("message").strip()

    entrypoint = ""
    traceback_line = ""
    traceback_match = None
    for traceback_match in _TRACEBACK_FILE_RE.finditer(raw):
        pass
    if traceback_match is not None:
        entrypoint = traceback_match.group("path").strip()
        traceback_line = traceback_match.group("line").strip()

    signature_parts = [kind]
    if module:
        signature_parts.append(module)
    elif message:
        signature_parts.append(message[:120])
    if entrypoint:
        signature_parts.append(Path(entrypoint).name)

    return {
        "kind": kind,
        "message": message,
        "module": module,
        "entrypoint": entrypoint,
        "traceback_line": traceback_line,
        "raw_text": raw[:4000],
        "signature": "|".join(signature_parts),
    }


def _diff_covers_line(diff: str, line: int) -> bool:
    """Return True if a unified-diff hunk covers the given old-file line number."""
    for match in _DIFF_HUNK_RE.finditer(diff):
        old_start = int(match.group("old_start"))
        old_count = int(match.group("old_count") or "1")
        if old_start <= line < old_start + old_count:
            return True
    return False


def _patch_acts_as_verifier(report: dict[str, Any], state: Any) -> bool:
    """Check if a successful file_patch on the exact traceback line serves as verification.

    For exact-line runtime errors (e.g. AttributeError at line 248), a successful
    file_patch that modifies the hunk containing that line is accepted as proof
    the error was addressed, even when no shell_exec verifier was run.
    """
    entrypoint = str(report.get("entrypoint") or "").strip()
    traceback_line = str(report.get("traceback_line") or "").strip()
    if not entrypoint or not traceback_line:
        return False
    try:
        target_line = int(traceback_line)
    except ValueError:
        return False

    artifacts = getattr(state, "artifacts", {})
    if not isinstance(artifacts, dict):
        return False

    entrypoint_resolved = Path(entrypoint).resolve()
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            continue
        if artifact.get("kind") != "file_patch":
            continue
        meta = artifact.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        if not meta.get("success") or not meta.get("changed"):
            continue
        patch_path = str(meta.get("path") or meta.get("source_path") or "").strip()
        if not patch_path:
            continue
        if Path(patch_path).resolve() != entrypoint_resolved:
            continue
        diff = str(meta.get("diff") or "")
        if _diff_covers_line(diff, target_line):
            return True
    return False


def runtime_error_completion_block(state: Any, *, verifier_verdict: dict[str, Any] | None) -> dict[str, Any] | None:
    report = current_reported_runtime_error(state)
    if report is None:
        return None
    verifier = verifier_verdict if isinstance(verifier_verdict, dict) else {}
    if runtime_error_verifier_passes(report, verifier):
        _mark_runtime_error_verified(state, report, verifier)
        return None
    if _patch_acts_as_verifier(report, state):
        _mark_runtime_error_verified(state, report, {"command": "file_patch", "verdict": "pass"})
        return None
    return _runtime_error_block_payload(report, verifier, reason="reported_runtime_error_unverified")


def runtime_error_task_fail_block(
    state: Any,
    *,
    message: str,
    verifier_verdict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    report = current_reported_runtime_error(state)
    if report is None:
        return None
    verifier = verifier_verdict if isinstance(verifier_verdict, dict) else {}
    if not verifier:
        return _runtime_error_block_payload(report, verifier, reason="reported_runtime_error_failure_without_verifier")
    if runtime_error_verifier_passes(report, verifier):
        return None
    if not runtime_error_verifier_is_relevant(report, verifier):
        return _runtime_error_block_payload(report, verifier, reason="reported_runtime_error_failure_irrelevant_verifier")
    combined = _verifier_text(verifier).lower()
    msg = str(message or "").lower()
    signature_terms = _signature_terms(report)
    if not any(term and (term.lower() in combined or term.lower() in msg) for term in signature_terms):
        return _runtime_error_block_payload(report, verifier, reason="reported_runtime_error_failure_unsupported")
    return None


def runtime_error_ask_human_block(state: Any, *, question: str) -> dict[str, Any] | None:
    report = current_reported_runtime_error(state)
    if report is None:
        return None
    text = str(question or "").lower()
    asks_missing_info = any(
        phrase in text
        for phrase in (
            "which command",
            "what command",
            "missing information",
            "which file",
            "what path",
            "approval",
            "credentials",
            "permission",
        )
    )
    asks_user_to_retry = any(
        phrase in text
        for phrase in (
            "please run",
            "let me know if",
            "try again",
            "test it yourself",
            "would you like me to verify",
            "ready for testing",
        )
    )
    if asks_missing_info and not asks_user_to_retry:
        return None
    return _runtime_error_block_payload(report, {}, reason="reported_runtime_error_ask_human_before_repair")


def current_reported_runtime_error(state: Any) -> dict[str, Any] | None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    report = scratchpad.get("_reported_runtime_error")
    if not isinstance(report, dict):
        return None
    if str(report.get("status") or "open").strip().lower() in {"verified_fixed", "closed", "ignored"}:
        return None
    return report


def runtime_error_verifier_passes(report: dict[str, Any], verifier: dict[str, Any]) -> bool:
    verdict = str(verifier.get("verdict") or verifier.get("status") or "").strip().lower()
    if verdict != "pass":
        return False
    if not runtime_error_verifier_is_relevant(report, verifier):
        return False
    text = _verifier_output_text(verifier).lower()
    return not any(term and term.lower() in text for term in _signature_terms(report))


def runtime_error_verifier_is_relevant(report: dict[str, Any], verifier: dict[str, Any]) -> bool:
    command = str(verifier.get("command") or verifier.get("target") or "").strip().lower()
    if not command:
        return False
    entrypoint = str(report.get("entrypoint") or "").strip()
    module = str(report.get("module") or "").strip()
    if entrypoint:
        entry_name = Path(entrypoint).name.lower()
        entry_path = entrypoint.replace("\\", "/").lower()
        if entry_name and entry_name in command:
            return True
        if entry_path and entry_path in command.replace("\\", "/"):
            return True
    if module:
        if re.search(rf"\bimport\s+{re.escape(module.lower())}\b", command):
            return True
        if module.lower() in command and "python" in command:
            return True
    return False


def runtime_error_required_command(report: dict[str, Any]) -> str:
    entrypoint = str(report.get("entrypoint") or "").strip()
    module = str(report.get("module") or "").strip()
    if module and entrypoint:
        directory = str(Path(entrypoint).parent)
        python = _python_for_directory(directory)
        return f"cd {shlex.quote(directory)} && {shlex.quote(python)} -c {shlex.quote('import ' + module)}"
    if entrypoint:
        python = _python_for_directory(str(Path(entrypoint).parent))
        return f"{shlex.quote(python)} {shlex.quote(entrypoint)}"
    if module:
        return f"python3 -c {shlex.quote('import ' + module)}"
    return "run the smallest focused verifier that reproduces the reported traceback entrypoint"


def _runtime_error_block_payload(
    report: dict[str, Any],
    verifier: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    module = str(report.get("module") or "").strip()
    kind = str(report.get("kind") or "runtime error").strip()
    entrypoint = str(report.get("entrypoint") or "").strip()
    repeated = int(report.get("repeated_count", 1) or 1)
    notes = [
        f"The user reported `{kind}` and this runtime error is still open.",
        "Do not finish, fail, or ask the user to retry until the exact reported error is localized, patched, and verified.",
        "Run or inspect the traceback entrypoint/import target, apply a direct fix, then run a verifier that covers the same entrypoint or import.",
    ]
    if repeated >= 2:
        notes.append(
            "The same error was reported more than once; do not repeat the previous fix class. State a different concrete hypothesis before patching."
        )
    if module:
        notes.append(f"For this import error, inspect whether module `{module}` exists under the imported Python module name.")
    return {
        "reason": reason,
        "reported_runtime_error": {
            "kind": kind,
            "message": str(report.get("message") or ""),
            "module": module,
            "entrypoint": entrypoint,
            "traceback_line": str(report.get("traceback_line") or ""),
            "signature": str(report.get("signature") or ""),
            "repeated_count": repeated,
        },
        "last_verifier_verdict": verifier or None,
        "next_required_action": {
            "tool_names": ["file_read", "dir_list", "find_files", "file_patch", "shell_exec"],
            "required_arguments": {"command": runtime_error_required_command(report)},
            "notes": notes,
        },
    }


def _mark_runtime_error_verified(state: Any, report: dict[str, Any], verifier: dict[str, Any]) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    updated = dict(report)
    updated["status"] = "verified_fixed"
    updated["verified_after_report"] = True
    updated["verifier_command"] = str(verifier.get("command") or verifier.get("target") or "")
    scratchpad["_reported_runtime_error"] = updated


def _verifier_text(verifier: dict[str, Any]) -> str:
    return "\n".join(
        str(verifier.get(key) or "")
        for key in ("command", "target", "key_stdout", "key_stderr", "stdout", "stderr", "failure_mode", "notes")
        if str(verifier.get(key) or "").strip()
    )


def _verifier_output_text(verifier: dict[str, Any]) -> str:
    return "\n".join(
        str(verifier.get(key) or "")
        for key in ("key_stdout", "key_stderr", "stdout", "stderr", "failure_mode", "notes")
        if str(verifier.get(key) or "").strip()
    )


def _signature_terms(report: dict[str, Any]) -> list[str]:
    terms = [str(report.get("kind") or "").strip(), str(report.get("module") or "").strip()]
    message = str(report.get("message") or "").strip()
    if message:
        terms.append(message)
    return [term for term in terms if term]


def _runtime_error_prior_fix_classes(scratchpad: dict[str, Any], report: dict[str, Any]) -> list[str]:
    previous = scratchpad.get("_reported_runtime_error_prior_fix_classes")
    classes = list(previous) if isinstance(previous, list) else []
    if int(report.get("repeated_count", 1) or 1) >= 2 and "previous_hypothesis_failed" not in classes:
        classes.append("previous_hypothesis_failed")
    return classes[-5:]


def _python_for_directory(directory: str) -> str:
    path = Path(directory)
    venv_python = path / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"
