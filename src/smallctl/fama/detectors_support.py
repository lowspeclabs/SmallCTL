from __future__ import annotations

from collections import Counter
from typing import Any

from .detector_classifiers import (
    WRITE_TOOLS,
    _READ_LOOP_TOOLS,
    _metadata_verifier,
    _verifier_failed,
    _looks_like_zero_tests,
)
from .signals import current_step


def _scratchpad_int(state: Any, key: str, default: int) -> int:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return default
    try:
        return int(scratchpad.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _has_human_gate(state: Any) -> bool:
    pending = getattr(state, "pending_interrupt", None)
    if isinstance(pending, dict) and pending:
        return True
    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    for key in ("_human_approval_pending", "_shell_human_retry_state"):
        if scratchpad.get(key):
            return True
    return False


def _result_text(result: Any, *, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata if isinstance(metadata, dict) else getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    parts = [
        str(getattr(result, "error", "") or ""),
        str(getattr(result, "output", "") or ""),
    ]
    for key in (
        "error",
        "message",
        "reason",
        "path",
        "target",
        "stderr",
        "stdout",
        "verifier_stderr",
        "verifier_stdout",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        for key in ("stderr", "stdout", "message", "error"):
            value = output.get(key)
            if value not in (None, ""):
                parts.append(str(value))
    return "\n".join(parts)


def _path_from_metadata_or_args(metadata: dict[str, Any]) -> str:
    for key in ("path", "target", "file", "filename"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    args = metadata.get("arguments")
    if isinstance(args, dict):
        for key in ("path", "target", "file", "filename"):
            value = str(args.get(key) or "").strip()
            if value:
                return value
    return ""


def _repeated_tool_from_history(state: Any, *, threshold: int) -> tuple[str | None, str | None]:
    history = getattr(state, "tool_history", None)
    if not isinstance(history, list) or not history:
        return None, None
    counts = Counter(str(item or "") for item in history if str(item or "").strip())
    for fingerprint, count in counts.most_common():
        if count < threshold:
            continue
        tool_name = fingerprint.split("|", 1)[0].strip()
        if tool_name:
            return tool_name, fingerprint
    return None, None


def _record_read_loop_recovery_payload(
    state: Any,
    *,
    tool_name: str,
    fingerprint: str | None,
) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    target = _read_loop_target_from_fingerprint(fingerprint)
    scratchpad["_read_loop_recovery_payload"] = {
        "tool_name": tool_name,
        "fingerprint": str(fingerprint or "")[:240],
        "target": target,
        "created_at_step": current_step(state),
        "last_evidence_summary": _read_loop_evidence_summary(state, tool_name=tool_name, target=target),
        "allowed_next_action": (
            f"Do not call {tool_name} with the same target again. Use visible evidence to patch, "
            "run a focused verifier, inspect a different missing target, or complete if enough evidence is present."
        ),
    }


def _read_loop_target_from_fingerprint(fingerprint: str | None) -> str:
    text = str(fingerprint or "").strip()
    if not text:
        return ""
    if "|" in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if len(parts) >= 2:
            return parts[1][:180]
    return text[:180]


def _read_loop_evidence_summary(state: Any, *, tool_name: str, target: str) -> str:
    if tool_name == "artifact_read":
        artifacts = getattr(state, "artifacts", None)
        if isinstance(artifacts, dict) and target:
            artifact = artifacts.get(target)
            title = str(getattr(artifact, "title", "") or "").strip()
            metadata = getattr(artifact, "metadata", None)
            if isinstance(metadata, dict):
                lines = metadata.get("line_count") or metadata.get("total_lines")
                if title and lines:
                    return f"Artifact {target} ({title}) has {lines} lines; use already-read preview/coverage before rereading."
                if title:
                    return f"Artifact {target} ({title}) was already read recently."
        return f"Recent artifact_read calls already targeted {target or 'the same artifact/range'}."
    if target:
        return f"Recent {tool_name} calls already targeted {target}."
    return f"Recent {tool_name} calls repeated without producing new evidence."


def _next_action_for_repeated_loop(
    *,
    repeated_tool: str | None,
    counters: list[tuple[str, int]],
) -> str:
    tool_name = str(repeated_tool or "").strip()
    counter_names = {name for name, _count in counters}
    if tool_name in _READ_LOOP_TOOLS:
        return (
            f"Do not call {tool_name} with the same target again. Use the already-visible evidence "
            "to make the next patch, run a focused verifier, or inspect a different missing target."
        )
    if tool_name in {"shell_exec", "ssh_exec"}:
        return (
            f"Do not rerun the same {tool_name} command unchanged. Read its prior output, then change "
            "one variable: path, command, patch, or verifier scope."
        )
    if tool_name in WRITE_TOOLS or "repeat_patch" in counter_names:
        return (
            "Do not retry the same write/patch unchanged. Inspect the current staged/target content, "
            "then make the smallest different patch."
        )
    if "no_progress" in counter_names or "no_actionable_progress" in counter_names:
        return (
            "Stop the current loop, state one concrete missing evidence item, and take a different "
            "tool action that can produce it."
        )
    return "Reuse the prior evidence and take the next smallest different action."


def _early_stop_evidence(metadata: dict[str, Any], *, error: str, state: Any) -> str:
    verifier = _metadata_verifier(metadata)
    if _verifier_failed(verifier):
        return _verifier_evidence(verifier)

    lowered_error = error.lower()
    if "latest verifier verdict is still failing" in lowered_error:
        return "task_complete rejected because the latest verifier verdict is failing"

    pending = metadata.get("pending_acceptance_criteria")
    if isinstance(pending, list) and pending:
        return "task_complete rejected with pending acceptance criteria"
    checklist = metadata.get("acceptance_checklist")
    if _checklist_has_pending(checklist):
        return "task_complete rejected with unsatisfied acceptance checklist"

    scratchpad = getattr(state, "scratchpad", None)
    scratch_verdict = scratchpad.get("_last_verifier_verdict") if isinstance(scratchpad, dict) else None
    if _verifier_failed(scratch_verdict):
        return _verifier_evidence(scratch_verdict)
    return ""


def _verifier_from_result_or_state(state: Any, *, result: Any | None) -> dict[str, Any] | None:
    metadata = getattr(result, "metadata", None) if result is not None else None
    metadata = metadata if isinstance(metadata, dict) else {}
    verifier = _metadata_verifier(metadata)
    if verifier is not None:
        return verifier
    nested = metadata.get("verifier")
    if isinstance(nested, dict) and nested:
        return nested
    verifier_verdict = str(metadata.get("verifier_verdict") or "").strip()
    if verifier_verdict:
        return {
            "verdict": verifier_verdict,
            "command": str(metadata.get("verifier_command") or metadata.get("command") or ""),
            "target": str(metadata.get("verifier_target") or metadata.get("target") or ""),
            "exit_code": metadata.get("verifier_exit_code"),
            "key_stdout": str(metadata.get("verifier_stdout") or ""),
            "key_stderr": str(metadata.get("verifier_stderr") or ""),
            "failure_mode": str(metadata.get("failure_mode") or ""),
        }
    current_verifier = getattr(state, "current_verifier_verdict", None)
    verdict = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    return verdict if isinstance(verdict, dict) and verdict else None


def _verifier_evidence(verifier: dict[str, Any]) -> str:
    verdict = str(verifier.get("verdict") or "unknown").strip()
    target = str(verifier.get("command") or verifier.get("target") or "").strip()
    output = " ".join(
        str(verifier.get(key) or "").lower()
        for key in ("key_stdout", "key_stderr", "failure_mode")
    )
    qualifier = "; zero tests discovered" if _looks_like_zero_tests(output) else ""
    error_snippet = ""
    if verdict != "pass":
        blocker = verifier.get("latest_blocker") or {}
        salient = str(blocker.get("salient_error") or "").strip()
        if salient:
            error_snippet = salient
        else:
            for key in ("key_stderr", "key_stdout", "failure_mode"):
                text = str(verifier.get(key) or "").strip()
                if text:
                    error_snippet = text
                    break
        if error_snippet:
            error_snippet = f" [{error_snippet[:200]}]"
    if target:
        return f"task_complete rejected with verifier verdict {verdict}: {target}{qualifier}{error_snippet}"
    return f"task_complete rejected with verifier verdict {verdict}{qualifier}{error_snippet}"


def _checklist_has_pending(checklist: Any) -> bool:
    if not isinstance(checklist, list):
        return False
    for item in checklist:
        if isinstance(item, dict) and not bool(item.get("satisfied")):
            return True
    return False
