from __future__ import annotations

import re
from typing import Any

from .tool_result_verification_constants import (
    _DOCKER_NO_SUCH_CONTAINER_RE,
    _FOG_RESOURCE_RE,
    _LS_NO_SUCH_FILE_RE,
    _REMOVAL_ABSENCE_PIPE_RE,
    _REMOVAL_ABSENCE_PROBE_RE,
    _REMOVAL_TASK_KEYWORDS,
)
from .tool_result_verification_helpers import (
    command_has_write_or_heredoc_shape,
    exit_code_matches,
    output_confirms_not_found,
)


def _task_has_removal_intent(state: Any) -> bool:
    """Return True when the original task description signals a removal intent."""
    original_task = _removal_task_text(state).lower()
    if not original_task:
        return False
    return any(kw in original_task for kw in _REMOVAL_TASK_KEYWORDS)


def _removal_task_text(state: Any) -> str:
    run_brief = getattr(state, "run_brief", None)
    original_task = str(getattr(run_brief, "original_task", "") or "")
    wm = getattr(state, "working_memory", None)
    current_goal = str(getattr(wm, "current_goal", "") or "")
    return " ".join(part for part in (original_task, current_goal) if part).strip()


def _removal_task_subject_terms(state: Any) -> set[str]:
    task_text = _removal_task_text(state)
    if not task_text:
        return set()
    candidates = re.findall(r"[a-z0-9][a-z0-9_.@/-]{2,}", task_text.lower())
    stop_words = {
        "cleanup", "clean", "remove", "removed", "removal", "delete", "deleted",
        "purge", "stop", "disable", "disabled", "mask", "masked", "verify",
        "remote", "server", "host", "file", "files", "service", "services",
        "systemd", "process", "processes", "user", "users", "database",
    }
    terms = set()
    for candidate in candidates:
        stripped = candidate.strip(".,:;()[]{}'\"")
        if stripped in stop_words:
            continue
        if stripped.startswith(("http", "/tmp/")):
            continue
        terms.add(stripped)
    return terms


def _command_mentions_removal_subject(command: str, state: Any) -> bool:
    lowered = str(command or "").lower()
    if _FOG_RESOURCE_RE.search(lowered):
        return True
    task_terms = _removal_task_subject_terms(state)
    return any(term in lowered for term in task_terms)


def _command_is_removal_absence_probe(command: str, state: Any) -> bool:
    cmd = str(command or "").strip()
    if not _REMOVAL_ABSENCE_PROBE_RE.search(cmd):
        return False
    if command_has_write_or_heredoc_shape(cmd):
        return False
    lowered = cmd.lower()
    has_absence_tool_shape = (
        re.search(r"(?:^|[;&|]\s*)find\s+", lowered) is not None
        or re.search(r"(?:^|[;&|]\s*)ls\s+", lowered) is not None
        or "grep" in lowered
        or "pgrep" in lowered
        or re.search(r"(?:^|[;&|]\s*)ps(?:\s|$)", lowered) is not None
        or ("systemctl" in lowered and _REMOVAL_ABSENCE_PIPE_RE.search(cmd) is not None)
        or re.search(r"(?:^|[;&|]\s*)docker(?:\s+container)?\s+rm\b", lowered) is not None
    )
    if not has_absence_tool_shape:
        return False
    return _command_mentions_removal_subject(cmd, state)


def _absence_probe_found_resources(
    *,
    command: str,
    stdout: str,
    stderr: str,
    exit_code: Any,
) -> bool:
    out = str(stdout or "").strip()
    err = str(stderr or "").strip()
    if _LS_NO_SUCH_FILE_RE.search(err) or _LS_NO_SUCH_FILE_RE.search(out):
        return False
    lowered_cmd = str(command or "").lower()
    if ("grep -q" in lowered_cmd or "grep --quiet" in lowered_cmd) and exit_code_matches(exit_code, {0}):
        return True
    if out:
        return True
    if err and not output_confirms_not_found("", err):
        return True
    return False


def _looks_like_ssh_transport_failure(*, exit_code: Any, stdout: str, stderr: str) -> bool:
    if exit_code_matches(exit_code, {255}):
        return True
    combined = f"{stdout}\n{stderr}".lower()
    return any(
        marker in combined
        for marker in (
            "no route to host",
            "connection refused",
            "connection timed out",
            "network is unreachable",
            "could not resolve hostname",
            "permission denied (publickey",
            "permission denied, please try again",
        )
    )


def _classify_removal_absence_probe(
    state: Any,
    *,
    command: str,
    exit_code: Any,
    stdout: str,
    stderr: str,
    tool_name: str = "",
) -> dict[str, Any]:
    default = {
        "is_absence_probe": False,
        "absence_confirmed": False,
        "found_resources": False,
        "reason": "",
    }
    if not _task_has_removal_intent(state):
        return default
    cmd = str(command or "").strip()
    if not cmd or not _command_is_removal_absence_probe(cmd, state):
        return default

    out = str(stdout or "").strip()
    err = str(stderr or "").strip()
    if str(tool_name or "").strip() == "ssh_exec" and _looks_like_ssh_transport_failure(
        exit_code=exit_code,
        stdout=out,
        stderr=err,
    ):
        return default
    combined = "\n".join(part for part in (out, err) if part).strip()
    found_resources = _absence_probe_found_resources(command=cmd, stdout=out, stderr=err, exit_code=exit_code)
    if found_resources:
        return {
            "is_absence_probe": True,
            "absence_confirmed": False,
            "found_resources": True,
            "reason": "absence probe found matching cleanup/removal resources",
        }

    if _LS_NO_SUCH_FILE_RE.search(err) or _LS_NO_SUCH_FILE_RE.search(out):
        return {
            "is_absence_probe": True,
            "absence_confirmed": True,
            "found_resources": False,
            "reason": "ls reported deleted resource is absent",
        }

    if _DOCKER_NO_SUCH_CONTAINER_RE.search(err) or _DOCKER_NO_SUCH_CONTAINER_RE.search(out):
        return {
            "is_absence_probe": True,
            "absence_confirmed": True,
            "found_resources": False,
            "reason": "docker reports container already absent",
        }

    lowered_cmd = cmd.lower()
    empty_output = not combined
    if empty_output and exit_code_matches(exit_code, {0, 1}):
        if "grep" in lowered_cmd:
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "grep absence probe returned no matches",
            }
        if "find" in lowered_cmd:
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "find absence probe returned no matches",
            }
        if "systemctl" in lowered_cmd and _REMOVAL_ABSENCE_PIPE_RE.search(cmd):
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "systemctl absence probe returned no matching units",
            }
        if "pgrep" in lowered_cmd or re.search(r"(?:^|\s)ps(?:\s|$)", lowered_cmd):
            return {
                "is_absence_probe": True,
                "absence_confirmed": True,
                "found_resources": False,
                "reason": "process absence probe returned no matches",
            }

    return {
        "is_absence_probe": True,
        "absence_confirmed": False,
        "found_resources": False,
        "reason": "absence probe did not prove resource absence",
    }
