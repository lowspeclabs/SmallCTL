from __future__ import annotations

import re
from typing import Any

from .tool_result_verification_constants import (
    _CURL_VERIFIER_FAILURE_RE,
    _NGINX_VERIFIER_COMMAND_RE,
    _NGINX_VERIFIER_FAILURE_RE,
    _TEST_FAILURE_COUNT_RE,
    _TEST_FAILURE_OUTPUT_RE,
    _TEST_FAILURE_SUMMARY_RE,
    _ZERO_TESTS_RAN_RE,
)
from .tool_result_verification_helpers import snip_text as _snip_text, verifier_kind_for_command, verifier_strength


_DOCKER_INVENTORY_HEADERS = (
    "container id",
    "driver    volume name",
    "network id",
    "repository",
    "docker version",
    "server version",
)

_DOCKER_NON_SWARM_DIAGNOSTIC_MARKERS = (
    "this node is not a swarm manager",
    "usage:  docker swarm command",
    "run 'docker swarm command --help'",
)


def _docker_segment_is_readonly_diagnostic(segment: str) -> bool:
    import shlex

    text = str(segment or "").strip()
    if not text:
        return True
    try:
        tokens = shlex.split(text)
    except ValueError:
        return False
    while tokens and tokens[0] in {"sudo", "timeout", "env"}:
        if tokens[0] == "timeout" and len(tokens) >= 2:
            tokens = tokens[2:]
        elif tokens[0] == "env":
            tokens = tokens[1:]
            while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
                tokens = tokens[1:]
        else:
            tokens = tokens[1:]
    if len(tokens) < 2 or tokens[0] not in {"docker", "podman"}:
        return False
    subcommand = tokens[1].lower()
    if subcommand in {"ps", "info", "version", "inspect", "logs"}:
        return True
    if subcommand in {"volume", "network", "image", "config", "container", "service", "stack", "node"}:
        return len(tokens) >= 3 and tokens[2].lower() in {"ls", "list", "inspect", "show"}
    if subcommand == "swarm":
        # Docker has no `swarm status` command, but models often use it as a
        # read-only diagnostic. Treat only this shape as diagnostic noise.
        return len(tokens) >= 3 and tokens[2].lower() == "status"
    return False


def _command_is_docker_readonly_diagnostic_chain(command: str) -> bool:
    normalized = str(command or "").strip()
    if not normalized:
        return False
    if re.search(r"\|\||\|", normalized):
        return False
    segments = [part.strip() for part in re.split(r"\s*(?:&&|;)\s*", normalized) if part.strip()]
    return bool(segments) and all(_docker_segment_is_readonly_diagnostic(segment) for segment in segments)


def _docker_diagnostic_has_useful_partial_output(*, command: str, stdout: str, stderr: str) -> bool:
    if not _command_is_docker_readonly_diagnostic_chain(command):
        return False
    out = str(stdout or "").strip().lower()
    err = str(stderr or "").strip().lower()
    if not out:
        return False
    if not any(header in out for header in _DOCKER_INVENTORY_HEADERS):
        return False
    if not err:
        return True
    return any(marker in err for marker in _DOCKER_NON_SWARM_DIAGNOSTIC_MARKERS)


def _semantic_verifier_failure(*, command: str, stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in (stdout, stderr) if str(part or "").strip())
    if not combined:
        return ""
    normalized_command = re.sub(r"\s+", " ", str(command or "").strip().lower())
    normalized_output = re.sub(r"\s+", " ", combined.strip().lower())
    if (
        "test -f" in normalized_command
        and "echo" in normalized_command
        and re.search(r"\bmissing\b", normalized_output)
        and not re.search(r"\bexists\b", normalized_output)
    ):
        return "file existence verifier reported MISSING"
    if _NGINX_VERIFIER_COMMAND_RE.search(command) or "nginx:" in combined.lower():
        match = _NGINX_VERIFIER_FAILURE_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    if "curl" in str(command or "").lower() or "curl:" in combined.lower():
        match = _CURL_VERIFIER_FAILURE_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    match = _ZERO_TESTS_RAN_RE.search(combined)
    if match:
        return _snip_text(match.group(0), limit=240)
    if verifier_kind_for_command(command) in {"test_suite", "run_target"}:
        match = _TEST_FAILURE_SUMMARY_RE.search(combined)
        if match:
            summary = _snip_text(match.group(0), limit=240)
            count_match = _TEST_FAILURE_COUNT_RE.search(combined)
            if count_match:
                failures = next(g for g in count_match.groups() if g is not None)
                summary = f"{summary} ({failures} test failure(s) detected)"
            return summary
        match = _TEST_FAILURE_OUTPUT_RE.search(combined)
        if match:
            return _snip_text(match.group(0), limit=240)
    return ""


def _task_or_history_requires_runtime_verifier(state: Any) -> bool:
    prior_command = _prior_failed_verifier_command(state)
    if verifier_strength(verifier_kind_for_command(prior_command)) > verifier_strength("syntax_only"):
        return True
    texts: list[str] = []
    run_brief = getattr(state, "run_brief", None)
    texts.append(str(getattr(run_brief, "original_task", "") or ""))
    working_memory = getattr(state, "working_memory", None)
    texts.append(str(getattr(working_memory, "current_goal", "") or ""))
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        handoff = scratchpad.get("_last_task_handoff")
        if isinstance(handoff, dict):
            texts.append(str(handoff.get("effective_task") or ""))
            texts.append(str(handoff.get("current_goal") or ""))
    combined = " ".join(texts).lower()
    return any(
        marker in combined
        for marker in (
            "run script",
            "run the script",
            "run it",
            "test",
            "tests",
            "unittest",
            "pytest",
            "verify functionality",
            "fix until complete",
        )
    )


def _prior_failed_verifier_command(state: Any) -> str:
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        failed = scratchpad.get("_last_failed_verifier")
        if isinstance(failed, dict):
            command = str(failed.get("command") or "").strip()
            if command:
                return command
    prior = getattr(state, "last_verifier_verdict", None)
    if isinstance(prior, dict) and str(prior.get("verdict") or "").strip().lower() == "fail":
        return str(prior.get("command") or "").strip()
    return ""


def _passing_verifier_is_weaker_than_prior_failure(
    state: Any,
    *,
    current_command: str,
    current_kind: str,
) -> bool:
    if verifier_strength(current_kind) > verifier_strength("syntax_only"):
        return False
    prior_command = _prior_failed_verifier_command(state)
    if not prior_command:
        return False
    prior_kind = verifier_kind_for_command(prior_command)
    if verifier_strength(prior_kind) <= verifier_strength(current_kind):
        return False
    normalized_current = re.sub(r"\s+", " ", str(current_command or "").strip().lower())
    normalized_prior = re.sub(r"\s+", " ", prior_command.strip().lower())
    return normalized_current != normalized_prior


def _insufficient_verifier_message(state: Any, *, command: str) -> str:
    prior_command = _prior_failed_verifier_command(state)
    if prior_command:
        return (
            f"Verifier `{command}` only checks syntax and is weaker than the prior failed verifier "
            f"`{prior_command}`; rerun the script/tests that failed."
        )
    return f"Verifier `{command}` only checks syntax; rerun the script/tests required by the task."


def _install_task_requires_strong_verifier(state: Any, *, command: str) -> tuple[bool, str]:
    """Check if the current task is an install task and the verifier is too weak.

    Returns (True, reason) if the verifier should be rejected as insufficient for
    an install/ setup/ deploy objective.
    """
    task = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "").strip().lower()
    if not task:
        return False, ""
    install_markers = ("install", "setup", "deploy", "configure")
    if not any(m in task for m in install_markers):
        return False, ""

    normalized = re.sub(r"\s+", " ", str(command or "").strip().lower())
    if re.fullmatch(r"(?:sudo\s+)?cat\s+/etc/(?:os-release|redhat-release|centos-release)", normalized):
        return False, ""
    kind = verifier_kind_for_command(command)
    if verifier_strength(kind) >= verifier_strength("install_service_status"):
        # Strong install verifier (service status, package, port, version)
        return False, ""
    # Weak verifiers for install tasks: only check file existence
    weak_patterns = [
        (r"\bls\s+-la?\s+\S+", "file existence check"),
        (r"\btest\s+-[ef]\s+\S+", "file existence check"),
        (r"\bcat\s+\S+", "file content check"),
    ]
    for pattern, check_type in weak_patterns:
        if re.search(pattern, normalized):
            return True, (
                f"This is an install task, but the verifier only does a {check_type} (`{command}`). "
                f"For install tasks, verify with service status, version command, or listening port."
            )
    return False, ""
