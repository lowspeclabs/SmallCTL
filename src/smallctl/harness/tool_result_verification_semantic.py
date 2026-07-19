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

_STRUCTURED_ERROR_HEADING_RE = re.compile(r"^##\s+error\b", re.IGNORECASE)
_TRAILING_NONZERO_EXIT_MARKER_RE = re.compile(r"^-{2,}\s*EXIT\s*:\s*([1-9][0-9]*)\s*$", re.IGNORECASE)
_FAMILY_INTERPRETER_RE = re.compile(r"^(?:python(?:3(?:\.\d+)?)?|node|ruby|perl|php|bash|sh)$")


def _app_level_failure_marker(*, stdout: str, stderr: str) -> str:
    """Detect app-level failure reports that contradict a zero shell exit code.

    Some CLIs print a structured error report (a leading ``## Error`` block)
    or an explicit trailing exit marker (``---EXIT:4``) while the surrounding
    pipeline still exits 0. Anchoring to the leading block and the trailing
    marker keeps benign mid-output mentions (e.g. file contents that contain
    ``## Error``) from tripping the detector.
    """
    for stream in (stdout, stderr):
        stripped = str(stream or "").strip()
        if not stripped:
            continue
        first_line = stripped.splitlines()[0].strip()
        if _STRUCTURED_ERROR_HEADING_RE.match(first_line):
            return "output begins with a structured '## Error' failure report"
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        match = _TRAILING_NONZERO_EXIT_MARKER_RE.match(lines[-1]) if lines else None
        if match:
            return f"trailing '---EXIT:{match.group(1)}' marker reports a nonzero app-level exit"
    return ""


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
    app_failure = _app_level_failure_marker(stdout=stdout, stderr=stderr)
    if app_failure:
        return app_failure
    normalized_command = re.sub(r"\s+", " ", str(command or "").strip().lower())
    normalized_output = re.sub(r"\s+", " ", combined.strip().lower())
    if (
        "test -f" in normalized_command
        and "echo" in normalized_command
        and re.search(r"\bmissing\b", normalized_output)
        and not re.search(r"\bexists\b", normalized_output)
    ):
        return "file existence verifier reported MISSING"
    if _command_is_pipe_to_shell(normalized_command) and _output_looks_like_404_body(combined):
        return "pipe-to-shell command fetched a 404 / HTML error body instead of a script"
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


_PIPE_TO_SHELL_RE = re.compile(
    r"(?:curl|wget)\b.*\|\s*(?:bash|sh|dash|zsh|ksh)\b",
    re.IGNORECASE,
)

_404_BODY_START_RE = re.compile(
    r"^(?:404\s*:|404\s+not\s+found|<html)",
    re.IGNORECASE,
)


def _command_is_pipe_to_shell(command: str) -> bool:
    """Return True when a command downloads and immediately executes a script."""
    return bool(_PIPE_TO_SHELL_RE.search(str(command or "")))


def _output_looks_like_404_body(output: str) -> bool:
    """Return True when the leading output looks like a 404 or HTML error page."""
    first_line = str(output or "").lstrip().splitlines()[0] if str(output or "").strip() else ""
    return bool(_404_BODY_START_RE.search(first_line))


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
        # An insufficient-verifier rejection must never become the baseline;
        # only genuine execution failures seed the prior-failed verifier.
        failure_mode = str(prior.get("failure_mode") or "").strip().lower()
        if failure_mode != "insufficient_verifier" and not prior.get("insufficient_verifier"):
            return str(prior.get("command") or "").strip()
    return ""


def _canonical_family_path_token(token: str) -> str:
    """Canonicalize a path-like verifier target without erasing its identity.

    Equal-strength verifiers against DIFFERENT path targets (``pytest
    tests/a.py`` vs ``pytest tests/b.py``) must not share a family signature,
    so the path itself is preserved in canonical relative/absolute form
    instead of collapsing to a wildcard.
    """
    raw = str(token or "").strip().strip("'\"")
    raw = raw.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return re.sub(r"/+", "/", raw).rstrip("/")


def _verifier_family_signature(command: str) -> tuple[str, ...]:
    """Return a stable family signature for a verifier command.

    The signature captures the tool/executable and its action words while
    discarding corrected-argument noise: shell wrappers (``cd``/``sudo``/
    ``timeout``), redirections, pipe tails (``| head -50``), and flags with
    their values. Path-like positional tokens keep their canonical identity so
    an equal-strength pass against a different file target cannot clear a
    prior failure, while a passing rerun with corrected args
    (``--node pve1`` -> ``--node pve``) addresses the prior failure instead of
    sidestepping it.
    """
    import shlex

    text = re.sub(r"\s+", " ", str(command or "").strip().lower())
    if not text:
        return ()
    text = re.split(r"\s*(?:&&|\|\||;)\s*", text)[-1].strip()
    text = re.split(r"\s*\|\s*", text)[0].strip()
    text = re.sub(r"(?:\s*\d?>&\d+)+\s*$", "", text)
    text = re.sub(r"\s*\d?>+\s*\S+\s*$", "", text).strip()
    if not text:
        return ()
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    while tokens:
        head = tokens[0]
        if head == "sudo":
            tokens = tokens[1:]
        elif head == "timeout" and len(tokens) >= 2:
            tokens = tokens[2:]
        elif head == "env":
            tokens = tokens[1:]
            while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
                tokens = tokens[1:]
        else:
            break
    if not tokens:
        return ()
    signature = [tokens[0].rsplit("/", 1)[-1]]
    rest = tokens[1:]
    if _FAMILY_INTERPRETER_RE.match(signature[0]) and rest and not rest[0].startswith("-"):
        script = rest[0]
        if "/" in script or re.search(r"\.[a-z0-9]+$", script):
            signature.append(_canonical_family_path_token(script))
            rest = rest[1:]
    consume_next = False
    for token in rest:
        if consume_next:
            consume_next = False
            continue
        if token.startswith("-"):
            # Flags are corrected-argument surface; a bare long flag consumes
            # the following token as its value (``--node pve``).
            if token.startswith("--") and "=" not in token:
                consume_next = True
            continue
        if "/" in token or re.search(r"\.[a-z0-9]{1,5}$", token):
            signature.append(_canonical_family_path_token(token))
        else:
            signature.append(token)
    return tuple(signature)


def _diagnostic_task_exemption(state: Any) -> bool:
    """Pure-diagnostic exemption based on the immutable original task only.

    The working-memory goal is mutable and can drift mid-run (moving the
    goalpost); ``run_brief.original_task`` is the authoritative objective.
    """
    from ..diagnostic_tasks import (
        _DIAGNOSTIC_MARKERS,
        _MUTATION_REMEDIATION_MARKERS,
        _NEGATIVE_VERIFICATION_MARKERS,
    )

    task = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "").strip().casefold()
    if not task:
        return False
    padded = f" {task} "
    if any(marker in padded for marker in _DIAGNOSTIC_MARKERS):
        if any(marker in padded for marker in _MUTATION_REMEDIATION_MARKERS):
            return False
        return True
    return any(marker in task for marker in _NEGATIVE_VERIFICATION_MARKERS)


def _passing_verifier_is_weaker_than_prior_failure(
    state: Any,
    *,
    current_command: str,
    current_kind: str,
) -> bool:
    # Pure diagnostic/observation tasks gather multiple distinct read-only checks.
    # A later diagnostic command should not be considered "weaker" than an earlier
    # failed diagnostic command; each probe contributes independent evidence.
    if _diagnostic_task_exemption(state):
        return False
    current_strength = verifier_strength(current_kind)
    prior_command = _prior_failed_verifier_command(state)
    if not prior_command:
        return False
    prior_kind = verifier_kind_for_command(prior_command)
    prior_strength = verifier_strength(prior_kind)
    # A strictly stronger verifier can overwrite a prior failure, even if the
    # command differs (e.g., a more comprehensive integration test after a
    # narrower unit test failed).
    if current_strength > prior_strength:
        return False
    # If the current verifier is strictly weaker than a prior failed verifier,
    # it cannot overwrite that failure.
    if current_strength < prior_strength:
        return True
    # Even if strengths are equal, a read-only diagnostic (e.g., journalctl, cat)
    # should not overwrite a prior functional status/command verifier that failed.
    if current_strength == prior_strength and current_kind == "diagnostic" and prior_kind != "diagnostic":
        return True
    normalized_current = re.sub(r"\s+", " ", str(current_command or "").strip().lower())
    normalized_prior = re.sub(r"\s+", " ", prior_command.strip().lower())
    if normalized_current == normalized_prior:
        return False
    if current_kind == prior_kind:
        current_signature = _verifier_family_signature(current_command)
        if current_signature and current_signature == _verifier_family_signature(prior_command):
            # Same verifier family addressing the same objective with corrected
            # arguments: the pass answers the prior failure and clears it.
            return False
    return True


def _insufficient_verifier_message(state: Any, *, command: str) -> str:
    prior_command = _prior_failed_verifier_command(state)
    if prior_command:
        return (
            f"Verifier `{command}` is weaker than or equal to the prior failed verifier "
            f"`{prior_command}`; rerun the script/tests that failed."
        )
    return f"Verifier `{command}` is too weak; rerun the script/tests required by the task."


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

    # tee/cat combined with an interactive prompt is never a valid install verifier.
    if re.search(r"\b(?:tee|cat)\b", normalized) and re.search(
        r"\b(?:Choice:\s*\[\d+\]|Are you sure you wish to continue|Should .*?\?\s*\([yYnN]/|Hit \[?Enter\]?|password:)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True, (
            f"This is an install task, but the verifier `{command}` only masks an interactive prompt "
            f"with tee/cat. For install tasks, verify with service status, package status, listening port, or version command."
        )

    kind = verifier_kind_for_command(command)
    if verifier_strength(kind) >= verifier_strength("install_service_status"):
        return False, ""

    return True, (
        f"This is an install task, but the verifier `{command}` is not a post-condition check. "
        f"For install tasks, verify with service status, package status, listening port, or version command."
    )
