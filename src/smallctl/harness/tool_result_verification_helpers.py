from __future__ import annotations

import re
from typing import Any

from ..docker_retry_normalization import classify_docker_failure
from .tool_result_verification_constants import (
    _BINARY_PROBE_RE,
    _FOG_RESOURCE_RE,
    _NOT_FOUND_MARKERS,
    _REMOVAL_ABSENCE_PIPE_RE,
    _REMOVAL_ABSENCE_PROBE_RE,
    _REMOVAL_TASK_KEYWORDS,
)

_VERIFIER_KIND_STRENGTH = {
    "syntax_only": 1,
    "lint_typecheck": 2,
    "diagnostic": 2,
    "run_target": 3,
    "test_suite": 4,
    "install_service_status": 4,
    "install_package_status": 4,
    "install_port_listener": 4,
    "install_version_command": 4,
}


def verifier_kind_for_command(command: str) -> str:
    normalized = re.sub(r"\s+", " ", str(command or "").strip().lower())
    if not normalized:
        return ""
    padded = f" {normalized}"
    if "py_compile" in padded:
        return "syntax_only"
    if any(marker in padded for marker in (" pytest", " unittest", " npm test", "npm run test", "pnpm test", "yarn test", "go test", "cargo test", "vitest", "jest")):
        return "test_suite"
    if any(marker in padded for marker in (" ruff", " mypy", " eslint", "cargo clippy")):
        return "lint_typecheck"
    if re.search(r"\bpython(?:3(?:\.\d+)?)?\b.*\.py\b", normalized):
        if re.search(r"\b(nodes list|help|(--?help))\b", normalized):
            return "diagnostic"
        return "run_target"
    # Install-task verifiers: service status, package presence, listening ports, version commands
    if re.search(r"\bsystemctl\s+(?:status|is-active|is-enabled)\b", normalized):
        return "install_service_status"
    if re.search(r"\b(?:dpkg\s+-[lL]|apt\s+list|apt-cache\s+policy|rpm\s+-[qQ]|dpkg-query\s+-[lW])\b", normalized):
        return "install_package_status"
    if re.search(r"\b(?:ss\s+-tlnp|netstat\s+-tlnp|lsof\s+-i\s+:\d+|ss\s+-plnt)\b", normalized):
        return "install_port_listener"
    if re.search(r"\b(?:\S+\s+--version|\S+\s+-version|\S+\s+-v\b|which\s+\S+|whereis\s+\S+)\b", normalized):
        return "install_version_command"
    return "diagnostic"


def verifier_strength(kind: str) -> int:
    return _VERIFIER_KIND_STRENGTH.get(str(kind or "").strip(), 0)


def command_is_binary_probe(command: str) -> bool:
    return _BINARY_PROBE_RE.match(str(command or "").strip()) is not None


def output_confirms_not_found(stdout: str, stderr: str) -> bool:
    """Return True when stdout/stderr text confirms the binary is absent."""
    combined = (str(stdout or "") + " " + str(stderr or "")).lower()
    # A completely empty combined output with exit 127 also qualifies — the
    # shell itself swallowed the "not found" message (happens on some remotes).
    if not combined.strip():
        return True
    return any(marker in combined for marker in _NOT_FOUND_MARKERS)


def classify_execution_failure(text: str) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    docker_class = classify_docker_failure(lowered)
    if docker_class:
        return docker_class
    if "syntaxerror" in lowered or "parseerror" in lowered:
        return "syntax"
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "import"
    if "malformed stanza" in lowered and "sources" in lowered:
        return "apt_sources_malformed"
    if "deb822" in lowered or "invalid line" in lowered and "sources.list" in lowered:
        return "apt_sources_malformed"
    if (
        "timed out" in lowered
        or "timeout" in lowered
        or "connection timed out" in lowered
        or "connection refused" in lowered
        or "no route to host" in lowered
        or "network is unreachable" in lowered
        or "could not resolve hostname" in lowered
    ):
        return "environment"
    if "permission denied" in lowered or "password" in lowered or "sudo" in lowered:
        return "environment"
    if "assert" in lowered or "failed" in lowered or "traceback" in lowered:
        return "test"
    if "no such file" in lowered or "not found" in lowered:
        return "path"
    return "logic"


def looks_like_infinite_loop(command: str, error: str, stdout: str, stderr: str) -> bool:
    cmd = str(command or "").lower()
    err = str(error or "").lower()
    out = str(stdout or "").lower()
    err_out = str(stderr or "").lower()
    if "loop" in err or "recursion" in err or "maximum recursion" in err:
        return True
    if "timeout" in err and ("while" in cmd or "loop" in cmd):
        return True
    if "killed" in err_out and "timeout" in err_out:
        return True
    return False


def heredoc_delimiters_balanced(command: str) -> bool:
    """Check that every heredoc opener (<< or <<-) has a matching delimiter line."""
    text = str(command or "")
    if "<<" not in text:
        return True
    # Find all heredoc delimiters in the command.
    # Match `<< 'EOF'`, `<< "EOF"`, `<<EOF`, `<<-EOF`, etc.
    delimiters = re.findall(r"<<\s*-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", text)
    if not delimiters:
        return True
    lines = text.splitlines()
    for delim in delimiters:
        # The delimiter must appear on its own line (or with leading whitespace for <<-)
        found = False
        for line in lines:
            stripped = line.strip()
            if stripped == delim or stripped == f"'{delim}'" or stripped == f'"{delim}"':
                found = True
                break
        if not found:
            return False
    return True


def command_has_write_or_heredoc_shape(command: str) -> bool:
    lowered = str(command or "").strip().lower()
    if not lowered:
        return False
    if "<<" in lowered:
        return True
    if re.search(r"(?:^|[;&|]\s*)(?:mv|cp|rm|sed\s+-i|perl\s+-i)\b", lowered):
        return True
    if re.search(r"(?:^|[^<])>>?(?:[^&]|$)", lowered):
        return True
    if re.search(r"(?:^|[;&|]\s*)(?:cat|printf|echo)\b[^\n;&|]*(?:>>|>)", lowered):
        return True
    if re.search(r"(?:^|[;&|]\s*)tee(?:\s+-a)?\s+", lowered):
        return True
    return False


def exit_code_matches(exit_code: Any, values: set[int]) -> bool:
    try:
        return int(exit_code) in values
    except (TypeError, ValueError):
        return exit_code is None and 0 in values


def snip_text(value: Any, *, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


# Install-task absence probes: e.g. `dpkg -l | grep -i fog` returning exit 1 with no output.
_INSTALL_ABSENCE_LEFT_RE = re.compile(
    r"\b(?:dpkg\s+-[lL]|apt\s+(?:list|show|search)|rpm\s+-[qQ]|systemctl\s+(?:status|is-active|is-enabled)|service\s+\S+\s+status)\b",
    re.IGNORECASE,
)
_INSTALL_ABSENCE_RIGHT_RE = re.compile(
    r"\|\s*(?:grep|egrep|fgrep)\b",
    re.IGNORECASE,
)


def command_is_install_absence_probe(command: str) -> bool:
    """Return True for pipe commands that probe for package/service absence."""
    cmd = str(command or "").strip()
    if not cmd:
        return False
    if not _INSTALL_ABSENCE_RIGHT_RE.search(cmd):
        return False
    return _INSTALL_ABSENCE_LEFT_RE.search(cmd) is not None


def install_absence_probe_confirmed(
    *,
    command: str,
    exit_code: Any,
    stdout: str,
    stderr: str,
) -> bool:
    """Return True when an install absence probe confirms the resource is absent."""
    if not command_is_install_absence_probe(command):
        return False
    if not exit_code_matches(exit_code, {0, 1}):
        return False
    out = str(stdout or "").strip()
    err = str(stderr or "").strip()
    combined = "\n".join(part for part in (out, err) if part).strip()
    # Empty/whitespace-only output after grep with exit 1 means no matches.
    if not combined:
        return True
    # If output is very short and looks like a header/no-data marker, treat as absence.
    if len(combined) <= 80 and not any(marker in combined.lower() for marker in ("ii ", "active", "running", "installed")):
        return True
    return False
