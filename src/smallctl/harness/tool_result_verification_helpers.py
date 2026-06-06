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
        return "run_target"
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
    if "timed out" in lowered or "timeout" in lowered or "connection timed out" in lowered:
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
