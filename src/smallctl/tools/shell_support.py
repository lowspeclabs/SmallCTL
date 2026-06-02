from __future__ import annotations

import shlex
import re
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail
from .fs_sessions import _same_target_path, _write_session_can_finalize
from .fs_write_session_policy import _write_session_resume_metadata


_ARGPARSE_REQUIRED_ARGS_PATTERN = re.compile(
    r"(?:error:\s*)?the following arguments are required:\s*(.+)",
    re.IGNORECASE,
)
_YES_PIPE_PATTERN = re.compile(
    r"(?:^|[;&(]\s*)yes(?:\s+[^|;&]+)?\s*\|\s*(?P<target>[^;&|]+)",
    re.IGNORECASE | re.DOTALL,
)
_SINGLE_ANSWER_PIPE_PATTERN = re.compile(
    r"(?:^|[;&(]\s*)echo\s+(?P<answer>(?:-[A-Za-z]+\s+)?(?:['\"]?\s*[YyNn](?:[Ee][Ss]|[Oo])?\s*['\"]?))\s*\|\s*(?P<target>[^;&|]+)",
    re.IGNORECASE | re.DOTALL,
)
_INVALID_INPUT_MARKERS = (
    "invalid input, please try again",
    "answer not recognized",
)
_REMOTE_INSTALLER_PREFLIGHT_KEY = "_remote_installer_preflight"
_DEB822_FIELDS = ("Types:", "URIs:", "Suites:", "Components:")
_SHELL_CONTROL_TOKENS = {"&&", ";", "||", "|"}
_DISPOSABLE_PATH_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
}
_DISPOSABLE_PATH_SUFFIXES = {".pyc", ".pyo", ".tmp"}
_SOURCE_OR_TEST_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".cs",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".sh",
    ".sql",
}
_SOURCE_OR_TEST_DIR_NAMES = {
    "src",
    "lib",
    "app",
    "tests",
    "test",
    "scripts",
    "components",
    "pages",
    "packages",
}


def _apt_deb822_preflight_guard(command: str, *, tool_name: str) -> dict[str, Any] | None:
    raw = str(command or "").strip()
    lowered = raw.lower()
    if not re.search(r"\b(?:apt|apt-get)\b", lowered):
        return None
    if not re.search(r"\b(?:install|update|upgrade|dist-upgrade|full-upgrade|autoremove|purge|remove)\b", lowered):
        return None
    if all(field.lower() in lowered for field in _DEB822_FIELDS) and "debian.sources" in lowered:
        return None
    validator = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path('/etc/apt/sources.list.d/debian.sources')\n"
        "s = p.read_text()\n"
        "missing = [k for k in ('Types:', 'URIs:', 'Suites:', 'Components:') if k not in s]\n"
        "assert not missing, 'debian.sources missing deb822 fields: ' + ', '.join(missing)\n"
        "print('deb822 OK')\n"
        "PY"
    )
    return fail(
        f"`{tool_name}` blocked apt package operation until `/etc/apt/sources.list.d/debian.sources` is validated as deb822.",
        metadata={
            "reason": "apt_deb822_preflight_required",
            "command": raw,
            "required_fields": list(_DEB822_FIELDS),
            "next_required_action": {
                "tool_name": tool_name,
                "required_arguments": {"command": validator},
                "notes": [
                    "Run the validator first, or combine this exact validation before apt with &&.",
                    "For small sources files, prefer full-file ssh_file_write over boundary-anchored replacement.",
                ],
            },
        },
    )


def compose_remote_command(*parts: str, via_script: bool = False, script_path: str = "/tmp/smallctl_probe.sh") -> str:
    """Compose a remote SSH command with safe quoting. For long or complex commands, upload a script."""
    if via_script or len(shlex.join(parts)) > 400:
        script_body = "\n".join(parts)
        return f"cat > {script_path} << 'EOF'\n{script_body}\nEOF\nbash {script_path}"
    return " ".join(shlex.quote(p) for p in parts)


def _safe_resolve_path(path: str | Path, *, cwd: str | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd or Path.cwd()) / candidate
    try:
        return candidate.resolve()
    except Exception:
        return candidate.absolute()


def _is_within_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _extract_shell_delete_targets(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        return []
    try:
        parts = shlex.split(raw)
    except ValueError:
        return []
    targets: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if token in {"rm", "rmdir"}:
            index += 1
            while index < len(parts):
                current = parts[index]
                if current in _SHELL_CONTROL_TOKENS:
                    break
                if current.startswith("-"):
                    index += 1
                    continue
                targets.append(current)
                index += 1
            continue
        if token == "find":
            segment: list[str] = []
            index += 1
            while index < len(parts) and parts[index] not in _SHELL_CONTROL_TOKENS:
                segment.append(parts[index])
                index += 1
            if "-delete" in segment and segment:
                targets.append(segment[0])
            continue
        if token == "git" and index + 1 < len(parts) and parts[index + 1] == "clean":
            targets.append(".")
            index += 2
            continue
        index += 1
    return targets


def _is_disposable_delete_target(path: Path) -> bool:
    if path.name in _DISPOSABLE_PATH_NAMES:
        return True
    if path.suffix in _DISPOSABLE_PATH_SUFFIXES:
        return True
    return any(part in _DISPOSABLE_PATH_NAMES for part in path.parts)


def _looks_like_source_or_test_artifact(path: Path) -> bool:
    if path.suffix in _SOURCE_OR_TEST_SUFFIXES:
        return True
    return any(part in _SOURCE_OR_TEST_DIR_NAMES for part in path.parts)


def _protected_working_set_paths(state: LoopState) -> set[str]:
    protected: set[str] = set()
    cwd = str(getattr(state, "cwd", "") or Path.cwd())

    def add_path(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            protected.add(str(_safe_resolve_path(text, cwd=cwd)))

    challenge_progress = getattr(state, "challenge_progress", None)
    if challenge_progress is not None:
        for path in getattr(challenge_progress, "last_code_change_paths", []) or []:
            add_path(path)
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        for key in ("protected_working_set", "_protected_working_set"):
            raw = scratchpad.get(key)
            if isinstance(raw, dict):
                for path in raw.keys():
                    add_path(path)
            elif isinstance(raw, (list, tuple, set)):
                for path in raw:
                    add_path(path)
        for key in ("generated_paths", "_generated_paths", "_task_target_paths"):
            raw = scratchpad.get(key)
            if isinstance(raw, (list, tuple, set)):
                for path in raw:
                    add_path(path)
    return protected


def _target_contains_protected_path(target: Path, protected_paths: set[str]) -> bool:
    for protected in protected_paths:
        protected_path = Path(protected)
        if protected_path == target or _is_within_path(protected_path, target):
            return True
    return False


def _explicit_delete_requested(state: LoopState, target: Path) -> bool:
    task_text_parts = [
        str(getattr(getattr(state, "run_brief", None), "original_task", "") or ""),
        str(getattr(getattr(state, "working_memory", None), "current_goal", "") or ""),
    ]
    text = "\n".join(task_text_parts).lower()
    if not any(word in text for word in ("delete", "remove", "rm -rf", "clean up", "cleanup")):
        return False
    target_name = target.name.lower()
    target_text = str(target).lower()
    return target_name in text or target_text in text


def _shell_workspace_destructive_delete_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    targets = _extract_shell_delete_targets(command)
    if not targets:
        return None
    cwd = str(getattr(state, "cwd", "") or Path.cwd())
    workspace = _safe_resolve_path(cwd)
    protected_paths = _protected_working_set_paths(state)
    blocked: list[dict[str, Any]] = []
    allowed: list[str] = []
    for raw_target in targets:
        resolved = _safe_resolve_path(raw_target, cwd=cwd)
        if _is_disposable_delete_target(resolved):
            allowed.append(str(resolved))
            continue
        if not _is_within_path(resolved, workspace):
            continue
        if _explicit_delete_requested(state, resolved):
            allowed.append(str(resolved))
            continue
        reasons: list[str] = []
        if str(resolved) in protected_paths:
            reasons.append("protected_working_set")
        if _target_contains_protected_path(resolved, protected_paths):
            reasons.append("contains_protected_working_set_path")
        if _looks_like_source_or_test_artifact(resolved):
            reasons.append("source_or_test_artifact")
        # Unknown workspace directory deletes are a common destructive reset pattern.
        if not reasons and ("rm -r" in command or "rm -rf" in command or "rm -fr" in command or "rmdir" in command):
            reasons.append("unknown_workspace_delete")
        if reasons:
            blocked.append(
                {
                    "raw_target": raw_target,
                    "resolved_target": str(resolved),
                    "reasons": reasons,
                }
            )
    if not blocked:
        return None
    return fail(
        "Shell command blocked: destructive delete targets implementation artifacts. "
        "Deletion is not an acceptable repair/reset operation for code changes unless "
        "the user explicitly requested that exact deletion.",
        metadata={
            "reason": "workspace_destructive_delete_blocked",
            "error_kind": "workspace_destructive_delete_blocked",
            "command": command,
            "blocked_targets": blocked,
            "allowed_targets": allowed,
            "next_required_tool": {
                "tool_name": "file_read",
                "required_fields": ["path"],
                "notes": [
                    "Read the target and repair it with file_patch.",
                    "For generated code you own, use file_write with replace_strategy='overwrite'.",
                    "If deletion is intentional, ask_human with the exact path and reason.",
                ],
            },
        },
    )


def classify_shell_outcome(command: str, returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    """Classify expected miss commands as empty_result when stdout/stderr semantics are clear."""
    cmd = str(command or "").strip()
    if returncode == 0:
        return {"status": "success", "kind": "ok"}
    absence_patterns = [
        (r'\bfind\s+.*-name\s+', "absence_probe"),
        (r'\bwhich\s+', "absence_probe"),
        (r'\bgrep\s+-[qLl]', "absence_probe"),
        (r'\btest\s+-[efdx]', "absence_probe"),
    ]
    for pat, kind in absence_patterns:
        if re.search(pat, cmd) and not stdout.strip():
            return {"status": "success", "kind": "empty_result", "exit_code": returncode}
    return {"status": "failure", "kind": "error", "exit_code": returncode}


def _interactive_installer_yes_pipe_guard(
    command: str,
    *,
    tool_name: str,
) -> dict[str, Any] | None:
    raw = str(command or "").strip()
    if not raw:
        return None

    for match in _YES_PIPE_PATTERN.finditer(raw):
        target = " ".join(str(match.group("target") or "").split())
        target_lower = target.lower()
        if not _looks_like_interactive_installer_target(target_lower):
            continue
        return fail(
            f"`{tool_name}` blocked `yes |` automation for an interactive installer: `{raw}`. "
            "Use the installer's non-interactive mode when available, such as `--autoaccept` or `-y`; "
            "otherwise use a config/preseed file or an explicit `printf` script with known answers.",
            metadata={
                "command": raw,
                "reason": "unsafe_yes_pipe_interactive_installer",
                "detected_target": target,
                "next_required_action": {
                    "strategy": "use_structured_noninteractive_install",
                    "preferred_inputs": [
                        "--autoaccept or -y if the installer documents it",
                        "a preseed/config file such as .fogsettings",
                        "an explicit printf script with known prompt answers",
                    ],
                },
            },
        )
    for match in _SINGLE_ANSWER_PIPE_PATTERN.finditer(raw):
        target = " ".join(str(match.group("target") or "").split())
        target_lower = target.lower()
        if not _looks_like_interactive_installer_target(target_lower):
            continue
        return fail(
            f"`{tool_name}` blocked single-answer `echo |` automation for an interactive installer: `{raw}`. "
            "A lone Y/N answer is brittle for multi-prompt installers. Use the installer's non-interactive mode "
            "when available, such as `--autoaccept` or `-y`; otherwise use a config/preseed file or an explicit "
            "`printf` script with the complete known answer stream.",
            metadata={
                "command": raw,
                "reason": "unsafe_single_answer_pipe_interactive_installer",
                "detected_target": target,
                "detected_answer": " ".join(str(match.group("answer") or "").split()),
                "next_required_action": {
                    "strategy": "use_structured_noninteractive_install",
                    "preferred_inputs": [
                        "--autoaccept or -y if the installer documents it",
                        "a preseed/config file such as .fogsettings",
                        "an explicit printf script with the complete known prompt answers",
                    ],
                },
            },
        )
    return None


def _installer_command_suggested_timeout(command: str, timeout_sec: int) -> int:
    raw = str(command or "")
    try:
        current_timeout = int(timeout_sec)
    except (TypeError, ValueError):
        current_timeout = 60
    words = _split_shell_words(raw.lower())
    if any(_looks_like_interactive_installer_word(word) for word in words) and current_timeout <= 60:
        return 600
    return max(1, current_timeout)


def _remote_installer_preflight_guard(
    command: str,
    *,
    host: str,
    user: str | None,
    state: LoopState | None,
) -> dict[str, Any] | None:
    raw = str(command or "").strip()
    if state is None or not raw or not _looks_like_remote_installer_mutation(raw):
        return None
    cwd, script_path = _remote_installer_cwd_and_script(raw)
    key = "|".join([str(host or "").strip().lower(), str(user or "").strip().lower(), cwd])
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    preflights = scratchpad.get(_REMOTE_INSTALLER_PREFLIGHT_KEY)
    if not isinstance(preflights, dict):
        preflights = {}
        scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = preflights
    entry = preflights.get(key)
    if isinstance(entry, dict):
        status = str(entry.get("status") or "").strip()
        created = int(entry.get("created_at_step", 0) or 0)
        if status == "clean" and int(getattr(state, "step_count", 0) or 0) - created <= 8:
            return None
        if status in {"missing_critical_files", "corrupt"}:
            return fail(
                "Remote installer preflight found missing or corrupted critical installer files. "
                "Repair the environment with a fresh clone or clean reset before running the installer.",
                metadata={
                    "reason": "remote_installer_preflight_failed",
                    "host": host,
                    "user": user,
                    "command": raw,
                    "cwd": cwd,
                    "script_path": script_path,
                    "preflight": entry,
                    "next_required_action": "fresh clone or clean reset; do not patch individual installer files",
                },
            )
    checks = _remote_installer_preflight_checks(cwd=cwd, script_path=script_path)
    preflights[key] = {
        "host": host,
        "user": user or "",
        "cwd": cwd,
        "script_path": script_path,
        "checks": checks,
        "created_at_step": int(getattr(state, "step_count", 0) or 0),
        "status": "required",
    }
    checks_text = " && ".join(checks)
    return fail(
        "Remote installer preflight required before running this high-risk installer mutation. "
        "Run narrow repo/integrity checks first, then retry the installer after the preflight is clean. "
        f"Required checks: {checks_text}",
        metadata={
            "reason": "remote_installer_preflight_required",
            "host": host,
            "user": user,
            "command": raw,
            "cwd": cwd,
            "script_path": script_path,
            "required_checks": checks,
            "next_required_action": checks_text,
        },
    )


def _looks_like_remote_installer_mutation(command: str) -> bool:
    for segment in _simple_shell_command_segments(command):
        try:
            words = _split_shell_words(segment)
        except ValueError:
            words = []
        if not words:
            continue
        executable = Path(words[0]).name.lower()
        if executable == "make" and len(words) > 1 and words[1].lower() == "install":
            return True
        if executable in {"bash", "sh", "dash", "zsh", "ksh"} and len(words) > 1:
            script_name = Path(words[1]).name.lower()
            if script_name == "installfog.sh" or (
                script_name.endswith(".sh")
                and ("install" in script_name or "bootstrap" in script_name)
            ):
                return True
            continue
        if executable == "installfog.sh":
            return True
        if executable.endswith(".sh") and ("install" in executable or "bootstrap" in executable):
            return True
    return False


def _simple_shell_command_segments(command: str) -> list[str]:
    raw = str(command or "")
    if not raw.strip():
        return []
    # Good enough for safety classification: split on common shell command
    # separators while preserving ordinary arguments like installer paths.
    parts = re.split(r"\s*(?:&&|\|\||[;|])\s*", raw)
    return [part.strip() for part in parts if part.strip()]


def _remote_installer_cwd_and_script(command: str) -> tuple[str, str]:
    raw = str(command or "").strip()
    cwd = ""
    cd_match = re.search(r"(?:^|[;&]\s*)cd\s+([^;&|]+?)\s*(?:&&|\|\||;|\||$)", raw)
    if cd_match:
        cwd = str(cd_match.group(1) or "").strip().strip("'\"")

    script = ""
    for segment in _simple_shell_command_segments(raw):
        try:
            words = _split_shell_words(segment)
        except ValueError:
            words = []
        if not words:
            continue
        executable = Path(words[0]).name.lower()
        if executable in {"bash", "sh", "dash", "zsh", "ksh"} and len(words) > 1:
            candidate = words[1]
            candidate_name = Path(candidate).name.lower()
            if candidate_name.endswith(".sh") and ("install" in candidate_name or "bootstrap" in candidate_name):
                script = candidate
                break
        elif executable.endswith(".sh") and ("install" in executable or "bootstrap" in executable):
            script = words[0]
            break
        elif executable == "make" and len(words) > 1 and words[1].lower() == "install":
            script = "make install"
            break

    if script and script != "make install":
        if script.startswith("./") and cwd:
            script = cwd.rstrip("/") + "/" + script[2:]
        elif script.startswith("../") and cwd:
            script = cwd.rstrip("/") + "/" + script
        elif not script.startswith("/") and cwd:
            script = cwd.rstrip("/") + "/" + script
    return cwd, script


def _remote_installer_preflight_checks(*, cwd: str, script_path: str) -> list[str]:
    prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
    checks = ["pwd"]
    if cwd:
        checks.append(f"cd {shlex.quote(cwd)} && git rev-parse --show-toplevel")
        checks.append(f"cd {shlex.quote(cwd)} && git status --short")
    if script_path and script_path != "make install":
        checks.append(f"test -x {shlex.quote(script_path)}")
    elif cwd:
        checks.append(prefix + "test -f Makefile")
    return checks


def _mark_remote_installer_preflight_clean(
    state: LoopState | None,
    *,
    host: str,
    user: str | None,
    cwd: str,
) -> None:
    if state is None:
        return
    key = "|".join([str(host or "").strip().lower(), str(user or "").strip().lower(), cwd])
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    preflights = scratchpad.get(_REMOTE_INSTALLER_PREFLIGHT_KEY)
    if not isinstance(preflights, dict):
        preflights = {}
        scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = preflights
    entry = preflights.get(key)
    if not isinstance(entry, dict):
        entry = {}
        preflights[key] = entry
    entry["status"] = "clean"
    entry["host"] = host
    entry["user"] = user or ""
    entry["cwd"] = cwd
    entry["created_at_step"] = int(getattr(state, "step_count", 0) or 0)


def _expose_interactive_session_tools(state: LoopState | None) -> None:
    if state is None:
        return
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    scratchpad["_expose_interactive_session_tools"] = True


def _looks_like_interactive_installer_target(target_lower: str) -> bool:
    if not target_lower:
        return False
    words = _split_shell_words(target_lower)
    if not words:
        return False
    executable = Path(words[0]).name
    if executable in {"bash", "sh", "dash", "zsh", "ksh", "env"}:
        return any(_looks_like_interactive_installer_word(word) for word in words[1:])
    return any(_looks_like_interactive_installer_word(word) for word in words)


def _looks_like_interactive_installer_word(word: str) -> bool:
    name = Path(str(word or "")).name.lower()
    return bool(name) and (name.endswith(".sh") or "install" in name)


class InvalidInputLoopDetector:
    def __init__(self, *, threshold: int = 3, max_tail_chars: int = 4096) -> None:
        self.threshold = max(1, int(threshold))
        self.max_tail_chars = max(512, int(max_tail_chars))
        self.count = 0
        self.output_tail = ""

    def observe(self, chunk: str) -> dict[str, Any] | None:
        text = str(chunk or "")
        if not text:
            return None
        self.output_tail = (self.output_tail + text)[-self.max_tail_chars :]
        lowered = text.lower()
        self.count += sum(lowered.count(marker) for marker in _INVALID_INPUT_MARKERS)
        if self.count < self.threshold:
            return None
        return {
            "reason": "interactive_invalid_input_loop",
            "invalid_input_count": self.count,
            "output_tail": self.output_tail,
            "diagnosis": (
                "The command is repeatedly rejecting automated input. Stop this run and replace "
                "blanket input piping with documented non-interactive flags, a config/preseed file, "
                "or an explicit prompt answer script."
            ),
        }


def _extract_missing_argparse_arguments(error_text: str) -> list[str]:
    match = _ARGPARSE_REQUIRED_ARGS_PATTERN.search(str(error_text or ""))
    if not match:
        return []

    missing = match.group(1).strip()
    if not missing:
        return []

    missing = missing.replace(" and ", ", ")
    values = [part.strip(" .`'\"") for part in missing.split(",")]
    return [value for value in values if value]


def _build_argparse_missing_args_question(command: str, missing_args: list[str]) -> str:
    missing_text = ", ".join(missing_args) if missing_args else "required arguments"
    return (
        f"The command `{command}` is missing required arguments: {missing_text}. "
        "What values should I use?"
    )


def _detect_unsupported_shell_syntax(command: str) -> str | None:
    if "<<<" in command:
        return (
            "Command uses Bash-only here-string redirection (`<<<`), but smallctl runs shell "
            "commands through /bin/sh on Unix. Rewrite it with POSIX syntax (for example, "
            "use `printf` piped into the command) or wrap the whole command in `bash -lc`."
        )
    return None


_DETACHED_COMMAND_MARKERS = (
    "nohup ",
    "setsid ",
    "disown",
    "daemonize ",
    "systemd-run ",
    "tmux ",
    "screen ",
)
_FOLLOW_FLAGS = {"-f", "--follow", "--tail=follow"}
_INSPECTION_FLAGS = {"-h", "--help", "-v", "--version", "version"}
_SERVICE_MANAGER_COMMANDS = {"systemctl", "service", "supervisorctl", "rc-service", "launchctl"}
_PACKAGE_RUNNERS = {"npm", "pnpm", "yarn", "bun"}
_FOREGROUND_SUBCOMMANDS = {"dev", "develop", "serve", "server", "start", "watch"}
_FOREGROUND_BINARIES = {
    "air",
    "caddy",
    "flask",
    "gunicorn",
    "http-server",
    "nodemon",
    "rails",
    "redis-server",
    "uvicorn",
    "vite",
    "webpack-dev-server",
}


def _foreground_command_guard(
    command: str,
    *,
    tool_name: str,
    allow_background_parameter: bool = False,
) -> dict[str, Any] | None:
    reason = _likely_long_running_foreground_reason(command)
    if reason is None:
        return None

    background_text = " or `background=True`" if allow_background_parameter else ""
    return fail(
        f"`{tool_name}` blocked a likely long-running foreground command: `{command}`. "
        "Start services with a service manager, a detached/background command, or a bounded command "
        f"such as `timeout 20s ...`{background_text}, then verify with a separate health check.",
        metadata={
            "command": command,
            "reason": "long_running_foreground_command",
            "foreground_detection": reason,
            "next_required_action": {
                "strategy": "detach_or_bound_then_verify",
                "notes": [
                    "Use a service manager for daemons when available.",
                    "Use a detached/background launch when the command is expected to keep running.",
                    "Use a bounded `timeout` wrapper only when sampling foreground output is intentional.",
                    "Run a separate verification command after launch.",
                ],
            },
        },
    )


def _likely_long_running_foreground_reason(command: str) -> str | None:
    raw = str(command or "").strip()
    if not raw:
        return None
    if _has_detached_or_bounded_marker(raw):
        return None

    commands = [raw]
    words = _split_shell_words(raw)
    if len(words) >= 3 and words[0] in {"bash", "sh", "/bin/bash", "/bin/sh"} and words[1] in {"-c", "-lc"}:
        commands.append(words[2])

    for candidate in commands:
        for segment in _split_shell_command_segments(candidate):
            reason = _likely_long_running_simple_command_reason(segment)
            if reason is not None:
                return reason
    return None


def _has_detached_or_bounded_marker(command: str) -> bool:
    raw = str(command or "").strip()
    if not raw:
        return False
    if raw.endswith("&") and not raw.endswith("&&"):
        return True
    lowered = raw.lower()
    if lowered.startswith("timeout ") or lowered.startswith("/usr/bin/timeout "):
        return True
    return any(marker in lowered for marker in _DETACHED_COMMAND_MARKERS)


def _likely_long_running_simple_command_reason(command: str) -> str | None:
    words = _split_shell_words(command)
    if not words:
        return None
    words = _strip_environment_and_wrappers(words)
    if not words:
        return None

    executable = Path(words[0]).name.lower()
    args = [word.lower() for word in words[1:]]
    if any(arg in _INSPECTION_FLAGS for arg in args):
        return None
    if executable in _SERVICE_MANAGER_COMMANDS:
        return None
    if executable == "docker":
        if len(args) >= 2 and args[0] == "logs" and any(arg in _FOLLOW_FLAGS for arg in args[1:]):
            return "follow_output"
        if "run" in args and "-d" in args:
            return None
        if len(args) >= 3 and args[0] == "compose" and args[1] == "up" and "-d" in args[2:]:
            return None
        if len(args) >= 2 and args[0] == "compose" and args[1] == "logs" and any(arg in _FOLLOW_FLAGS for arg in args[2:]):
            return "follow_output"
    if executable == "kubectl" and len(args) >= 1 and args[0] == "logs" and any(arg in _FOLLOW_FLAGS for arg in args[1:]):
        return "follow_output"
    if executable in {"tail", "journalctl"} and any(arg in _FOLLOW_FLAGS for arg in args):
        return "follow_output"
    if executable in _PACKAGE_RUNNERS:
        if args[:2] == ["run", "dev"] or (args and args[0] in _FOREGROUND_SUBCOMMANDS):
            return "package_runner_foreground"
        if executable in {"yarn", "pnpm", "bun"} and args and args[0] == "run" and len(args) > 1 and args[1] in _FOREGROUND_SUBCOMMANDS:
            return "package_runner_foreground"
    if executable in {"python", "python3"} and len(args) >= 2 and args[0] == "-m" and args[1] in {"http.server", "uvicorn"}:
        return "python_module_server"
    if executable in _FOREGROUND_BINARIES:
        if executable == "caddy" and args and args[0] == "start":
            return None
        if executable == "caddy" and args and args[0] == "reload":
            return None
        if args and args[0] in {"run", *list(_FOREGROUND_SUBCOMMANDS)}:
            return "service_foreground_subcommand"
        if executable in {"uvicorn", "gunicorn", "redis-server", "http-server", "vite", "webpack-dev-server", "nodemon"}:
            return "service_foreground_binary"
    return None


def _split_shell_command_segments(command: str) -> list[str]:
    words = _split_shell_words(command)
    if not words:
        return []
    segments: list[list[str]] = [[]]
    for word in words:
        if word in {"&&", "||", ";", "|"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(word)
    return [" ".join(shlex.quote(part) for part in segment) for segment in segments if segment]


def _strip_environment_and_wrappers(words: list[str]) -> list[str]:
    stripped = list(words)
    while stripped and "=" in stripped[0] and not stripped[0].startswith("="):
        key, _value = stripped[0].split("=", 1)
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            break
        stripped.pop(0)
    while stripped and Path(stripped[0]).name.lower() in {"sudo", "doas", "env", "command"}:
        stripped.pop(0)
        while stripped and stripped[0].startswith("-"):
            stripped.pop(0)
    return stripped


def _shell_workspace_relative_hint(command: str, cwd: str | None = None) -> str | None:
    raw_command = str(command or "")
    match = re.search(r"(?<![\w/])(/temp(?:/[^\s\"'`]+)*)", raw_command)
    if match is None:
        return None

    suspicious_path = match.group(1)
    trimmed = suspicious_path.lstrip("/")
    if not trimmed:
        return None

    base = Path(cwd) if cwd else Path.cwd()
    workspace_candidate = (base / Path(trimmed)).resolve()
    if not (workspace_candidate.exists() or workspace_candidate.parent.exists()):
        return None

    return (
        f"That command used the root-level `{suspicious_path}` path. "
        f"If you meant the workspace copy, retry with `{('./' + trimmed)}` instead."
    )


def _shell_write_session_target_path_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    session = getattr(state, "write_session", None)
    if session is None:
        return None

    status = str(getattr(session, "status", "") or "").strip().lower()
    if status == "complete":
        return None

    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    if not target_path:
        return None

    has_unpromoted_changes = bool(
        getattr(session, "write_sections_completed", [])
        or str(getattr(session, "write_last_staged_hash", "") or "").strip()
        or bool(getattr(session, "write_pending_finalize", False))
    )
    if not has_unpromoted_changes:
        return None

    if not _command_targets_path(command, target_path=target_path, cwd=state.cwd):
        return None

    from ..harness.tool_visibility import _has_finalizable_write_session
    can_finalize = _has_finalizable_write_session(state)
    if can_finalize:
        next_required_tool = {
            "tool_name": "finalize_write_session",
            "required_fields": [],
            "required_arguments": {},
            "optional_fields": [],
            "notes": [
                "Promote the staged file to the target path before running shell verification on the target.",
            ],
        }
        finalize_hint = "or finalize it with `finalize_write_session` "
    else:
        next_required_tool = _write_session_resume_metadata(session, path=target_path)
        finalize_hint = ""

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    return fail(
        f"Write Session `{session_id}` for `{target_path}` is still {status or 'open'} and staged-only. "
        f"The command targets the unpromoted target path. Resume the write session with `file_write` "
        f"{finalize_hint}before running shell checks on `{target_path}`.",
        metadata={
            "command": command,
            "reason": "write_session_unpromoted_target_path",
            "write_session_id": session_id,
            "write_session_status": status or "open",
            "write_session_mode": str(getattr(session, "write_session_mode", "") or "").strip(),
            "target_path": target_path,
            "staging_path": staging_path,
            "next_section_name": str(getattr(session, "write_next_section", "") or "").strip(),
            "next_required_tool": next_required_tool,
        },
    )


def _shell_write_session_artifact_delete_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    session = getattr(state, "write_session", None)
    if session is None:
        return None

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if not session_id:
        return None

    destructive_targets = _write_session_delete_targets(command)
    if not destructive_targets:
        return None

    cwd = str(getattr(state, "cwd", "") or "")
    protected_paths = _protected_write_session_paths(session, cwd=cwd)
    matched_targets = [
        target
        for target in destructive_targets
        if _targets_write_session_artifact(
            target,
            session_id=session_id,
            protected_paths=protected_paths,
            cwd=cwd,
        )
    ]
    if not matched_targets:
        return None

    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    return fail(
        f"Shell command blocked: `{command}` would delete Write Session `{session_id}` recovery artifacts. "
        "Staging, original, and attempt snapshots are recovery state, not disposable scratch files. "
        "Finalize or resume the write session, or ask explicitly for cleanup once recovery is no longer needed.",
        metadata={
            "command": command,
            "reason": "write_session_artifact_delete_blocked",
            "error_kind": "write_session_artifact_delete_blocked",
            "write_session_id": session_id,
            "write_session_status": str(getattr(session, "status", "") or "").strip(),
            "target_path": target_path,
            "staging_path": staging_path,
            "matched_targets": matched_targets,
            "next_required_tool": {
                "tool_name": "finalize_write_session" if _write_session_can_finalize(session) else "file_write",
                "required_fields": [] if _write_session_can_finalize(session) else ["path", "content", "write_session_id", "section_name"],
                "required_arguments": {} if _write_session_can_finalize(session) else _write_session_resume_metadata(session, path=target_path).get("required_arguments", {}),
                "optional_fields": [] if _write_session_can_finalize(session) else ["next_section_name"],
                "notes": [
                    "Do not delete .smallctl/write_sessions artifacts while recovery may still need them.",
                    "If cleanup is intentional, ask the user for explicit approval after completion.",
                ],
            },
        },
    )


def _shell_execution_authoring_guard(state: LoopState, command: str) -> dict[str, Any] | None:
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    if plan is not None and not getattr(plan, "approved", False):
        return fail(
            "Shell execution is blocked until the spec contract is approved.",
            metadata={
                "command": command,
                "reason": "spec_not_approved",
                "plan_id": getattr(plan, "plan_id", ""),
            },
        )

    if state.contract_phase() == "author":
        if not state.files_changed_this_cycle:
            return fail(
                "Shell execution is blocked until the authoring contract has produced a target artifact.",
                metadata={
                    "command": command,
                    "reason": "authoring_target_missing",
                    "contract_phase": state.contract_phase(),
                    "files_changed_this_cycle": state.files_changed_this_cycle,
                },
            )
    return None


def _write_session_delete_targets(command: str) -> list[str]:
    commands = [str(command or "")]
    tokens = _split_shell_words(command)
    if len(tokens) >= 3 and tokens[0] in {"bash", "sh", "/bin/bash", "/bin/sh"} and tokens[1] in {"-c", "-lc"}:
        commands.append(tokens[2])

    targets: list[str] = []
    for raw_command in commands:
        words = _split_shell_words(raw_command)
        for index, word in enumerate(words):
            if word not in {"rm", "/bin/rm"}:
                continue
            for candidate in words[index + 1 :]:
                if candidate in {"&&", "||", ";", "|"}:
                    break
                if candidate.startswith("-"):
                    continue
                for path_candidate in _token_path_candidates(candidate):
                    targets.append(path_candidate)
    return targets


def _split_shell_words(command: str) -> list[str]:
    try:
        return shlex.split(str(command or ""), posix=True)
    except ValueError:
        return str(command or "").split()


def _protected_write_session_paths(session: Any, *, cwd: str | None = None) -> set[str]:
    protected: set[str] = set()
    for raw_path in (
        getattr(session, "write_staging_path", ""),
        getattr(session, "write_original_snapshot_path", ""),
        getattr(session, "write_last_attempt_snapshot_path", ""),
    ):
        path_text = str(raw_path or "").strip()
        if not path_text:
            continue
        protected.add(path_text)
        try:
            base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
            resolved = (Path(path_text) if Path(path_text).is_absolute() else base / path_text).resolve()
            protected.add(str(resolved))
        except Exception:
            pass
    return protected


def _targets_write_session_artifact(
    target: str,
    *,
    session_id: str,
    protected_paths: set[str],
    cwd: str | None = None,
) -> bool:
    raw = str(target or "").strip().strip("'\"`")
    if not raw:
        return False

    normalized = raw.replace("\\", "/")
    if ".smallctl/write_sessions/" in normalized and session_id in normalized:
        return True
    if normalized.endswith(".smallctl/write_sessions") or normalized.endswith(".smallctl/write_sessions/"):
        return True

    if any("*" in part or "?" in part or "[" in part for part in (raw, normalized)):
        if ".smallctl/write_sessions/" in normalized and session_id in normalized:
            return True

    for protected in protected_paths:
        if not protected:
            continue
        if _same_target_path(raw, protected, cwd):
            return True
        if any(mark in raw for mark in ("*", "?", "[")):
            prefix = raw.split("*", 1)[0].split("?", 1)[0].split("[", 1)[0]
            if prefix and str(protected).startswith(prefix):
                return True
    return False


def _command_targets_path(command: str, *, target_path: str, cwd: str | None = None) -> bool:
    raw_command = str(command or "")
    target = str(target_path or "").strip()
    if not raw_command.strip() or not target:
        return False

    aliases = _target_path_aliases(target, cwd=cwd)
    if not aliases:
        return False

    for alias in aliases:
        if _path_alias_mentioned(raw_command, alias):
            return True

    tokens: list[str]
    try:
        tokens = shlex.split(raw_command, posix=True)
    except ValueError:
        tokens = raw_command.split()

    for token in tokens:
        for candidate in _token_path_candidates(token):
            if any(candidate == alias for alias in aliases):
                return True
            if _same_target_path(candidate, target, cwd):
                return True
    return False


def _target_path_aliases(target_path: str, *, cwd: str | None = None) -> list[str]:
    aliases: set[str] = set()
    raw = str(target_path or "").strip()
    if not raw:
        return []

    aliases.add(raw)
    if raw.startswith("./"):
        aliases.add(raw[2:])
    elif not raw.startswith("/"):
        aliases.add(f"./{raw}")

    try:
        base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
        resolved = (Path(raw) if Path(raw).is_absolute() else (base / raw)).resolve()
        aliases.add(str(resolved))
        try:
            rel = resolved.relative_to(base)
        except Exception:
            rel = None
        if rel is not None:
            rel_str = str(rel)
            if rel_str:
                aliases.add(rel_str)
                aliases.add(f"./{rel_str}")
    except Exception:
        pass

    return [alias for alias in aliases if alias]


def _path_alias_mentioned(command: str, alias: str) -> bool:
    if not alias:
        return False
    pattern = rf"(?<![A-Za-z0-9_./-]){re.escape(alias)}(?![A-Za-z0-9_./-])"
    return bool(re.search(pattern, command))


def _token_path_candidates(token: str) -> list[str]:
    normalized = str(token or "").strip().strip("'\"`")
    if not normalized:
        return []

    while normalized.startswith("("):
        normalized = normalized[1:].strip()
    while normalized.endswith((";", "|", "&", ",", ")")):
        normalized = normalized[:-1].strip()
    if not normalized:
        return []

    candidates = [normalized]
    if "=" in normalized and not normalized.startswith("="):
        _, value = normalized.split("=", 1)
        value = value.strip().strip("'\"`")
        while value.endswith((";", "|", "&", ",", ")")):
            value = value[:-1].strip()
        if value:
            candidates.append(value)
    return candidates


def _shell_status_update_interval(timeout_sec: int) -> float:
    return max(1.0, min(max(1, timeout_sec) / 3.0, 10.0))


def _build_shell_status_update(command: str, *, elapsed_sec: float, timeout_sec: int) -> str:
    elapsed_text = f"{elapsed_sec:.0f}s"
    timeout_text = f"{max(1, timeout_sec)}s"
    return f"[still running after {elapsed_text} of {timeout_text}] {command}"
