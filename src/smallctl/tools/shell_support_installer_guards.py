from __future__ import annotations

import shlex
import re
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail
from .shell_parsing import (
    _simple_shell_command_segments,
    _split_shell_words,
)
from .shell_support_constants import (
    _REMOTE_INSTALLER_PREFLIGHT_KEY,
    _SINGLE_ANSWER_PIPE_PATTERN,
    _YES_PIPE_PATTERN,
)


def _guard_fail(
    message: str,
    *,
    reason: str,
    command: str,
    error_kind: str | None = None,
    next_required_tool: dict[str, Any] | None = None,
    next_required_action: dict[str, Any] | str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent guard failure result."""
    metadata: dict[str, Any] = {
        "reason": reason,
        "command": command,
    }
    if error_kind is not None:
        metadata["error_kind"] = error_kind
    if next_required_tool is not None:
        metadata["next_required_tool"] = next_required_tool
    if next_required_action is not None:
        metadata["next_required_action"] = next_required_action
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return fail(message, metadata=metadata)


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
        return _guard_fail(
            f"`{tool_name}` blocked `yes |` automation for an interactive installer: `{raw}`. "
            "Use the installer's non-interactive mode when available, such as `--autoaccept` or `-y`; "
            "otherwise use a config/preseed file or an explicit `printf` script with known answers.",
            reason="unsafe_yes_pipe_interactive_installer",
            command=raw,
            next_required_action={
                "strategy": "use_structured_noninteractive_install",
                "preferred_inputs": [
                    "--autoaccept or -y if the installer documents it",
                    "a preseed/config file such as .fogsettings",
                    "an explicit printf script with known prompt answers",
                ],
            },
            extra_metadata={
                "detected_target": target,
            },
        )
    for match in _SINGLE_ANSWER_PIPE_PATTERN.finditer(raw):
        target = " ".join(str(match.group("target") or "").split())
        target_lower = target.lower()
        if not _looks_like_interactive_installer_target(target_lower):
            continue
        return _guard_fail(
            f"`{tool_name}` blocked single-answer `echo |` automation for an interactive installer: `{raw}`. "
            "A lone Y/N answer is brittle for multi-prompt installers. Use the installer's non-interactive mode "
            "when available, such as `--autoaccept` or `-y`; otherwise use a config/preseed file or an explicit "
            "`printf` script with the complete known answer stream.",
            reason="unsafe_single_answer_pipe_interactive_installer",
            command=raw,
            next_required_action={
                "strategy": "use_structured_noninteractive_install",
                "preferred_inputs": [
                    "--autoaccept or -y if the installer documents it",
                    "a preseed/config file such as .fogsettings",
                    "an explicit printf script with the complete known prompt answers",
                ],
            },
            extra_metadata={
                "detected_target": target,
                "detected_answer": " ".join(str(match.group("answer") or "").split()),
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
            return _guard_fail(
                "Remote installer preflight found missing or corrupted critical installer files. "
                "Repair the environment with a fresh clone or clean reset before running the installer.",
                reason="remote_installer_preflight_failed",
                command=raw,
                next_required_action="fresh clone or clean reset; do not patch individual installer files",
                extra_metadata={
                    "host": host,
                    "user": user,
                    "cwd": cwd,
                    "script_path": script_path,
                    "preflight": entry,
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
    return _guard_fail(
        "Remote installer preflight required before running this high-risk installer mutation. "
        "Run narrow repo/integrity checks first, then retry the installer after the preflight is clean. "
        f"Required checks: {checks_text}",
        reason="remote_installer_preflight_required",
        command=raw,
        next_required_action=checks_text,
        extra_metadata={
            "host": host,
            "user": user,
            "cwd": cwd,
            "script_path": script_path,
            "required_checks": checks,
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
