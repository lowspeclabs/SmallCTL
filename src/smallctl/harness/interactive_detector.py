from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..state import LoopState
from ..tools.control_phase_gates import task_involves_interactive_program as _task_involves_interactive_program
from ..tools.installer_preflight import run_installer_preflight_probes
from ..tools.network_ssh_helpers import detect_interactive_prompt as _detect_interactive_prompt


@dataclass(frozen=True)
class InteractiveNeed:
    """Classification of whether and how a task needs interactive session handling."""

    transport: str  # 'local' | 'remote' | 'hybrid'
    mode: str  # 'scripted' | 'reactive'
    confidence: float


# Extended marker sets beyond the existing task_involves_interactive_program
_INTERACTIVE_KEYWORDS_STRONG: tuple[str, ...] = (
    "stdin",
    "user input",
    "prompt for",
    "REPL",
    "menu-driven",
    "choose option",
)

_INTERACTIVE_KEYWORDS_MEDIUM: tuple[str, ...] = (
    "yes/no",
    "Y/N",
    "press enter",
    "type your",
    "what is your",
)

_SERVER_DAEMON_MARKERS: tuple[str, ...] = (
    "listen on port",
    "socket server",
    "chat server",
    "accept connection",
)


async def classify_interactive_need(
    task: str,
    state: LoopState,
) -> InteractiveNeed | None:
    """Classify whether a task needs interactive session handling.

    Composes three existing detectors:
    1. Keyword classifier for interactive/GUI programs
    2. Prompt detector on last tool output
    3. Installer interactivity probe (if already computed)

    Returns None if no interactive need detected, or an InteractiveNeed
    with transport, mode, and confidence.
    """
    text = str(task or "").strip().lower()
    if not text:
        return None

    signals: list[tuple[str, float]] = []

    # 1. GUI/TUI/game keywords (strong)
    if _task_involves_interactive_program(state):
        signals.append(("gui_tui_game", 1.0))

    # 2. Explicit interactive keywords (strong)
    if any(marker in text for marker in _INTERACTIVE_KEYWORDS_STRONG):
        signals.append(("explicit_interactive", 1.0))

    # 3. CLI dialog markers (medium)
    if any(marker in text for marker in _INTERACTIVE_KEYWORDS_MEDIUM):
        signals.append(("cli_dialog", 0.6))

    # 4. Server/daemon markers (medium)
    if any(marker in text for marker in _SERVER_DAEMON_MARKERS):
        signals.append(("server_daemon", 0.6))

    # 5. Installer markers (medium) — reuse already-computed probe if present
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    preflight_probes = scratchpad.get("_preflight_probes")
    if isinstance(preflight_probes, dict) and preflight_probes.get("is_interactive"):
        signals.append(("installer_interactive", 0.6))

    # 6. History — output matches a prompt (strong)
    last_tool_output = _last_tool_output_text(state)
    if last_tool_output:
        detected = _detect_interactive_prompt(last_tool_output)
        if detected is not None:
            signals.append(("history_prompt_match", 1.0))

    # 7. History — shell_exec timed out with partial output (medium)
    if _last_shell_exec_timed_out_with_output(state):
        signals.append(("timeout_partial_output", 0.6))

    if not signals:
        return None

    # Determine transport
    has_remote = _has_remote_target(text, state)
    has_local = _has_local_target(text, state)
    if has_remote and has_local:
        transport = "hybrid"
    elif has_remote:
        transport = "remote"
    else:
        transport = "local"

    # Determine mode: scripted vs reactive
    # Scripted: all inputs known up front (installer, fixed quiz)
    # Reactive: each input depends on last output (game, REPL)
    if _looks_like_scripted_interaction(text, state):
        mode = "scripted"
    else:
        mode = "reactive"

    # Confidence: any single strong signal or 2+ medium signals → high confidence
    strong_count = sum(1 for _, weight in signals if weight >= 0.8)
    medium_count = sum(1 for _, weight in signals if 0.4 <= weight < 0.8)
    if strong_count >= 1 or medium_count >= 2:
        confidence = 0.85
    elif strong_count >= 1 or medium_count >= 1:
        confidence = 0.65
    else:
        confidence = 0.45

    return InteractiveNeed(transport=transport, mode=mode, confidence=confidence)


def _last_tool_output_text(state: LoopState) -> str:
    """Extract the most recent tool output text for prompt detection."""
    # Check the latest artifact for shell/ssh output
    artifacts = getattr(state, "artifacts", {})
    if not artifacts:
        return ""
    # Find the most recent artifact with shell/ssh output
    candidates = []
    for aid, art in artifacts.items():
        tool_name = str(getattr(art, "tool_name", "") or getattr(art, "kind", "") or "").strip()
        if tool_name not in {"shell_exec", "ssh_exec"}:
            continue
        created = str(getattr(art, "created_at", "") or "").strip()
        candidates.append((created, art))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, latest = candidates[0]
    # Try to get output text
    inline = str(getattr(latest, "inline_content", "") or "").strip()
    if inline:
        return inline
    preview = str(getattr(latest, "preview_text", "") or "").strip()
    if preview:
        return preview
    summary = str(getattr(latest, "summary", "") or "").strip()
    return summary


def _last_shell_exec_timed_out_with_output(state: LoopState) -> bool:
    """Check if the most recent shell_exec timed out after emitting bytes."""
    tool_records = getattr(state, "tool_execution_records", {})
    if not tool_records:
        return False
    # Find the most recent shell_exec record
    candidates = []
    for key, record in tool_records.items():
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "") != "shell_exec":
            continue
        candidates.append((str(key), record))
    if not candidates:
        return False
    # Sort by key (usually step-based) and take the last
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, latest = candidates[0]
    result = latest.get("result")
    if not isinstance(result, dict):
        return False
    if result.get("success") is not False:
        return False
    error = str(result.get("error") or "").lower()
    if "timed out" not in error and "timeout" not in error:
        return False
    # Check if there was partial output
    output = result.get("output")
    if isinstance(output, dict):
        stdout = str(output.get("stdout") or "").strip()
        stderr = str(output.get("stderr") or "").strip()
        return bool(stdout or stderr)
    output_str = str(output or "").strip()
    return bool(output_str)


def _has_remote_target(text: str, state: LoopState) -> bool:
    """Check if the task or state references a remote target."""
    lowered = text.lower()
    from ..harness.task_classifier_constants import IP_ADDRESS_PATTERN
    has_ip = bool(IP_ADDRESS_PATTERN.search(lowered))
    has_ssh_marker = any(m in lowered for m in ("ssh", "scp", "sftp", "remote host", "remote server"))
    if has_ip or has_ssh_marker:
        return True
    # Check scratchpad for resolved remote followup
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    resolved = scratchpad.get("_resolved_remote_followup")
    if isinstance(resolved, dict) and (resolved.get("host") or resolved.get("target")):
        return True
    return False


def _has_local_target(text: str, state: LoopState) -> bool:
    """Check if the task or state references a local target."""
    lowered = text.lower()
    local_markers = ("./", "../", "/home/", "/tmp/", "local", "localhost")
    if any(m in lowered for m in local_markers):
        return True
    # Default to local if no explicit remote
    return True


def _looks_like_scripted_interaction(text: str, state: LoopState) -> bool:
    """Determine if interaction is scripted (inputs known up front) vs reactive."""
    lowered = text.lower()
    # Strong scripted markers
    scripted_markers = (
        "answer yes to all",
        "answer no to all",
        "preseed",
        "non-interactive",
        "noninteractive",
        "silent install",
        "unattended",
        "auto accept",
        "autoaccept",
        "batch mode",
        "batchmode",
        "fixed answers",
        "known answers",
        "answer sequence",
        "input sequence",
        "all prompts",
        "every prompt",
    )
    if any(m in lowered for m in scripted_markers):
        return True
    # Check if the task describes a known input sequence
    if re.search(r"answer\s+(yes|no|y|n)\s+to\s+all", lowered):
        return True
    if re.search(r"(run|execute).*with\s+input", lowered):
        return True
    # If installer with explicit flags → scripted
    if "installer" in lowered and any(f in lowered for f in ("-y", "--yes", "--auto", "-f")):
        return True
    # Default to reactive for games, REPLs, etc.
    reactive_markers = (
        "game",
        "play",
        "guess",
        "number guessing",
        "quiz",
        "chat",
        "REPL",
        "interactive mode",
        "menu",
        "choose",
        "select option",
    )
    if any(m in lowered for m in reactive_markers):
        return False
    # Default: assume reactive for safety (model must see each prompt)
    return False
