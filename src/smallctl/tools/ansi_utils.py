"""ANSI escape code utilities for terminal output cleaning."""
from __future__ import annotations

import re
from typing import Any

# Pattern matching ANSI escape sequences
_ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1b\[[0-9;:?]*[A-Za-z])|"  # CSI sequences
    r"(?:\x1b\][0-9;]*(?:\x07|\x1b\\))|"  # OSC sequences
    r"(?:\x1b[\(\)][A-Za-z0-9])|"  # character set sequences
    r"(?:\x1b[#%\(\)]*[A-Za-z])|"  # other ESC sequences
    r"(?:\x1b[@-Z\\-_])|"  # single-char ESC sequences
    r"(?:\x9b[0-9;:?]*[A-Za-z])|"  # CSI with C1 control
    r"(?:\x9d[0-9;]*(?:\x07|\x9c))"  # OSC with C1 control
)

# TUI / dialog application signatures
_TUI_SIGNATURES = (
    "whiptail",
    "dialog",
    "ncurses",
    "curses",
    "\x1b=",        # application keypad mode, common in dialog/whiptail screens
    "\x1b[?1049h",  # alternate screen buffer
    "\x1b[?1049l",  # exit alternate screen buffer
    "\x1b[2J",       # clear screen
    "\x1b[H",        # home cursor
    "\x1b(B",        # ASCII charset selection before box/widget drawing
    "\x1b)0",        # line-drawing charset selection
    "┌", "└", "┐", "┘", "─", "│", "╔", "╚", "╗", "╝", "═", "║",  # box drawing
)

# Interactive installer/dialog signatures that commonly appear after ANSI
# stripping or when remote TERM is incompatible.
_INTERACTIVE_INSTALLER_SIGNATURES = (
    "error opening terminal",
    "installer exited",
    "static ip",
    "use the arrow keys",
    "press enter",
    "press <enter>",
    "press return",
    "select an option",
    "choose an option",
    "open source software",
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def has_ansi(text: str) -> bool:
    """Check if text contains ANSI escape sequences."""
    return bool(_ANSI_ESCAPE_RE.search(text))


def detect_tui_application(text: str) -> dict[str, Any] | None:
    """Detect if the output indicates a TUI/dialog application.
    
    Returns a dict with detected info, or None if no TUI detected.
    """
    text = str(text or "")
    lowered = text.lower()
    
    for sig in _TUI_SIGNATURES:
        if sig in text or sig in lowered:
            return {
                "tui_detected": True,
                "signature": sig,
                "kind": "dialog_tui",
            }
    
    for sig in _INTERACTIVE_INSTALLER_SIGNATURES:
        if sig in lowered:
            return {
                "tui_detected": True,
                "signature": sig,
                "kind": "interactive_installer",
            }

    # Many ncurses/dialog frames are stored as `qqqq`, `x...x`, etc. when the
    # line-drawing charset escape has been stripped or partially captured.
    if re.search(r"\bq{4,}[^a-z0-9]{0,20}(?:ok|yes|no|cancel|back|select|install)\b", lowered):
        return {
            "tui_detected": True,
            "signature": "line_drawing_charset",
            "kind": "dialog_tui",
        }
    
    return None


def strip_ansi_from_dict(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Return a copy of dict with ANSI stripped from specified keys."""
    result = dict(d)
    for key in keys:
        if key in result and isinstance(result[key], str):
            result[key] = strip_ansi(result[key])
    return result
