from __future__ import annotations

PHASES = ("explore", "plan", "author", "execute", "verify", "repair")


def normalize_phase(value: str | None) -> str:
    phase = (value or "explore").strip().lower()
    if phase not in PHASES:
        return "explore"
    return phase
