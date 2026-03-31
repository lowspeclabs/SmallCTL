from __future__ import annotations

PHASES = ("explore", "plan", "execute", "verify")


def normalize_phase(value: str | None) -> str:
    phase = (value or "explore").strip().lower()
    if phase not in PHASES:
        return "explore"
    return phase
