from __future__ import annotations

from typing import Any

from ..state import LoopState


REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"


def _clear_remote_mutation_requirement(state: LoopState | None, *, path: str, host: str) -> None:
    if state is None:
        return
    requirement = state.scratchpad.get(REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(requirement, dict):
        return
    guessed_paths = [str(item) for item in requirement.get("guessed_paths", []) if str(item).strip()]
    requirement_host = str(requirement.get("host") or "").strip().lower()
    if requirement_host and host and requirement_host != host.strip().lower():
        return
    if guessed_paths and path not in guessed_paths:
        return
    state.scratchpad.pop(REMOTE_MUTATION_VERIFICATION_KEY, None)
