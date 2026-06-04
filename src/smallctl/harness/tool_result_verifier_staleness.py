from __future__ import annotations

from typing import Any

from .remote_mutation_helpers import _STALE_VERIFIER_KEY


def _mark_verifier_stale_after_file_change(
    service: Any,
    *,
    tool_name: str,
    paths: list[str],
) -> None:
    state = getattr(service.harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if state is None or not isinstance(scratchpad, dict):
        return
    current_verifier = getattr(state, "current_verifier_verdict", None)
    verifier = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    if not isinstance(verifier, dict) or not verifier:
        return
    clean_paths = [str(path).strip() for path in paths if str(path).strip()]
    scratchpad[_STALE_VERIFIER_KEY] = {
        "reason": "file_changed_after_verifier",
        "tool_name": tool_name,
        "paths": clean_paths,
        "prior_verdict": dict(verifier),
    }
