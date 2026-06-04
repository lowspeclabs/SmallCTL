from __future__ import annotations

from typing import Any

ARTIFACT_COVERAGE_SCRATCHPAD_KEY = "_artifact_read_coverage"


def fully_read_artifact_ids(state: Any) -> set[str]:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return set()
    coverage = scratchpad.get(ARTIFACT_COVERAGE_SCRATCHPAD_KEY)
    if not isinstance(coverage, dict):
        return set()
    return {
        str(artifact_id).strip()
        for artifact_id, entry in coverage.items()
        if str(artifact_id).strip()
        and isinstance(entry, dict)
        and bool(entry.get("complete"))
    }
