from __future__ import annotations

from typing import Any

from ..normalization import dedupe_keep_tail
from .tool_result_support import note_anchor as _note_anchor


def record_failure_outcome(service: Any, *, tool_name: str, result: Any) -> None:
    if not result.success and result.error:
        if (
            not result.metadata.get("hallucination")
            and not result.metadata.get("approval_denied")
            and not result.metadata.get("suppress_failure_persistence")
        ):
            service.harness.state.working_memory.failures = dedupe_keep_tail(
                service.harness.state.working_memory.failures + [f"{tool_name}: {result.error}"],
                limit=8,
            )
            service.harness.state.recent_errors.append(f"{tool_name}: {result.error}")
            _note_anchor(service.harness, content=f"{tool_name} failed: {str(result.error).strip()}", tag="fail")
