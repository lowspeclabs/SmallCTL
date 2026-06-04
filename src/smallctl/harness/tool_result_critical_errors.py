from __future__ import annotations

import re
from typing import Any

from ..normalization import dedupe_keep_tail
from ..state_schema import MemoryEntry

_CRITICAL_ERROR_RE = re.compile(
    r"^(?:\s*[-*]+\s*)?(?:Error|ERROR|error|fatal|FATAL|Failed|FAILED|FAILURE|failure|Exception|EXCEPTION)\b.*",
    re.MULTILINE,
)


def _extract_and_pin_critical_errors(
    service: Any,
    *,
    tool_name: str,
    result: Any,
    artifact: Any,
) -> None:
    """Extract critical error lines from tool output and pin them in working memory."""
    text = ""
    if not result.success and result.error:
        text = str(result.error)
    elif isinstance(result.output, dict):
        if tool_name in {"shell_exec", "ssh_exec"}:
            stdout = str(result.output.get("stdout") or "")
            stderr = str(result.output.get("stderr") or "")
            text = stdout + "\n" + stderr
        else:
            text = str(result.output.get("content") or result.output.get("stdout") or result.output.get("stderr") or "")
    elif isinstance(result.output, str):
        text = result.output

    if not text:
        return

    matches = _CRITICAL_ERROR_RE.findall(text)
    if not matches:
        return

    # Deduplicate and limit
    seen: set[str] = set()
    errors: list[str] = []
    for line in matches:
        normalized = line.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        clipped = line.strip()[:280]
        if clipped:
            errors.append(clipped)
        if len(errors) >= 2:
            break

    if not errors:
        return

    wm = service.harness.state.working_memory
    current_step = service.harness.state.step_count
    current_phase = service.harness.state.current_phase
    new_facts: list[str] = []
    new_meta: list[MemoryEntry] = []

    for err in errors:
        prefix = f"CRITICAL: {err}"
        if prefix not in wm.known_facts:
            new_facts.append(prefix)
            new_meta.append(
                MemoryEntry(
                    content=prefix,
                    created_at_step=current_step,
                    created_phase=current_phase,
                    freshness="pinned",
                )
            )

    if new_facts:
        wm.known_facts = dedupe_keep_tail(wm.known_facts + new_facts, limit=12)
        # Align meta so pinned entries survive invalidation
        existing_lookup = {m.content: m for m in wm.known_fact_meta}
        aligned: list[MemoryEntry] = []
        for fact in wm.known_facts:
            if fact in existing_lookup:
                aligned.append(existing_lookup[fact])
            else:
                # Find matching new meta
                match = next((m for m in new_meta if m.content == fact), None)
                if match:
                    aligned.append(match)
                else:
                    aligned.append(
                        MemoryEntry(
                            content=fact,
                            created_at_step=current_step,
                            created_phase=current_phase,
                        )
                    )
        wm.known_fact_meta = aligned
