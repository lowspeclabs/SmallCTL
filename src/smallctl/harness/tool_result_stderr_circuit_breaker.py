from __future__ import annotations

from typing import Any

from .tool_result_stderr_signatures import stderr_signature_key, stderr_signature_line


def _record_stderr_signature_circuit_breaker(service: Any, *, tool_name: str, result: Any) -> None:
    if tool_name not in {"shell_exec", "ssh_exec"}:
        return
    signature_line = stderr_signature_line(result)
    if not signature_line:
        return
    key = stderr_signature_key(result)
    if not key:
        return
    state = service.harness.state
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    counts = scratchpad.setdefault("_stderr_signature_counts", {})
    if not isinstance(counts, dict):
        counts = {}
        scratchpad["_stderr_signature_counts"] = counts
    count = int(counts.get(key, 0) or 0) + 1
    counts[key] = count
    if count < 2:
        return
    scratchpad["_stderr_signature_circuit_breaker"] = {
        "signature": signature_line,
        "count": count,
        "tool_name": tool_name,
        "next_required_action": "Use a different repair strategy; do not retry the same command/fix against this stderr.",
    }
    state.recent_errors.append(f"stderr_signature_circuit_breaker: {signature_line}")
