from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from smallctl.fama.runtime import _handle_signal
from smallctl.fama.signals import FamaFailureKind, FamaSignal


def _make_harness() -> Any:
    logs: list[dict[str, Any]] = []

    def runlog(event: str, message: str, **data: Any) -> None:
        logs.append({"event": event, "message": message, **data})

    return SimpleNamespace(_runlog=runlog, logs=logs)


def _make_state() -> Any:
    scratchpad: dict[str, Any] = {}
    fama_state: dict[str, Any] = {"seen_signatures": []}
    return SimpleNamespace(
        scratchpad=scratchpad,
        _fama_state=fama_state,
        step_count=1,
    )


def _loop_guard_signal(step: int = 1, evidence: str = "no_actionable_progress=4") -> FamaSignal:
    return FamaSignal(
        kind=FamaFailureKind.LOOPING,
        severity=2,
        source="loop_guard",
        evidence=evidence,
        step=step,
        tool_name=None,
        failure_class="repeated_action",
        next_safe_action="Stop repeating.",
    )


def test_loop_guard_suppressed_twice_then_escalated() -> None:
    harness = _make_harness()
    state = _make_state()
    config = SimpleNamespace(
        fama_enabled=True,
        fama_max_active_mitigations=3,
        fama_signal_window=5,
        loop_guard_stagnation_threshold=3,
    )
    signal = _loop_guard_signal(step=5)

    # First suppression
    result1 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result1 is True  # processed
    assert any(log["event"] == "fama_signal_detected" for log in harness.logs)

    # Second call with same signature should be suppressed
    harness.logs.clear()
    result2 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result2 is False  # suppressed
    assert any(log["event"] == "fama_signal_suppressed" for log in harness.logs)
    assert state.scratchpad["_fama"]["_fama_loop_guard_suppression_count:looping:loop_guard:"] == 1

    # Third suppression
    harness.logs.clear()
    result3 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result3 is False  # suppressed
    assert state.scratchpad["_fama"]["_fama_loop_guard_suppression_count:looping:loop_guard:"] == 2

    # Fourth suppression should escalate
    harness.logs.clear()
    result4 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result4 is True  # escalated and processed
    assert any(log["event"] == "fama_signal_escalated" for log in harness.logs)
    # Count is cleared after successful pass-through
    assert "_fama_loop_guard_suppression_count:looping:loop_guard:" not in state.scratchpad["_fama"]


def test_loop_guard_suppression_count_cleared_on_new_signal() -> None:
    harness = _make_harness()
    state = _make_state()
    config = SimpleNamespace(
        fama_enabled=True,
        fama_max_active_mitigations=3,
        fama_signal_window=5,
        loop_guard_stagnation_threshold=3,
    )
    signal = _loop_guard_signal(step=5)

    _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert state.scratchpad["_fama"]["_fama_loop_guard_suppression_count:looping:loop_guard:"] == 1

    # New signal with a different repeated_tool creates a different signature
    signal2 = _loop_guard_signal(step=6, evidence="repeated_tool=file_read; no_actionable_progress=5")
    harness.logs.clear()
    result = _handle_signal(harness, state=state, config=config, signal=signal2, dedupe=True)
    assert result is True
    # Old suppression count for the original signature should remain
    assert state.scratchpad["_fama"].get("_fama_loop_guard_suppression_count:looping:loop_guard:") == 1
    # New signature should have no suppression count yet (or was cleared after pass-through)
    assert "_fama_loop_guard_suppression_count:looping:loop_guard:file_read" not in state.scratchpad["_fama"]
