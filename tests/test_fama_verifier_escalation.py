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
    return SimpleNamespace(
        scratchpad={},
        recent_messages=[],
        step_count=1,
    )


def _verifier_signal(step: int = 1) -> FamaSignal:
    return FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="verifier",
        evidence="exit_code=3; systemctl status docker",
        step=step,
        tool_name="ssh_exec",
        failure_class="verifier_failed",
    )


def test_verifier_failure_suppressed_twice_then_escalated() -> None:
    """Regression: repeated verifier failures must escalate without crashing.

    The escalation path used ``signal.model_copy()``, which does not exist on
    the dataclass-based ``FamaSignal``. It now uses ``dataclasses.replace``.
    """
    harness = _make_harness()
    state = _make_state()
    config = SimpleNamespace(
        fama_enabled=True,
        fama_max_active_mitigations=3,
        fama_signal_window=5,
        loop_guard_stagnation_threshold=3,
    )
    signal = _verifier_signal(step=1)

    result1 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result1 is True
    assert any(log["event"] == "fama_signal_detected" for log in harness.logs)

    harness.logs.clear()
    result2 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result2 is False
    assert any(log["event"] == "fama_signal_suppressed" for log in harness.logs)

    harness.logs.clear()
    result3 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result3 is False
    assert any(log["event"] == "fama_signal_suppressed" for log in harness.logs)

    # Fourth repetition should escalate and be processed.
    harness.logs.clear()
    result4 = _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    assert result4 is True
    assert any(log["event"] == "fama_signal_escalated" for log in harness.logs)
    assert any(log["event"] == "fama_signal_detected" for log in harness.logs)
