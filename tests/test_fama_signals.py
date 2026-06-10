from __future__ import annotations

from types import SimpleNamespace

from smallctl.fama.runtime import _handle_signal, expire_for_turn
from smallctl.fama.signals import FamaFailureKind, FamaSignal, get_fama_state, push_fama_signal
from smallctl.fama.state import activate_mitigations, active_mitigations, expire_mitigations
from smallctl.fama.router import route_signal
from smallctl.state import LoopState


class _Config:
    fama_enabled = True
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 2
    fama_signal_window = 8
    fama_done_gate_on_failure = True


def test_fama_signal_stores_under_versioned_scratchpad() -> None:
    state = LoopState(step_count=4)
    signal = FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="test",
        evidence="verifier failed",
        step=4,
        tool_name="task_complete",
    )

    push_fama_signal(state, signal, window=8)

    payload = state.scratchpad["_fama"]
    assert payload["version"] == 1
    assert payload["signals"][-1]["kind"] == "early_stop"
    assert payload["signals"][-1]["tool_name"] == "task_complete"


def test_fama_mitigation_expires_after_ttl() -> None:
    state = LoopState(step_count=10)
    signal = FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="test",
        evidence="verifier failed",
        step=10,
    )
    activate_mitigations(state, route_signal(signal, state=state, config=_Config()), max_active=2)

    assert {item.name for item in active_mitigations(state)} == {
        "done_gate",
        "acceptance_checklist_capsule",
    }

    assert expire_mitigations(state, step=12) == []
    expired = expire_mitigations(state, step=13)

    assert {item.name for item in expired} == {"done_gate", "acceptance_checklist_capsule"}
    assert active_mitigations(state) == []
    assert get_fama_state(state)["active_mitigations"] == []


def test_fama_state_helpers_tolerate_missing_step_field() -> None:
    state = SimpleNamespace(scratchpad={})
    signal = FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="test",
        evidence="verifier failed",
        step=0,
    )

    push_fama_signal(state, signal, window=8)

    assert state.scratchpad["_fama"]["last_observed_step"] == 0


def test_context_drift_routes_tool_plan_evidence_capsule() -> None:
    state = LoopState(step_count=2)
    signal = FamaSignal(
        kind=FamaFailureKind.CONTEXT_DRIFT,
        severity=2,
        source="test",
        evidence="context_missing",
        step=2,
        failure_class="context_missing",
    )

    mitigations = route_signal(signal, state=state, config=_Config())

    assert "evidence_gathering_needed" in {item.name for item in mitigations}


def test_fama_suppressed_signal_is_logged() -> None:
    state = LoopState(step_count=4)
    events: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        config=_Config(),
        _runlog=lambda event, message, **kwargs: events.append((event, kwargs)),
    )
    signal = FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="test",
        evidence="verifier failed",
        step=4,
        tool_name="task_complete",
    )

    assert _handle_signal(harness, state=state, config=harness.config, signal=signal, dedupe=True) is True
    assert _handle_signal(harness, state=state, config=harness.config, signal=signal, dedupe=True) is False

    suppressed = [data for event, data in events if event == "fama_signal_suppressed"]
    assert suppressed
    assert suppressed[-1]["kind"] == "early_stop"
    assert suppressed[-1]["tool_name"] == "task_complete"
    assert "early_stop" in str(suppressed[-1]["signature"])


def test_fama_active_mitigation_ttl_is_logged() -> None:
    state = LoopState(step_count=11)
    events: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        config=_Config(),
        _runlog=lambda event, message, **kwargs: events.append((event, kwargs)),
    )
    signal = FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="test",
        evidence="verifier failed",
        step=10,
        tool_name="task_complete",
    )
    activate_mitigations(state, route_signal(signal, state=state, config=_Config()), max_active=2)

    expire_for_turn(harness, mode="loop")

    ttl_events = [data for event, data in events if event == "fama_mitigation_ttl"]
    assert {event["mitigation"] for event in ttl_events} == {"done_gate", "acceptance_checklist_capsule"}
    assert all(event["remaining_steps"] > 0 for event in ttl_events)


def test_detect_repeated_failure_pattern_fires_after_three_identical_failures() -> None:
    from smallctl.fama.detectors import detect_repeated_failure_pattern
    state = LoopState(step_count=5)
    state.scratchpad["_repeated_failure_observations"] = [
        {
            "key": "file_read::not_found",
            "tool_name": "file_read",
            "domain": "",
            "pattern": "not_found",
            "count": 3,
            "last_step": 5,
            "first_step": 1,
        }
    ]
    signal = detect_repeated_failure_pattern(state, threshold=3)
    assert signal is not None
    assert signal.tool_name == "file_read"
    assert signal.failure_class == "not_found"
    assert "3 attempts" in signal.evidence


def test_detect_repeated_failure_pattern_ignores_stale_observations() -> None:
    from smallctl.fama.detectors import detect_repeated_failure_pattern
    state = LoopState(step_count=20)
    state.scratchpad["_repeated_failure_observations"] = [
        {
            "key": "file_read::not_found",
            "tool_name": "file_read",
            "domain": "",
            "pattern": "not_found",
            "count": 3,
            "last_step": 5,
            "first_step": 1,
        }
    ]
    signal = detect_repeated_failure_pattern(state, threshold=3)
    assert signal is None


def test_detect_verifier_failure_mode_fires_on_persistent_failure_mode() -> None:
    from smallctl.fama.detectors import detect_verifier_failure_mode
    state = LoopState(step_count=5)
    state.last_verifier_verdict = {"failure_mode": "logic", "verdict": "fail"}
    state.scratchpad["_fama_verifier_failure_mode:logic"] = 2
    signal = detect_verifier_failure_mode(state)
    assert signal is not None
    assert signal.failure_class == "logic"
    assert "failure_mode=logic" in signal.evidence


def test_detect_verifier_failure_mode_is_none_on_first_observation() -> None:
    from smallctl.fama.detectors import detect_verifier_failure_mode
    state = LoopState(step_count=5)
    state.last_verifier_verdict = {"failure_mode": "logic", "verdict": "fail"}
    signal = detect_verifier_failure_mode(state)
    assert signal is None
