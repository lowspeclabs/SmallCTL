from __future__ import annotations

from types import SimpleNamespace

from smallctl.fama.signals import FamaFailureKind, FamaSignal, get_fama_state, push_fama_signal
from smallctl.fama.state import activate_mitigations, active_mitigations, expire_mitigations
from smallctl.fama.router import route_signal
from smallctl.state import LoopState


class _Config:
    fama_default_ttl_steps = 2


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
