from __future__ import annotations

from smallctl.fama.capsules import render_fama_capsules
from smallctl.fama.signals import ActiveMitigation
from smallctl.fama.state import activate_mitigations
from smallctl.state import LoopState


def _activate(state: LoopState, *names: str) -> None:
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name=name,
                reason="test",
                source_signal="early_stop:0",
                activated_step=index,
                expires_after_step=2,
            )
            for index, name in enumerate(names)
        ],
        max_active=max(2, len(names)),
    )


def test_fama_capsules_render_active_mitigation_lines() -> None:
    state = LoopState()
    _activate(state, "done_gate", "acceptance_checklist_capsule")

    lines = render_fama_capsules(state, token_budget=180)

    assert lines == [
        "Before task_complete, satisfy the latest verifier/acceptance evidence or call task_fail with the blocker.",
        "Use the acceptance checklist and latest verifier result as the finish gate.",
    ]


def test_fama_capsules_respect_disabled_config() -> None:
    state = LoopState()
    state.scratchpad["_fama_config"] = {"enabled": False, "mode": "lite", "capsule_token_budget": 180}
    _activate(state, "done_gate")

    assert render_fama_capsules(state, token_budget=180) == []


def test_fama_capsules_respect_line_and_token_caps(monkeypatch) -> None:
    state = LoopState()
    names = [f"capsule_{index}" for index in range(7)]
    _activate(state, *names)
    monkeypatch.setattr(
        "smallctl.fama.capsules.CAPSULE_TEXT",
        {name: f"capsule line {index}" for index, name in enumerate(names)},
    )

    lines = render_fama_capsules(state, token_budget=180)
    tiny_budget_lines = render_fama_capsules(state, token_budget=1)

    assert len(lines) == 5
    assert tiny_budget_lines == []
