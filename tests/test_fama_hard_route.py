from __future__ import annotations

import pytest
from types import SimpleNamespace
from typing import Any

from smallctl.fama.detectors import detect_tool_plan_hard_route
from smallctl.harness.run_mode import ModeDecisionService


def _make_state(**kwargs: Any) -> Any:
    scratchpad = dict(kwargs.pop("scratchpad", {}))
    return SimpleNamespace(scratchpad=scratchpad, **kwargs)


def test_detect_hard_route_sets_flag_on_evidence_before_patch() -> None:
    state = _make_state()
    state.scratchpad["_recovery_metrics"] = {"tool_plan_evidence_before_patch_count": 1}
    assert detect_tool_plan_hard_route(state) is True
    assert state.scratchpad["_fama_force_tool_plan_next_turn"] is True


def test_detect_hard_route_sets_flag_on_repeated_read() -> None:
    state = _make_state()
    state.scratchpad["_recovery_metrics"] = {"tool_plan_repeated_read_count": 2}
    assert detect_tool_plan_hard_route(state) is True
    assert state.scratchpad["_fama_force_tool_plan_next_turn"] is True


def test_detect_hard_route_sets_flag_on_wrong_path() -> None:
    state = _make_state()
    state.scratchpad["_recovery_metrics"] = {"tool_plan_wrong_path_count": 1}
    assert detect_tool_plan_hard_route(state) is True
    assert state.scratchpad["_fama_force_tool_plan_next_turn"] is True


def test_detect_hard_route_no_flag_when_clean() -> None:
    state = _make_state()
    state.scratchpad["_recovery_metrics"] = {}
    assert detect_tool_plan_hard_route(state) is False
    assert "_fama_force_tool_plan_next_turn" not in state.scratchpad


@pytest.mark.asyncio
async def test_mode_decision_returns_tool_plan_when_flag_set() -> None:
    state = _make_state()
    state.scratchpad["_fama_force_tool_plan_next_turn"] = True
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *a, **k: None,
        client=SimpleNamespace(model="test"),
        event_handler=None,
    )
    service = ModeDecisionService(harness)
    result = await service.decide("read src")
    assert result == "tool_plan"
    assert "_fama_force_tool_plan_next_turn" not in state.scratchpad


@pytest.mark.asyncio
async def test_mode_decision_does_not_route_when_flag_missing() -> None:
    state = _make_state()
    state.planning_mode_enabled = False
    state.active_plan = None
    state.recent_messages = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *a, **k: None,
        client=SimpleNamespace(model="test"),
        event_handler=None,
    )
    service = ModeDecisionService(harness)
    # It will fall through to other heuristics; smalltalk should return chat
    result = await service.decide("hello")
    assert result == "chat"
