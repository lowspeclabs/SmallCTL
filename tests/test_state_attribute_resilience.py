from __future__ import annotations

from smallctl.state import LoopState
from smallctl.state_schema import PromptBudgetSnapshot


def test_prompt_budget_snapshot_alias_returns_prompt_budget() -> None:
    """Accessing the deprecated prompt_budget_snapshot alias returns prompt_budget."""
    state = LoopState()
    result = state.prompt_budget_snapshot  # type: ignore[attr-defined]
    assert isinstance(result, PromptBudgetSnapshot)
    assert result is state.prompt_budget


def test_unknown_attribute_still_raises() -> None:
    """Truly unknown attributes should still raise AttributeError."""
    state = LoopState()
    try:
        _ = state.this_attribute_definitely_does_not_exist  # type: ignore[attr-defined]
        raise AssertionError("Expected AttributeError")
    except AttributeError as exc:
        assert "this_attribute_definitely_does_not_exist" in str(exc)


def test_safe_get_returns_existing_attribute() -> None:
    """safe_get returns the actual value when the attribute exists."""
    state = LoopState()
    state.current_phase = "plan"
    assert state.safe_get("current_phase", "explore") == "plan"


def test_safe_get_returns_default_for_missing_attribute() -> None:
    """safe_get returns the default for a truly missing attribute."""
    state = LoopState()
    result = state.safe_get("nonexistent_field", ["default"])
    assert result == ["default"]


def test_safe_set_returns_true_for_existing_attribute() -> None:
    """safe_set returns True when the attribute exists and is updated."""
    state = LoopState()
    assert state.safe_set("current_phase", "execute") is True
    assert state.current_phase == "execute"


def test_safe_set_returns_false_for_missing_attribute() -> None:
    """safe_set returns False when the attribute does not exist."""
    state = LoopState()
    assert state.safe_set("nonexistent_field", "value") is False
