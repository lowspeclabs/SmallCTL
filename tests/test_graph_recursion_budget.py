from __future__ import annotations

from types import SimpleNamespace

from smallctl.config import resolve_config
from smallctl.graph.runtime_base import publish_graph_step_budget, resolve_graph_recursion_limit
from smallctl.prompts import build_system_prompt
from smallctl.state import LoopState


def _harness_for_state(state: LoopState, **config_values: object) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(**config_values), state=state)


def test_resolve_config_exposes_graph_recursion_defaults() -> None:
    config = resolve_config({})

    assert config.graph_recursion_limit == 1024
    assert config.graph_coding_recursion_limit == 2048


def test_coding_runs_use_higher_recursion_limit() -> None:
    state = LoopState(task_mode="local_execute")
    harness = _harness_for_state(
        state,
        graph_recursion_limit=1024,
        graph_coding_recursion_limit=2048,
    )

    assert resolve_graph_recursion_limit(harness) == 2048


def test_non_coding_runs_use_base_recursion_limit() -> None:
    state = LoopState(task_mode="chat")
    harness = _harness_for_state(
        state,
        graph_recursion_limit=1024,
        graph_coding_recursion_limit=2048,
    )

    assert resolve_graph_recursion_limit(harness) == 1024


def test_prompt_includes_graph_step_budget_guidance() -> None:
    state = LoopState(task_mode="local_execute")
    state.step_count = 24
    publish_graph_step_budget(_harness_for_state(state), recursion_limit=2048)

    prompt = build_system_prompt(state, "author")

    assert "STEP BUDGET: You have approximately 2024 graph steps remaining out of 2048." in prompt
    assert "do not re-verify unless the latest test output shows a real failure" in prompt
