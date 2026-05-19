from __future__ import annotations

from smallctl.config import SmallctlConfig, resolve_config
from smallctl.graph.hard_step_detector import HardStepDetector
from smallctl.state import LoopState, PlanStep


def test_detector_disabled_by_default() -> None:
    step = PlanStep(step_id="S1", title="hard", difficulty="hard")

    assert HardStepDetector().should_scale(step=step, state=LoopState(), config=SmallctlConfig()) is False


def test_detector_triggers_on_explicit_hard_when_enabled() -> None:
    config = SmallctlConfig(test_time_scaling_enabled=True)
    step = PlanStep(step_id="S1", title="hard", difficulty="hard")

    decision = HardStepDetector().decide(step=step, state=LoopState(), config=config)

    assert decision.should_scale is True
    assert decision.reason == "explicit_hard"


def test_detector_triggers_on_retry_by_default_when_enabled() -> None:
    config = SmallctlConfig(test_time_scaling_enabled=True)
    step = PlanStep(step_id="S1", title="retrying", retry_count=1)

    decision = HardStepDetector().decide(step=step, state=LoopState(), config=config)

    assert decision.should_scale is True
    assert decision.reason == "retry"


def test_detector_respects_explicit_only_trigger() -> None:
    config = SmallctlConfig(test_time_scaling_enabled=True, test_time_scaling_trigger="explicit")
    step = PlanStep(step_id="S1", title="retrying", retry_count=1)

    assert HardStepDetector().should_scale(step=step, state=LoopState(), config=config) is False


def test_detector_heuristic_triggers_on_risky_tool_allowlist() -> None:
    config = SmallctlConfig(test_time_scaling_enabled=True, test_time_scaling_trigger="heuristic")
    step = PlanStep(step_id="S1", title="risky", tool_allowlist=["file_patch"])

    decision = HardStepDetector().decide(step=step, state=LoopState(), config=config)

    assert decision.should_scale is True
    assert decision.reason == "risk_tool_allowlist"


def test_test_time_scaling_config_coerces_cli_values() -> None:
    config = resolve_config(
        {
            "test_time_scaling_enabled": "true",
            "test_time_scaling_runtimes": "staged_execution,tool_plan",
            "test_time_scaling_max_candidates": "4",
            "test_time_scaling_score_threshold": "0.75",
            "test_time_scaling_all_fail_action": "fail_step",
        }
    )

    assert config.test_time_scaling_enabled is True
    assert config.test_time_scaling_runtimes == ["staged_execution", "tool_plan"]
    assert config.test_time_scaling_max_candidates == 4
    assert config.test_time_scaling_score_threshold == 0.75
    assert config.test_time_scaling_all_fail_action == "fail_step"
