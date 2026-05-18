from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness.core_facade import _inject_recovery_metrics


def test_inject_recovery_metrics_copies_scratchpad_metrics() -> None:
    state = SimpleNamespace(
        scratchpad={
            "_recovery_metrics": {
                "tool_plan_invocations": 2,
                "tool_plan_total_tokens": 800,
            }
        }
    )
    result: dict[str, object] = {"status": "completed"}
    _inject_recovery_metrics(result, state)
    assert result["recovery_metrics"] == {
        "tool_plan_invocations": 2,
        "tool_plan_total_tokens": 800,
    }


def test_inject_recovery_metrics_skips_when_empty() -> None:
    state = SimpleNamespace(scratchpad={"_recovery_metrics": {}})
    result: dict[str, object] = {"status": "completed"}
    _inject_recovery_metrics(result, state)
    assert "recovery_metrics" not in result


def test_inject_recovery_metrics_skips_when_missing() -> None:
    state = SimpleNamespace(scratchpad={})
    result: dict[str, object] = {"status": "completed"}
    _inject_recovery_metrics(result, state)
    assert "recovery_metrics" not in result


def test_inject_recovery_metrics_skips_when_no_scratchpad() -> None:
    state = SimpleNamespace()
    result: dict[str, object] = {"status": "completed"}
    _inject_recovery_metrics(result, state)
    assert "recovery_metrics" not in result
