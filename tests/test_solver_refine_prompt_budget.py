from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.harness.refine_service import RefineService


def _make_harness(
    *,
    enabled: bool = True,
    budget: int = 700,
    model_fn=None,
) -> Any:
    config = SimpleNamespace(
        solver_refine_enabled=enabled,
        solver_refine_token_budget=budget,
    )
    return SimpleNamespace(
        config=config,
        _runlog=lambda *a, **k: None,
        _model_call=model_fn,
    )


@pytest.mark.asyncio
async def test_refine_returns_none_when_disabled() -> None:
    harness = _make_harness(enabled=False)
    service = RefineService(harness)
    result = await service.run_bounded_refine("draft", "obs")
    assert result is None


@pytest.mark.asyncio
async def test_refine_returns_none_when_budget_exceeded() -> None:
    harness = _make_harness(enabled=True, budget=10)
    service = RefineService(harness)
    result = await service.run_bounded_refine("draft" * 500, "obs" * 500)
    assert result is None


@pytest.mark.asyncio
async def test_refine_returns_none_when_no_model_call() -> None:
    harness = _make_harness(enabled=True)
    service = RefineService(harness)
    result = await service.run_bounded_refine("draft", "obs")
    assert result is None


@pytest.mark.asyncio
async def test_refine_parses_pass_response() -> None:
    async def model_fn(*, messages, tools):
        return '{"verdict": "pass", "issues": [], "revised_output": ""}'

    harness = _make_harness(enabled=True, model_fn=model_fn)
    service = RefineService(harness)
    result = await service.run_bounded_refine("draft", "obs")
    assert result is not None
    assert result.verdict == "pass"


@pytest.mark.asyncio
async def test_refine_falls_back_to_raw_observations_when_rewoo_frame_exceeds_budget() -> None:
    captured: dict[str, Any] = {}

    async def model_fn(*, messages, tools):
        captured["prompt"] = messages[-1]["content"]
        return '{"verdict": "pass", "issues": [], "revised_output": ""}'

    harness = _make_harness(enabled=True, budget=700, model_fn=model_fn)
    service = RefineService(harness)
    result = await service.run_bounded_refine(
        "draft",
        "small raw observation",
        context_frame="REWOO EVIDENCE\n" + ("oversized frame " * 200),
    )

    assert result is not None
    assert result.verdict == "pass"
    assert "small raw observation" in captured["prompt"]
    assert "oversized frame" not in captured["prompt"]
