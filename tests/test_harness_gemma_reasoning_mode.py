from __future__ import annotations

from smallctl.harness import Harness


def test_harness_sets_tags_reasoning_mode_for_gemma_despite_off_profile() -> None:
    harness = Harness(
        endpoint="http://openrouter.ai/api/v1",
        model="google/gemma-4-26b-a4b-it",
        provider_profile="openrouter",
        api_key="test-key",
        reasoning_mode="off",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "tags"


def test_harness_preserves_off_reasoning_mode_for_non_gemma() -> None:
    harness = Harness(
        endpoint="http://openrouter.ai/api/v1",
        model="qwen/qwen3.5:9b",
        provider_profile="openrouter",
        api_key="test-key",
        reasoning_mode="off",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "off"
