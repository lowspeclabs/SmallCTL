from __future__ import annotations

from smallctl.harness import Harness


def test_harness_sets_tags_reasoning_mode_for_known_gemma_despite_off_profile() -> None:
    harness = Harness(
        endpoint="http://openrouter.ai/api/v1",
        model="google/gemma-4-26b-a4b-it",
        provider_profile="openrouter",
        api_key="test-key",
        reasoning_mode="off",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "tags"


def test_harness_defaults_gemma_4_it_to_tags_when_auto() -> None:
    # The 31b IT variant is hosted on OpenRouter and emits <think> tags, so it
    # should be treated as a tagged Gemma-4 IT model rather than falling back to
    # reasoning-off mode.
    harness = Harness(
        endpoint="http://openrouter.ai/api/v1",
        model="google/gemma-4-31b-it",
        provider_profile="openrouter",
        api_key="test-key",
        reasoning_mode="auto",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "tags"
    assert harness.state.scratchpad.get("_thinking_tags_disabled") is None


def test_harness_preserves_explicit_tags_for_unknown_gemma() -> None:
    harness = Harness(
        endpoint="http://openrouter.ai/api/v1",
        model="google/gemma-4-31b-it",
        provider_profile="openrouter",
        api_key="test-key",
        reasoning_mode="tags",
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


def test_harness_recognizes_bare_gemma_4_quantized_suffix_as_tagged() -> None:
    # Some backends/gguf filenames drop the "-it" suffix even though the
    # checkpoint is instruction-tuned and emits <think> tags.
    harness = Harness(
        endpoint="http://localhost:8080/v1",
        model="gemma-4-e4b",
        provider_profile="llamacpp",
        api_key="test-key",
        reasoning_mode="auto",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "tags"
    assert harness.state.scratchpad.get("_thinking_tags_disabled") is None
