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


def test_harness_uses_field_reasoning_for_small_gemma_4_it() -> None:
    # Small Gemma-4 IT checkpoints (e2b/e4b) emit native reasoning_content and
    # tool_calls deltas more reliably when not instructed to wrap reasoning in
    # explicit <think> tags.
    harness = Harness(
        endpoint="http://localhost:8080/v1",
        model="gemma-4-e4b",
        provider_profile="llamacpp",
        api_key="test-key",
        reasoning_mode="auto",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "field"
    assert harness.state.scratchpad.get("_thinking_tags_disabled") is True


def test_harness_recognizes_spaced_gemma_4_e4b_name_as_field() -> None:
    # Users often pass model names with spaces (e.g. "Gemma 4 e4b"). The
    # classifier must collapse separators so the harness selects the native
    # reasoning-field mode and disables explicit <think> tag instructions.
    harness = Harness(
        endpoint="http://localhost:8080/v1",
        model="Gemma 4 e4b",
        provider_profile="llamacpp",
        api_key="test-key",
        reasoning_mode="auto",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "field"
    assert harness.state.scratchpad.get("_thinking_tags_disabled") is True


def test_harness_disables_think_tags_for_non_exact_small_gemma_on_llamacpp() -> None:
    # Larger local Gemma-4 variants (e.g. 12b served by llama.cpp) are prone
    # to native reasoning-channel loops when prompted for explicit <think>
    # tags. The harness should proactively disable tag instructions.
    harness = Harness(
        endpoint="http://localhost:8080/v1",
        model="Gemma 4 12b",
        provider_profile="llamacpp",
        api_key="test-key",
        reasoning_mode="auto",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "off"
    assert harness.state.scratchpad.get("_thinking_tags_disabled") is True


def test_harness_overrides_explicit_tags_for_non_exact_small_gemma_on_llamacpp() -> None:
    # Even if the user/profile asks for tags, non-exact-small local Gemma-4
    # variants should not use explicit <think> instructions.
    harness = Harness(
        endpoint="http://localhost:8080/v1",
        model="gemma-4-12b-it",
        provider_profile="llamacpp",
        api_key="test-key",
        reasoning_mode="tags",
        runtime_context_probe=False,
    )
    assert harness.reasoning_mode == "off"
    assert harness.state.scratchpad.get("_thinking_tags_disabled") is True
