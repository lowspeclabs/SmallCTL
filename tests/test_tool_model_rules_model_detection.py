from __future__ import annotations

import pytest

from smallctl.graph.tool_model_rules_model_detection import (
    _model_is_exact_small_gemma_4_it,
    _model_is_gemma_4,
    _model_uses_gemma_rules,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-4-e2b-it",
        "gemma-4-e4b-it",
        "gemma-4-e2b",
        "gemma-4-e4b",
        "Gemma 4 e2b",
        "Gemma 4 E2B",
        "gemma 4 e2b",
        "google/gemma-4-e2b-it",
        "google_gemma-4-e2b-it",
        "gemma_4_e2b",
        "gemma-4-e2b-it@q4_k_m",
    ],
)
def test_exact_small_gemma_4_detection_matches_user_friendly_names(model_name: str) -> None:
    assert _model_is_exact_small_gemma_4_it(model_name) is True
    assert _model_uses_gemma_rules(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-3-4b",
        "google/gemma-3-4b-it",
    ],
)
def test_exact_small_gemma_4_detection_does_not_match_other_gemma_sizes(model_name: str) -> None:
    assert _model_is_exact_small_gemma_4_it(model_name) is False
    # Gemma-3 still uses the broader Gemma rules (noise stripping, etc.).
    assert _model_uses_gemma_rules(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-2b",
        "gemma-7b",
    ],
)
def test_gemma_rules_do_not_match_unsupported_gemma_sizes(model_name: str) -> None:
    assert _model_is_exact_small_gemma_4_it(model_name) is False
    assert _model_uses_gemma_rules(model_name) is False


@pytest.mark.parametrize(
    "model_name",
    [
        "qwen/qwen-2.5-7b-instruct",
        "openai/gpt-4o",
        "lfm2.5-8b-a1b",
        "",
        None,
    ],
)
def test_gemma_rules_do_not_match_non_gemma_models(model_name: str | None) -> None:
    assert _model_uses_gemma_rules(model_name) is False
    assert _model_is_exact_small_gemma_4_it(model_name) is False
    assert _model_is_gemma_4(model_name) is False


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-3-4b",
        "google/gemma-3-4b-it",
    ],
)
def test_gemma_4_detection_does_not_match_gemma_3_models(model_name: str) -> None:
    assert _model_is_gemma_4(model_name) is False
    assert _model_is_exact_small_gemma_4_it(model_name) is False
    # Gemma-3 still uses the broader Gemma rules (noise stripping, etc.).
    assert _model_uses_gemma_rules(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-4-12b-it",
        "gemma-4-12b",
        "Gemma 4 12b",
        "gemma 4 12b",
        "google/gemma-4-12b-it",
        "gemma-4-27b-it",
        "gemma-4-27b",
        # Existing small variants are still covered by the broader matcher.
        "gemma-4-e2b-it",
        "gemma-4-e4b-it",
    ],
)
def test_gemma_4_family_detection_matches_larger_variants(model_name: str) -> None:
    assert _model_is_gemma_4(model_name) is True
    assert _model_uses_gemma_rules(model_name) is True
