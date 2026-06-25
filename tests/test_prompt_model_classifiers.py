from __future__ import annotations

import pytest

from smallctl.prompt_model_classifiers import (
    is_exact_large_gemma_4_26b_a4b_it_model_name,
    is_exact_small_gemma_4_it_model_name,
    is_gemma_4_it_model_name,
    is_gemma_model_name,
    is_lfm25_8b_a1b_model_name,
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
        "Gemma 4 e4b",
        "google/gemma-4-e2b-it",
        "google_gemma-4-e2b-it",
        "gemma_4_e2b",
        "gemma-4-e2b-it@q4_k_m",
    ],
)
def test_exact_small_gemma_4_matches_user_friendly_names(model_name: str) -> None:
    assert is_exact_small_gemma_4_it_model_name(model_name) is True
    assert is_gemma_model_name(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-3-4b",
        "google/gemma-3-4b-it",
        "gemma-2b",
        "gemma-7b",
        "qwen/qwen-2.5-7b-instruct",
        "",
        None,
    ],
)
def test_exact_small_gemma_4_does_not_match_other_models(model_name: str | None) -> None:
    assert is_exact_small_gemma_4_it_model_name(model_name) is False


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-4-e4b-it",
        "gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ],
)
def test_gemma_4_it_matches_it_variants(model_name: str) -> None:
    assert is_gemma_4_it_model_name(model_name) is True
    assert is_gemma_model_name(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma-4-e4b",
        "gemma-4-26b-a4b",
        "gemma-4",
        "gemma-3-4b-it",
    ],
)
def test_gemma_4_it_does_not_match_non_it_variants(model_name: str) -> None:
    assert is_gemma_4_it_model_name(model_name) is False


@pytest.mark.parametrize(
    "model_name",
    [
        "google_gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it",
    ],
)
def test_exact_large_gemma_4_26b_matches(model_name: str) -> None:
    assert is_exact_large_gemma_4_26b_a4b_it_model_name(model_name) is True


def test_lfm25_matches_exact_models() -> None:
    assert is_lfm25_8b_a1b_model_name("lfm2.5-8b-a1b") is True
    assert is_lfm25_8b_a1b_model_name("liquid/lfm2.5-8b-a1b") is True
    assert is_lfm25_8b_a1b_model_name("lfm2.5-8b") is False
