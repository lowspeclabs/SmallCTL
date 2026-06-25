from __future__ import annotations

import re

from .normalization import collapse_model_name

_GEMMA_MODEL_MARKERS = (
    "google_gemma-4",
    "google_gemma",
    "gemma-4",
    "gemma-3",
    "gemma/",
)
_EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
    # Bare quantized suffixes are used by some backends/gguf filenames that
    # drop the "-it" token even though the checkpoint is instruction-tuned.
    "gemma-4-e2b",
    "gemma-4-e4b",
)
_EXACT_GEMMA_4_26B_A4B_IT_MODEL_SUFFIXES = (
    "google-gemma-4-26b-a4b-it",
)
_EXACT_LFM_25_8B_A1B_MODELS = (
    "lfm2.5-8b-a1b",
    "liquid/lfm2.5-8b-a1b",
)


def is_gemma_model_name(model_name: str) -> bool:
    normalized = collapse_model_name(model_name)
    return any(marker in normalized for marker in _GEMMA_MODEL_MARKERS)


def _matches_any_suffix(normalized: str, suffixes: tuple[str, ...]) -> bool:
    for suffix in suffixes:
        if (
            normalized == suffix
            or normalized.startswith(f"{suffix}-")
            or normalized.endswith(f"-{suffix}")
            or re.search(rf"(?:^|-){re.escape(suffix)}(?:$|-)", normalized)
        ):
            return True
    return False


def is_exact_small_gemma_4_it_model_name(model_name: str) -> bool:
    normalized = collapse_model_name(model_name)
    if not normalized:
        return False
    return _matches_any_suffix(normalized, _EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES)


def is_exact_large_gemma_4_26b_a4b_it_model_name(model_name: str) -> bool:
    normalized = collapse_model_name(model_name)
    if not normalized:
        return False
    return _matches_any_suffix(normalized, _EXACT_GEMMA_4_26B_A4B_IT_MODEL_SUFFIXES)


def is_gemma_4_it_model_name(model_name: str) -> bool:
    """True for Gemma-4 instruction-tuned variants that emit <think> tags.

    Covers the known small IT suffixes (e2b/e4b), the 26b A4B IT variant, and
    the 31b IT variant observed on OpenRouter.  Using the broader "gemma-4-"
    + "-it" heuristic lets the harness treat future Gemma-4 IT checkpoints the
    same way without waiting for an exact allow-list update.
    """
    normalized = collapse_model_name(model_name)
    if "gemma-4-" not in normalized:
        return False
    return normalized.endswith("-it")


def is_lfm25_8b_a1b_model_name(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in _EXACT_LFM_25_8B_A1B_MODELS
