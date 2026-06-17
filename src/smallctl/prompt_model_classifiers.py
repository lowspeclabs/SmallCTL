from __future__ import annotations

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
)
_EXACT_GEMMA_4_26B_A4B_IT_MODEL_SUFFIXES = (
    "google_gemma-4-26b-a4b-it",
)
_EXACT_LFM_25_8B_A1B_MODELS = (
    "lfm2.5-8b-a1b",
    "liquid/lfm2.5-8b-a1b",
)


def is_gemma_model_name(model_name: str) -> bool:
    lowered = str(model_name or "").strip().lower()
    return any(marker in lowered for marker in _GEMMA_MODEL_MARKERS)


def is_exact_small_gemma_4_it_model_name(model_name: str) -> bool:
    lowered = str(model_name or "").strip().lower()
    return any(lowered.endswith(suffix) for suffix in _EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES)


def is_exact_large_gemma_4_26b_a4b_it_model_name(model_name: str) -> bool:
    lowered = str(model_name or "").strip().lower()
    return any(lowered.endswith(suffix) for suffix in _EXACT_GEMMA_4_26B_A4B_IT_MODEL_SUFFIXES)


def is_lfm25_8b_a1b_model_name(model_name: str) -> bool:
    lowered = str(model_name or "").strip().lower()
    return lowered in _EXACT_LFM_25_8B_A1B_MODELS
