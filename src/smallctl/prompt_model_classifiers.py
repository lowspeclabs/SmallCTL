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
# User-facing and backend-agnostic slugs for Gemma-4 instruction-tuned
# checkpoints that do not always include the "-it" token (e.g. "Gemma 4 12b"
# or "gemma-4-12b" served by llama.cpp).  Treating them as IT lets the
# harness keep the <think>-tag reasoning protocol enabled instead of falling
# back to reasoning=off.
_KNOWN_GEMMA_4_IT_BASE_SUFFIXES = (
    "gemma-4-12b",
    "gemma-4-27b",
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

    Covers the known small IT suffixes (e2b/e4b), the 26b A4B IT variant, the
    31b IT variant observed on OpenRouter, and common user/backend slugs such
    as ``gemma-4-12b`` where the "-it" token is omitted.  Using the broader
    heuristic lets the harness treat future Gemma-4 IT checkpoints the same
    way without waiting for an exact allow-list update.
    """
    normalized = collapse_model_name(model_name)
    if "gemma-4-" not in normalized:
        return False
    if normalized.endswith("-it"):
        return True
    return _matches_any_suffix(normalized, _KNOWN_GEMMA_4_IT_BASE_SUFFIXES)


def is_lfm25_8b_a1b_model_name(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in _EXACT_LFM_25_8B_A1B_MODELS


def is_gemma_4_non_exact_small_model_name(model_name: str) -> bool:
    """True for Gemma-4 variants that are NOT the known-good small IT checkpoints.

    Larger/non-IT Gemma-4 variants (e.g. 12b, 27b) served by backends such as
    llama.cpp tend to emit native reasoning channels without producing visible
    assistant content or tool calls.  Planning-mode auto-escalation hurts for
    these models because they get stuck re-reasoning about the plan instead of
    acting, so the harness treats them like small models and keeps them in loop
    mode where reasoning-channel recovery is more robust.
    """
    normalized = collapse_model_name(model_name)
    if "gemma-4" not in normalized:
        return False
    return not is_exact_small_gemma_4_it_model_name(model_name)
