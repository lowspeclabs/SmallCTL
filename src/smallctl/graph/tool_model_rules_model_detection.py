from __future__ import annotations

import re

from ..normalization import collapse_model_name

_GLM_BOX_MODEL_MARKERS = (
    "zai-org/glm-4.6v-flash",
    "glm-4.6v-flash",
    "zai-org/glm",
)
_GPT_OSS_MODEL_MARKERS = (
    "openai/gpt-oss-20b",
    "gpt-oss-20b",
    "openai/gpt-oss",
)
_QWEN_MODEL_MARKERS = (
    "qwen/",
    "qwen2.5",
    "qwen-2.5",
    "qwen3",
    "qwen-3",
    "qwen3.5",
    "qwen-3.5",
)
_EXACT_QWEN_25_7B_INSTRUCT_MODELS = (
    "qwen/qwen-2.5-7b-instruct",
    "qwen-2.5-7b-instruct",
    "qwen2.5-7b-instruct",
)
_EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
    # Bare quantized suffixes are used by some backends/gguf filenames that
    # drop the "-it" token even though the checkpoint is instruction-tuned.
    "gemma-4-e2b",
    "gemma-4-e4b",
)
_GEMMA_MODEL_MARKERS = (
    "google_gemma-4",
    "google_gemma",
    "gemma-4",
    "gemma-3",
    "gemma/",
)
_EXACT_LFM_25_8B_A1B_MODELS = (
    "lfm2.5-8b-a1b",
    "liquid/lfm2.5-8b-a1b",
)


def _model_uses_glm_box_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _GLM_BOX_MODEL_MARKERS))


def _model_uses_gpt_oss_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _GPT_OSS_MODEL_MARKERS))


def _model_uses_gemma_rules(model_name: str | None) -> bool:
    normalized = collapse_model_name(model_name)
    return bool(normalized and any(marker in normalized for marker in _GEMMA_MODEL_MARKERS))


def _model_uses_qwen_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _QWEN_MODEL_MARKERS))


def _model_is_exact_qwen_25_7b_instruct(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in _EXACT_QWEN_25_7B_INSTRUCT_MODELS


def _model_is_exact_small_gemma_4_it(model_name: str | None) -> bool:
    normalized = collapse_model_name(model_name)
    if not normalized:
        return False
    for suffix in _EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES:
        if (
            normalized == suffix
            or normalized.startswith(f"{suffix}-")
            or normalized.endswith(f"-{suffix}")
            or re.search(rf"(?:^|-){re.escape(suffix)}(?:$|-)", normalized)
        ):
            return True
    return False


def _model_is_lfm25_8b_a1b(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in _EXACT_LFM_25_8B_A1B_MODELS


def _model_is_gemma_4(model_name: str | None) -> bool:
    """Return True for any Gemma-4 sized variant (small/12b/27b/etc).

    The existing `_model_is_exact_small_gemma_4_it` matcher is narrowly tuned
    to the e2b/e4b small instruction checkpoints.  Larger Gemma-4 variants
    such as 12b or 27b are reported by backends with names like
    ``Gemma 4 12b`` or ``gemma-4-12b-it`` and need the same reasoning-stream
    and parsing accommodations, so this broader matcher covers the whole
    Gemma-4 family.
    """
    normalized = collapse_model_name(model_name)
    if not normalized:
        return False
    return "gemma-4" in normalized


def _model_is_gemma_4_small(model_name: str | None) -> bool:
    """Return True for Gemma-4 variants at 12b and below.

    These smaller checkpoints (e.g. ``gemma-4-12b``, ``Gemma 4 12b``,
    ``gemma-4-12b-it``) are more prone to getting stuck emitting native
    reasoning tokens and do not recover with the wider reasoning budgets
    used for larger variants.  The 27b checkpoint and exact-small IT
    variants are excluded from this matcher.
    """
    normalized = collapse_model_name(model_name)
    if not normalized:
        return False
    if not _model_is_gemma_4(model_name):
        return False
    # Exclude the larger 27b variant and the exact-small e2b/e4b IT checkpoints,
    # which have their own dedicated handling.
    if "gemma-4-27b" in normalized or "gemma-4-27-b" in normalized:
        return False
    if _model_is_exact_small_gemma_4_it(model_name):
        return False
    # Match 12b and smaller explicit size tokens (1b, 2b, 4b, 8b, 12b), or a
    # bare instruction-tuned suffix with no size which is assumed to be the
    # common 12b instruction checkpoint.
    return bool(
        re.search(r"gemma-4-(?:1[0-2]b|[1-9]b|it)(?:-|$)", normalized)
        or normalized == "gemma-4-it"
        or normalized == "gemma-4"
    )
