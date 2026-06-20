from __future__ import annotations

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
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _GEMMA_MODEL_MARKERS))


def _model_uses_qwen_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _QWEN_MODEL_MARKERS))


def _model_is_exact_qwen_25_7b_instruct(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in _EXACT_QWEN_25_7B_INSTRUCT_MODELS


def _model_is_exact_small_gemma_4_it(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(
        normalized
        and any(
            normalized == suffix or normalized.endswith(f"/{suffix}")
            for suffix in _EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES
        )
    )


def _model_is_lfm25_8b_a1b(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in _EXACT_LFM_25_8B_A1B_MODELS
