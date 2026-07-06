from __future__ import annotations

import logging
import re
from typing import Any

from ..client.llamacpp_preflight import _is_swa_model
from ..client.request_budget import build_request_budget
from ..normalization import collapse_model_name


_PROVIDER_REQUEST_BUDGET_PROFILES = {"llamacpp"}

# Best-effort context ceilings for common model families. These act as a safety
# rail when the backend does not advertise a context limit and the user asks for
# a prompt budget that exceeds the model's known window.
_MODEL_NAME_CONTEXT_CEILINGS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"qwen3(\.5)?\b", re.IGNORECASE), 128_000),
    (re.compile(r"qwen2(\.5)?\b", re.IGNORECASE), 128_000),
    (re.compile(r"llama[-_]?3(\.[123])?\b", re.IGNORECASE), 128_000),
    (re.compile(r"llama[-_]?3\b", re.IGNORECASE), 8_192),
    (re.compile(r"gemma[-_]?(2|3)\b", re.IGNORECASE), 128_000),
    (re.compile(r"gemma[-_]?4\b", re.IGNORECASE), 128_000),
    (re.compile(r"phi[-_]?(3|4)\b", re.IGNORECASE), 128_000),
    (re.compile(r"mistral[-_]?(small|large)?\b", re.IGNORECASE), 32_768),
    (re.compile(r"mixtral\b", re.IGNORECASE), 32_768),
    (re.compile(r"deepseek[-_]?v3\b", re.IGNORECASE), 64_000),
    (re.compile(r"deepseek[-_]?r1\b", re.IGNORECASE), 64_000),
    (re.compile(r"deepseek[-_]?v4\b", re.IGNORECASE), 1_000_000),
    (re.compile(r"command[-_]?r\b", re.IGNORECASE), 128_000),
]


def _derive_context_limit_from_model_name(model_name: str | None) -> int | None:
    if not model_name:
        return None
    text = collapse_model_name(model_name)
    if not text:
        return None
    for pattern, limit in _MODEL_NAME_CONTEXT_CEILINGS:
        if pattern.search(text):
            return limit
    return None


def resolve_effective_prompt_budget(
    *,
    configured_max_prompt_tokens: int | None,
    configured_max_prompt_tokens_explicit: bool = True,
    server_context_limit: int | None,
    current_max_prompt_tokens: int | None = None,
    observed_n_keep: int | None = None,
    provider_profile: str | None = None,
    model_name: str | None = None,
) -> int | None:
    candidates: list[int] = []
    if configured_max_prompt_tokens is not None and configured_max_prompt_tokens_explicit:
        candidates.append(max(64, int(configured_max_prompt_tokens)))

        # Only cap an explicit user-configured prompt budget by a known model
        # context window. Avoid imposing a default ceiling when the user did not
        # ask for a specific budget.
        model_context_limit = _derive_context_limit_from_model_name(model_name)
        if model_context_limit is not None:
            candidates.append(model_context_limit)

    derived_server_budget = derive_prompt_budget_from_context_limit(
        server_context_limit,
        provider_profile=provider_profile,
    )
    if derived_server_budget is not None:
        candidates.append(derived_server_budget)

    if (
        current_max_prompt_tokens is not None
        and observed_n_keep is not None
        and server_context_limit is not None
        and observed_n_keep >= server_context_limit
    ):
        overflow = observed_n_keep - server_context_limit + 128
        candidates.append(max(64, int(current_max_prompt_tokens) - overflow))

    if not candidates:
        return None
    return min(candidates)


def apply_server_context_limit(
    harness: Any,
    context_limit: int | None,
    *,
    source: str,
    observed_n_keep: int | None = None,
) -> int | None:
    if context_limit is None:
        return harness.context_policy.max_prompt_tokens

    normalized_limit = max(1, int(context_limit))
    # Runtime probes reflect the server that is actually listening now.  Allow
    # them to expand a stale lower limit after a local llama.cpp restart with a
    # larger --ctx-size, while still letting overflow reports shrink the budget.
    if (
        harness.server_context_limit is None
        or normalized_limit < harness.server_context_limit
        or source == "runtime_probe"
    ):
        harness.discovered_server_context_limit = normalized_limit
        harness.server_context_limit = normalized_limit

    configured_budget = getattr(harness, "configured_max_prompt_tokens", None)
    configured_budget_explicit = bool(getattr(harness, "configured_max_prompt_tokens_explicit", True))
    overflow_detected = (
        observed_n_keep is not None
        and harness.server_context_limit is not None
        and observed_n_keep >= harness.server_context_limit
    )
    model_name = getattr(getattr(harness, "client", None), "model", None)
    model_context_limit = _derive_context_limit_from_model_name(model_name)
    effective_max_prompt_tokens = resolve_effective_prompt_budget(
        configured_max_prompt_tokens=configured_budget,
        configured_max_prompt_tokens_explicit=configured_budget_explicit,
        server_context_limit=harness.server_context_limit,
        current_max_prompt_tokens=harness.context_policy.max_prompt_tokens,
        observed_n_keep=observed_n_keep if overflow_detected else None,
        provider_profile=getattr(harness, "provider_profile", None),
        model_name=model_name,
    )

    # Apply SWA-aware hard cap for small Gemma-4 on llama.cpp so each full
    # reprocess stays cheap even when prefix caching cannot be reused.
    if (
        _is_swa_model(model_name, getattr(harness, "provider_profile", None))
        and effective_max_prompt_tokens is not None
        and effective_max_prompt_tokens > harness.context_policy.swa_prompt_cap
    ):
        previous = effective_max_prompt_tokens
        effective_max_prompt_tokens = harness.context_policy.swa_prompt_cap
        harness._runlog(
            "swa_prompt_cap_applied",
            "capped prompt budget for SWA/hybrid-memory model",
            swa_prompt_cap=harness.context_policy.swa_prompt_cap,
            previous_budget=previous,
        )

    if (
        configured_budget is not None
        and configured_budget_explicit
        and effective_max_prompt_tokens is not None
        and effective_max_prompt_tokens < configured_budget
    ):
        logging.getLogger("smallctl.harness").warning(
            "Prompt budget reduced from configured %s to effective %s tokens "
            "(server context limit: %s). Leave more headroom for completion tokens or increase the backend context size.",
            configured_budget,
            effective_max_prompt_tokens,
            harness.server_context_limit,
        )

    if (
        configured_budget is not None
        and configured_budget_explicit
        and model_context_limit is not None
        and configured_budget > model_context_limit
    ):
        harness._runlog(
            "context_limit",
            "configured prompt budget exceeds known model context window",
            configured_max_prompt_tokens=configured_budget,
            model_context_limit=model_context_limit,
            effective_max_prompt_tokens=effective_max_prompt_tokens,
            model_name=model_name,
        )

    # Boost context budget for local coding tasks to reduce lane dropping
    from ..harness.task_classifier import task_is_local_coding_target
    task_text = ""
    run_brief = getattr(harness.state, "run_brief", None)
    if run_brief is not None:
        task_text = str(getattr(run_brief, "original_task", "") or "")
    if task_is_local_coding_target(task_text) and harness.server_context_limit is not None:
        local_coding_budget = int(harness.server_context_limit * 0.9)
        if effective_max_prompt_tokens is None or effective_max_prompt_tokens < local_coding_budget:
            effective_max_prompt_tokens = local_coding_budget

    if effective_max_prompt_tokens is not None:
        harness.context_policy.max_prompt_tokens = effective_max_prompt_tokens
    preserved_configured_budget = (
        configured_budget is not None
        and configured_budget_explicit
        and effective_max_prompt_tokens == max(64, int(configured_budget))
    )

    partition_context = harness.context_policy.max_prompt_tokens or harness.server_context_limit
    if partition_context is not None:
        harness.context_policy.recalculate_quotas(
            partition_context,
            backend_profile=getattr(harness, "provider_profile", None),
        )
        model_name = getattr(getattr(harness, "client", None), "model", None)
        harness.context_policy.apply_model_profile(model_name)
        harness.context_policy.recalculate_quotas(partition_context)
        harness.state.recent_message_limit = harness.context_policy.recent_message_limit
        harness.config.context_limit = harness.server_context_limit

    # One-time terminal banner so headless users see their context budget.
    # Skip in TUI mode (event_handler is set) to avoid corrupting the terminal.
    if (
        not getattr(harness, "_context_banner_printed", False)
        and getattr(harness, "event_handler", None) is None
    ):
        harness._context_banner_printed = True
        total_ctx = harness.server_context_limit or 0
        free_ctx = harness.context_policy.max_prompt_tokens or 0
        if total_ctx > 0:
            import sys

            sys.stderr.write(
                f"[CONTEXT] {free_ctx:,} / {total_ctx:,} tokens available "
                f"({free_ctx / total_ctx * 100:.0f}% free)\n"
            )
            sys.stderr.flush()

    harness._runlog(
        "context_limit",
        "server context limit applied",
        source=source,
        context_limit=harness.server_context_limit,
        configured_max_prompt_tokens=configured_budget,
        max_prompt_tokens=harness.context_policy.max_prompt_tokens,
        partition_context=partition_context,
        preserved_configured_budget=preserved_configured_budget,
        hot_message_limit=harness.context_policy.hot_message_limit,
        recent_message_limit=harness.context_policy.recent_message_limit,
        observed_n_keep=observed_n_keep,
    )
    return harness.context_policy.max_prompt_tokens


def derive_prompt_budget_from_context_limit(
    context_limit: int | None,
    *,
    provider_profile: str | None = None,
) -> int | None:
    if context_limit is None:
        return None
    limit = max(1, int(context_limit))
    profile = str(provider_profile or "").strip().lower()
    if profile in _PROVIDER_REQUEST_BUDGET_PROFILES:
        return build_request_budget(limit).effective_prompt_budget
    # Use a fixed reserve for completion/tools rather than a percentage,
    # so large context windows are not disproportionately penalized.
    reserve = min(4096, limit // 8)
    return max(64, limit - reserve)
