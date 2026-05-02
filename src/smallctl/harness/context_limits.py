from __future__ import annotations

from typing import Any


def resolve_effective_prompt_budget(
    *,
    configured_max_prompt_tokens: int | None,
    configured_max_prompt_tokens_explicit: bool = True,
    server_context_limit: int | None,
    current_max_prompt_tokens: int | None = None,
    observed_n_keep: int | None = None,
) -> int | None:
    candidates: list[int] = []
    if configured_max_prompt_tokens is not None and configured_max_prompt_tokens_explicit:
        candidates.append(max(64, int(configured_max_prompt_tokens)))

    derived_server_budget = derive_prompt_budget_from_context_limit(server_context_limit)
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
    if harness.server_context_limit is None or normalized_limit < harness.server_context_limit:
        harness.discovered_server_context_limit = normalized_limit
        harness.server_context_limit = normalized_limit

    configured_budget = getattr(harness, "configured_max_prompt_tokens", None)
    configured_budget_explicit = bool(getattr(harness, "configured_max_prompt_tokens_explicit", True))
    overflow_detected = (
        observed_n_keep is not None
        and harness.server_context_limit is not None
        and observed_n_keep >= harness.server_context_limit
    )
    preserve_configured_budget = (
        configured_budget is not None
        and configured_budget_explicit
        and not overflow_detected
    )
    if preserve_configured_budget:
        effective_max_prompt_tokens = max(64, int(configured_budget))
    else:
        effective_max_prompt_tokens = resolve_effective_prompt_budget(
            configured_max_prompt_tokens=configured_budget,
            configured_max_prompt_tokens_explicit=configured_budget_explicit,
            server_context_limit=harness.server_context_limit,
            current_max_prompt_tokens=harness.context_policy.max_prompt_tokens,
            observed_n_keep=observed_n_keep,
        )
    if effective_max_prompt_tokens is not None:
        harness.context_policy.max_prompt_tokens = effective_max_prompt_tokens

    partition_context = harness.context_policy.max_prompt_tokens or harness.server_context_limit
    if partition_context is not None:
        harness.context_policy.recalculate_quotas(
            partition_context,
            backend_profile=harness.provider_profile,
        )
        model_name = getattr(getattr(harness, "client", None), "model", None)
        harness.context_policy.apply_model_profile(model_name)
        harness.context_policy.recalculate_quotas(partition_context)
        harness.state.recent_message_limit = harness.context_policy.recent_message_limit
        harness._harness_kwargs["context_limit"] = harness.server_context_limit

    harness._runlog(
        "context_limit",
        "server context limit applied",
        source=source,
        context_limit=harness.server_context_limit,
        configured_max_prompt_tokens=configured_budget,
        max_prompt_tokens=harness.context_policy.max_prompt_tokens,
        partition_context=partition_context,
        preserved_configured_budget=preserve_configured_budget,
        hot_message_limit=harness.context_policy.hot_message_limit,
        recent_message_limit=harness.context_policy.recent_message_limit,
        observed_n_keep=observed_n_keep,
    )
    return harness.context_policy.max_prompt_tokens


def derive_prompt_budget_from_context_limit(context_limit: int | None) -> int | None:
    if context_limit is None:
        return None
    limit = max(1, int(context_limit))
    # Use a fixed reserve for completion/tools rather than a percentage,
    # so large context windows are not disproportionately penalized.
    reserve = min(2048, limit // 4)
    return max(64, limit - reserve)
