from __future__ import annotations

from typing import Any

from ..client.request_budget import build_request_budget


_PROVIDER_REQUEST_BUDGET_PROFILES = {"llamacpp"}


def resolve_effective_prompt_budget(
    *,
    configured_max_prompt_tokens: int | None,
    configured_max_prompt_tokens_explicit: bool = True,
    server_context_limit: int | None,
    current_max_prompt_tokens: int | None = None,
    observed_n_keep: int | None = None,
    provider_profile: str | None = None,
) -> int | None:
    candidates: list[int] = []
    if configured_max_prompt_tokens is not None and configured_max_prompt_tokens_explicit:
        candidates.append(max(64, int(configured_max_prompt_tokens)))

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
    effective_max_prompt_tokens = resolve_effective_prompt_budget(
        configured_max_prompt_tokens=configured_budget,
        configured_max_prompt_tokens_explicit=configured_budget_explicit,
        server_context_limit=harness.server_context_limit,
        current_max_prompt_tokens=harness.context_policy.max_prompt_tokens,
        observed_n_keep=observed_n_keep if overflow_detected else None,
        provider_profile=getattr(harness, "provider_profile", None),
    )
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
