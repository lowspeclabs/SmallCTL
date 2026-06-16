from __future__ import annotations

import logging
from dataclasses import MISSING, fields
from typing import Any

from .config import SmallctlConfig
from .harness.config import HarnessConfig

# Fields that exist in HarnessConfig but are intentionally not in SmallctlConfig.
# These are either runtime-only, derived, or injected by the caller.
HARNESS_ONLY_FIELDS = {
    "allow_interactive_shell_approval",
    "artifact_read_inline_token_limit",
    "artifact_start_index",
    "max_repair_steps",
    "repair_phase_max_steps",
    "policy",
    "run_logger",
    "shell_approval_session_default",
    "strategy",
    "strategy_prompt",
    "tool_result_inline_token_limit",
}

# Fields that exist in SmallctlConfig but are local-only and should never reach Harness.
LOCAL_ONLY_FIELDS = {
    "cleanup",
    "compatibility_warnings",
    "config_path",
    "debug",
    "graph_thread_id",
    "log_file",
    "preset",
    "restore_graph_state",
    "shell_preflight_block_destructive",
    "shell_preflight_cache_enabled",
    "shell_preflight_enabled",
    "shell_preflight_level",
    "shell_preflight_require_approval_for_package_changes",
    "shell_preflight_require_approval_for_service_changes",
    "shell_preflight_transparent",
    "shell_preflight_warn_large_output",
    "show_system_messages",
    "staged_reasoning",
    "no_fama",
    "task",
    "tui",
}


def project_config_to_harness_kwargs(
    config: SmallctlConfig,
    *,
    run_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Project a SmallctlConfig onto HarnessConfig kwargs.

    This uses reflection to automatically map every field that exists in both
    dataclasses, then applies explicit overrides for derived / injected values.

    Both TUI and non-TUI paths must use this single projection to prevent
    feature-parity regressions.
    """
    smallctl_names = {f.name for f in fields(SmallctlConfig)}
    harness_names = {f.name for f in fields(HarnessConfig)}

    harness_field_map = {f.name: f for f in fields(HarnessConfig)}
    kwargs: dict[str, Any] = {}

    # Automatic 1:1 mapping for fields present in both configs.
    # When the config object is a mock (e.g. SimpleNamespace in tests) that
    # lacks a field, fall back to the HarnessConfig default so the projection
    # stays robust.
    for name in harness_names:
        if name in HARNESS_ONLY_FIELDS:
            continue
        if name not in smallctl_names:
            raise ValueError(
                f"HarnessConfig field '{name}' is missing from SmallctlConfig. "
                f"Add it to SmallctlConfig or to HARNESS_ONLY_FIELDS with a default."
            )
        if hasattr(config, name):
            kwargs[name] = getattr(config, name)
        else:
            field = harness_field_map[name]
            if field.default is not MISSING:
                kwargs[name] = field.default
            elif field.default_factory is not MISSING:
                kwargs[name] = field.default_factory()
            else:
                raise ValueError(
                    f"Required config field '{name}' is missing from {type(config).__name__}"
                )

    # Explicit overrides for derived / injected values
    if run_logger is not None:
        kwargs["run_logger"] = run_logger

    if hasattr(config, "staged_reasoning"):
        strategy = {"thought_architecture": "staged_reasoning"} if config.staged_reasoning else None
    else:
        strategy = None
    kwargs["strategy"] = strategy

    max_prompt_tokens_explicit = bool(
        getattr(config, "max_prompt_tokens_explicit", getattr(config, "max_prompt_tokens", None) is not None)
    )
    kwargs["max_prompt_tokens_explicit"] = max_prompt_tokens_explicit

    return kwargs
