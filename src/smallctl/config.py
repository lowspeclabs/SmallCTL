from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .config_support import (
    LOCAL_CONFIG,
    _env_config,
    _normalize_graph_checkpointer,
    _read_yaml,
    _to_bool,
    _to_float,
    _to_int,
    _to_int_allow_zero,
)
from .phases import normalize_phase
from .presets import get_preset_defaults
from .provider_profiles import resolve_provider_profile


PROVIDER_PROFILES: dict[str, dict[str, Any]] = {
    "auto": {"runtime_context_probe": True},
    "generic": {"runtime_context_probe": True},
    "openai": {
        "chat_endpoint": "/chat/completions",
        "reasoning_mode": "off",
        "runtime_context_probe": False,
    },
    "ollama": {
        "chat_endpoint": "/chat/completions",
        "reasoning_mode": "off",
        "runtime_context_probe": False,
    },
    "vllm": {
        "chat_endpoint": "/chat/completions",
        "reasoning_mode": "field",
        "runtime_context_probe": False,
    },
    "lmstudio": {
        "chat_endpoint": "/chat/completions",
        "reasoning_mode": "tags",
        "runtime_context_probe": False,
    },
    "openrouter": {
        "chat_endpoint": "/chat/completions",
        "reasoning_mode": "off",
        "runtime_context_probe": True,
    },
    "llamacpp": {
        "chat_endpoint": "/chat/completions",
        "runtime_context_probe": True,
    },
}


@dataclass
class SmallctlConfig:
    endpoint: str = "http://localhost:8000/v1"
    model: str = "qwen3.5:4b"
    phase: str = "explore"
    provider_profile: str = "generic"
    indexer: bool = False
    tool_profiles: list[str] | None = None
    reasoning_mode: str = "auto"
    thinking_visibility: bool = True
    thinking_start_tag: str = "<think>"
    thinking_end_tag: str = "</think>"
    chat_endpoint: str = "/chat/completions"
    checkpoint_on_exit: bool = False
    checkpoint_path: str | None = None
    graph_checkpointer: str = "memory"
    graph_checkpoint_path: str | None = None
    restore_graph_state: bool = False
    graph_thread_id: str | None = None
    fresh_run: bool = False
    fresh_run_turns: int = 1
    planning_mode: bool = False
    contract_flow_ui: bool = False
    staged_reasoning: bool = False
    staged_execution_enabled: bool = False
    staged_step_prompt_tokens: int = 4096
    log_file: str | None = None
    debug: bool = False
    cleanup: bool = False
    tui: bool = False
    task: str | None = None
    config_path: str | None = None
    preset: str | None = None
    api_key: str | None = "local-dev-key"
    context_limit: int | None = None
    max_prompt_tokens: int | None = None
    max_prompt_tokens_explicit: bool = False
    reserve_completion_tokens: int = 1024
    reserve_tool_tokens: int = 512
    first_token_timeout_sec: int | None = None
    healthcheck_url: str | None = None
    restart_command: str | None = None
    startup_grace_period_sec: int = 20
    max_restarts_per_hour: int = 2
    backend_healthcheck_url: str | None = None
    backend_restart_command: str | None = None
    backend_unload_command: str | None = None
    backend_healthcheck_timeout_sec: int = 5
    backend_restart_grace_sec: int = 20
    summarize_at_ratio: float = 0.8
    recent_message_limit: int = 6
    max_summary_items: int = 3
    max_artifact_snippets: int = 4
    artifact_snippet_token_limit: int = 400
    multi_file_artifact_snippet_limit: int = 8
    multi_file_primary_file_limit: int = 3
    remote_task_artifact_snippet_limit: int = 8
    remote_task_primary_file_limit: int = 2
    compatibility_warnings: list[str] = field(default_factory=list)
    runtime_context_probe: bool = True
    summarizer_endpoint: str | None = None
    summarizer_model: str | None = None
    summarizer_api_key: str | None = None
    min_exploration_steps: int = 1
    artifact_summarization_threshold: int = 1200
    chunk_mode_min_bytes: int = 4096
    chunk_mode_new_file_only: bool = True
    chunk_mode_supported_models: list[str] = field(default_factory=lambda: ["qwen3.5", "llama3.1", "deepseek-v3"])
    small_model_soft_write_chars: int = 2000
    small_model_hard_write_chars: int = 4000
    new_file_chunk_mode_line_estimate: int = 100
    allow_multi_section_turns_for_small_edits: bool = True
    failed_local_patch_limit: int = 2
    enable_write_intent_recovery: bool = True
    enable_assistant_code_write_recovery: bool = True
    write_recovery_min_confidence: str = "high"
    write_recovery_allow_raw_text_targets: bool = True
    loop_guard_enabled: bool = True
    loop_guard_stagnation_threshold: int = 3
    loop_guard_level2_threshold: int = 5
    loop_guard_recent_writes_limit: int = 5
    loop_guard_tail_lines: int = 50
    loop_guard_similarity_threshold: float = 0.9
    loop_guard_cumulative_write_gate: bool = True
    loop_guard_checkpoint_gate: bool = True
    loop_guard_diff_gate: bool = True
    fama_enabled: bool = True
    fama_mode: str = "lite"
    fama_default_ttl_steps: int = 2
    fama_max_active_mitigations: int = 2
    fama_signal_window: int = 8
    fama_done_gate_on_failure: bool = True
    fama_capsule_token_budget: int = 180
    fama_llm_judge_enabled: bool = False
    fama_llm_judge_min_severity: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_config(cli: dict[str, Any]) -> SmallctlConfig:
    # Source precedence (before provider defaults):
    # user config path -> local config -> env/.env -> CLI
    env_cfg = _env_config()
    user_cfg: dict[str, Any] = {}
    local_cfg: dict[str, Any] = {}

    user_config_path = cli.get("config_path") or env_cfg.get("config_path")
    if user_config_path:
        user_cfg = _read_yaml(Path(user_config_path))
        user_cfg["config_path"] = user_config_path
    local_cfg = _read_yaml(Path.cwd() / LOCAL_CONFIG)

    merged: dict[str, Any] = {}
    if "provider_profile" not in merged:
        merged["provider_profile"] = "auto"

    cli_clean = {k: v for k, v in cli.items() if v is not None}
    if "tool_profiles" in cli_clean:
        from .tools.profiles import parse_public_profiles

        cli_clean["tool_profiles"] = parse_public_profiles(cli_clean["tool_profiles"])
    if "debug" in cli_clean:
        cli_clean["debug"] = _to_bool(cli_clean["debug"])
    if "thinking_visibility" in cli_clean:
        cli_clean["thinking_visibility"] = _to_bool(cli_clean["thinking_visibility"])
    if "checkpoint_on_exit" in cli_clean:
        cli_clean["checkpoint_on_exit"] = _to_bool(cli_clean["checkpoint_on_exit"])
    if "restore_graph_state" in cli_clean:
        cli_clean["restore_graph_state"] = _to_bool(cli_clean["restore_graph_state"])
    if "fresh_run" in cli_clean:
        cli_clean["fresh_run"] = _to_bool(cli_clean["fresh_run"])
    if "fresh_run_turns" in cli_clean:
        cli_clean["fresh_run_turns"] = _to_int_allow_zero(cli_clean["fresh_run_turns"])
    if "planning_mode" in cli_clean:
        cli_clean["planning_mode"] = _to_bool(cli_clean["planning_mode"])
    if "contract_flow_ui" in cli_clean:
        cli_clean["contract_flow_ui"] = _to_bool(cli_clean["contract_flow_ui"])
    if "staged_reasoning" in cli_clean:
        cli_clean["staged_reasoning"] = _to_bool(cli_clean["staged_reasoning"])
    if "staged_execution_enabled" in cli_clean:
        cli_clean["staged_execution_enabled"] = _to_bool(cli_clean["staged_execution_enabled"])
    if "indexer" in cli_clean:
        cli_clean["indexer"] = _to_bool(cli_clean["indexer"])
    if "enable_write_intent_recovery" in cli_clean:
        cli_clean["enable_write_intent_recovery"] = _to_bool(cli_clean["enable_write_intent_recovery"])
    if "enable_assistant_code_write_recovery" in cli_clean:
        cli_clean["enable_assistant_code_write_recovery"] = _to_bool(cli_clean["enable_assistant_code_write_recovery"])
    if "write_recovery_allow_raw_text_targets" in cli_clean:
        cli_clean["write_recovery_allow_raw_text_targets"] = _to_bool(
            cli_clean["write_recovery_allow_raw_text_targets"]
        )
    for key in (
        "loop_guard_enabled",
        "loop_guard_cumulative_write_gate",
        "loop_guard_checkpoint_gate",
        "loop_guard_diff_gate",
        "fama_enabled",
        "fama_done_gate_on_failure",
        "fama_llm_judge_enabled",
    ):
        if key in cli_clean:
            cli_clean[key] = _to_bool(cli_clean[key])
    if "graph_checkpointer" in cli_clean:
        cli_clean["graph_checkpointer"] = _normalize_graph_checkpointer(cli_clean["graph_checkpointer"])
    if "graph_checkpoint_path" in cli_clean and "graph_checkpointer" not in cli_clean:
        cli_clean["graph_checkpointer"] = "file"
    if "context_limit" in cli_clean:
        parsed_limit = _to_int(cli_clean["context_limit"])
        if parsed_limit is None:
            cli_clean.pop("context_limit", None)
        else:
            cli_clean["context_limit"] = parsed_limit
    for key in (
        "max_prompt_tokens",
        "reserve_completion_tokens",
        "reserve_tool_tokens",
        "first_token_timeout_sec",
        "startup_grace_period_sec",
        "max_restarts_per_hour",
        "backend_healthcheck_timeout_sec",
        "backend_restart_grace_sec",
        "recent_message_limit",
        "max_summary_items",
        "max_artifact_snippets",
        "artifact_snippet_token_limit",
        "multi_file_artifact_snippet_limit",
        "multi_file_primary_file_limit",
        "remote_task_artifact_snippet_limit",
        "remote_task_primary_file_limit",
        "loop_guard_stagnation_threshold",
        "loop_guard_level2_threshold",
        "loop_guard_recent_writes_limit",
        "loop_guard_tail_lines",
        "staged_step_prompt_tokens",
        "fama_default_ttl_steps",
        "fama_max_active_mitigations",
        "fama_signal_window",
        "fama_capsule_token_budget",
        "fama_llm_judge_min_severity",
    ):
        if key in cli_clean:
            parsed_limit = _to_int(cli_clean[key])
            if parsed_limit is None:
                cli_clean.pop(key, None)
            else:
                cli_clean[key] = parsed_limit
    if "summarize_at_ratio" in cli_clean:
        parsed_ratio = _to_float(cli_clean["summarize_at_ratio"])
        if parsed_ratio is None:
            cli_clean.pop("summarize_at_ratio", None)
        else:
            cli_clean["summarize_at_ratio"] = parsed_ratio
    if "loop_guard_similarity_threshold" in cli_clean:
        parsed_ratio = _to_float(cli_clean["loop_guard_similarity_threshold"])
        if parsed_ratio is None:
            cli_clean.pop("loop_guard_similarity_threshold", None)
        else:
            cli_clean["loop_guard_similarity_threshold"] = parsed_ratio
    if "healthcheck_url" not in cli_clean and "backend_healthcheck_url" in cli_clean:
        cli_clean["healthcheck_url"] = cli_clean["backend_healthcheck_url"]
    if "restart_command" not in cli_clean and "backend_restart_command" in cli_clean:
        cli_clean["restart_command"] = cli_clean["backend_restart_command"]
    if "startup_grace_period_sec" not in cli_clean and "backend_restart_grace_sec" in cli_clean:
        cli_clean["startup_grace_period_sec"] = cli_clean["backend_restart_grace_sec"]

    explicit_merged: dict[str, Any] = {}
    explicit_merged.update(user_cfg)
    explicit_merged.update(local_cfg)
    explicit_merged.update(env_cfg)
    explicit_merged.update(cli_clean)
    preset_name = str(explicit_merged.get("preset") or "").strip().lower() or None
    preset_defaults = get_preset_defaults(preset_name)

    if preset_defaults:
        merged.update(preset_defaults)
    merged.update(user_cfg)
    merged.update(local_cfg)
    merged.update(env_cfg)
    merged.update(cli_clean)
    if preset_name:
        merged["preset"] = preset_name
    if merged.get("staged_reasoning") and "staged_execution_enabled" not in explicit_merged:
        merged["staged_execution_enabled"] = True
    for key in ("fama_enabled", "fama_done_gate_on_failure", "fama_llm_judge_enabled"):
        if key in merged:
            merged[key] = _to_bool(merged[key])
    for key in (
        "fama_default_ttl_steps",
        "fama_max_active_mitigations",
        "fama_signal_window",
        "fama_capsule_token_budget",
        "fama_llm_judge_min_severity",
    ):
        if key in merged:
            parsed_limit = _to_int(merged[key])
            if parsed_limit is None:
                merged.pop(key, None)
            else:
                merged[key] = parsed_limit

    explicit_prompt_budget = "max_prompt_tokens" in explicit_merged
    merged["max_prompt_tokens_explicit"] = explicit_prompt_budget
    if not explicit_prompt_budget and "context_limit" in merged:
        merged["max_prompt_tokens"] = merged["context_limit"]
    compatibility_warnings: list[str] = []
    if "context_limit" in merged and not explicit_prompt_budget:
        compatibility_warnings.append(
            "Legacy SMALLCTL_CONTEXT_LIMIT is being treated as SMALLCTL_MAX_PROMPT_TOKENS."
        )
    elif "context_limit" in merged and explicit_prompt_budget:
        compatibility_warnings.append(
            "Legacy SMALLCTL_CONTEXT_LIMIT remains set; prefer SMALLCTL_MAX_PROMPT_TOKENS for per-turn budgeting."
        )
    if preset_name and not preset_defaults:
        compatibility_warnings.append(f"Unknown preset '{preset_name}' ignored.")
    merged["phase"] = normalize_phase(str(merged.get("phase", "explore")))
    merged["graph_checkpointer"] = _normalize_graph_checkpointer(merged.get("graph_checkpointer", "memory"))
    if merged.get("graph_checkpoint_path") and merged["graph_checkpointer"] == "memory":
        merged["graph_checkpointer"] = "file"
    if merged.get("fresh_run"):
        merged["restore_graph_state"] = False
    _apply_provider_profile(merged, cli_clean, compatibility_warnings)
    profile = str(merged.get("provider_profile", "auto")).strip().lower()
    if profile == "lmstudio":
        if not str(merged.get("backend_unload_command") or "").strip() and not str(
            merged.get("backend_restart_command") or ""
        ).strip():
            compatibility_warnings.append(
                "LM Studio native API unload will be attempted automatically; configure SMALLCTL_BACKEND_RESTART_COMMAND if you want restart fallback when unload does not recover."
            )
        configured_timeout = merged.get("first_token_timeout_sec")
        if configured_timeout is not None:
            try:
                normalized_timeout = float(configured_timeout)
            except (TypeError, ValueError):
                normalized_timeout = 0.0
            if 0.0 < normalized_timeout < 45.0:
                compatibility_warnings.append(
                    "LM Studio first_token_timeout_sec below 45s may cause false 'backend wedged' detections on slower local generations."
                )
    merged["compatibility_warnings"] = compatibility_warnings

    allowed_keys = {f.name for f in fields(SmallctlConfig)}
    unknown_keys = sorted(k for k in merged if k not in allowed_keys)
    if unknown_keys:
        compatibility_warnings.append(f"Ignoring unsupported config keys: {', '.join(unknown_keys)}.")
    filtered = {k: v for k, v in merged.items() if k in allowed_keys}
    return SmallctlConfig(**filtered)


def _apply_provider_profile(
    merged: dict[str, Any],
    cli_clean: dict[str, Any],
    compatibility_warnings: list[str],
) -> None:
    profile, profile_warnings = resolve_provider_profile(
        merged.get("endpoint"),
        merged.get("model"),
        merged.get("provider_profile", "auto"),
    )
    if profile_warnings:
        compatibility_warnings.extend(profile_warnings)
    merged["provider_profile"] = profile
    defaults = PROVIDER_PROFILES.get(profile, PROVIDER_PROFILES["generic"])
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
    merged.update({k: v for k, v in cli_clean.items() if k in defaults})
    merged["provider_profile"] = profile
