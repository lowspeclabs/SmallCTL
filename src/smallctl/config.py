from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .client import detect_provider_profile
from .phases import normalize_phase
from .tools.profiles import parse_public_profiles

try:
    import yaml
except Exception:  # pragma: no cover - fallback if dependency missing
    yaml = None


ENV_PREFIX = "SMALLCTL_"
LOCAL_CONFIG = ".smallctl.yaml"
PROVIDER_PROFILES: dict[str, dict[str, Any]] = {
    "auto": {
        "runtime_context_probe": True,
    },
    "generic": {
        "runtime_context_probe": True,
    },
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
    inventory: str | None = None
    tool_profiles: list[str] | None = None
    use_ansible: bool = True
    ansible_check_mode_in_plan: bool = True
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
    log_file: str | None = None
    debug: bool = False
    cleanup: bool = False
    tui: bool = False
    task: str | None = None
    config_path: str | None = None
    api_key: str | None = "local-dev-key"
    context_limit: int | None = None
    max_prompt_tokens: int | None = None
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _to_int_allow_zero(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to read YAML config files.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return data


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_graph_checkpointer(value: Any) -> str:
    backend = str(value or "memory").strip().lower()
    return backend if backend in {"memory", "file"} else "memory"


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        parsed[key] = value
    return parsed


def _env_config() -> dict[str, Any]:
    dotenv = _read_dotenv(Path.cwd() / ".env")
    env_or_dotenv = lambda key: os.getenv(key) or dotenv.get(key)
    raw = {
        "endpoint": env_or_dotenv(f"{ENV_PREFIX}ENDPOINT"),
        "model": env_or_dotenv(f"{ENV_PREFIX}MODEL"),
        "phase": env_or_dotenv(f"{ENV_PREFIX}PHASE"),
        "provider_profile": env_or_dotenv(f"{ENV_PREFIX}PROVIDER_PROFILE"),
        "inventory": env_or_dotenv(f"{ENV_PREFIX}INVENTORY"),
        "tool_profiles": env_or_dotenv(f"{ENV_PREFIX}TOOL_PROFILES"),
        "use_ansible": env_or_dotenv(f"{ENV_PREFIX}USE_ANSIBLE"),
        "ansible_check_mode_in_plan": env_or_dotenv(f"{ENV_PREFIX}ANSIBLE_CHECK_MODE_IN_PLAN"),
        "reasoning_mode": env_or_dotenv(f"{ENV_PREFIX}REASONING_MODE"),
        "thinking_visibility": env_or_dotenv(f"{ENV_PREFIX}THINKING_VISIBILITY"),
        "thinking_start_tag": env_or_dotenv(f"{ENV_PREFIX}THINKING_START_TAG"),
        "thinking_end_tag": env_or_dotenv(f"{ENV_PREFIX}THINKING_END_TAG"),
        "chat_endpoint": env_or_dotenv(f"{ENV_PREFIX}CHAT_ENDPOINT"),
        "checkpoint_on_exit": env_or_dotenv(f"{ENV_PREFIX}CHECKPOINT_ON_EXIT"),
        "checkpoint_path": env_or_dotenv(f"{ENV_PREFIX}CHECKPOINT_PATH"),
        "graph_checkpointer": env_or_dotenv(f"{ENV_PREFIX}GRAPH_CHECKPOINTER"),
        "graph_checkpoint_path": env_or_dotenv(f"{ENV_PREFIX}GRAPH_CHECKPOINT_PATH"),
        "restore_graph_state": env_or_dotenv(f"{ENV_PREFIX}RESTORE_GRAPH_STATE"),
        "graph_thread_id": env_or_dotenv(f"{ENV_PREFIX}GRAPH_THREAD_ID"),
        "fresh_run": env_or_dotenv(f"{ENV_PREFIX}FRESH_RUN"),
        "fresh_run_turns": env_or_dotenv(f"{ENV_PREFIX}FRESH_RUN_TURNS"),
        "planning_mode": env_or_dotenv(f"{ENV_PREFIX}PLANNING_MODE"),
        "contract_flow_ui": env_or_dotenv(f"{ENV_PREFIX}CONTRACT_FLOW_UI"),
        "log_file": env_or_dotenv(f"{ENV_PREFIX}LOG_FILE"),
        "debug": env_or_dotenv(f"{ENV_PREFIX}DEBUG"),
        "config_path": env_or_dotenv(f"{ENV_PREFIX}CONFIG"),
        "indexer": env_or_dotenv(f"{ENV_PREFIX}INDEXER"),
        "api_key": env_or_dotenv(f"{ENV_PREFIX}API_KEY"),
        "context_limit": env_or_dotenv(f"{ENV_PREFIX}CONTEXT_LIMIT"),
        "max_prompt_tokens": env_or_dotenv(f"{ENV_PREFIX}MAX_PROMPT_TOKENS"),
        "reserve_completion_tokens": env_or_dotenv(f"{ENV_PREFIX}RESERVE_COMPLETION_TOKENS"),
        "reserve_tool_tokens": env_or_dotenv(f"{ENV_PREFIX}RESERVE_TOOL_TOKENS"),
        "first_token_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}FIRST_TOKEN_TIMEOUT_SEC"),
        "healthcheck_url": env_or_dotenv(f"{ENV_PREFIX}HEALTHCHECK_URL"),
        "restart_command": env_or_dotenv(f"{ENV_PREFIX}RESTART_COMMAND"),
        "startup_grace_period_sec": env_or_dotenv(f"{ENV_PREFIX}STARTUP_GRACE_PERIOD_SEC"),
        "max_restarts_per_hour": env_or_dotenv(f"{ENV_PREFIX}MAX_RESTARTS_PER_HOUR"),
        "backend_healthcheck_url": env_or_dotenv(f"{ENV_PREFIX}BACKEND_HEALTHCHECK_URL"),
        "backend_restart_command": env_or_dotenv(f"{ENV_PREFIX}BACKEND_RESTART_COMMAND"),
        "backend_unload_command": env_or_dotenv(f"{ENV_PREFIX}BACKEND_UNLOAD_COMMAND"),
        "backend_healthcheck_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}BACKEND_HEALTHCHECK_TIMEOUT_SEC"),
        "backend_restart_grace_sec": env_or_dotenv(f"{ENV_PREFIX}BACKEND_RESTART_GRACE_SEC"),
        "summarize_at_ratio": env_or_dotenv(f"{ENV_PREFIX}SUMMARIZE_AT_RATIO"),
        "recent_message_limit": env_or_dotenv(f"{ENV_PREFIX}RECENT_MESSAGE_LIMIT"),
        "max_summary_items": env_or_dotenv(f"{ENV_PREFIX}MAX_SUMMARY_ITEMS"),
        "max_artifact_snippets": env_or_dotenv(f"{ENV_PREFIX}MAX_ARTIFACT_SNIPPETS"),
        "artifact_snippet_token_limit": env_or_dotenv(f"{ENV_PREFIX}ARTIFACT_SNIPPET_TOKEN_LIMIT"),
        "summarizer_endpoint": env_or_dotenv(f"{ENV_PREFIX}SUMMARIZER_ENDPOINT"),
        "summarizer_model": env_or_dotenv(f"{ENV_PREFIX}SUMMARIZER_MODEL"),
        "summarizer_api_key": env_or_dotenv(f"{ENV_PREFIX}SUMMARIZER_API_KEY"),
        "min_exploration_steps": env_or_dotenv(f"{ENV_PREFIX}MIN_EXPLORATION_STEPS"),
        "artifact_summarization_threshold": env_or_dotenv(f"{ENV_PREFIX}ARTIFACT_SUMMARIZATION_THRESHOLD"),
        "chunk_mode_min_bytes": env_or_dotenv(f"{ENV_PREFIX}CHUNK_MODE_MIN_BYTES"),
        "chunk_mode_new_file_only": env_or_dotenv(f"{ENV_PREFIX}CHUNK_MODE_NEW_FILE_ONLY"),
        "chunk_mode_supported_models": env_or_dotenv(f"{ENV_PREFIX}CHUNK_MODE_SUPPORTED_MODELS"),
        "small_model_soft_write_chars": env_or_dotenv(f"{ENV_PREFIX}SMALL_MODEL_SOFT_WRITE_CHARS"),
        "small_model_hard_write_chars": env_or_dotenv(f"{ENV_PREFIX}SMALL_MODEL_HARD_WRITE_CHARS"),
        "new_file_chunk_mode_line_estimate": env_or_dotenv(f"{ENV_PREFIX}NEW_FILE_CHUNK_MODE_LINE_ESTIMATE"),
        "allow_multi_section_turns_for_small_edits": env_or_dotenv(f"{ENV_PREFIX}ALLOW_MULTI_SECTION_TURNS_FOR_SMALL_EDITS"),
        "failed_local_patch_limit": env_or_dotenv(f"{ENV_PREFIX}FAILED_LOCAL_PATCH_LIMIT"),
        "enable_write_intent_recovery": env_or_dotenv(f"{ENV_PREFIX}ENABLE_WRITE_INTENT_RECOVERY"),
        "enable_assistant_code_write_recovery": env_or_dotenv(f"{ENV_PREFIX}ENABLE_ASSISTANT_CODE_WRITE_RECOVERY"),
        "write_recovery_min_confidence": env_or_dotenv(f"{ENV_PREFIX}WRITE_RECOVERY_MIN_CONFIDENCE"),
        "write_recovery_allow_raw_text_targets": env_or_dotenv(f"{ENV_PREFIX}WRITE_RECOVERY_ALLOW_RAW_TEXT_TARGETS"),
    }
    cfg = {k: v for k, v in raw.items() if v not in (None, "")}
    if "use_ansible" in cfg:
        cfg["use_ansible"] = _to_bool(cfg["use_ansible"])
    if "tool_profiles" in cfg:
        cfg["tool_profiles"] = parse_public_profiles(cfg["tool_profiles"])
    if "debug" in cfg:
        cfg["debug"] = _to_bool(cfg["debug"])
    if "ansible_check_mode_in_plan" in cfg:
        cfg["ansible_check_mode_in_plan"] = _to_bool(cfg["ansible_check_mode_in_plan"])
    if "thinking_visibility" in cfg:
        cfg["thinking_visibility"] = _to_bool(cfg["thinking_visibility"])
    if "checkpoint_on_exit" in cfg:
        cfg["checkpoint_on_exit"] = _to_bool(cfg["checkpoint_on_exit"])
    if "restore_graph_state" in cfg:
        cfg["restore_graph_state"] = _to_bool(cfg["restore_graph_state"])
    if "fresh_run" in cfg:
        cfg["fresh_run"] = _to_bool(cfg["fresh_run"])
    if "fresh_run_turns" in cfg:
        cfg["fresh_run_turns"] = _to_int_allow_zero(cfg["fresh_run_turns"])
    if "planning_mode" in cfg:
        cfg["planning_mode"] = _to_bool(cfg["planning_mode"])
    if "contract_flow_ui" in cfg:
        cfg["contract_flow_ui"] = _to_bool(cfg["contract_flow_ui"])
    if "indexer" in cfg:
        cfg["indexer"] = _to_bool(cfg["indexer"])
    if "chunk_mode_new_file_only" in cfg:
        cfg["chunk_mode_new_file_only"] = _to_bool(cfg["chunk_mode_new_file_only"])
    if "allow_multi_section_turns_for_small_edits" in cfg:
        cfg["allow_multi_section_turns_for_small_edits"] = _to_bool(cfg["allow_multi_section_turns_for_small_edits"])
    if "enable_write_intent_recovery" in cfg:
        cfg["enable_write_intent_recovery"] = _to_bool(cfg["enable_write_intent_recovery"])
    if "enable_assistant_code_write_recovery" in cfg:
        cfg["enable_assistant_code_write_recovery"] = _to_bool(cfg["enable_assistant_code_write_recovery"])
    if "write_recovery_allow_raw_text_targets" in cfg:
        cfg["write_recovery_allow_raw_text_targets"] = _to_bool(cfg["write_recovery_allow_raw_text_targets"])
    if "chunk_mode_supported_models" in cfg:
        cfg["chunk_mode_supported_models"] = [s.strip() for s in str(cfg["chunk_mode_supported_models"]).split(",") if s.strip()]
    if "graph_checkpointer" in cfg:
        cfg["graph_checkpointer"] = _normalize_graph_checkpointer(cfg["graph_checkpointer"])
    if "graph_checkpoint_path" in cfg and "graph_checkpointer" not in cfg:
        cfg["graph_checkpointer"] = "file"
    if "context_limit" in cfg:
        parsed_limit = _to_int(cfg["context_limit"])
        if parsed_limit is None:
            cfg.pop("context_limit", None)
        else:
            cfg["context_limit"] = parsed_limit
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
        "artifact_summarization_threshold",
        "chunk_mode_min_bytes",
        "small_model_soft_write_chars",
        "small_model_hard_write_chars",
        "new_file_chunk_mode_line_estimate",
        "failed_local_patch_limit",
    ):
        if key in cfg:
            parsed_limit = _to_int(cfg[key])
            if parsed_limit is None:
                cfg.pop(key, None)
            else:
                cfg[key] = parsed_limit
    if "summarize_at_ratio" in cfg:
        parsed_ratio = _to_float(cfg["summarize_at_ratio"])
        if parsed_ratio is None:
            cfg.pop("summarize_at_ratio", None)
        else:
            cfg["summarize_at_ratio"] = parsed_ratio
    if "healthcheck_url" not in cfg and "backend_healthcheck_url" in cfg:
        cfg["healthcheck_url"] = cfg["backend_healthcheck_url"]
    if "restart_command" not in cfg and "backend_restart_command" in cfg:
        cfg["restart_command"] = cfg["backend_restart_command"]
    if "startup_grace_period_sec" not in cfg and "backend_restart_grace_sec" in cfg:
        cfg["startup_grace_period_sec"] = cfg["backend_restart_grace_sec"]
    return cfg


def resolve_config(cli: dict[str, Any]) -> SmallctlConfig:
    # Precedence: CLI > env vars > local .smallctl.yaml > user config path.
    # Env remains the shared source of truth for all entry points, while the
    # config files provide fallback defaults when the environment is missing.
    env_cfg = _env_config()
    merged: dict[str, Any] = {}

    user_config_path = cli.get("config_path") or env_cfg.get("config_path")
    if user_config_path:
        merged.update(_read_yaml(Path(user_config_path)))
        merged["config_path"] = user_config_path

    merged.update(_read_yaml(Path.cwd() / LOCAL_CONFIG))
    merged.update(env_cfg)

    if "provider_profile" not in merged:
        merged["provider_profile"] = "auto"

    cli_clean = {k: v for k, v in cli.items() if v is not None}
    if "no_ansible" in cli_clean:
        cli_clean["use_ansible"] = not bool(cli_clean.pop("no_ansible"))
    if "use_ansible" in cli_clean:
        cli_clean["use_ansible"] = _to_bool(cli_clean["use_ansible"])
    if "tool_profiles" in cli_clean:
        cli_clean["tool_profiles"] = parse_public_profiles(cli_clean["tool_profiles"])
    if "debug" in cli_clean:
        cli_clean["debug"] = _to_bool(cli_clean["debug"])
    if "ansible_check_mode_in_plan" in cli_clean:
        cli_clean["ansible_check_mode_in_plan"] = _to_bool(cli_clean["ansible_check_mode_in_plan"])
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
    if "indexer" in cli_clean:
        cli_clean["indexer"] = _to_bool(cli_clean["indexer"])
    if "enable_write_intent_recovery" in cli_clean:
        cli_clean["enable_write_intent_recovery"] = _to_bool(cli_clean["enable_write_intent_recovery"])
    if "enable_assistant_code_write_recovery" in cli_clean:
        cli_clean["enable_assistant_code_write_recovery"] = _to_bool(cli_clean["enable_assistant_code_write_recovery"])
    if "write_recovery_allow_raw_text_targets" in cli_clean:
        cli_clean["write_recovery_allow_raw_text_targets"] = _to_bool(cli_clean["write_recovery_allow_raw_text_targets"])
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
    if "healthcheck_url" not in cli_clean and "backend_healthcheck_url" in cli_clean:
        cli_clean["healthcheck_url"] = cli_clean["backend_healthcheck_url"]
    if "restart_command" not in cli_clean and "backend_restart_command" in cli_clean:
        cli_clean["restart_command"] = cli_clean["backend_restart_command"]
    if "startup_grace_period_sec" not in cli_clean and "backend_restart_grace_sec" in cli_clean:
        cli_clean["startup_grace_period_sec"] = cli_clean["backend_restart_grace_sec"]
    merged.update(cli_clean)
    explicit_prompt_budget = "max_prompt_tokens" in merged
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
    merged["phase"] = normalize_phase(str(merged.get("phase", "explore")))
    merged["graph_checkpointer"] = _normalize_graph_checkpointer(
        merged.get("graph_checkpointer", "memory")
    )
    if merged.get("graph_checkpoint_path") and merged["graph_checkpointer"] == "memory":
        merged["graph_checkpointer"] = "file"
    if merged.get("fresh_run"):
        merged["restore_graph_state"] = False
    _apply_provider_profile(merged, cli_clean)
    merged["compatibility_warnings"] = compatibility_warnings

    allowed_keys = {f.name for f in fields(SmallctlConfig)}
    unknown_keys = sorted(k for k in merged if k not in allowed_keys)
    if unknown_keys:
        compatibility_warnings.append(
            f"Ignoring unsupported config keys: {', '.join(unknown_keys)}."
        )
    filtered = {k: v for k, v in merged.items() if k in allowed_keys}
    return SmallctlConfig(**filtered)


def _apply_provider_profile(merged: dict[str, Any], cli_clean: dict[str, Any]) -> None:
    profile = str(merged.get("provider_profile", "auto")).strip().lower()
    if profile == "auto":
        profile = detect_provider_profile(merged.get("endpoint"), merged.get("model"))
    merged["provider_profile"] = profile
    defaults = PROVIDER_PROFILES.get(profile, PROVIDER_PROFILES["generic"])
    for key, value in defaults.items():
        # Apply profile defaults only when not explicitly provided by merged inputs.
        if key not in merged:
            merged[key] = value
    # CLI always wins, including explicit profile-related overrides.
    merged.update({k: v for k, v in cli_clean.items() if k in defaults or k == "provider_profile"})
