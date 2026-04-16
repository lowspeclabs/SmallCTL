from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .tools.profiles import parse_public_profiles

try:
    import yaml
except Exception:  # pragma: no cover - fallback if dependency missing
    yaml = None


ENV_PREFIX = "SMALLCTL_"
LOCAL_CONFIG = ".smallctl.yaml"


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


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to read YAML config files.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return data


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


def _normalize_graph_checkpointer(value: Any) -> str:
    backend = str(value or "memory").strip().lower()
    return backend if backend in {"memory", "file"} else "memory"


def _env_config() -> dict[str, Any]:
    dotenv = _read_dotenv(Path.cwd() / ".env")
    env_or_dotenv = lambda key: os.getenv(key) or dotenv.get(key)
    raw = {
        "endpoint": env_or_dotenv(f"{ENV_PREFIX}ENDPOINT"),
        "model": env_or_dotenv(f"{ENV_PREFIX}MODEL"),
        "phase": env_or_dotenv(f"{ENV_PREFIX}PHASE"),
        "provider_profile": env_or_dotenv(f"{ENV_PREFIX}PROVIDER_PROFILE"),
        "tool_profiles": env_or_dotenv(f"{ENV_PREFIX}TOOL_PROFILES"),
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
        "staged_reasoning": env_or_dotenv(f"{ENV_PREFIX}STAGED_REASONING"),
        "log_file": env_or_dotenv(f"{ENV_PREFIX}LOG_FILE"),
        "debug": env_or_dotenv(f"{ENV_PREFIX}DEBUG"),
        "config_path": env_or_dotenv(f"{ENV_PREFIX}CONFIG"),
        "preset": env_or_dotenv(f"{ENV_PREFIX}PRESET"),
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
        "loop_guard_enabled": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_ENABLED"),
        "loop_guard_stagnation_threshold": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_STAGNATION_THRESHOLD"),
        "loop_guard_level2_threshold": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_LEVEL2_THRESHOLD"),
        "loop_guard_recent_writes_limit": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_RECENT_WRITES_LIMIT"),
        "loop_guard_tail_lines": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_TAIL_LINES"),
        "loop_guard_similarity_threshold": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_SIMILARITY_THRESHOLD"),
        "loop_guard_cumulative_write_gate": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_CUMULATIVE_WRITE_GATE"),
        "loop_guard_checkpoint_gate": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_CHECKPOINT_GATE"),
        "loop_guard_diff_gate": env_or_dotenv(f"{ENV_PREFIX}LOOP_GUARD_DIFF_GATE"),
    }
    cfg = {k: v for k, v in raw.items() if v not in (None, "")}
    if "tool_profiles" in cfg:
        cfg["tool_profiles"] = parse_public_profiles(cfg["tool_profiles"])
    if "debug" in cfg:
        cfg["debug"] = _to_bool(cfg["debug"])
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
    if "staged_reasoning" in cfg:
        cfg["staged_reasoning"] = _to_bool(cfg["staged_reasoning"])
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
    for key in (
        "loop_guard_enabled",
        "loop_guard_cumulative_write_gate",
        "loop_guard_checkpoint_gate",
        "loop_guard_diff_gate",
    ):
        if key in cfg:
            cfg[key] = _to_bool(cfg[key])
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
        "loop_guard_stagnation_threshold",
        "loop_guard_level2_threshold",
        "loop_guard_recent_writes_limit",
        "loop_guard_tail_lines",
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
    if "loop_guard_similarity_threshold" in cfg:
        parsed_ratio = _to_float(cfg["loop_guard_similarity_threshold"])
        if parsed_ratio is None:
            cfg.pop("loop_guard_similarity_threshold", None)
        else:
            cfg["loop_guard_similarity_threshold"] = parsed_ratio
    if "healthcheck_url" not in cfg and "backend_healthcheck_url" in cfg:
        cfg["healthcheck_url"] = cfg["backend_healthcheck_url"]
    if "restart_command" not in cfg and "backend_restart_command" in cfg:
        cfg["restart_command"] = cfg["backend_restart_command"]
    if "startup_grace_period_sec" not in cfg and "backend_restart_grace_sec" in cfg:
        cfg["startup_grace_period_sec"] = cfg["backend_restart_grace_sec"]
    return cfg
