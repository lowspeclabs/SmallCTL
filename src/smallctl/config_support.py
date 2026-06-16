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


def _normalize_run_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower().replace("-", "_")
    return mode if mode in {"auto", "chat", "loop", "planning", "indexer", "tool_plan"} else "auto"


_BOOL_CONFIG_KEYS = {
    "debug",
    "thinking_visibility",
    "checkpoint_on_exit",
    "restore_graph_state",
    "runtime_context_probe",
    "fresh_run",
    "planning_mode",
    "contract_flow_ui",
    "staged_reasoning",
    "staged_execution_enabled",
    "indexer",
    "enable_write_intent_recovery",
    "enable_assistant_code_write_recovery",
    "write_recovery_allow_raw_text_targets",
    "chunk_mode_new_file_only",
    "allow_multi_section_turns_for_small_edits",
    "loop_guard_enabled",
    "loop_guard_cumulative_write_gate",
    "loop_guard_checkpoint_gate",
    "loop_guard_diff_gate",
    "fama_enabled",
    "fama_disabled",
    "fama_done_gate_on_failure",
    "fama_llm_judge_enabled",
    "reflexion_enabled",
    "reflexion_persist_cross_task",
    "subtask_ledger_enabled",
    "tool_plan_runtime_enabled",
    "tool_plan_auto_select",
    "tool_plan_readonly_only",
    "tool_plan_allow_web",
    "tool_plan_allow_artifact_read",
    "tool_plan_allow_git",
    "tool_plan_fallback_to_loop_on_invalid_plan",
    "tool_dag_enabled",
    "tool_dag_preserve_result_order",
    "solver_refine_enabled",
    "solver_refine_on_final_answer",
    "solver_refine_on_patch_plan",
    "solver_refine_on_task_complete",
    "rewoo_lane_frames_enabled",
    "rewoo_planner_frame_enabled",
    "rewoo_solver_frame_enabled",
    "rewoo_refiner_frame_enabled",
    "test_time_scaling_enabled",
    "test_time_scaling_mutating_parallel_enabled",
    "escalation_enabled",
    "escalation_expose_tool",
    "escalation_auto_trigger",
    "escalation_require_tool_plan_evidence",
    "escalation_redact_secrets",
    "show_system_messages",
    "verbose",
}

_INT_ALLOW_ZERO_CONFIG_KEYS = {"fresh_run_turns"}

_INT_CONFIG_KEYS = {
    "context_limit",
    "max_prompt_tokens",
    "reserve_completion_tokens",
    "reserve_tool_tokens",
    "first_token_timeout_sec",
    "startup_grace_period_sec",
    "max_restarts_per_hour",
    "backend_healthcheck_timeout_sec",
    "backend_restart_grace_sec",
    "graph_node_timeout_sec",
    "graph_model_call_timeout_sec",
    "graph_dispatch_tools_timeout_sec",
    "graph_idle_watchdog_sec",
    "graph_recursion_limit",
    "graph_coding_recursion_limit",
    "needs_human_timeout_sec",
    "recent_message_limit",
    "max_summary_items",
    "max_artifact_snippets",
    "artifact_snippet_token_limit",
    "multi_file_artifact_snippet_limit",
    "multi_file_primary_file_limit",
    "remote_task_artifact_snippet_limit",
    "remote_task_primary_file_limit",
    "min_exploration_steps",
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
    "staged_step_prompt_tokens",
    "fama_default_ttl_steps",
    "fama_max_active_mitigations",
    "fama_signal_window",
    "fama_capsule_token_budget",
    "fama_llm_judge_min_severity",
    "reflexion_max_items",
    "reflexion_inject_top_k",
    "subtask_max_active",
    "subtask_max_history",
    "subtask_inject_completed_limit",
    "tool_plan_max_steps",
    "tool_plan_max_repair_attempts",
    "schema_validation_max_repair_attempts",
    "tool_plan_observation_token_limit",
    "tool_plan_max_observation_chars_per_step",
    "tool_plan_solver_fresh_output_limit",
    "tool_dag_max_parallel",
    "tool_dag_timeout_sec",
    "solver_refine_max_passes",
    "solver_refine_token_budget",
    "rewoo_frame_token_budget",
    "test_time_scaling_max_candidates",
    "test_time_scaling_min_candidates",
    "test_time_scaling_parallel_max",
    "test_time_scaling_timeout_sec",
    "escalation_max_prompt_chars",
    "escalation_max_response_tokens",
    "escalation_timeout_sec",
    "escalation_max_per_task",
    "escalation_cooldown_turns",
    "escalation_repeated_failure_threshold",
}

_FLOAT_CONFIG_KEYS = {
    "summarize_at_ratio",
    "loop_guard_similarity_threshold",
    "test_time_scaling_score_threshold",
    "escalation_temperature",
}

_COMMA_LIST_CONFIG_KEYS = {"test_time_scaling_runtimes", "chunk_mode_supported_models"}


def _apply_typed_config_values(values: dict[str, Any]) -> None:
    for key in _BOOL_CONFIG_KEYS:
        if key in values:
            values[key] = _to_bool(values[key])
    for key in _INT_ALLOW_ZERO_CONFIG_KEYS:
        if key in values:
            parsed = _to_int_allow_zero(values[key])
            if parsed is None:
                values.pop(key, None)
            else:
                values[key] = parsed
    for key in _INT_CONFIG_KEYS:
        if key in values:
            parsed = _to_int(values[key])
            if parsed is None:
                values.pop(key, None)
            else:
                values[key] = parsed
    for key in _FLOAT_CONFIG_KEYS:
        if key in values:
            parsed = _to_float(values[key])
            if parsed is None:
                values.pop(key, None)
            else:
                values[key] = parsed
    for key in _COMMA_LIST_CONFIG_KEYS:
        if key in values and isinstance(values[key], str):
            values[key] = [item.strip() for item in values[key].split(",") if item.strip()]


def _apply_config_aliases(values: dict[str, Any]) -> None:
    if "run_mode" in values:
        values["run_mode"] = _normalize_run_mode(values["run_mode"])
    if "graph_checkpointer" in values:
        values["graph_checkpointer"] = _normalize_graph_checkpointer(values["graph_checkpointer"])
    if "graph_checkpoint_path" in values and "graph_checkpointer" not in values:
        values["graph_checkpointer"] = "file"
    if "healthcheck_url" not in values and "backend_healthcheck_url" in values:
        values["healthcheck_url"] = values["backend_healthcheck_url"]
    if "restart_command" not in values and "backend_restart_command" in values:
        values["restart_command"] = values["backend_restart_command"]
    if "startup_grace_period_sec" not in values and "backend_restart_grace_sec" in values:
        values["startup_grace_period_sec"] = values["backend_restart_grace_sec"]


def _env_raw_config(env_or_dotenv: Any) -> dict[str, Any]:
    return {
        "endpoint": env_or_dotenv(f"{ENV_PREFIX}ENDPOINT"),
        "model": env_or_dotenv(f"{ENV_PREFIX}MODEL"),
        "phase": env_or_dotenv(f"{ENV_PREFIX}PHASE"),
        "provider_profile": env_or_dotenv(f"{ENV_PREFIX}PROVIDER_PROFILE"),
        "run_mode": env_or_dotenv(f"{ENV_PREFIX}RUN_MODE"),
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
        "graph_node_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}GRAPH_NODE_TIMEOUT_SEC"),
        "graph_model_call_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}GRAPH_MODEL_CALL_TIMEOUT_SEC"),
        "graph_dispatch_tools_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}GRAPH_DISPATCH_TOOLS_TIMEOUT_SEC"),
        "graph_idle_watchdog_sec": env_or_dotenv(f"{ENV_PREFIX}GRAPH_IDLE_WATCHDOG_SEC"),
        "graph_recursion_limit": env_or_dotenv(f"{ENV_PREFIX}GRAPH_RECURSION_LIMIT"),
        "graph_coding_recursion_limit": env_or_dotenv(f"{ENV_PREFIX}GRAPH_CODING_RECURSION_LIMIT"),
        "needs_human_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}NEEDS_HUMAN_TIMEOUT_SEC"),
        "restore_graph_state": env_or_dotenv(f"{ENV_PREFIX}RESTORE_GRAPH_STATE"),
        "graph_thread_id": env_or_dotenv(f"{ENV_PREFIX}GRAPH_THREAD_ID"),
        "fresh_run": env_or_dotenv(f"{ENV_PREFIX}FRESH_RUN"),
        "fresh_run_turns": env_or_dotenv(f"{ENV_PREFIX}FRESH_RUN_TURNS"),
        "planning_mode": env_or_dotenv(f"{ENV_PREFIX}PLANNING_MODE"),
        "contract_flow_ui": env_or_dotenv(f"{ENV_PREFIX}CONTRACT_FLOW_UI"),
        "staged_reasoning": env_or_dotenv(f"{ENV_PREFIX}STAGED_REASONING"),
        "staged_execution_enabled": env_or_dotenv(f"{ENV_PREFIX}STAGED_EXECUTION"),
        "staged_step_prompt_tokens": env_or_dotenv(f"{ENV_PREFIX}STAGED_STEP_PROMPT_TOKENS"),
        "tool_plan_runtime_enabled": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_RUNTIME_ENABLED"),
        "tool_plan_auto_select": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_AUTO_SELECT"),
        "tool_plan_readonly_only": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_READONLY_ONLY"),
        "tool_plan_max_steps": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_MAX_STEPS"),
        "tool_plan_max_repair_attempts": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_MAX_REPAIR_ATTEMPTS"),
        "schema_validation_max_repair_attempts": env_or_dotenv(f"{ENV_PREFIX}SCHEMA_VALIDATION_MAX_REPAIR_ATTEMPTS"),
        "tool_plan_observation_token_limit": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_OBSERVATION_TOKEN_LIMIT"),
        "tool_plan_max_observation_chars_per_step": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_MAX_OBSERVATION_CHARS_PER_STEP"),
        "tool_plan_solver_fresh_output_limit": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_SOLVER_FRESH_OUTPUT_LIMIT"),
        "tool_plan_allow_web": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_ALLOW_WEB"),
        "tool_plan_allow_git": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_ALLOW_GIT"),
        "tool_plan_allow_artifact_read": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_ALLOW_ARTIFACT_READ"),
        "tool_plan_fallback_to_loop_on_invalid_plan": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_FALLBACK_TO_LOOP_ON_INVALID_PLAN"),
        "tool_dag_enabled": env_or_dotenv(f"{ENV_PREFIX}TOOL_DAG_ENABLED"),
        "tool_dag_max_parallel": env_or_dotenv(f"{ENV_PREFIX}TOOL_DAG_MAX_PARALLEL"),
        "tool_dag_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}TOOL_DAG_TIMEOUT_SEC"),
        "tool_dag_preserve_result_order": env_or_dotenv(f"{ENV_PREFIX}TOOL_DAG_PRESERVE_RESULT_ORDER"),
        "solver_refine_enabled": env_or_dotenv(f"{ENV_PREFIX}SOLVER_REFINE_ENABLED"),
        "solver_refine_max_passes": env_or_dotenv(f"{ENV_PREFIX}SOLVER_REFINE_MAX_PASSES"),
        "solver_refine_on_final_answer": env_or_dotenv(f"{ENV_PREFIX}SOLVER_REFINE_ON_FINAL_ANSWER"),
        "solver_refine_on_patch_plan": env_or_dotenv(f"{ENV_PREFIX}SOLVER_REFINE_ON_PATCH_PLAN"),
        "solver_refine_on_task_complete": env_or_dotenv(f"{ENV_PREFIX}SOLVER_REFINE_ON_TASK_COMPLETE"),
        "solver_refine_token_budget": env_or_dotenv(f"{ENV_PREFIX}SOLVER_REFINE_TOKEN_BUDGET"),
        "rewoo_lane_frames_enabled": env_or_dotenv(f"{ENV_PREFIX}REWOO_LANE_FRAMES_ENABLED"),
        "rewoo_planner_frame_enabled": env_or_dotenv(f"{ENV_PREFIX}REWOO_PLANNER_FRAME_ENABLED"),
        "rewoo_solver_frame_enabled": env_or_dotenv(f"{ENV_PREFIX}REWOO_SOLVER_FRAME_ENABLED"),
        "rewoo_refiner_frame_enabled": env_or_dotenv(f"{ENV_PREFIX}REWOO_REFINER_FRAME_ENABLED"),
        "rewoo_frame_token_budget": env_or_dotenv(f"{ENV_PREFIX}REWOO_FRAME_TOKEN_BUDGET"),
        "test_time_scaling_enabled": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_ENABLED"),
        "test_time_scaling_runtimes": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_RUNTIMES"),
        "test_time_scaling_trigger": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_TRIGGER"),
        "test_time_scaling_max_candidates": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_MAX_CANDIDATES"),
        "test_time_scaling_min_candidates": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_MIN_CANDIDATES"),
        "test_time_scaling_policy": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_POLICY"),
        "test_time_scaling_strategy": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_STRATEGY"),
        "test_time_scaling_score_threshold": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_SCORE_THRESHOLD"),
        "test_time_scaling_parallel_max": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_PARALLEL_MAX"),
        "test_time_scaling_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_TIMEOUT_SEC"),
        "test_time_scaling_mutating_parallel_enabled": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_MUTATING_PARALLEL_ENABLED"),
        "test_time_scaling_all_fail_action": env_or_dotenv(f"{ENV_PREFIX}TEST_TIME_SCALING_ALL_FAIL_ACTION"),
        "escalation_enabled": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_ENABLED"),
        "escalation_expose_tool": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_EXPOSE_TOOL"),
        "escalation_auto_trigger": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_AUTO_TRIGGER"),
        "escalation_endpoint": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_ENDPOINT"),
        "escalation_model": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_MODEL"),
        "escalation_provider_profile": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_PROVIDER_PROFILE"),
        "escalation_api_key": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_API_KEY"),
        "escalation_api_key_env": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_API_KEY_ENV"),
        "escalation_chat_endpoint": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_CHAT_ENDPOINT"),
        "escalation_max_prompt_chars": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_MAX_PROMPT_CHARS"),
        "escalation_max_response_tokens": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_MAX_RESPONSE_TOKENS"),
        "escalation_temperature": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_TEMPERATURE"),
        "escalation_timeout_sec": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_TIMEOUT_SEC"),
        "escalation_max_per_task": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_MAX_PER_TASK"),
        "escalation_cooldown_turns": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_COOLDOWN_TURNS"),
        "escalation_repeated_failure_threshold": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_REPEATED_FAILURE_THRESHOLD"),
        "escalation_require_tool_plan_evidence": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_REQUIRE_TOOL_PLAN_EVIDENCE"),
        "escalation_redact_secrets": env_or_dotenv(f"{ENV_PREFIX}ESCALATION_REDACT_SECRETS"),
        "log_file": env_or_dotenv(f"{ENV_PREFIX}LOG_FILE"),
        "debug": env_or_dotenv(f"{ENV_PREFIX}DEBUG"),
        "config_path": env_or_dotenv(f"{ENV_PREFIX}CONFIG"),
        "preset": env_or_dotenv(f"{ENV_PREFIX}PRESET"),
        "indexer": env_or_dotenv(f"{ENV_PREFIX}INDEXER"),
        "api_key": env_or_dotenv(f"{ENV_PREFIX}API_KEY"),
        "context_limit": env_or_dotenv(f"{ENV_PREFIX}CONTEXT_LIMIT"),
        "max_prompt_tokens": env_or_dotenv(f"{ENV_PREFIX}MAX_PROMPT_TOKENS"),
        "max_prompt_tokens_explicit": env_or_dotenv(f"{ENV_PREFIX}MAX_PROMPT_TOKENS_EXPLICIT"),
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
        "multi_file_artifact_snippet_limit": env_or_dotenv(f"{ENV_PREFIX}MULTI_FILE_ARTIFACT_SNIPPET_LIMIT"),
        "multi_file_primary_file_limit": env_or_dotenv(f"{ENV_PREFIX}MULTI_FILE_PRIMARY_FILE_LIMIT"),
        "remote_task_artifact_snippet_limit": env_or_dotenv(f"{ENV_PREFIX}REMOTE_TASK_ARTIFACT_SNIPPET_LIMIT"),
        "remote_task_primary_file_limit": env_or_dotenv(f"{ENV_PREFIX}REMOTE_TASK_PRIMARY_FILE_LIMIT"),
        "runtime_context_probe": env_or_dotenv(f"{ENV_PREFIX}RUNTIME_CONTEXT_PROBE"),
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
        "fama_enabled": env_or_dotenv(f"{ENV_PREFIX}FAMA_ENABLED"),
        "fama_disabled": env_or_dotenv(f"{ENV_PREFIX}FAMA_DISABLED"),
        "fama_mode": env_or_dotenv(f"{ENV_PREFIX}FAMA_MODE"),
        "fama_default_ttl_steps": env_or_dotenv(f"{ENV_PREFIX}FAMA_DEFAULT_TTL_STEPS"),
        "fama_max_active_mitigations": env_or_dotenv(f"{ENV_PREFIX}FAMA_MAX_ACTIVE_MITIGATIONS"),
        "fama_signal_window": env_or_dotenv(f"{ENV_PREFIX}FAMA_SIGNAL_WINDOW"),
        "fama_done_gate_on_failure": env_or_dotenv(f"{ENV_PREFIX}FAMA_DONE_GATE_ON_FAILURE"),
        "fama_capsule_token_budget": env_or_dotenv(f"{ENV_PREFIX}FAMA_CAPSULE_TOKEN_BUDGET"),
        "fama_llm_judge_enabled": env_or_dotenv(f"{ENV_PREFIX}FAMA_LLM_JUDGE_ENABLED"),
        "fama_llm_judge_min_severity": env_or_dotenv(f"{ENV_PREFIX}FAMA_LLM_JUDGE_MIN_SEVERITY"),
        "reflexion_enabled": env_or_dotenv(f"{ENV_PREFIX}REFLEXION_ENABLED"),
        "reflexion_max_items": env_or_dotenv(f"{ENV_PREFIX}REFLEXION_MAX_ITEMS"),
        "reflexion_inject_top_k": env_or_dotenv(f"{ENV_PREFIX}REFLEXION_INJECT_TOP_K"),
        "reflexion_persist_cross_task": env_or_dotenv(f"{ENV_PREFIX}REFLEXION_PERSIST_CROSS_TASK"),
        "reflexion_min_failure_severity": env_or_dotenv(f"{ENV_PREFIX}REFLEXION_MIN_FAILURE_SEVERITY"),
        "subtask_ledger_enabled": env_or_dotenv(f"{ENV_PREFIX}SUBTASK_LEDGER_ENABLED"),
        "subtask_max_active": env_or_dotenv(f"{ENV_PREFIX}SUBTASK_MAX_ACTIVE"),
        "subtask_max_history": env_or_dotenv(f"{ENV_PREFIX}SUBTASK_MAX_HISTORY"),
        "subtask_inject_completed_limit": env_or_dotenv(f"{ENV_PREFIX}SUBTASK_INJECT_COMPLETED_LIMIT"),
        "show_system_messages": env_or_dotenv(f"{ENV_PREFIX}SHOW_SYSTEM_MESSAGES"),
        "verbose": env_or_dotenv(f"{ENV_PREFIX}VERBOSE"),
    }


def _env_config_key_names() -> set[str]:
    return set(_env_raw_config(lambda _key: None).keys())


def _env_config() -> dict[str, Any]:
    dotenv = _read_dotenv(Path.cwd() / ".env")
    env_or_dotenv = lambda key: os.getenv(key) or dotenv.get(key)
    raw = _env_raw_config(env_or_dotenv)
    cfg = {k: v for k, v in raw.items() if v not in (None, "")}
    if "tool_profiles" in cfg:
        cfg["tool_profiles"] = parse_public_profiles(cfg["tool_profiles"])
    _apply_typed_config_values(cfg)
    _apply_config_aliases(cfg)
    return cfg
