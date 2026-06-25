from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .cleanup import run_cleanup
from .config import resolve_config
from .harness import Harness, HarnessConfig
from .logging_utils import create_run_logger, log_kv, setup_logging
from .memory_cli import build_memory_parser, handle_memory_command
from .presets import list_presets


def _escalation_harness_kwargs(config: object) -> dict[str, object]:
    return {
        "escalation_enabled": getattr(config, "escalation_enabled", False),
        "escalation_expose_tool": getattr(config, "escalation_expose_tool", True),
        "escalation_auto_trigger": getattr(config, "escalation_auto_trigger", False),
        "escalation_endpoint": getattr(config, "escalation_endpoint", None),
        "escalation_model": getattr(config, "escalation_model", None),
        "escalation_provider_profile": getattr(config, "escalation_provider_profile", "auto"),
        "escalation_api_key": getattr(config, "escalation_api_key", None),
        "escalation_api_key_env": getattr(config, "escalation_api_key_env", None),
        "escalation_chat_endpoint": getattr(config, "escalation_chat_endpoint", "/chat/completions"),
        "escalation_max_prompt_chars": getattr(config, "escalation_max_prompt_chars", 48000),
        "escalation_max_response_tokens": getattr(config, "escalation_max_response_tokens", 1600),
        "escalation_temperature": getattr(config, "escalation_temperature", 0.2),
        "escalation_timeout_sec": getattr(config, "escalation_timeout_sec", 120),
        "escalation_max_per_task": getattr(config, "escalation_max_per_task", 3),
        "escalation_cooldown_turns": getattr(config, "escalation_cooldown_turns", 2),
        "escalation_repeated_failure_threshold": getattr(config, "escalation_repeated_failure_threshold", 2),
        "escalation_require_tool_plan_evidence": getattr(config, "escalation_require_tool_plan_evidence", True),
        "escalation_redact_secrets": getattr(config, "escalation_redact_secrets", True),
    }


def build_harness_config_kwargs(
    config: object,
    *,
    run_logger: logging.Logger,
    task: str | None = None,
) -> dict[str, object]:
    strategy = {"thought_architecture": "staged_reasoning"} if config.staged_reasoning else None
    max_prompt_tokens_explicit = bool(
        getattr(config, "max_prompt_tokens_explicit", config.max_prompt_tokens is not None)
    )
    return {
        "endpoint": config.endpoint,
        "model": config.model,
        "phase": config.phase,
        "provider_profile": config.provider_profile,
        "api_key": config.api_key,
        "tool_profiles": config.tool_profiles,
        "reasoning_mode": config.reasoning_mode,
        "thinking_visibility": config.thinking_visibility,
        "thinking_start_tag": config.thinking_start_tag,
        "thinking_end_tag": config.thinking_end_tag,
        "chat_endpoint": config.chat_endpoint,
        "runtime_context_probe": getattr(config, "runtime_context_probe", True),
        "checkpoint_on_exit": config.checkpoint_on_exit,
        "checkpoint_path": config.checkpoint_path,
        "graph_checkpointer": config.graph_checkpointer,
        "graph_checkpoint_path": config.graph_checkpoint_path,
        "graph_node_timeout_sec": getattr(config, "graph_node_timeout_sec", 300),
        "graph_model_call_timeout_sec": getattr(config, "graph_model_call_timeout_sec", 600),
        "graph_dispatch_tools_timeout_sec": getattr(config, "graph_dispatch_tools_timeout_sec", 300),
        "graph_idle_watchdog_sec": getattr(config, "graph_idle_watchdog_sec", 300),
        "graph_recursion_limit": getattr(config, "graph_recursion_limit", 1024),
        "graph_coding_recursion_limit": getattr(config, "graph_coding_recursion_limit", 2048),
        "needs_human_timeout_sec": getattr(config, "needs_human_timeout_sec", 600),
        "fresh_run": config.fresh_run,
        "fresh_run_turns": config.fresh_run_turns,
        "planning_mode": config.planning_mode,
        "contract_flow_ui": config.contract_flow_ui,
        "strategy": strategy,
        "context_limit": config.context_limit,
        "max_prompt_tokens": config.max_prompt_tokens,
        "max_prompt_tokens_explicit": max_prompt_tokens_explicit,
        "max_completion_tokens": getattr(config, "max_completion_tokens", None),
        "reserve_completion_tokens": config.reserve_completion_tokens,
        "reserve_tool_tokens": config.reserve_tool_tokens,
        "first_token_timeout_sec": config.first_token_timeout_sec,
        "healthcheck_url": config.healthcheck_url,
        "restart_command": config.restart_command,
        "startup_grace_period_sec": config.startup_grace_period_sec,
        "max_restarts_per_hour": config.max_restarts_per_hour,
        "backend_healthcheck_url": config.backend_healthcheck_url,
        "backend_restart_command": config.backend_restart_command,
        "backend_unload_command": config.backend_unload_command,
        "backend_healthcheck_timeout_sec": config.backend_healthcheck_timeout_sec,
        "backend_restart_grace_sec": config.backend_restart_grace_sec,
        "summarize_at_ratio": config.summarize_at_ratio,
        "recent_message_limit": config.recent_message_limit,
        "max_summary_items": config.max_summary_items,
        "max_artifact_snippets": config.max_artifact_snippets,
        "artifact_snippet_token_limit": config.artifact_snippet_token_limit,
        "artifact_summarization_threshold": getattr(config, "artifact_summarization_threshold", 1200),
        "multi_file_artifact_snippet_limit": config.multi_file_artifact_snippet_limit,
        "multi_file_primary_file_limit": config.multi_file_primary_file_limit,
        "remote_task_artifact_snippet_limit": config.remote_task_artifact_snippet_limit,
        "remote_task_primary_file_limit": config.remote_task_primary_file_limit,
        "summarizer_endpoint": getattr(config, "summarizer_endpoint", None),
        "summarizer_model": getattr(config, "summarizer_model", None),
        "summarizer_api_key": getattr(config, "summarizer_api_key", None),
        "indexer": config.indexer,
        "run_mode": getattr(config, "run_mode", "auto"),
        "tool_plan_runtime_enabled": getattr(config, "tool_plan_runtime_enabled", False),
        "tool_plan_auto_select": getattr(config, "tool_plan_auto_select", False),
        "tool_plan_readonly_only": getattr(config, "tool_plan_readonly_only", True),
        "tool_plan_max_steps": getattr(config, "tool_plan_max_steps", 6),
        "tool_plan_max_repair_attempts": getattr(config, "tool_plan_max_repair_attempts", 1),
        "schema_validation_max_repair_attempts": getattr(config, "schema_validation_max_repair_attempts", 2),
        "tool_call_repair_enabled": getattr(config, "tool_call_repair_enabled", True),
        "tool_call_repair_log_only": getattr(config, "tool_call_repair_log_only", False),
        "tool_call_repair_max_actions_per_call": getattr(config, "tool_call_repair_max_actions_per_call", 4),
        "tool_plan_observation_token_limit": getattr(config, "tool_plan_observation_token_limit", 900),
        "tool_plan_max_observation_chars_per_step": getattr(config, "tool_plan_max_observation_chars_per_step", 600),
        "tool_plan_solver_fresh_output_limit": getattr(config, "tool_plan_solver_fresh_output_limit", 1200),
        "tool_plan_allow_web": getattr(config, "tool_plan_allow_web", True),
        "tool_plan_allow_artifact_read": getattr(config, "tool_plan_allow_artifact_read", True),
        "tool_plan_allow_git": getattr(config, "tool_plan_allow_git", False),
        "tool_plan_fallback_to_loop_on_invalid_plan": getattr(config, "tool_plan_fallback_to_loop_on_invalid_plan", True),
        "tool_dag_enabled": getattr(config, "tool_dag_enabled", False),
        "tool_dag_max_parallel": getattr(config, "tool_dag_max_parallel", 4),
        "tool_dag_timeout_sec": getattr(config, "tool_dag_timeout_sec", 30),
        "tool_dag_preserve_result_order": getattr(config, "tool_dag_preserve_result_order", True),
        "min_exploration_steps": getattr(config, "min_exploration_steps", 1),
        "chunk_mode_min_bytes": getattr(config, "chunk_mode_min_bytes", 4096),
        "chunk_mode_new_file_only": getattr(config, "chunk_mode_new_file_only", True),
        "chunk_mode_supported_models": getattr(config, "chunk_mode_supported_models", ["qwen3.5", "llama3.1", "deepseek-v3"]),
        "small_model_soft_write_chars": getattr(config, "small_model_soft_write_chars", 2000),
        "small_model_hard_write_chars": getattr(config, "small_model_hard_write_chars", 4000),
        "new_file_chunk_mode_line_estimate": getattr(config, "new_file_chunk_mode_line_estimate", 100),
        "allow_multi_section_turns_for_small_edits": getattr(config, "allow_multi_section_turns_for_small_edits", True),
        "failed_local_patch_limit": getattr(config, "failed_local_patch_limit", 2),
        "enable_write_intent_recovery": getattr(config, "enable_write_intent_recovery", True),
        "enable_assistant_code_write_recovery": getattr(config, "enable_assistant_code_write_recovery", True),
        "write_recovery_min_confidence": getattr(config, "write_recovery_min_confidence", "high"),
        "write_recovery_allow_raw_text_targets": getattr(config, "write_recovery_allow_raw_text_targets", True),
        "solver_refine_enabled": getattr(config, "solver_refine_enabled", False),
        "solver_refine_max_passes": getattr(config, "solver_refine_max_passes", 1),
        "solver_refine_on_final_answer": getattr(config, "solver_refine_on_final_answer", True),
        "solver_refine_on_patch_plan": getattr(config, "solver_refine_on_patch_plan", True),
        "solver_refine_on_task_complete": getattr(config, "solver_refine_on_task_complete", True),
        "solver_refine_token_budget": getattr(config, "solver_refine_token_budget", 700),
        "rewoo_lane_frames_enabled": getattr(config, "rewoo_lane_frames_enabled", False),
        "rewoo_planner_frame_enabled": getattr(config, "rewoo_planner_frame_enabled", False),
        "rewoo_solver_frame_enabled": getattr(config, "rewoo_solver_frame_enabled", False),
        "rewoo_refiner_frame_enabled": getattr(config, "rewoo_refiner_frame_enabled", False),
        "rewoo_frame_token_budget": getattr(config, "rewoo_frame_token_budget", 1200),
        "staged_execution_enabled": getattr(config, "staged_execution_enabled", False),
        "staged_step_prompt_tokens": getattr(config, "staged_step_prompt_tokens", 4096),
        "test_time_scaling_enabled": getattr(config, "test_time_scaling_enabled", False),
        "test_time_scaling_runtimes": getattr(config, "test_time_scaling_runtimes", ["staged_execution"]),
        "test_time_scaling_trigger": getattr(config, "test_time_scaling_trigger", "retry_or_explicit"),
        "test_time_scaling_max_candidates": getattr(config, "test_time_scaling_max_candidates", 3),
        "test_time_scaling_min_candidates": getattr(config, "test_time_scaling_min_candidates", 2),
        "test_time_scaling_policy": getattr(config, "test_time_scaling_policy", "proposal_then_execute"),
        "test_time_scaling_strategy": getattr(config, "test_time_scaling_strategy", "diverse_nudges"),
        "test_time_scaling_score_threshold": getattr(config, "test_time_scaling_score_threshold", 0.85),
        "test_time_scaling_parallel_max": getattr(config, "test_time_scaling_parallel_max", 1),
        "test_time_scaling_timeout_sec": getattr(config, "test_time_scaling_timeout_sec", 120),
        "test_time_scaling_mutating_parallel_enabled": getattr(config, "test_time_scaling_mutating_parallel_enabled", False),
        "test_time_scaling_all_fail_action": getattr(config, "test_time_scaling_all_fail_action", "fallback_normal_retry"),
        **_escalation_harness_kwargs(config),
        "run_logger": run_logger,
        "fama_enabled": False if getattr(config, "fama_disabled", False) else getattr(config, "fama_enabled", True),
        "fama_disabled": getattr(config, "fama_disabled", False),
        "fama_mode": getattr(config, "fama_mode", "lite"),
        "fama_default_ttl_steps": getattr(config, "fama_default_ttl_steps", 2),
        "fama_max_active_mitigations": getattr(config, "fama_max_active_mitigations", 2),
        "fama_signal_window": getattr(config, "fama_signal_window", 8),
        "fama_done_gate_on_failure": getattr(config, "fama_done_gate_on_failure", True),
        "fama_capsule_token_budget": getattr(config, "fama_capsule_token_budget", 180),
        "fama_llm_judge_enabled": getattr(config, "fama_llm_judge_enabled", False),
        "fama_llm_judge_min_severity": getattr(config, "fama_llm_judge_min_severity", 3),
        "loop_guard_enabled": getattr(config, "loop_guard_enabled", True),
        "loop_guard_stagnation_threshold": getattr(config, "loop_guard_stagnation_threshold", 3),
        "loop_guard_level2_threshold": getattr(config, "loop_guard_level2_threshold", 5),
        "loop_guard_recent_writes_limit": getattr(config, "loop_guard_recent_writes_limit", 5),
        "loop_guard_tail_lines": getattr(config, "loop_guard_tail_lines", 50),
        "loop_guard_similarity_threshold": getattr(config, "loop_guard_similarity_threshold", 0.9),
        "loop_guard_cumulative_write_gate": getattr(config, "loop_guard_cumulative_write_gate", True),
        "loop_guard_checkpoint_gate": getattr(config, "loop_guard_checkpoint_gate", True),
        "loop_guard_diff_gate": getattr(config, "loop_guard_diff_gate", True),
        "reflexion_enabled": getattr(config, "reflexion_enabled", True),
        "reflexion_max_items": getattr(config, "reflexion_max_items", 5),
        "reflexion_inject_top_k": getattr(config, "reflexion_inject_top_k", 3),
        "reflexion_persist_cross_task": getattr(config, "reflexion_persist_cross_task", False),
        "reflexion_min_failure_severity": getattr(config, "reflexion_min_failure_severity", "warning"),
        "subtask_ledger_enabled": getattr(config, "subtask_ledger_enabled", True),
        "subtask_max_active": getattr(config, "subtask_max_active", 1),
        "subtask_max_history": getattr(config, "subtask_max_history", 12),
        "subtask_inject_completed_limit": getattr(config, "subtask_inject_completed_limit", 3),
        "sudo_password": getattr(config, "sudo_password", None),
        "verbose": getattr(config, "verbose", False),
        "debug_subsystems": getattr(config, "debug_subsystems", None),
        "debug_tokens": getattr(config, "debug_tokens", False),
        "log_max_mb": getattr(config, "log_max_mb", 100),
        **({"task": task} if task is not None else {}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smallctl", description="smallctl CLI")
    parser.add_argument("--task", help="Task string to run")
    parser.add_argument("--endpoint", help="OpenAI-compatible API base URL")
    parser.add_argument("--model", help="Model name")
    parser.add_argument(
        "--preset",
        choices=list_presets(),
        help="Named preset for common run profiles",
    )
    parser.add_argument("--phase", help="Initial phase (explore|plan|execute|verify)")
    parser.add_argument(
        "--provider-profile",
        choices=["auto", "generic", "openai", "ollama", "vllm", "lmstudio", "openrouter", "llamacpp"],
        help="Compatibility profile for OpenAI-compatible provider behavior",
    )
    parser.add_argument(
        "--run-mode",
        choices=["auto", "chat", "loop", "planning", "indexer", "tool_plan"],
        help="Explicit runtime mode",
    )
    parser.add_argument(
        "--tool-profiles",
        help="Comma-separated static tool profiles to expose: core,data,network,mutate,indexer",
    )
    parser.add_argument("--config", dest="config_path", help="User config path")
    parser.add_argument(
        "--reasoning-mode",
        choices=["auto", "tags", "field", "off"],
        help="Reasoning extraction mode for streamed responses",
    )
    parser.add_argument(
        "--hide-thinking",
        dest="thinking_visibility",
        action="store_false",
        default=None,
        help="Hide thinking output in terminal stream",
    )
    parser.add_argument(
        "--thinking-start-tag",
        help="Start tag for tag-based thinking parsing",
    )
    parser.add_argument(
        "--thinking-end-tag",
        help="End tag for tag-based thinking parsing",
    )
    parser.add_argument(
        "--chat-endpoint",
        help="OpenAI-compatible chat endpoint path",
    )
    parser.add_argument(
        "--checkpoint-on-exit",
        action="store_true",
        default=None,
        help="Persist loop checkpoint after task completion/failure",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Optional explicit checkpoint output path",
    )
    parser.add_argument(
        "--graph-checkpointer",
        choices=["memory", "file"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--graph-checkpoint-path",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume",
        dest="restore_graph_state",
        action="store_true",
        default=None,
        help="Resume from the latest saved chat/session state",
    )
    parser.add_argument(
        "--restore-graph-state",
        dest="restore_graph_state",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        default=None,
        help="Start without loading prior experience memory or graph state",
    )
    parser.add_argument(
        "--fresh-run-turns",
        type=int,
        help="How many initial turns fresh-run memory suppression should stay active",
    )
    parser.add_argument(
        "--planning-mode",
        action="store_true",
        default=None,
        help="Start in planning mode",
    )
    parser.add_argument(
        "--contract-flow-ui",
        dest="contract_flow_ui",
        action="store_true",
        default=None,
        help="Show refined contract-phase and verifier details in the UI",
    )
    parser.add_argument(
        "--no-contract-flow-ui",
        dest="contract_flow_ui",
        action="store_false",
        default=None,
        help="Hide refined contract-phase and verifier details in the UI",
    )
    parser.add_argument(
        "--staged-reasoning",
        dest="staged_reasoning",
        action="store_true",
        default=None,
        help="Enable the staged reasoning strategy toggle for rollout testing",
    )
    parser.add_argument(
        "--no-staged-reasoning",
        dest="staged_reasoning",
        action="store_false",
        default=None,
        help="Disable the staged reasoning strategy toggle",
    )
    parser.add_argument(
        "--staged-execution",
        dest="staged_execution_enabled",
        action="store_true",
        default=None,
        help="Execute approved plans with the staged execution runtime",
    )
    parser.add_argument(
        "--no-staged-execution",
        dest="staged_execution_enabled",
        action="store_false",
        default=None,
        help="Disable staged execution for approved plans",
    )
    parser.add_argument(
        "--staged-step-prompt-tokens",
        type=int,
        help="Optional prompt token budget for each staged step",
    )
    parser.add_argument(
        "--graph-thread-id",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,
        help="Show verbose backend/interrupt messages in assistant turns",
    )
    parser.add_argument(
        "--show-system-messages",
        "--show-system-message",
        dest="show_system_messages",
        action="store_true",
        default=None,
        help="Show system-level TUI messages and shutdown alerts",
    )
    parser.add_argument(
        "--hide-system-messages",
        "--hide-system-message",
        dest="show_system_messages",
        action="store_false",
        default=None,
        help="Hide system-level TUI messages and shutdown alerts",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove Python cache artifacts before starting harness",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch Textual UI shell",
    )
    parser.add_argument(
        "--indexer",
        action="store_true",
        help="Run in code indexer mode",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--debug-subsystem",
        dest="debug_subsystems",
        action="append",
        default=None,
        help="Enable debug logging for a subsystem (client, graph, tools, context, fama, ui, memory, state). May be repeated.",
    )
    parser.add_argument("--debug-tokens", action="store_true", default=None, help="Log every model token instead of sampling")
    parser.add_argument("--log-max-mb", type=int, help="Per-run log size cap in megabytes (default: 100)")
    parser.add_argument("--api-key", dest="api_key", help="API key for endpoint")
    parser.add_argument("--context-limit", type=int, help="Context window/token budget override")
    parser.add_argument("--max-prompt-tokens", type=int, help="Per-request prompt token budget")
    parser.add_argument("--max-completion-tokens", type=int, help="Per-turn maximum completion token budget")
    parser.add_argument("--reserve-completion-tokens", type=int, help="Reserved completion tokens")
    parser.add_argument("--reserve-tool-tokens", type=int, help="Reserved tool-call tokens")
    parser.add_argument(
        "--backend-unload-command",
        help="Shell command to unload a wedged backend model before retrying generation",
    )
    parser.add_argument("--summarize-at-ratio", type=float, help="Prompt usage ratio that triggers compaction")
    parser.add_argument("--recent-message-limit", type=int, help="Recent raw message retention limit")
    parser.add_argument("--max-summary-items", type=int, help="Maximum retrieved summary items")
    parser.add_argument("--max-artifact-snippets", type=int, help="Maximum retrieved artifact snippets")
    parser.add_argument(
        "--artifact-snippet-token-limit",
        type=int,
        help="Approximate token cap for each retrieved artifact snippet",
    )
    parser.add_argument(
        "--min-exploration-steps",
        type=int,
        help="Minimum required steps in DISCOVERY phase before allowing task completion",
    )
    parser.add_argument("--rewoo-lane-frames", dest="rewoo_lane_frames_enabled", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rewoo-planner-frame", dest="rewoo_planner_frame_enabled", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rewoo-solver-frame", dest="rewoo_solver_frame_enabled", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rewoo-refiner-frame", dest="rewoo_refiner_frame_enabled", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rewoo-frame-token-budget", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--fama-enabled",
        action="store_true",
        default=None,
        help="Explicitly enable FAMA failure-aware mitigation. Use this to override fama_enabled: false from config or .env.",
    )
    parser.add_argument(
        "--fama-disabled",
        action="store_true",
        help="Explicitly disable FAMA failure-aware mitigation at runtime. This is the only override that prevents the loop-mode guard from auto-enabling FAMA.",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers)
    
    return parser


def _resolve_session_id(harness: object | None, fallback: str | None = None) -> str:
    if harness is not None:
        state = getattr(harness, "state", None)
        thread_id = str(getattr(state, "thread_id", "") or "").strip()
        if thread_id:
            return thread_id
        conversation_id = str(getattr(harness, "conversation_id", "") or "").strip()
        if conversation_id:
            return conversation_id
    if fallback:
        return fallback
    return ""


def _print_shutdown_alert(session_id: str, status: str = "alert", *, reason: str | None = None, include_message: bool = True) -> None:
    payload = {
        "status": status,
        "session_id": session_id or "unknown",
    }
    if reason:
        payload["reason"] = reason
    if include_message:
        payload["message"] = (
            "smallctl closed via Ctrl+C"
            if status == "alert"
            else "smallctl TUI closed"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Handle subcommands first
    if args.command == "memory":
        return handle_memory_command(args)

    config = resolve_config(vars(args))

    setup_logging(
        config.debug,
        log_file=config.log_file,
        stream_to_terminal=not args.tui,
        debug_subsystems=config.debug_subsystems,
    )
    run_logger = create_run_logger(
        "logs",
        debug_subsystems=config.debug_subsystems,
        log_max_mb=config.log_max_mb,
        debug_tokens=config.debug_tokens,
    )

    def _emit_log_dir_finalized(final_dir: str) -> None:
        print(json.dumps({"status": "log_dir_finalized", "run_log_dir": final_dir}))
        run_logger.log("harness", "log_dir_finalized", run_log_dir=final_dir)

    run_logger._finalize_listener = _emit_log_dir_finalized

    log = logging.getLogger("smallctl")
    log_kv(
        log,
        logging.INFO,
        "smallctl_initialized",
        debug=config.debug,
        phase=config.phase,
        provider_profile=config.provider_profile,
        staged_reasoning=config.staged_reasoning,
        tui=bool(args.tui),
        run_log_dir=str(run_logger.run_dir),
    )
    if not args.tui:
        print(json.dumps({"status": "logging_ready", "run_log_dir": str(run_logger.run_dir)}))

        if config.debug:
            print(json.dumps(config.to_dict(), indent=2, sort_keys=True))

    for warning in config.compatibility_warnings:
        log_kv(log, logging.WARNING, "config_compatibility_warning", warning=warning)

    if args.cleanup:
        cleanup_result = run_cleanup(".")
        print(
            json.dumps(
                {"status": "cleanup_complete", **cleanup_result},
                indent=2,
                sort_keys=True,
            )
        )

    if args.tui:
        show_system_setting = getattr(config, "show_system_messages", None)
        show_system_messages = bool(show_system_setting) if show_system_setting is not None else False
        include_shutdown_message = show_system_setting is not False
        if show_system_messages and not sys.stdin.isatty():
            print(
                json.dumps(
                    {
                        "status": "warning",
                        "reason": "TUI launched without an interactive terminal (tty); rendering may fail or appear blank. Run without --tui or allocate a pseudo-tty (e.g., script, ssh -t).",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
        try:
            from .ui import SmallctlApp
        except Exception as exc:
            print(json.dumps({"status": "failed", "reason": f"TUI unavailable: {exc}"}, indent=2))
            return 1
        harness_kwargs = build_harness_config_kwargs(config, run_logger=run_logger, task=config.task)
        harness_kwargs["restore_graph_state_on_startup"] = config.restore_graph_state
        harness_kwargs["restore_thread_id"] = config.graph_thread_id
        harness_kwargs["show_system_messages"] = show_system_messages
        app = SmallctlApp(harness_kwargs=harness_kwargs)
        try:
            app.run()
        except KeyboardInterrupt:
            harness = getattr(app, "harness", None)
            if harness is not None:
                try:
                    harness.note_task_shutdown("keyboard_interrupt")
                except Exception:
                    pass
                try:
                    asyncio.run(harness.teardown())
                except Exception as exc:
                    log.warning("Harness teardown failed after Ctrl+C: %s", exc)
            _print_shutdown_alert(
                _resolve_session_id(
                    getattr(app, "harness", None),
                    fallback=str(getattr(app, "restore_thread_id", None) or "").strip() or None,
                ),
                reason="keyboard_interrupt",
                include_message=include_shutdown_message,
            )
            return 130
        except Exception as exc:
            log.exception("tui_fatal_error")
            print(f"\n[FATAL ERROR] TUI crashed: {exc}")
            # Ensure terminal mode is reset if possible
            sys.stdout.write("\033[?1049l")  # exit alternate screen buffer
            sys.stdout.write("\033[?1000l\033[?1002l\033[?1003l\033[?1006l\033[?1015l")
            sys.stdout.flush()
            return 1
        finally:
            # Force secondary terminal reset code just in case textual cleanup was partial
            sys.stdout.write("\033[?1049l")  # exit alternate screen buffer
            sys.stdout.write("\033[?1000l\033[?1006l\033[?25h")
            sys.stdout.flush()
        if getattr(app, "closed_by_ctrl_c", False):
            _print_shutdown_alert(
                _resolve_session_id(
                    getattr(app, "harness", None),
                    fallback=str(getattr(app, "restore_thread_id", None) or "").strip() or None,
                ),
                status="exited",
                reason="user_quit_confirmed",
                include_message=include_shutdown_message,
            )
            return 130
    elif config.task or config.restore_graph_state:
        harness = Harness(HarnessConfig(**build_harness_config_kwargs(config, run_logger=run_logger)))
        if config.restore_graph_state and not config.fresh_run:
            restored = harness.restore_graph_state(thread_id=config.graph_thread_id)
            if not restored:
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "reason": "No persisted graph state found.",
                            "thread_id": config.graph_thread_id,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 1
            if not config.task:
                print(
                    json.dumps(
                        {
                            "status": "graph_state_restored",
                            "thread_id": harness.state.thread_id,
                            "phase": harness.state.current_phase,
                            "step_count": harness.state.step_count,
                            "has_pending_interrupt": harness.has_pending_interrupt(),
                            "interrupt": harness.get_pending_interrupt(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
        interrupted = False
        try:
            result = asyncio.run(harness.run_auto(config.task))
        except KeyboardInterrupt:
            interrupted = True
            harness.note_task_shutdown("keyboard_interrupt")
            result = None
        except Exception as exc:
            log.exception("Harness run failed")
            result = {
                "status": "failed",
                "reason": str(exc),
                "error": {"type": "runtime", "message": str(exc), "details": {}},
            }
        finally:
            try:
                asyncio.run(harness.teardown())
            except Exception as exc:
                log.warning("Harness teardown failed: %s", exc)
        if interrupted:
            _print_shutdown_alert(_resolve_session_id(harness), reason="keyboard_interrupt")
            return 130
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("No task provided. Use --task to run a task.")

    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
