from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .cleanup import run_cleanup
from .config import resolve_config
from .harness import Harness
from .logging_utils import create_run_logger, log_kv, setup_logging
from .memory_cli import build_memory_parser, handle_memory_command, memory_cli
from .presets import list_presets


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
        "--graph-thread-id",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path",
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
    parser.add_argument("--api-key", dest="api_key", help="API key for endpoint")
    parser.add_argument("--context-limit", type=int, help="Context window/token budget override")
    parser.add_argument("--max-prompt-tokens", type=int, help="Per-request prompt token budget")
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
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers)
    
    return parser


def _resolve_session_id(harness: object | None) -> str:
    if harness is None:
        return ""
    state = getattr(harness, "state", None)
    thread_id = str(getattr(state, "thread_id", "") or "").strip()
    if thread_id:
        return thread_id
    conversation_id = str(getattr(harness, "conversation_id", "") or "").strip()
    if conversation_id:
        return conversation_id
    return ""


def _print_shutdown_alert(session_id: str) -> None:
    print(
        json.dumps(
            {
                "status": "alert",
                "message": "smallctl closed via Ctrl+C",
                "session_id": session_id or "unknown",
            },
            indent=2,
            sort_keys=True,
        )
    )


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Handle subcommands first
    if args.command == "memory":
        return handle_memory_command(args)

    config = resolve_config(vars(args))

    setup_logging(config.debug, log_file=config.log_file, stream_to_terminal=not args.tui)
    run_logger = create_run_logger("logs")
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
        try:
            from .ui import SmallctlApp
        except Exception as exc:
            print(json.dumps({"status": "failed", "reason": f"TUI unavailable: {exc}"}, indent=2))
            return 1
        strategy = {"thought_architecture": "staged_reasoning"} if config.staged_reasoning else None
        harness_kwargs = {
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
            "checkpoint_on_exit": config.checkpoint_on_exit,
            "checkpoint_path": config.checkpoint_path,
            "graph_checkpointer": config.graph_checkpointer,
            "graph_checkpoint_path": config.graph_checkpoint_path,
            "fresh_run": config.fresh_run,
            "fresh_run_turns": config.fresh_run_turns,
            "planning_mode": config.planning_mode,
            "contract_flow_ui": config.contract_flow_ui,
            "strategy": strategy,
            "restore_graph_state_on_startup": config.restore_graph_state,
            "restore_thread_id": config.graph_thread_id,
            "context_limit": config.context_limit,
            "max_prompt_tokens": config.max_prompt_tokens,
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
            "indexer": config.indexer,
            "run_logger": run_logger,
            "task": config.task,
        }
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
            _print_shutdown_alert(_resolve_session_id(getattr(app, "harness", None)))
            return 130
        except Exception as exc:
            log.exception("tui_fatal_error")
            print(f"\n[FATAL ERROR] TUI crashed: {exc}")
            # Ensure terminal mode is reset if possible
            sys.stdout.write("\033[?1000l\033[?1002l\033[?1003l\033[?1006l\033[?1015l")
            sys.stdout.flush()
            return 1
        finally:
            # Force secondary terminal reset code just in case textual cleanup was partial
            sys.stdout.write("\033[?1000l\033[?1006l\033[?25h")
            sys.stdout.flush()
        if getattr(app, "closed_by_ctrl_c", False):
            _print_shutdown_alert(_resolve_session_id(getattr(app, "harness", None)))
    elif config.task or config.restore_graph_state:
        strategy = {"thought_architecture": "staged_reasoning"} if config.staged_reasoning else None
        harness = Harness(
            endpoint=config.endpoint,
            model=config.model,
            phase=config.phase,
            provider_profile=config.provider_profile,
            api_key=config.api_key,
            tool_profiles=config.tool_profiles,
            reasoning_mode=config.reasoning_mode,
            thinking_visibility=config.thinking_visibility,
            thinking_start_tag=config.thinking_start_tag,
            thinking_end_tag=config.thinking_end_tag,
            chat_endpoint=config.chat_endpoint,
            checkpoint_on_exit=config.checkpoint_on_exit,
            checkpoint_path=config.checkpoint_path,
            graph_checkpointer=config.graph_checkpointer,
            graph_checkpoint_path=config.graph_checkpoint_path,
            fresh_run=config.fresh_run,
            fresh_run_turns=config.fresh_run_turns,
            contract_flow_ui=config.contract_flow_ui,
            strategy=strategy,
            context_limit=config.context_limit,
            max_prompt_tokens=config.max_prompt_tokens,
            reserve_completion_tokens=config.reserve_completion_tokens,
            reserve_tool_tokens=config.reserve_tool_tokens,
            first_token_timeout_sec=config.first_token_timeout_sec,
            healthcheck_url=config.healthcheck_url,
            restart_command=config.restart_command,
            startup_grace_period_sec=config.startup_grace_period_sec,
            max_restarts_per_hour=config.max_restarts_per_hour,
            backend_healthcheck_url=config.backend_healthcheck_url,
            backend_restart_command=config.backend_restart_command,
            backend_unload_command=config.backend_unload_command,
            backend_healthcheck_timeout_sec=config.backend_healthcheck_timeout_sec,
            backend_restart_grace_sec=config.backend_restart_grace_sec,
            summarize_at_ratio=config.summarize_at_ratio,
            recent_message_limit=config.recent_message_limit,
            max_summary_items=config.max_summary_items,
            max_artifact_snippets=config.max_artifact_snippets,
            artifact_snippet_token_limit=config.artifact_snippet_token_limit,
            indexer=config.indexer,
            run_logger=run_logger,
        )
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
            _print_shutdown_alert(_resolve_session_id(harness))
            return 130
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("No task provided. Use --task to run a task.")

    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
