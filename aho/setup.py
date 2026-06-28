"""
aho/setup.py
------------
Trial-runner setup pieces for harness_runner.py.

This module extracts configuration, environment prep, and harness
initialization from the main runner to keep the top-level runner
as a thin pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.smallctl.harness import Harness
from src.smallctl.logging_utils import create_run_logger

from aho.logging_aho import AHOLogger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("aho/harness_config.json")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the harness configuration from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Strategy / system prompt construction
# ---------------------------------------------------------------------------

_THOUGHT_ARCHITECTURES = {
    "think_before_every_tool_call": (
        "Before every tool call, write a brief <think> block explaining why "
        "you are calling that tool and what you expect to learn."
    ),
    "silent": (
        "Do NOT include any thinking or reasoning blocks. "
        "Only emit tool calls and the final answer."
    ),
    "chain_of_thought": (
        "Reason step-by-step in plain text before using any tool."
    ),
    "reflection_after_tool": (
        "After each tool result, write one sentence summarising what you learned "
        "before deciding your next action."
    ),
    "reflection_plus": (
        "After each tool result, write a <think> block asking: 'Based on my goals, "
        "what is still missing (e.g., which keywords am I still lacking information for)?' "
        "Then decide your next action."
    ),
    "multi_phase_discovery": (
        "You MUST operate in three distinct phases:\n"
        "1. DISCOVERY: Use gathering tools to find all required facts. Transition when you have all constants.\n"
        "2. VERIFICATION: Call memory_update to list all required constants found so far.\n"
        "3. SYNTHESIS: Only then, implement the final code and call task_complete."
    ),
    "verification_chain": (
        "1. RETRIEVAL: Call long_context_lookup & summarize_report. READ them ONCE.\n"
        "2. DRAFT: Write the Python class.\n"
        "3. REVIEW: You MUST call `review_draft` to verify your code. Provide matching_keywords for the task requirements when available.\n"
        "4. COMPLETE: Finalize and call `task_complete`."
    ),
}

_DELIMITER_INSTRUCTIONS = {
    "xml": (
        "All tool calls MUST use the <tool_call> XML wrapper. "
        "Never place JSON outside a <tool_call> tag."
    ),
    "json": (
        "Express tool calls as raw JSON objects. "
        "Do not use XML tags around them."
    ),
    "markdown": (
        "Wrap tool calls in ```json code fences."
    ),
}

_ERROR_HANDLING = {
    "retry_with_hint": (
        "If a tool call fails, read the error message carefully and retry ONCE "
        "with a corrected argument. If it fails again, call task_fail."
    ),
    "fail_fast": (
        "If any tool call returns an error, immediately call task_fail with the reason."
    ),
    "ignore_and_continue": (
        "If a tool call fails, skip it and continue with the next step."
    ),
}


def build_system_prompt(strategy: dict[str, Any]) -> str:
    """Build the system prompt from strategy configuration."""
    thought_key = strategy.get("thought_architecture", "think_before_every_tool_call")
    thought = _THOUGHT_ARCHITECTURES.get(
        thought_key,
        _THOUGHT_ARCHITECTURES["think_before_every_tool_call"],
    )
    delimiter = _DELIMITER_INSTRUCTIONS.get(
        strategy.get("delimiter_style", "xml"),
        _DELIMITER_INSTRUCTIONS["xml"],
    )
    error_h = _ERROR_HANDLING.get(
        strategy.get("error_handling", "retry_with_hint"),
        _ERROR_HANDLING["retry_with_hint"],
    )
    addendum: str = strategy.get("system_prompt_addendum", "")
    forbidden: list[str] = strategy.get("forbidden_patterns", [])
    known_failures: list[str] = strategy.get("known_failures", [])

    parts = [
        "You are a helpful assistant with access to several tools: "
        "`weather_lookup`, `clothing_suggest`, `long_context_lookup`, `summarize_report`, and `artifact_read`.",
        "Large tool results are stored as artifacts (e.g., 'artifact A0001'). "
        "Use `artifact_read(artifact_id='A0001')` to read their full content.",
        thought,
        delimiter,
        error_h,
        "When you have a final answer, call task_complete with your recommendation.",
    ]
    if forbidden:
        parts.append(
            "FORBIDDEN patterns — never do any of the following: "
            + "; ".join(forbidden)
        )
    if known_failures:
        parts.append(
            "CRITICAL: Past attempts failed for the following reasons. DO NOT repeat these mistakes:\n- "
            + "\n- ".join(known_failures)
        )
    if addendum:
        parts.append(addendum)
    return "\n\n".join(p.strip() for p in parts if p.strip())


def build_task_string(cfg: dict[str, Any]) -> str:
    """Extract the task description from config."""
    return cfg.get("task", {}).get(
        "description",
        "Find the current weather in Paris, France, then recommend one specific "
        "clothing item for that weather.",
    )


def maybe_inject_known_failures(strategy: dict[str, Any], results_path: Path) -> dict[str, Any]:
    """
    Inject known failures from recent results into the strategy.
    Returns a new strategy dict with known_failures populated.
    """
    strategy = strategy.copy()
    try:
        if not results_path.exists():
            return strategy
        failures = []
        with open(results_path, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    for fm in r.get("failure_modes", []):
                        if "accuracy" in fm or "hallucinated" in fm:
                            failures.append(fm)
        if failures:
            # Deduplicate and take top 3
            unique_failures: list[str] = []
            for f in reversed(failures):
                if f not in unique_failures:
                    unique_failures.append(f)
            strategy["known_failures"] = unique_failures[:3]
    except Exception:
        pass
    return strategy


def apply_inline_results_to_prompt(system_prompt: str, inline_enabled: bool) -> str:
    """Modify system prompt for inline tool results mode."""
    if not inline_enabled:
        return system_prompt

    # Replace the conflicting artifact_read instruction
    system_prompt = system_prompt.replace(
        "Large tool results are stored as artifacts (e.g., 'artifact A0001'). Use `artifact_read(artifact_id='A0001')` to read their full content.",
        "Tool results include KEY FACTS directly in the response - you do NOT need to call artifact_read."
    )

    system_prompt += (
        "\n\n[INLINE TOOL RESULTS ENABLED]\n"
        "Tool results now include KEY FACTS directly in the response.\n"
        "• You do NOT need to call artifact_read - facts are already provided\n"
        "• You do NOT need to re-query the same information\n"
        "• Use the KEY FACTS immediately to implement the solution\n"
        "• Call task_complete as soon as you have all required information\n"
        "\nWorkflow: lookup → see facts → implement → complete"
    )
    return system_prompt


# ---------------------------------------------------------------------------
# Mock tool registry setup
# ---------------------------------------------------------------------------

def make_mock_registry(harness: Any, trial_seed: int, distilled_mode: bool = True) -> Any:
    """
    Builds a ToolRegistry that includes the standard core tools PLUS the
    two AHO mock tools. We don't modify src/smallctl/tools/register.py;
    instead we register the mocks into a fresh registry after the fact.

    NOTE: distilled_mode now defaults to True for inline tool results.
    Tools return KEY FACTS directly instead of full documents.
    Set distilled_mode=False for legacy full-document behavior.
    """
    from src.smallctl.tools.register import build_registry, register_mock_tool
    from src.smallctl.tools.base import ToolSpec, build_tool_schema
    from src.smallctl.tools.profiles import CORE_PROFILE
    from aho.mock_tools import clothing_suggest, review_draft, weather_lookup
    from aho.tool_deduplication import (
        cached_long_context_lookup as long_context_lookup,
        cached_summarize_report as summarize_report,
    )

    registry = build_registry(harness, registry_profiles=None)
    strategy_id = getattr(harness, "strategy_id", "") or (getattr(harness.config, "strategy_id", "") if hasattr(harness, "config") else "")
    is_challenge = bool(strategy_id and "challenge" in str(strategy_id))
    if not is_challenge:
        # Filter for SLM efficiency: ONLY Carbon task tools
        keep = {
            "artifact_read",
            "memory_update",
            "task_complete",
            "task_fail",
            "loop_status",
            "long_context_lookup",
            "summarize_report",
            "review_draft",
        }
        registry._specs = {k: v for k, v in registry._specs.items() if k in keep}


    # weather_lookup
    register_mock_tool(
        registry,
        name="weather_lookup",
        description="Get weather data.",
        handler=lambda **kw: weather_lookup(trial_seed=trial_seed, **kw),
        required=["city"],
        properties={"city": {"type": "string"}},
    )

    # clothing_suggest
    register_mock_tool(
        registry,
        name="clothing_suggest",
        description="Suggest clothing.",
        handler=clothing_suggest,
        required=["condition"],
        properties={"condition": {"type": "string"}},
    )

    # long_context_lookup
    registry.register(
        ToolSpec(
            name="long_context_lookup",
            description=(
                "Fetch climate policy KEY FACTS. "
                "Returns distilled facts (~80 tokens) by default. "
                "Set distilled=false ONLY if you need the full verbose document (~3000 tokens)."
            ),
            schema=build_tool_schema(
                required=["topic"],
                properties={
                    "topic": {"type": "string"},
                    "distilled": {"type": "boolean", "default": distilled_mode}
                },
            ),
            handler=lambda **kw: long_context_lookup(
                distilled=kw.get("distilled", distilled_mode),
                **{k: v for k, v in kw.items() if k != "distilled"}
            ),
            tier="tier1",
            category="mock",
            risk="low",
            allowed_phases={"explore", "plan", "execute", "verify"},
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
        )
    )

    # summarize_report
    registry.register(
        ToolSpec(
            name="summarize_report",
            description=(
                "Fetch carbon market KEY FACTS. "
                "Returns distilled facts (~65 tokens) by default. "
                "Set distilled=false ONLY if you need the full verbose report (~2500 tokens)."
            ),
            schema=build_tool_schema(
                required=["subject"],
                properties={
                    "subject": {"type": "string"},
                    "distilled": {"type": "boolean", "default": distilled_mode}
                },
            ),
            handler=lambda **kw: summarize_report(
                distilled=kw.get("distilled", distilled_mode),
                **{k: v for k, v in kw.items() if k != "distilled"}
            ),
            tier="tier1",
            category="mock",
            risk="low",
            allowed_phases={"explore", "plan", "execute", "verify"},
            allowed_modes={"loop"},
            profiles={CORE_PROFILE},
        )
    )

    # review_draft
    register_mock_tool(
        registry,
        name="review_draft",
        description="Review a synthesis draft before finalisation.",
        handler=review_draft,
        required=["draft"],
        properties={
            "draft": {"type": "string"},
            "matching_keywords": {"type": "array", "items": {"type": "string"}},
        },
    )

    return registry


# ---------------------------------------------------------------------------
# Harness factory
# ---------------------------------------------------------------------------

def create_harness(
    cfg: dict[str, Any],
    system_prompt: str,
    strategy: dict[str, Any],
    logger: AHOLogger,
) -> Harness:
    """Create and configure a Harness instance from config."""
    from src.smallctl.logging_utils import create_run_logger

    run_logger = create_run_logger(f"aho/logs/trials/{cfg.get('strategy_id', 'unknown')}")
    provider = cfg.get("provider_profile", "lmstudio")
    max_steps = int(strategy.get("max_steps", 10))

    harness = Harness(
        endpoint=cfg["endpoint"],
        model=cfg["model"],
        max_prompt_tokens=cfg.get("max_prompt_tokens"),
        context_limit=cfg.get("context_limit", 8192),
        reasoning_mode="tags" if provider == "lmstudio" else "off",
        chat_endpoint="/chat/completions",
        api_key=cfg.get("api_key", "local-dev-key"),
        use_ansible=False,
        run_logger=run_logger,
        summarize_at_ratio=cfg.get("summarize_at_ratio", 0.8),
        summarizer_endpoint=cfg.get("summarizer_endpoint"),
        summarizer_model=cfg.get("summarizer_model"),
        summarizer_api_key=cfg.get("summarizer_api_key"),
        reserve_completion_tokens=512,
        reserve_tool_tokens=256,
        strategy=strategy,
        strategy_prompt=system_prompt,
        provider_profile=provider,
        fama_enabled=cfg.get("fama_enabled", True),
        loop_guard_enabled=cfg.get("loop_guard_enabled", True),
        graph_node_timeout_sec=cfg.get("graph_node_timeout_sec", 300),
        graph_model_call_timeout_sec=cfg.get("graph_model_call_timeout_sec", 600),
        graph_dispatch_tools_timeout_sec=cfg.get("graph_dispatch_tools_timeout_sec", 300),
        graph_idle_watchdog_sec=cfg.get("graph_idle_watchdog_sec", 300),
        needs_human_timeout_sec=cfg.get("needs_human_timeout_sec", 600),
        artifact_summarization_threshold=cfg.get("artifact_summarization_threshold", 1200),
        staged_execution_enabled=cfg.get("staged_execution_enabled", False),
        staged_step_prompt_tokens=cfg.get("staged_step_prompt_tokens", 4096),
        tool_plan_allow_git=cfg.get("tool_plan_allow_git", False),
        min_exploration_steps=cfg.get("min_exploration_steps", 1),
        chunk_mode_min_bytes=cfg.get("chunk_mode_min_bytes", 4096),
        chunk_mode_new_file_only=cfg.get("chunk_mode_new_file_only", True),
        chunk_mode_supported_models=cfg.get("chunk_mode_supported_models", ["qwen3.5", "llama3.1", "deepseek-v3", "deepseek-v4"]),
        small_model_soft_write_chars=cfg.get("small_model_soft_write_chars", 2000),
        small_model_hard_write_chars=cfg.get("small_model_hard_write_chars", 4000),
        new_file_chunk_mode_line_estimate=cfg.get("new_file_chunk_mode_line_estimate", 100),
        allow_multi_section_turns_for_small_edits=cfg.get("allow_multi_section_turns_for_small_edits", True),
        failed_local_patch_limit=cfg.get("failed_local_patch_limit", 2),
        enable_write_intent_recovery=cfg.get("enable_write_intent_recovery", True),
        enable_assistant_code_write_recovery=cfg.get("enable_assistant_code_write_recovery", True),
        write_recovery_min_confidence=cfg.get("write_recovery_min_confidence", "high"),
        write_recovery_allow_raw_text_targets=cfg.get("write_recovery_allow_raw_text_targets", True),
        solver_refine_enabled=cfg.get("solver_refine_enabled", False),
        solver_refine_max_passes=cfg.get("solver_refine_max_passes", 1),
        solver_refine_on_final_answer=cfg.get("solver_refine_on_final_answer", True),
        solver_refine_on_patch_plan=cfg.get("solver_refine_on_patch_plan", True),
        solver_refine_on_task_complete=cfg.get("solver_refine_on_task_complete", True),
        solver_refine_token_budget=cfg.get("solver_refine_token_budget", 700),
        test_time_scaling_enabled=cfg.get("test_time_scaling_enabled", False),
        test_time_scaling_runtimes=cfg.get("test_time_scaling_runtimes", ["staged_execution"]),
        test_time_scaling_trigger=cfg.get("test_time_scaling_trigger", "retry_or_explicit"),
        test_time_scaling_max_candidates=cfg.get("test_time_scaling_max_candidates", 3),
        test_time_scaling_min_candidates=cfg.get("test_time_scaling_min_candidates", 2),
        test_time_scaling_policy=cfg.get("test_time_scaling_policy", "proposal_then_execute"),
        test_time_scaling_strategy=cfg.get("test_time_scaling_strategy", "diverse_nudges"),
        test_time_scaling_score_threshold=cfg.get("test_time_scaling_score_threshold", 0.85),
        test_time_scaling_parallel_max=cfg.get("test_time_scaling_parallel_max", 1),
        test_time_scaling_timeout_sec=cfg.get("test_time_scaling_timeout_sec", 120),
        test_time_scaling_mutating_parallel_enabled=cfg.get("test_time_scaling_mutating_parallel_enabled", False),
        test_time_scaling_all_fail_action=cfg.get("test_time_scaling_all_fail_action", "fallback_normal_retry"),
    )
    harness.guards.max_steps = max_steps
    harness.state.scratchpad["_max_steps"] = max_steps
    harness.strategy_id = cfg.get("strategy_id")
    if hasattr(harness, "config"):
        harness.config.strategy_id = cfg.get("strategy_id")
    return harness


# ---------------------------------------------------------------------------
# Dispatcher/interceptor setup
# ---------------------------------------------------------------------------

def create_interceptors(cfg: dict[str, Any], registry: Any, harness: Harness) -> list[Any]:
    """
    Create tool interceptors based on configuration.
    Returns a list of interceptors to be applied to the dispatcher.
    """
    from aho.artifact_guard import ArtifactDedupeInterceptor
    from aho.tool_guard import ToolCallGuard
    from aho.progress_guard import ProgressGuard
    from aho.auto_batch_dispatcher import AutoBatchDispatcher

    interceptors: list[Any] = []

    # 1. Redundancy Guard
    interceptors.append(ArtifactDedupeInterceptor())

    # 2. Auto-batching
    if cfg.get("auto_batch_enabled", True):
        batch_dispatcher = AutoBatchDispatcher(
            batch_window_ms=cfg.get("auto_batch_window_ms", 50.0),
            max_batch_size=cfg.get("auto_batch_max_size", 5),
        )
        interceptors.append(batch_dispatcher)
        harness._batch_dispatcher = batch_dispatcher

    # 3. Tool Call Guard
    if cfg.get("tool_guard_enabled", False):
        guard = ToolCallGuard(valid_tools=set(registry._specs.keys()))
        interceptors.append(guard)
        harness._tool_guard = guard

    # 4. Progress Guard
    if cfg.get("adaptive_steps", False):
        max_loops = cfg.get("max_loops", 2)
        progress_guard = ProgressGuard(
            message_provider=lambda: harness.state.recent_messages,
            max_loops=max_loops
        )
        interceptors.append(progress_guard)
        harness._progress_guard = progress_guard

    return interceptors



# ---------------------------------------------------------------------------
# Metrics recording
# ---------------------------------------------------------------------------

def record_trial_metrics(
    cfg: dict[str, Any],
    trial_id: int,
    harness: Harness,
    result: dict[str, Any],
    elapsed: float,
) -> None:
    """Record inline results metrics for a completed trial."""
    try:
        from aho.inline_results_metrics import record_inline_metrics

        inline_cfg = cfg.get("inline_tool_results", {})
        used_inline = inline_cfg.get("enabled", cfg.get("distilled_mode", True))

        # Estimate baseline tokens (without inline)
        baseline_tokens = harness.state.token_usage * 3  # Approximate

        record_inline_metrics(
            trial_id=str(trial_id),
            tokens_inline=harness.state.token_usage,
            tokens_baseline=baseline_tokens,
            facts_extracted=sum(1 for r in harness.state.tool_execution_records.values()
                               if r.get("tool_name") in ["long_context_lookup", "summarize_report"]),
            validation_confidence=0.8 if used_inline else 0.0,
            task_completed=result.get("status") == "completed",
            fallback_count=0,
            duration=elapsed,
            branch="inline_results" if used_inline else "baseline",
        )
    except Exception:
        pass


def record_self_improving_success(
    cfg: dict[str, Any],
    trial_id: int,
    task_string: str,
    result: dict[str, Any],
) -> None:
    """Record success for self-improving mode if enabled."""
    try:
        if cfg.get("strategy", {}).get("dynamic_few_shot_mode") == "self_improving":
            from aho.self_improving import get_batch_library
            get_batch_library().add_success(
                trial_id=trial_id,
                task=task_string,
                result=result
            )
    except Exception:
        pass


def log_batch_statistics(harness: Harness, logger: AHOLogger) -> None:
    """Log auto-batch statistics if enabled."""
    if hasattr(harness, '_batch_dispatcher'):
        try:
            batch_stats = harness._batch_dispatcher.get_stats()
            logger.info("runner", "auto_batch_stats",
                       f"Batching: {batch_stats['batched_calls']}/{batch_stats['total_calls']} calls batched "
                       f"({batch_stats['batch_rate']:.1%}), speedup: {batch_stats.get('speedup', 'N/A')}",
                       **batch_stats)
        except Exception as e:
            logger.debug("runner", "batch_stats_error", str(e))
