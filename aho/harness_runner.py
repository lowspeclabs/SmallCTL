from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

# Add repo root to path so `from src.smallctl import ...` resolves correctly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.smallctl.harness import Harness
from src.smallctl.tools.dispatcher import ToolDispatcher, PipelineDispatcher, ToolInterceptor

from aho.logging_aho import AHOLogger, create_aho_logger, setup_aho_logging
from aho.tool_deduplication import reset_tool_cache
from aho.setup import (
    load_config,
    build_system_prompt,
    build_task_string,
    maybe_inject_known_failures,
    apply_inline_results_to_prompt,
    make_mock_registry,
    create_harness,
    create_interceptors,
    record_trial_metrics,
    record_self_improving_success,
    log_batch_statistics,
    CONFIG_PATH,
)

_LOG = logging.getLogger("aho.runner")


# Config loading moved to aho/setup.py


# ---------------------------------------------------------------------------
# Task / system prompt construction
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
            "FORBIDDEN patterns \u2014 never do any of the following: "
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
    return cfg.get("task", {}).get(
        "description",
        "Find the current weather in Paris, France, then recommend one specific "
        "clothing item for that weather.",
    )


# Mock tool registry injection moved to aho/setup.py (make_mock_registry)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

async def run_single_trial(
    cfg: dict[str, Any],
    trial_id: int,
    logger: AHOLogger,
) -> dict[str, Any]:
    """
    Run one trial of the current strategy.

    Returns a trial result dict.  Never raises — all exceptions are caught
    and recorded as errors so the researcher loop keeps going.
    """
    from src.smallctl.harness import Harness
    from src.smallctl.logging_utils import create_run_logger

    # Reset tool deduplication cache for clean trial state
    reset_tool_cache()
    logger.debug("runner", "cache_reset", f"Tool deduplication cache reset for trial {trial_id}")

    strategy = maybe_inject_known_failures(
        cfg.get("strategy", {}).copy(),
        Path("aho/results.jsonl"),
    )

    # Modify system prompt for inline tool results BEFORE building
    inline_enabled = cfg.get("inline_tool_results", {}).get("enabled", cfg.get("distilled_mode", True))
    system_prompt = apply_inline_results_to_prompt(
        build_system_prompt(strategy),
        inline_enabled,
    )

    task_string = build_task_string(cfg)
    trial_seed = trial_id  # deterministic per trial

    logger.info("runner", "trial_start", f"trial {trial_id} starting",
                trial_id=trial_id, strategy_id=cfg.get("strategy_id"), max_steps=int(strategy.get("max_steps", 10)))
    t0 = time.monotonic()
    harness = None

    try:
        harness = create_harness(cfg, system_prompt, strategy, logger)
        harness.state.scratchpad["_trial_seed"] = trial_seed

        inline_cfg = cfg.get("inline_tool_results", {})
        distilled_mode = inline_cfg.get("enabled", cfg.get("distilled_mode", True))
        mock_registry = make_mock_registry(harness, trial_seed, distilled_mode=distilled_mode)
        harness.registry = mock_registry
        if hasattr(harness.dispatcher, "registry"):
            harness.dispatcher.registry = mock_registry

        if cfg.get("mock_llm"):
            from aho.mock_client import MockOpenAICompatClient
            harness.client = MockOpenAICompatClient(
                base_url=cfg["endpoint"],
                model=cfg["model"]
            )
            logger.info("runner", "mock_llm", "Injected MockOpenAICompatClient for deterministic deterministic playback")

        # Keep the default inline_token_limit (250 tokens) so that large tool
        # results from long_context_lookup / summarize_report are NOT inlined
        # verbatim into conversation message content.  They are stored as compact
        # artifact references instead, which prevents LM Studio 500 errors caused
        # by oversized request payloads.  The model can still retrieve full content
        # via artifact_read if needed.
        # (Previously set to 1000 but that caused ~3100-token docs to be fully
        #  inlined, blowing the 7000-token prompt limit.)

        interceptors = create_interceptors(cfg, mock_registry, harness)
        if interceptors:
            harness.dispatcher = PipelineDispatcher(harness.dispatcher, interceptors)

        try:
            result = await asyncio.wait_for(
                harness.run_task(task_string),
                timeout=float(cfg.get("trial_timeout_sec", 120)),
            )
        finally:
            try:
                teardown_fn = getattr(harness, "teardown", None)
                if callable(teardown_fn):
                    await teardown_fn()
            except Exception as t_exc:
                logger.error("runner", "teardown_error", str(t_exc))

        elapsed = time.monotonic() - t0
        logger.info("runner", "trial_complete",
                    f"trial {trial_id} done in {elapsed:.1f}s — status={result.get('status')}",
                    trial_id=trial_id, elapsed=elapsed, status=result.get("status"))
        log_batch_statistics(harness, logger)
        record_trial_metrics(cfg, trial_id, harness, result, elapsed)
        record_self_improving_success(cfg, trial_id, task_string, result)

        state_dict = harness.state.to_dict() if hasattr(harness, "state") else {}
        return {
            "trial_id": trial_id,
            "result": result,
            "token_usage": harness.state.token_usage,
            "tool_records": {
                k: dict(v) for k, v in harness.state.tool_execution_records.items()
            },
            "step_count": harness.state.step_count,
            "elapsed_sec": round(elapsed, 2),
            "failure_events": state_dict.get("failure_events", []),
            "stagnation_counters": state_dict.get("stagnation_counters", {}),
            "tool_history": state_dict.get("tool_history", []),
            "error": None,
        }

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        msg = f"trial {trial_id} timed out after {elapsed:.1f}s"
        logger.bug("trial_timeout", msg,
                   exception="asyncio.TimeoutError",
                   context={"trial_id": trial_id, "strategy_id": cfg.get("strategy_id")})
        state_dict = harness.state.to_dict() if (harness and hasattr(harness, "state")) else {}
        return {
            "trial_id": trial_id,
            "result": {"status": "failed", "reason": "timeout"},
            "token_usage": 0,
            "tool_records": {},
            "step_count": 0,
            "elapsed_sec": round(elapsed, 2),
            "failure_events": state_dict.get("failure_events", []),
            "stagnation_counters": state_dict.get("stagnation_counters", {}),
            "tool_history": state_dict.get("tool_history", []),
            "error": "timeout",
        }

    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc()
        logger.bug("trial_exception", f"trial {trial_id} raised {type(exc).__name__}",
                   exception=tb,
                   context={"trial_id": trial_id, "strategy_id": cfg.get("strategy_id")})

        # One-Click Failing Test Generation
        if cfg.get("generate_test_on_crash"):
            test_dir = Path("tests")
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / f"test_crash_{uuid.uuid4().hex[:6]}.py"
            try:
                # Retrieve trace of recent messages if available
                state_dump = getattr(harness.state, "scratchpad", {}) if (harness and hasattr(harness, "state")) else {}
                test_code = f"""# Auto-generated crash reproduction
import pytest
from src.smallctl.harness import Harness

@pytest.mark.asyncio
async def test_auto_crash_repro():
    # Exception: {type(exc).__name__}: {exc}
    # Tracked state: {str(state_dump).replace('"', '')}
    # Instantiate harness and replay initial parts
    assert False, "Reproducing: {str(exc).replace('"', '')}"
"""
                test_file.write_text(test_code)
                logger.debug("runner", "test_generated", f"Wrote failing test to {test_file}")
            except Exception as test_ex:
                logger.error("runner", "test_generation_failed", str(test_ex))

        state_dict = harness.state.to_dict() if (harness and hasattr(harness, "state")) else {}
        return {
            "trial_id": trial_id,
            "result": {"status": "failed", "reason": str(exc)},
            "token_usage": 0,
            "tool_records": {},
            "step_count": 0,
            "elapsed_sec": round(elapsed, 2),
            "failure_events": state_dict.get("failure_events", []),
            "stagnation_counters": state_dict.get("stagnation_counters", {}),
            "tool_history": state_dict.get("tool_history", []),
            "error": f"{type(exc).__name__}: {exc}",
        }



# ---------------------------------------------------------------------------
# N-trial runner
# ---------------------------------------------------------------------------

async def run_n_trials(
    cfg: dict[str, Any],
    logger: AHOLogger,
    only_trial: int | None = None,
) -> list[dict[str, Any]]:
    n = cfg.get("n_trials", 5)
    trial_ids = [only_trial] if only_trial is not None else list(range(n))
    
    # Reset self-improving library for new batch
    try:
        from aho.self_improving import reset_batch_library
        reset_batch_library()
    except Exception:
        pass

    logger.info("runner", "batch_start", f"launching {len(trial_ids)} trial(s)",
                n_trials=len(trial_ids), strategy_id=cfg.get("strategy_id"))

    # Limit concurrency to avoid overwhelming local backends
    sem = asyncio.Semaphore(1)

    async def run_with_sem(tid):
        async with sem:
            return await run_single_trial(cfg, tid, logger)

    # Gather runs with concurrency limit
    results = await asyncio.gather(
        *[run_with_sem(tid) for tid in trial_ids],
        return_exceptions=False,  # exceptions are caught per-trial above
    )
    # Log self-improving stats if enabled
    try:
        from aho.self_improving import get_batch_library
        stats = get_batch_library().get_stats()
        if stats["count"] > 0:
            logger.info("runner", "self_improving_stats",
                       f"Accumulated {stats['count']} successful examples for learning",
                       examples_count=stats["count"], avg_tokens=stats["avg_tokens"])
    except Exception:
        pass
    
    logger.info("runner", "batch_complete", f"{len(results)} trial(s) finished",
                n_completed=len(results))
    return list(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AHO harness runner")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to harness_config.json")
    parser.add_argument("--trial-id", type=int, default=None, help="Run a single trial by ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--summarizer-endpoint", help="OpenAI-compatible endpoint for context summarization")
    parser.add_argument("--summarizer-model", help="Model to use for context summarization")
    parser.add_argument("--summarizer-api-key", help="API key for the summarizer endpoint")
    parser.add_argument("--mock-llm", action="store_true", help="Use a mocked deterministic LLM instead of real API")
    parser.add_argument("--generate-failing-test", action="store_true", help="Generate pytest for crash analysis")
    args = parser.parse_args(argv)

    setup_aho_logging(args.debug)
    logger = create_aho_logger()

    cfg = load_config(Path(args.config))
    cfg["mock_llm"] = args.mock_llm
    cfg["generate_test_on_crash"] = args.generate_failing_test

    logger.info("runner", "runner_start",
                f"strategy={cfg.get('strategy_id')} model={cfg.get('model')}",
                strategy_id=cfg.get("strategy_id"), model=cfg.get("model"))

    # CLI overrides for summarizer
    if args.summarizer_endpoint:
        cfg["summarizer_endpoint"] = args.summarizer_endpoint
    if args.summarizer_model:
        cfg["summarizer_model"] = args.summarizer_model
    if args.summarizer_api_key:
        cfg["summarizer_api_key"] = args.summarizer_api_key

    with contextlib.redirect_stdout(sys.stderr):
        trials = asyncio.run(run_n_trials(cfg, logger, only_trial=args.trial_id))

    # Write trial results to stdout for researcher.py to parse
    print(json.dumps(trials, indent=2, default=str))


if __name__ == "__main__":
    main()
