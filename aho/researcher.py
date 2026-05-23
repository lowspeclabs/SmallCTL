"""
aho/researcher.py
-----------------
The recursive improvement loop.

Analogous to the LOOP FOREVER section of karpathy/autoresearch's program.md.

Git integration (requires git in PATH)
--------------------------------------
- Each KEPT strategy is committed:  git commit aho/harness_config.json -m "..."
- DISCARDED strategies are reverted: git checkout -- aho/harness_config.json
- The full git log therefore IS the experiment history
- results.jsonl records the short git hash for cross-referencing

Harness code change detection
------------------------------
- Before every iteration the SHA-256 digest of src/smallctl/ + aho/*.py is
  computed and compared to the previous iteration.
- If the harness code changed (e.g. you edited harness_runner.py), a warning
  is emitted to the researcher channel and logged to bug_tracker.jsonl so the
  score shift is understood as a confounding factor, not a strategy improvement.

Algorithm
---------
    LOOP FOREVER:
        1. Load harness_config.json  (current strategy)
        2. Detect harness code drift (warn + log if changed)
        3. Read recent results.jsonl
        4. Determine best_score from kept entries
        5. Propose ONE targeted mutation via the LLM
        6. Apply the patch → save mutated config
        7. Run harness_runner.py  (N parallel trials via subprocess)
        8. Score with eval.py
        9. Log to results.jsonl (with git hash)
       10. Keep → git commit  |  Discard → git checkout --
       11. Log metrics, sleep, repeat

Error recovery
--------------
- Mutation proposal failure  → skip iteration, increment consecutive_failures
- Runner crash / non-zero exit → git checkout --, log bug, increment counter
- JSON parse failure          → git checkout --, log bug, increment counter
- consecutive_failures >= MAX_CONSECUTIVE_FAILURES → pause 60s, reset counter
- All errors logged to aho/bug_tracker.jsonl for cross-run analysis

Metrics emitted each iteration
-------------------------------
  mean_harness_score, pass_at_n, mean_token_usage, n_bugs, failure_modes[]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aho.eval import score_strategy
from aho.harness_evolver import maybe_evolve_harness
from aho.logging_aho import create_aho_logger, setup_aho_logging, AHOLogger
from aho.git_utils import (
    git_short_hash,
    git_commit_config,
    git_discard_config,
    git_is_available,
)
from aho.config import (
    load_config,
    save_config,
    load_recent_results,
    best_score_from_results,
    log_result,
    CONFIG_PATH,
    RESULTS_PATH,
)
from aho.mutation import (
    propose_mutation,
    apply_patch,
)

# Directories to watch for harness code drift
_DRIFT_WATCH_DIRS = [
    Path("src/smallctl"),
    Path("aho"),
]
# Files excluded from the drift hash (they change every run by design)
_DRIFT_EXCLUDE_SUFFIXES = {".pyc", ".jsonl", ".bak", ".log"}

MAX_CONSECUTIVE_FAILURES = 5
PAUSE_ON_FAILURE_SEC = 60


# Git helpers moved to aho/git_utils.py


# Harness code drift detection moved inline to research_loop


# Researcher LLM prompt moved to aho/mutation.py (RESEARCHER_SYSTEM_PROMPT)


# Config / results helpers moved to aho/config.py


# Mutation proposal moved to aho/mutation.py


# ---------------------------------------------------------------------------
# Runner subprocess
# ---------------------------------------------------------------------------

def _run_trials(cfg: dict[str, Any], logger: AHOLogger, timeout: int = 300) -> list[dict[str, Any]]:
    """
    Launch harness_runner.py in a subprocess, parse its JSON stdout.
    Returns trial list or raises RuntimeError on failure.
    """
    logger.info("runner", "subprocess_start", "launching harness_runner.py",
                strategy_id=cfg.get("strategy_id"), n_trials=cfg.get("n_trials", 5))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "aho/harness_runner.py"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"harness_runner timed out after {timeout}s") from exc

    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        stderr_tail = proc.stderr[-1000:] if proc.stderr else "(no stderr)"
        logger.bug("runner_crash",
                   f"harness_runner exited {proc.returncode}",
                   exception=stderr_tail,
                   context={"strategy_id": cfg.get("strategy_id"), "elapsed": elapsed})
        raise RuntimeError(f"harness_runner exited {proc.returncode}: {stderr_tail}")

    try:
        trials = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        preview = proc.stdout[:300] if proc.stdout else "(empty)"
        logger.bug("runner_parse_error",
                   "could not parse harness_runner stdout as JSON",
                   exception=str(exc),
                   context={"stdout_preview": preview, "strategy_id": cfg.get("strategy_id")})
        raise RuntimeError(f"JSON parse error from harness_runner: {exc}") from exc

    logger.info("runner", "subprocess_done",
                f"got {len(trials)} trial(s) in {elapsed:.1f}s",
                n_trials=len(trials), elapsed=round(elapsed, 2))
    return trials


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------

async def research_loop(
    logger: AHOLogger,
    *,
    max_iterations: int | None = None,
    dry_run: bool = False,
) -> None:
    """
    The main LOOP FOREVER.

    Parameters
    ----------
    max_iterations  — stop after N iterations (for testing; None = run forever)
    dry_run         — skip the actual subprocess call (for unit tests)
    """
    iteration = 0
    consecutive_failures = 0
    best_score = 0.0
    prev_harness_hash: str | None = None
    use_git = git_is_available()

    if use_git:
        logger.info("researcher", "git_available",
                    f"git detected — using commit/checkout for rollback (HEAD={git_short_hash()})")
    else:
        logger.warning("researcher", "git_unavailable",
                       "git not available — rollback disabled; install git for full experiment history")

    # Bootstrap best_score from any existing results
    existing = load_recent_results(50)
    if existing:
        best_score = best_score_from_results(existing)
        logger.info("researcher", "bootstrap",
                    f"loaded {len(existing)} existing results; best_score={best_score:.4f}",
                    best_score=best_score, n_existing=len(existing))

    while True:
        if max_iterations is not None and iteration >= max_iterations:
            logger.info("researcher", "loop_done",
                        f"reached max_iterations={max_iterations}", iterations=iteration)
            break

        logger.set_iteration(iteration)
        logger.info("researcher", "iteration_start", f"=== iteration {iteration} ===",
                    iteration=iteration, best_score=best_score,
                    consecutive_failures=consecutive_failures)

        # ---------------------------------------------------------------
        # Pause if too many consecutive failures
        # ---------------------------------------------------------------
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning("researcher", "pause_on_failures",
                           f"pausing {PAUSE_ON_FAILURE_SEC}s after {consecutive_failures} consecutive failures",
                           consecutive_failures=consecutive_failures)
            await asyncio.sleep(PAUSE_ON_FAILURE_SEC)
            consecutive_failures = 0

        # ---------------------------------------------------------------
        # 1. Load config + recent history
        # ---------------------------------------------------------------
        try:
            cfg = load_config()
        except Exception as exc:
            logger.bug("config_load_error", f"could not read config: {exc}",
                       exception=traceback.format_exc())
            consecutive_failures += 1
            await asyncio.sleep(5)
            iteration += 1
            continue

        recent_results = load_recent_results(10)

        # ---------------------------------------------------------------
        # 2. Harness code drift detection
        # ---------------------------------------------------------------
        current_harness_hash = _compute_harness_hash()
        if prev_harness_hash is not None and prev_harness_hash != current_harness_hash:
            logger.bug(
                "harness_code_drift",
                f"Harness source code changed between iter {iteration-1} and {iteration}. "
                "Score shift may reflect code change, not strategy improvement.",
                context={
                    "prev_hash": prev_harness_hash,
                    "current_hash": current_harness_hash,
                    "iteration": iteration,
                },
            )
            print(
                f"⚠ [drift] Harness code changed (hash {prev_harness_hash} → {current_harness_hash}). "
                "Score comparison for this iteration may be confounded.",
                flush=True,
            )
        prev_harness_hash = current_harness_hash

        # ---------------------------------------------------------------
        # 3. If this is iteration 0, run baseline without mutation
        # ---------------------------------------------------------------
        is_baseline = (cfg.get("version", 0) == 0 and iteration == 0)
        if is_baseline:
            logger.info("researcher", "baseline_run",
                        "running baseline trial (no mutation)", version=0)
            mutated = cfg
        else:
            # -----------------------------------------------------------
            # 4. Propose mutation
            # -----------------------------------------------------------
            try:
                patch = await propose_mutation(cfg, recent_results, logger)
                mutated = apply_patch(cfg, patch)
                save_config(mutated)
                logger.info("researcher", "config_saved",
                            f"saved v{mutated['version']}",
                            version=mutated["version"], strategy_id=mutated["strategy_id"])
            except Exception as exc:
                logger.bug("mutation_error", f"mutation proposal failed: {exc}",
                           exception=traceback.format_exc(),
                           context={"iteration": iteration})
                if use_git:
                    git_discard_config(CONFIG_PATH)
                consecutive_failures += 1
                iteration += 1
                await asyncio.sleep(3)
                continue

        # ---------------------------------------------------------------
        # 5. Run trials
        # ---------------------------------------------------------------
        if dry_run:
            logger.info("researcher", "dry_run", "skipping subprocess (dry_run=True)")
            trials: list[dict[str, Any]] = []
        else:
            try:
                trials = _run_trials(mutated, logger)
            except RuntimeError as exc:
                logger.error("researcher", "runner_failed", str(exc),
                             strategy_id=mutated.get("strategy_id"))
                if use_git:
                    git_discard_config(CONFIG_PATH)
                log_result(mutated, {
                    "pass_at_n": 0.0,
                    "mean_harness_score": 0.0,
                    "mean_token_usage": 0.0,
                    "failure_modes": ["runner_crash"],
                    "bugs": [str(exc)],
                    "n_bugs": 1,
                }, kept=False, git_hash=git_short_hash(), harness_hash=current_harness_hash)
                consecutive_failures += 1
                iteration += 1
                await asyncio.sleep(3)
                continue

        # ---------------------------------------------------------------
        # 6. Score
        # ---------------------------------------------------------------
        ground_truth: dict[str, Any] = mutated.get("task", {}).get("ground_truth", {})
        if not ground_truth:
            ground_truth = {
                "expected_keywords": ["umbrella", "raincoat", "jacket", "coat",
                                      "sweater", "t-shirt", "sunscreen", "gloves"],
                "required_tool_calls": ["weather_lookup", "clothing_suggest"],
            }

        try:
            scores = score_strategy(trials, mutated, ground_truth)
        except Exception as exc:
            logger.bug("scoring_error", f"eval failed: {exc}",
                       exception=traceback.format_exc(),
                       context={"iteration": iteration})
            if use_git:
                git_discard_config(CONFIG_PATH)
            consecutive_failures += 1
            iteration += 1
            await asyncio.sleep(3)
            continue

        # ---------------------------------------------------------------
        # 7. Emit metrics
        # ---------------------------------------------------------------
        logger.metrics(iteration, mutated.get("strategy_id", "?"), scores)

        current_score: float = scores["mean_harness_score"]
        kept = current_score > best_score

        # ---------------------------------------------------------------
        # 8. Keep → git commit  |  Discard → git checkout --
        # ---------------------------------------------------------------
        git_hash = git_short_hash()
        if kept:
            best_score = current_score
            if use_git:
                try:
                    git_hash = git_commit_config(
                        CONFIG_PATH,
                        mutated.get("strategy_id", "unknown"), current_score
                    )
                    logger.info("researcher", "git_commit",
                                f"committed as {git_hash}",
                                git_hash=git_hash, strategy_id=mutated.get("strategy_id"))
                except subprocess.CalledProcessError as exc:
                    logger.warning("researcher", "git_commit_failed",
                                   f"git commit failed (non-fatal): {exc.stderr}",
                                   stderr=exc.stderr)
            logger.info("researcher", "keep",
                        f"KEEP — new best {best_score:.4f}  git={git_hash}",
                        version=mutated.get("version"), strategy_id=mutated.get("strategy_id"),
                        git_hash=git_hash)
            print(
                f"[iter {iteration:03d}] "
                f"strategy={mutated.get('strategy_id'):<12} "
                f"score={scores['mean_harness_score']:.4f}  "
                f"pass@N={scores['pass_at_n']:.2f}  "
                f"tokens={scores['mean_token_usage']:.0f}  "
                f"bugs={scores['n_bugs']}  "
                f"git={git_hash}  ✓ KEEP",
                flush=True,
            )
            consecutive_failures = 0
        else:
            if use_git:
                git_discard_config(CONFIG_PATH)
            logger.info("researcher", "discard",
                        f"DISCARD — {current_score:.4f} ≤ {best_score:.4f}",
                        version=mutated.get("version"))
            print(
                f"[iter {iteration:03d}] "
                f"strategy={mutated.get('strategy_id'):<12} "
                f"score={scores['mean_harness_score']:.4f}  "
                f"pass@N={scores['pass_at_n']:.2f}  "
                f"tokens={scores['mean_token_usage']:.0f}  "
                f"bugs={scores['n_bugs']}  "
                f"git={git_hash}  ✗ discard",
                flush=True,
            )
            consecutive_failures = 0  # a non-improving run is not a failure

        log_result(mutated, scores, kept=kept,
                   git_hash=git_hash, harness_hash=current_harness_hash)

        # ---------------------------------------------------------------
        # 9. Layer 2: harness code evolution (every N iterations)
        # ---------------------------------------------------------------
        harness_evo_result = await maybe_evolve_harness(
            cfg=mutated,
            recent_results=load_recent_results(10),
            scores=scores,
            logger=logger,
            use_git=use_git,
            iteration=iteration,
        )
        if harness_evo_result and harness_evo_result.get("status") == "applied":
            # Log harness evolution as its own results.jsonl entry
            log_result(
                mutated,
                {**scores, "n_bugs": 0, "failure_modes": [], "bugs": []},
                kept=True,
                git_hash=harness_evo_result.get("git_hash", git_hash),
                harness_hash=_compute_harness_hash(),
            )
            # Recompute hash so drift detector knows baseline shifted
            prev_harness_hash = _compute_harness_hash()

        iteration += 1
        await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AHO researcher loop")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Stop after N iterations (default: run forever)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip actual trial execution (for testing)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    setup_aho_logging(args.debug)
    logger = create_aho_logger()

    logger.info("researcher", "researcher_start",
                "AHO researcher starting",
                max_iterations=args.max_iterations, dry_run=args.dry_run)

    try:
        asyncio.run(research_loop(
            logger,
            max_iterations=args.max_iterations,
            dry_run=args.dry_run,
        ))
    except KeyboardInterrupt:
        logger.info("researcher", "interrupted", "KeyboardInterrupt — stopping gracefully")
        print("\n[researcher] Stopped by user.")


if __name__ == "__main__":
    main()
