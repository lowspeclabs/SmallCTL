from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aho.config import load_recent_results
from aho.eval import score_strategy
from aho.git_manager.vcs import GitManager
from aho.harness_runner import run_n_trials
from aho.hypothesis.engine import Hypothesis, HypothesisEngine
from aho.logging_aho import create_aho_logger, setup_aho_logging
from aho.static_analysis.analyzer import HarnessStaticAnalyzer
from aho.config import load_config as load_harness_config


@dataclass
class ChallengeOutcome:
    challenge_id: str
    status: str
    baseline: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    hypothesis: dict[str, Any] | None = None
    branch_name: str | None = None
    benchmark: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


class ChallengeOptimizer:
    def __init__(
        self,
        root_dir: Path,
        challenges_path: Path,
        *,
        dry_run: bool = False,
        challenge_id: str | None = None,
        allow_dirty: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.challenges_path = challenges_path
        self.challenge_id = challenge_id
        self.dry_run = dry_run
        self.allow_dirty = allow_dirty
        self.logger = create_aho_logger()
        self.git_mgr = GitManager(root_dir)
        self.analyzer = HarnessStaticAnalyzer(root_dir)
        self.static_analysis = self.analyzer.analyze()
        self.historical_data = load_recent_results(n=10)
        self.harness_cfg = load_harness_config(root_dir / "aho" / "harness_config.json")
        self._repo_dirty = self.git_mgr.is_dirty(untracked_files=True)

        self.engine: HypothesisEngine | None = None
        if not self.dry_run:
            try:
                from dotenv import load_dotenv

                load_dotenv(root_dir / ".env")
            except Exception:
                pass
            endpoint = os.getenv("SMALLCTL_ENDPOINT") or "http://192.168.1.9:1234/v1"
            model = os.getenv("SMALLCTL_MODEL") or "qwen3.5-4b"
            api_key = os.getenv("SMALLCTL_API_KEY") or "local-dev-key"
            self.engine = HypothesisEngine(endpoint, model, api_key)

    def load_challenges(self) -> list[dict[str, Any]]:
        challenges = json.loads(self.challenges_path.read_text(encoding="utf-8"))
        if not isinstance(challenges, list):
            raise ValueError("challenges.json must contain a list of challenge objects")

        filtered: list[dict[str, Any]] = []
        for challenge in challenges:
            if not isinstance(challenge, dict):
                continue
            if self.challenge_id and challenge.get("id") != self.challenge_id:
                continue
            if "id" not in challenge or "description" not in challenge or "ground_truth" not in challenge:
                raise ValueError(f"Challenge is missing required fields: {challenge}")
            filtered.append(challenge)

        if self.challenge_id and not filtered:
            raise ValueError(f"No challenge matched id '{self.challenge_id}'")

        return filtered

    def _challenge_cfg(self, challenge: dict[str, Any], n_trials: int | None = None) -> dict[str, Any]:
        cfg = json.loads(json.dumps(self.harness_cfg))
        cfg["task"] = {"description": challenge["description"]}
        cfg["strategy_id"] = f"challenge-{challenge['id']}"
        if n_trials is not None:
            cfg["n_trials"] = n_trials
        return cfg

    async def run_challenges(self, challenges: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for challenge in challenges:
            challenge_id = challenge["id"]
            baseline_trials = int(challenge.get("baseline_trials", self.harness_cfg.get("n_trials", 1)))
            cfg = self._challenge_cfg(challenge, n_trials=baseline_trials)

            self.logger.info(
                "runner",
                "challenge_baseline_start",
                f"running baseline for {challenge_id}",
                challenge_id=challenge_id,
                n_trials=baseline_trials,
            )

            trials = await run_n_trials(cfg, self.logger)
            results[challenge_id] = {
                "cfg": cfg,
                "trials": trials,
                "scores": self.score_results(challenge, trials, cfg),
            }

        return results

    def score_results(
        self,
        challenge: dict[str, Any],
        trials: list[dict[str, Any]],
        cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = cfg or self._challenge_cfg(challenge)
        scores = score_strategy(trials, cfg, challenge["ground_truth"])
        self.logger.metrics(
            0,
            str(challenge["id"]),
            scores,
        )
        return scores

    async def propose_fixes(
        self,
        challenge: dict[str, Any],
        baseline: dict[str, Any],
    ) -> Hypothesis | None:
        if self.dry_run:
            return None
        if self.engine is None:
            raise RuntimeError("Hypothesis engine is not initialized")

        challenge_history = {
            "challenge_id": challenge["id"],
            "description": challenge["description"],
            "baseline_score": baseline["scores"].get("mean_harness_score", 0.0),
            "failure_modes": baseline["scores"].get("failure_modes", []),
            "bugs": baseline["scores"].get("bugs", []),
            "ground_truth": challenge["ground_truth"],
        }
        historical_data = [challenge_history, *self.historical_data]

        self.logger.info(
            "researcher",
            "propose_fix",
            f"proposing fix for {challenge['id']}",
            challenge_id=challenge["id"],
            failure_modes=baseline["scores"].get("failure_modes", []),
        )

        hypothesis = await self.engine.generate_hypothesis(self.static_analysis.model_dump(), historical_data)
        self.logger.info(
            "researcher",
            "hypothesis_generated",
            f"hypothesis {hypothesis.id} for {challenge['id']}",
            challenge_id=challenge["id"],
            hypothesis=hypothesis.model_dump(),
        )
        return hypothesis

    async def apply_and_test(
        self,
        challenge: dict[str, Any],
        hypothesis: Hypothesis,
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        if self.dry_run:
            return {
                "status": "dry_run",
                "branch_name": None,
                "candidate": None,
                "benchmark": None,
            }

        if self._repo_dirty and not self.allow_dirty:
            raise RuntimeError(
                "Working tree is dirty. Run the challenge loop on a clean workspace "
                "or pass --allow-dirty if you understand the tradeoff."
            )

        branch_name = self.git_mgr.create_experiment_branch(f"{challenge['id']}-{hypothesis.id}")
        applied = self.git_mgr.apply_hypothesis_patch(hypothesis.model_dump())
        if not applied:
            self.git_mgr.revert_to_original()
            return {
                "status": "patch_failed",
                "branch_name": branch_name,
                "candidate": None,
                "benchmark": None,
            }

        benchmark_trials = int(challenge.get("benchmark_trials", baseline["cfg"].get("n_trials", 1)))
        candidate_cfg = self._challenge_cfg(challenge, n_trials=benchmark_trials)
        self.logger.info(
            "runner",
            "challenge_candidate_start",
            f"running candidate for {challenge['id']} on {branch_name}",
            challenge_id=challenge["id"],
            branch=branch_name,
            n_trials=benchmark_trials,
        )
        trials = await run_n_trials(candidate_cfg, self.logger)
        candidate_scores = self.score_results(challenge, trials, candidate_cfg)

        benchmark = self.benchmark_improvement(baseline["scores"], candidate_scores)
        if benchmark["improved"]:
            self.finalize_success(branch_name, hypothesis)
        else:
            self.git_mgr.revert_to_original()

        return {
            "status": "improved" if benchmark["improved"] else "not_improved",
            "branch_name": branch_name,
            "candidate": {
                "cfg": candidate_cfg,
                "trials": trials,
                "scores": candidate_scores,
            },
            "benchmark": benchmark,
        }

    def benchmark_improvement(
        self,
        baseline_scores: dict[str, Any],
        candidate_scores: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_score = float(baseline_scores.get("mean_harness_score", 0.0))
        candidate_score = float(candidate_scores.get("mean_harness_score", 0.0))
        baseline_pass = float(baseline_scores.get("pass_at_n", 0.0))
        candidate_pass = float(candidate_scores.get("pass_at_n", 0.0))
        delta = round(candidate_score - baseline_score, 4)
        improved = delta > 0 or candidate_pass > baseline_pass
        return {
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "delta_score": delta,
            "baseline_pass_at_n": baseline_pass,
            "candidate_pass_at_n": candidate_pass,
            "improved": improved,
        }

    def finalize_success(self, branch_name: str, hypothesis: Hypothesis) -> None:
        self.git_mgr.commit_hypothesis(hypothesis.model_dump())
        self.git_mgr.merge_success(branch_name)

    def summarize_meta(self, outcomes: list[ChallengeOutcome], *, dry_run: bool = False) -> Path:
        baseline_outcomes = [outcome for outcome in outcomes if outcome.baseline is not None]
        baseline_scores = [
            float((outcome.baseline or {}).get("scores", {}).get("pass_at_n", 0.0))
            for outcome in baseline_outcomes
        ]
        baseline_mean_pass_at_n = round(sum(baseline_scores) / len(baseline_scores), 4) if baseline_scores else 0.0
        baseline_mean_harness_score = round(
            sum(float((outcome.baseline or {}).get("scores", {}).get("mean_harness_score", 0.0)) for outcome in baseline_outcomes) / len(baseline_outcomes),
            4,
        ) if baseline_outcomes else 0.0
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "challenge_filter": self.challenge_id,
            "total": len(outcomes),
            "baseline_mean_pass_at_n": baseline_mean_pass_at_n,
            "baseline_mean_harness_score": baseline_mean_harness_score,
            "counts": {
                "improved": sum(1 for outcome in outcomes if outcome.status == "improved"),
                "not_improved": sum(1 for outcome in outcomes if outcome.status == "not_improved"),
                "perfect": sum(1 for outcome in outcomes if outcome.status == "perfect"),
                "patch_failed": sum(1 for outcome in outcomes if outcome.status == "patch_failed"),
                "dry_run": sum(1 for outcome in outcomes if outcome.status == "dry_run"),
                "baseline": sum(1 for outcome in outcomes if outcome.status == "baseline"),
            },
            "outcomes": [
                {
                    "challenge_id": outcome.challenge_id,
                    "status": outcome.status,
                    "branch_name": outcome.branch_name,
                    "baseline": outcome.baseline,
                    "candidate": outcome.candidate,
                    "benchmark": outcome.benchmark,
                    "notes": outcome.notes,
                    "hypothesis": outcome.hypothesis,
                }
                for outcome in outcomes
            ],
        }

        summary_path = self.root_dir / "aho" / f"challenge_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        self.logger.info(
            "eval",
            "challenge_summary",
            f"wrote challenge summary to {summary_path}",
            summary_path=str(summary_path),
            summary=summary,
        )
        return summary_path


async def _run_optimizer(args: argparse.Namespace) -> Path:
    root = _REPO_ROOT
    setup_aho_logging(debug=True)

    optimizer = ChallengeOptimizer(
        root,
        root / args.challenges,
        dry_run=args.dry_run,
        challenge_id=args.challenge_id,
        allow_dirty=args.allow_dirty,
    )
    challenges = optimizer.load_challenges()
    print(f"Loaded {len(challenges)} challenge(s).")

    if args.dry_run:
        outcomes = [
            ChallengeOutcome(
                challenge_id=challenge["id"],
                status="dry_run",
                notes=[
                    "would run baseline trials",
                    "would score with eval.score_strategy",
                    "would propose a patch with HypothesisEngine",
                    "would branch, apply, benchmark, and merge if improvement is positive",
                ],
            )
            for challenge in challenges
        ]
        return optimizer.summarize_meta(outcomes, dry_run=True)

    baseline_runs = await optimizer.run_challenges(challenges)
    outcomes: list[ChallengeOutcome] = []

    if args.baseline_only:
        for challenge in challenges:
            challenge_id = challenge["id"]
            baseline = baseline_runs[challenge_id]
            outcomes.append(
                ChallengeOutcome(
                    challenge_id=challenge_id,
                    status="baseline",
                    baseline=baseline,
                    notes=["baseline-only run; no intervention attempted"],
                )
            )
        return optimizer.summarize_meta(outcomes, dry_run=False)

    for challenge in challenges:
        challenge_id = challenge["id"]
        baseline = baseline_runs[challenge_id]
        baseline_scores = baseline["scores"]

        if baseline_scores.get("mean_harness_score", 0.0) >= 1.0:
            outcomes.append(
                ChallengeOutcome(
                    challenge_id=challenge_id,
                    status="perfect",
                    baseline=baseline,
                    notes=["baseline already achieved a perfect score"],
                )
            )
            continue

        hypothesis = await optimizer.propose_fixes(challenge, baseline)
        if hypothesis is None:
            outcomes.append(
                ChallengeOutcome(
                    challenge_id=challenge_id,
                    status="patch_failed",
                    baseline=baseline,
                    notes=["dry run or missing hypothesis engine"],
                )
            )
            continue

        branch_result = await optimizer.apply_and_test(challenge, hypothesis, baseline)
        outcome = ChallengeOutcome(
            challenge_id=challenge_id,
            status=branch_result["status"],
            baseline=baseline,
            candidate=branch_result.get("candidate"),
            benchmark=branch_result.get("benchmark"),
            branch_name=branch_result.get("branch_name"),
            hypothesis=hypothesis.model_dump(),
        )
        outcomes.append(outcome)

    return optimizer.summarize_meta(outcomes, dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="AHO Challenge Optimizer")
    parser.add_argument("--challenges", default="aho/challenges_30.json", help="Path to challenges JSON")
    parser.add_argument("--challenge-id", default=None, help="Run only the matching challenge id")
    parser.add_argument("--dry-run", action="store_true", help="Plan the workflow without running trials or LLM calls")
    parser.add_argument("--baseline-only", action="store_true", help="Run the baseline harness for each challenge and skip any improvement attempts")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow git worktree mutations on a dirty workspace")
    args = parser.parse_args()

    summary_path = asyncio.run(_run_optimizer(args))
    print(f"Workflow complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
