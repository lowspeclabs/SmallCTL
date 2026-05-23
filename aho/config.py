"""
aho/config.py
-------------
Configuration loading and saving utilities for the researcher module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("aho/harness_config.json")
RESULTS_PATH = Path("aho/results.jsonl")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the harness configuration from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(cfg: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    """Save the harness configuration to JSON."""
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_recent_results(n: int = 10, path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Load the N most recent results from the results file."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows[-n:]


def best_score_from_results(results: list[dict[str, Any]]) -> float:
    """Extract the best score from kept results."""
    kept = [r for r in results if r.get("kept")]
    if not kept:
        return 0.0
    return max(r.get("mean_harness_score", 0.0) for r in kept)


def log_result(
    cfg: dict[str, Any],
    scores: dict[str, Any],
    kept: bool,
    git_hash: str = "",
    harness_hash: str = "",
    path: Path = RESULTS_PATH,
) -> None:
    """Log a result entry to the results file."""
    from datetime import datetime, timezone

    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "harness_hash": harness_hash,
        "strategy_id": cfg.get("strategy_id"),
        "version": cfg.get("version", 0),
        "strategy": cfg.get("strategy", {}),
        "pass_at_n": scores.get("pass_at_n", 0.0),
        "mean_harness_score": scores.get("mean_harness_score", 0.0),
        "mean_token_usage": scores.get("mean_token_usage", 0.0),
        "failure_modes": scores.get("failure_modes", []),
        "bugs": scores.get("bugs", []),
        "n_bugs": scores.get("n_bugs", 0),
        "kept": kept,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
