"""
aho/eval.py
-----------
Scoring engine for the Agentic Harness Optimizer.

Replaces autoresearch's `val_bpb` with the AHO Harness Score:

    S = (W1 * Accuracy) + (W2 * Format_Adherence) - (W3 * Token_Latency_Penalty)

Public API
----------
score_trial(trial, cfg, ground_truth)   -> dict
score_strategy(trials, cfg, ground_truth) -> dict
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_LOG = logging.getLogger("aho.eval")

# Known-valid tool names from harness_config task definition
_VALID_MOCK_TOOLS = {
    "weather_lookup",
    "clothing_suggest",
    "long_context_lookup",
    "summarize_report",
    "artifact_read",
    "task_complete",
    "task_fail",
    "shell_exec",
    "file_write",
    "memory_update",
    "review_draft",
}


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def score_trial(trial: dict[str, Any], cfg: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single trial trace.

    Returns
    -------
    dict with keys:
        accuracy            float [0, 1]
        format_score        float [0, 1]
        token_usage         int
        latency_penalty     float
        stagnation_penalty  float
        harness_score       float (the composite S)
        failure_modes       list[str]
        bugs                list[str]  — unexpected parse errors / crashes
    """
    scoring = cfg.get("scoring", {})
    w1: float = float(scoring.get("w_accuracy", 0.6))
    w2: float = float(scoring.get("w_format", 0.3))
    w3: float = float(scoring.get("w_latency", 0.05))
    w4: float = float(scoring.get("w_stagnation", 0.05))
    penalty_rate: float = float(scoring.get("latency_penalty_per_100_tokens", 0.01))

    result: dict[str, Any] = trial.get("result", {})
    tool_records: dict[str, Any] = trial.get("tool_records", {})
    token_usage: int = int(trial.get("token_usage", 0))
    step_count: int = int(trial.get("step_count", 0))
    error: str | None = trial.get("error")

    failure_events = trial.get("failure_events", [])
    stagnation_counters = trial.get("stagnation_counters", {})
    tool_history = trial.get("tool_history", [])

    failure_modes: list[str] = []
    bugs: list[str] = []

    # Parse and record failure events
    for fe in failure_events:
        if not isinstance(fe, dict):
            continue
        fc = fe.get("failure_class", "unknown")
        msg = fe.get("message", "")
        sev = fe.get("severity", "info")
        failure_modes.append(f"harness_failure: [{sev}] {fc}: {msg}")
        if sev == "hard":
            bugs.append(f"Hard harness failure class: {fc} - {msg}")

    # -----------------------------------------------------------------------
    # Error / crash — immediate zero
    # -----------------------------------------------------------------------
    if error or result.get("status") == "failed":
        reason = error or result.get("reason", "unknown")
        failure_modes.append(f"trial_error: {reason}")
        return {
            "accuracy": 0.0,
            "format_score": 0.0,
            "token_usage": token_usage,
            "latency_penalty": 0.0,
            "stagnation_penalty": 1.0,
            "harness_score": 0.0,
            "failure_modes": failure_modes,
            "bugs": [f"trial crashed: {reason}"],
            "step_count": step_count,
        }

    # -----------------------------------------------------------------------
    # Accuracy — did the model produce one of the expected keywords?
    # -----------------------------------------------------------------------
    expected_keywords: list[str] = ground_truth.get("expected_keywords", [])
    required_tools: list[str] = ground_truth.get("required_tool_calls", [])

    final_answer = str(result.get("summary", "") or result.get("message", "")).lower()
    accuracy = 1.0 if any(kw.lower() in final_answer for kw in expected_keywords) else 0.0
    if accuracy == 0.0:
        failure_modes.append("accuracy: answer missing expected keyword(s)")

    # -----------------------------------------------------------------------
    # Format adherence
    # -----------------------------------------------------------------------
    format_score, fmt_failures, fmt_bugs = _check_format_adherence(
        tool_records=tool_records,
        strategy=cfg.get("strategy", {}),
        required_tools=required_tools,
    )
    failure_modes.extend(fmt_failures)
    bugs.extend(fmt_bugs)

    # -----------------------------------------------------------------------
    # Token latency
    # -----------------------------------------------------------------------
    latency_penalty = min(1.0, (token_usage / 100) * penalty_rate)
    latency_score = 1.0 - latency_penalty
    if latency_penalty > 0.5:
        failure_modes.append(f"latency: token_usage={token_usage} is high for an SLM")

    # -----------------------------------------------------------------------
    # Stagnation Penalty Calculation
    # -----------------------------------------------------------------------
    stagnation_val = sum(stagnation_counters.values()) if isinstance(stagnation_counters, dict) else 0
    failure_val = 0.0
    for fe in failure_events:
        if not isinstance(fe, dict):
            continue
        sev = fe.get("severity", "").lower()
        if sev == "warning":
            failure_val += 0.1
        elif sev == "recoverable":
            failure_val += 0.2
        elif sev == "hard":
            failure_val += 0.5

    history_val = 0.0
    if isinstance(tool_history, list) and len(tool_history) > 1:
        for i in range(len(tool_history) - 1):
            if tool_history[i] == tool_history[i + 1]:
                history_val += 0.15

    stagnation_penalty = min(1.0, (stagnation_val * 0.1) + failure_val + history_val)
    if stagnation_penalty > 0.3:
        failure_modes.append(f"stagnation: loop guard or FAMA penalty={stagnation_penalty:.2f} is high")

    # -----------------------------------------------------------------------
    # Composite score
    # -----------------------------------------------------------------------
    S = (w1 * accuracy) + (w2 * format_score) - (w3 * (1.0 - latency_score)) - (w4 * stagnation_penalty)
    S = max(0.0, round(S, 4))

    return {
        "accuracy": accuracy,
        "format_score": round(format_score, 4),
        "token_usage": token_usage,
        "latency_penalty": round(latency_penalty, 4),
        "stagnation_penalty": round(stagnation_penalty, 4),
        "harness_score": S,
        "failure_modes": failure_modes,
        "bugs": bugs,
        "step_count": step_count,
    }


def score_strategy(
    trials: list[dict[str, Any]],
    cfg: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate N trial scores into a strategy-level report.

    Returns
    -------
    dict with keys:
        pass_at_n               float
        mean_harness_score      float
        mean_token_usage        float
        trial_scores            list[dict]
        failure_modes           list[str]  (deduplicated, with counts)
        bugs                    list[str]
        n_bugs                  int
    """
    if not trials:
        return {
            "pass_at_n": 0.0,
            "mean_harness_score": 0.0,
            "mean_token_usage": 0.0,
            "trial_scores": [],
            "failure_modes": ["no_trials_returned"],
            "bugs": ["harness_runner returned empty list"],
            "n_bugs": 1,
        }

    scored = [score_trial(t, cfg, ground_truth) for t in trials]

    n = len(scored)
    pass_n = sum(1 for s in scored if s["accuracy"] == 1.0) / n
    mean_S = sum(s["harness_score"] for s in scored) / n
    mean_tokens = sum(s["token_usage"] for s in scored) / n

    # Aggregate failure modes with frequency counts
    all_modes: list[str] = []
    all_bugs: list[str] = []
    for s in scored:
        all_modes.extend(s.get("failure_modes", []))
        all_bugs.extend(s.get("bugs", []))

    failure_summary = _summarize_modes(all_modes, n_trials=n)
    bug_summary = _summarize_modes(all_bugs, n_trials=n)

    return {
        "pass_at_n": round(pass_n, 4),
        "mean_harness_score": round(mean_S, 4),
        "mean_token_usage": round(mean_tokens, 1),
        "trial_scores": scored,
        "failure_modes": failure_summary,
        "bugs": bug_summary,
        "n_bugs": len(all_bugs),
    }


# ---------------------------------------------------------------------------
# Format adherence
# ---------------------------------------------------------------------------


def _check_format_adherence(
    tool_records: dict[str, Any],
    strategy: dict[str, Any],
    required_tools: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Inspect tool_execution_records from harness.state.

    Returns (score [0,1], failure_modes, bugs).
    """
    score = 1.0
    failures: list[str] = []
    bugs: list[str] = []

    forbidden_patterns: list[str] = strategy.get("forbidden_patterns", [])
    tool_call_format: str = strategy.get("tool_call_format", "strict_xml")

    called_tools: set[str] = set()

    for op_id, record in tool_records.items():
        tool_name: str = str(record.get("tool_name", ""))
        if tool_name:
            called_tools.add(tool_name)

        # --- hallucinated tool name ---
        if tool_name and tool_name not in _VALID_MOCK_TOOLS:
            score -= 0.25
            failures.append(f"hallucinated_tool: {tool_name}")

        # --- check args for JSON-inside-think patterns ---
        args_raw = json.dumps(record.get("args", {}))
        if re.search(r"<think[^>]*>.*?(\{|\[)", args_raw, re.DOTALL | re.IGNORECASE):
            score -= 0.5
            failures.append("json_inside_think_block")

        # --- check result for syntax errors ---
        result_payload = record.get("result", {})
        if isinstance(result_payload, dict):
            err = result_payload.get("error")
            if err and "parse" in str(err).lower():
                score -= 0.25
                failures.append(f"tool_parse_error: {err}")
                bugs.append(f"parse_error in op {op_id}: {err}")

        # --- forbidden patterns in args ---
        for pat in forbidden_patterns:
            if pat.lower() in args_raw.lower():
                score -= 0.2
                failures.append(f"forbidden_pattern_found: {pat}")

        # --- strict_xml: tool args must not be raw markdown block ---
        if tool_call_format == "strict_xml":
            if "```" in args_raw:
                score -= 0.15
                failures.append("markdown_code_fence_in_tool_args")

    # --- required tools were called ---
    for req in required_tools:
        if req not in called_tools:
            score -= 0.3
            failures.append(f"missing_required_tool: {req}")

    return max(0.0, score), failures, bugs


def _summarize_modes(modes: list[str], n_trials: int) -> list[str]:
    """
    Deduplicate mode strings and attach occurrence counts.
    Example: ["accuracy: answer missing (3/5)", ...]
    """
    counts: dict[str, int] = {}
    for m in modes:
        counts[m] = counts.get(m, 0) + 1
    summary = []
    for mode, count in sorted(counts.items(), key=lambda x: -x[1]):
        summary.append(f"{mode} ({count}/{n_trials})")
    return summary
