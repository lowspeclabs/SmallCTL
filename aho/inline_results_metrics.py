"""Metrics collection for inline tool results rollout."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class InlineResultsMetrics:
    """Metrics for inline tool results performance tracking."""
    
    # Trial identification
    trial_id: str
    branch: str = "inline_results"  # vs "baseline"
    
    # Token metrics
    tokens_saved: int = 0
    tokens_total_inline: int = 0
    tokens_total_baseline: int = 0
    
    # Step metrics
    steps_saved: int = 0
    artifact_read_calls_prevented: int = 0
    
    # Accuracy metrics
    facts_extracted: int = 0
    facts_validated: int = 0
    validation_confidence: float = 0.0
    fallback_count: int = 0
    
    # Success metrics
    task_completed: bool = False
    task_error: str | None = None
    duration_seconds: float = 0.0
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "branch": self.branch,
            "tokens_saved": self.tokens_saved,
            "tokens_total_inline": self.tokens_total_inline,
            "tokens_total_baseline": self.tokens_total_baseline,
            "steps_saved": self.steps_saved,
            "artifact_read_calls_prevented": self.artifact_read_calls_prevented,
            "facts_extracted": self.facts_extracted,
            "facts_validated": self.facts_validated,
            "validation_confidence": self.validation_confidence,
            "fallback_count": self.fallback_count,
            "task_completed": self.task_completed,
            "task_error": self.task_error,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InlineResultsMetrics":
        """Create from dictionary."""
        return cls(**data)


class MetricsCollector:
    """Collects and aggregates metrics for inline tool results."""
    
    def __init__(self, output_dir: str = "aho/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[InlineResultsMetrics] = []
    
    def record(self, metric: InlineResultsMetrics) -> None:
        """Record a single metric."""
        self.metrics.append(metric)
        
        # Append to file
        metrics_file = self.output_dir / "inline_results_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric.to_dict()) + "\n")
    
    def get_summary(self, branch: str | None = None) -> dict[str, Any]:
        """Get summary statistics for collected metrics."""
        filtered = self.metrics
        if branch:
            filtered = [m for m in filtered if m.branch == branch]
        
        if not filtered:
            return {"error": "No metrics found"}
        
        total = len(filtered)
        completed = sum(1 for m in filtered if m.task_completed)
        
        return {
            "total_trials": total,
            "completed": completed,
            "success_rate": completed / total if total > 0 else 0,
            "avg_tokens_saved": sum(m.tokens_saved for m in filtered) / total,
            "avg_steps_saved": sum(m.steps_saved for m in filtered) / total,
            "avg_confidence": sum(m.validation_confidence for m in filtered) / total,
            "total_fallbacks": sum(m.fallback_count for m in filtered),
            "token_reduction_pct": (
                sum(m.tokens_saved for m in filtered) /
                sum(m.tokens_total_baseline for m in filtered) * 100
                if sum(m.tokens_total_baseline for m in filtered) > 0 else 0
            ),
        }
    
    def compare_branches(self) -> dict[str, Any]:
        """Compare inline vs baseline branches."""
        inline_summary = self.get_summary(branch="inline_results")
        baseline_summary = self.get_summary(branch="baseline")
        
        return {
            "inline": inline_summary,
            "baseline": baseline_summary,
            "comparison": {
                "success_rate_delta": (
                    inline_summary.get("success_rate", 0) - 
                    baseline_summary.get("success_rate", 0)
                ),
                "token_reduction": inline_summary.get("token_reduction_pct", 0),
                "confidence": inline_summary.get("avg_confidence", 0),
            },
        }


class ABTestRunner:
    """A/B test runner for inline tool results."""
    
    def __init__(self, collector: MetricsCollector, split_ratio: float = 0.5):
        self.collector = collector
        self.split_ratio = split_ratio
        self.trial_counter = 0
    
    def should_use_inline(self, trial_id: str | None = None) -> bool:
        """
        Determine if this trial should use inline results (A) or baseline (B).
        
        Uses deterministic hashing for reproducibility.
        """
        if trial_id is None:
            self.trial_counter += 1
            trial_id = f"trial_{self.trial_counter}"
        
        # Hash trial_id to get deterministic assignment
        import hashlib
        hash_val = int(hashlib.md5(trial_id.encode()).hexdigest(), 16)
        
        # Assign to branch based on split ratio
        return (hash_val % 100) < (self.split_ratio * 100)
    
    def get_branch(self, trial_id: str | None = None) -> str:
        """Get branch name for trial."""
        return "inline_results" if self.should_use_inline(trial_id) else "baseline"


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_inline_metrics(
    trial_id: str,
    tokens_inline: int,
    tokens_baseline: int,
    facts_extracted: int,
    validation_confidence: float,
    task_completed: bool,
    fallback_count: int = 0,
    duration: float = 0.0,
    branch: str = "inline_results",
) -> None:
    """
    Convenience function to record inline results metrics.
    """
    collector = get_metrics_collector()
    
    metric = InlineResultsMetrics(
        trial_id=trial_id,
        branch=branch,
        tokens_saved=tokens_baseline - tokens_inline,
        tokens_total_inline=tokens_inline,
        tokens_total_baseline=tokens_baseline,
        steps_saved=1,  # Prevented artifact_read call
        artifact_read_calls_prevented=1,
        facts_extracted=facts_extracted,
        facts_validated=facts_extracted,  # Assume all validated for now
        validation_confidence=validation_confidence,
        fallback_count=fallback_count,
        task_completed=task_completed,
        duration_seconds=duration,
    )
    
    collector.record(metric)


def print_comparison_report() -> None:
    """Print A/B test comparison report."""
    collector = get_metrics_collector()
    comparison = collector.compare_branches()
    
    print("\n" + "=" * 60)
    print("INLINE TOOL RESULTS - A/B TEST COMPARISON")
    print("=" * 60)
    
    print("\n📊 INLINE BRANCH:")
    inline = comparison.get("inline", {})
    print(f"  Trials: {inline.get('total_trials', 0)}")
    print(f"  Success Rate: {inline.get('success_rate', 0):.1%}")
    print(f"  Avg Tokens Saved: {inline.get('avg_tokens_saved', 0):.0f}")
    print(f"  Token Reduction: {inline.get('token_reduction_pct', 0):.1f}%")
    print(f"  Avg Confidence: {inline.get('avg_confidence', 0):.2f}")
    
    print("\n📊 BASELINE BRANCH:")
    baseline = comparison.get("baseline", {})
    print(f"  Trials: {baseline.get('total_trials', 0)}")
    print(f"  Success Rate: {baseline.get('success_rate', 0):.1%}")
    
    print("\n📈 COMPARISON:")
    comp = comparison.get("comparison", {})
    delta = comp.get('success_rate_delta', 0)
    print(f"  Success Rate Δ: {delta:+.1%}")
    print(f"  Token Reduction: {comp.get('token_reduction', 0):.1f}%")
    print(f"  Avg Confidence: {comp.get('confidence', 0):.2f}")
    
    if delta >= -0.05:  # Within 5% of baseline
        print("\n✅ INLINE RESULTS APPROVED FOR ROLLOUT")
    else:
        print("\n⚠️  INLINE RESULTS NEED INVESTIGATION")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test metrics collection
    print("Testing metrics collection...")
    
    # Record some test metrics
    for i in range(5):
        record_inline_metrics(
            trial_id=f"test_{i}",
            tokens_inline=80,
            tokens_baseline=3100,
            facts_extracted=3,
            validation_confidence=0.85,
            task_completed=True,
            fallback_count=0,
        )
    
    # Record a baseline metric
    record_inline_metrics(
        trial_id="baseline_1",
        tokens_inline=3100,
        tokens_baseline=3100,
        facts_extracted=0,
        validation_confidence=0.0,
        task_completed=True,
        branch="baseline",
    )
    
    # Print report
    print_comparison_report()
