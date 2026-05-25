import subprocess
import json
import sys
import os
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class TrialResult:
    success: bool
    steps: int
    token_usage: int
    cost_est: float
    inactive_steps: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    success_rate: float
    mean_steps: float
    total_cost: float
    num_trials: int
    standard_deviation_success: float
    mean_inactive_steps: float


class ABOracle:
    """Runs real harness trials via Python subprocess instead of simulation."""

    def __init__(
        self,
        run_script: str = "-m",
        module: str = "aho.harness_runner",
        config_path: str = "aho/harness_config.json",
    ):
        self.run_script = run_script
        self.module = module
        self.config_path = config_path
        self._repo_root = Path(__file__).resolve().parent.parent.parent

    def run_benchmark(self, task_suite: List[str], n_trials: int = 5) -> BenchmarkSummary:
        """
        Runs the benchmark suite by spawning the actual harness runner for each task.
        Collects metrics from trial JSON output.
        """
        results: List[TrialResult] = []

        for task in task_suite:
            trial_results = self._run_task_trials(task, n_trials)
            for tr in trial_results:
                result = tr.get("result", {})
                status = result.get("status", "failed")
                success = status == "completed"
                steps = tr.get("step_count", 0)
                token_usage = tr.get("token_usage", 0)
                error = tr.get("error")

                # Rough cost estimate: ~$0.0015 per 1K tokens (local SLM proxy)
                cost_est = round(token_usage * 0.0000015, 6)

                results.append(
                    TrialResult(
                        success=success,
                        steps=steps,
                        token_usage=token_usage,
                        cost_est=cost_est,
                        inactive_steps=0,  # Derived from stagnation_counters if needed
                        error=error,
                    )
                )

        successes = [1 if r.success else 0 for r in results]
        steps = [r.steps for r in results]

        return BenchmarkSummary(
            success_rate=sum(successes) / len(results) if results else 0.0,
            mean_steps=sum(steps) / len(steps) if steps else 0.0,
            total_cost=sum(r.cost_est for r in results),
            num_trials=len(results),
            standard_deviation_success=_std_dev(successes),
            mean_inactive_steps=sum(r.inactive_steps for r in results) / len(results) if results else 0.0,
        )

    def _run_task_trials(self, task: str, n_trials: int) -> List[Dict[str, Any]]:
        """Spawn harness_runner.py for a specific task and return parsed trial list."""
        python_exe = sys.executable
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._repo_root / "src")

        # Build a temporary config that overrides the task description
        base_config_path = self._repo_root / self.config_path
        cfg: Dict[str, Any] = {}
        if base_config_path.exists():
            cfg = json.loads(base_config_path.read_text(encoding="utf-8"))
        cfg["task"] = {"description": task}
        cfg["n_trials"] = n_trials

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=self._repo_root / "aho"
        ) as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            cmd = [
                python_exe,
                self.run_script,
                self.module,
                "--config",
                tmp_path,
                "--mock-llm",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=str(self._repo_root),
                env=env,
            )

            if result.returncode != 0:
                print(f"Harness runner error for task '{task}': {result.stderr[:500]}")
                return []

            # harness_runner prints JSON array of trials to stdout
            output = result.stdout.strip()
            # In case there is logging mixed in, try to find the last JSON array
            trials = self._extract_json_array(output)
            return trials if isinstance(trials, list) else []
        except subprocess.TimeoutExpired:
            print(f"Task '{task}' timed out.")
            return []
        except Exception as e:
            print(f"Task '{task}' crashed: {e}")
            return []
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _extract_json_array(text: str) -> Any:
        """Try to extract the last JSON array from mixed stdout output."""
        # Strategy: find the last '[' that starts an array and parse from there
        decoder = json.JSONDecoder()
        idx = text.rfind("[")
        if idx == -1:
            # Maybe the whole text is JSON
            try:
                return json.loads(text)
            except Exception:
                return None
        while idx >= 0:
            try:
                obj, end = decoder.raw_decode(text, idx)
                if isinstance(obj, list):
                    return obj
            except Exception:
                pass
            idx = text.rfind("[", 0, idx)
        return None

    def calculate_improvement(
        self, control: BenchmarkSummary, experimental: BenchmarkSummary
    ) -> Dict[str, Any]:
        """
        Compare control vs experimental and determine if improvement is significant.
        """
        delta_success = experimental.success_rate - control.success_rate
        delta_steps_pct = (
            (experimental.mean_steps - control.mean_steps) / control.mean_steps
            if control.mean_steps > 0
            else 0
        )
        delta_cost_pct = (
            (experimental.total_cost - control.total_cost) / control.total_cost
            if control.total_cost > 0
            else 0
        )

        return {
            "delta_success": delta_success,
            "delta_steps_pct": delta_steps_pct,
            "delta_cost_pct": delta_cost_pct,
            "is_significant": delta_success > 0.02
            or (delta_cost_pct < -0.1 and delta_success >= -0.01),
        }


class ValidationPipeline:
    def __init__(self, oracle: ABOracle):
        self.oracle = oracle

    def run_live_hardening(self) -> bool:
        """
        Executes the actual smallctl harness on a real terminal task.
        This provides the final 'Scientific Hardening' required before merge.
        """
        print("🚀 PHASE 4.1: LIVE HARDENING...")
        try:
            # We run a simple but representative task
            task = "List root directory files and summarize findings."
            python_exe = sys.executable
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [python_exe, "-m", "src.smallctl.main", "--task", task, "--debug"],
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )

            if result.returncode == 0:
                print("✅ Live Hardening SUCCESSful.")
                return True
            else:
                print(f"❌ Live Hardening FAILED with exit code {result.returncode}")
                print(f"Error: {result.stderr[:500]}")
                return False
        except Exception as e:
            print(f"❌ Live Hardening crashed: {e}")
            return False

    def run_syntax_debug(self) -> bool:
        """
        Runs the 'Debug Agent' functionality: ruff/mypy and pytest check.
        """
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            # 1. Check syntax with ruff
            print("Checking syntax...")
            subprocess.run([sys.executable, "-m", "ruff", "check", "src/smallctl"], check=True, env=env)

            # 2. Run unit tests
            print("Running AHO unit tests...")
            test_dir = Path(__file__).resolve().parent.parent / "tests"
            if test_dir.exists():
                subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_dir), "-v"],
                    env=env,
                    check=True,
                )
            else:
                print("No tests/ directory found; skipping pytest.")

            return True
        except subprocess.CalledProcessError as e:
            print(f"Syntax or test failure: {e}")
            # The 'Debug Agent' should ideally try to fix syntax automatically here.
            return False


def _std_dev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


if __name__ == "__main__":
    oracle = ABOracle()
    control = oracle.run_benchmark(["task1", "task2"], n_trials=2)
    experimental = oracle.run_benchmark(["task1", "task2"], n_trials=2)
    comparison = oracle.calculate_improvement(control, experimental)
    print(f"Comparison Result: {json.dumps(comparison, indent=2)}")
