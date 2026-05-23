import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uuid
import argparse
import random

# Modules from other directories
from aho.static_analysis.analyzer import HarnessStaticAnalyzer
from aho.hypothesis.engine import HypothesisEngine, Hypothesis
from aho.git_manager.vcs import GitManager
from aho.validation.pipeline import ABOracle, ValidationPipeline, BenchmarkSummary

class KnowledgeStore:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text("[]\n")

    def log_trial(self, hypothesis: Hypothesis, result: Dict[str, Any]):
        """
        Record the trial outcome into a knowledge store.
        """
        data = {
            "hypothesis": hypothesis.dict(),
            "actual_outcome": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        # In a real system, append to JSONL or database
        trials = json.loads(self.storage_path.read_text())
        trials.append(data)
        self.storage_path.write_text(json.dumps(trials, indent=2))

    def get_past_outcomes(self) -> List[Dict[str, Any]]:
        return json.loads(self.storage_path.read_text())

class AHOOrchestrator:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.analyzer = HarnessStaticAnalyzer(root_dir)
        self.knowledge_store = KnowledgeStore(root_dir / "aho" / "knowledge_store" / "history.json")
        
        # Load API keys for the Hypothesis Engine
        import os
        from dotenv import load_dotenv
        # Look for .env in project root
        load_dotenv(root_dir / ".env")
        
        endpoint = os.getenv("SMALLCTL_ENDPOINT") or os.getenv("OPENROUTER_ENDPOINT") or "https://openrouter.ai/api/v1"
        model = os.getenv("SMALLCTL_MODEL") or os.getenv("OPENROUTER_MODEL") or "qwen/qwen3.5-9b"
        api_key = os.getenv("SMALLCTL_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "mock"
        
        self.engine = HypothesisEngine(endpoint, model, api_key)
        self.git_mgr = GitManager(root_dir)
        self.oracle = ABOracle()
        self.validator = ValidationPipeline(self.oracle)

    async def run_loop(self, target_crash: bool = False):
        print("=== Step 1: Data Gathering ===")
        analysis_result = self.analyzer.analyze()
        levers = analysis_result.levers
        print(f"Extracted {len(levers)} levers from harness.")

        print("\n=== Step 2: Hypothesis Generation ===")
        # Build history context from knowledge store
        history = self.knowledge_store.get_past_outcomes()
        
        bug_tracker_path = self.root_dir / "aho" / "bug_tracker.jsonl"
        hypothesis = None
        
        if target_crash and bug_tracker_path.exists():
            print("Targeting a known crash from bug_tracker.jsonl...")
            try:
                lines = [line.strip() for line in bug_tracker_path.read_text().splitlines() if line.strip()]
                if lines:
                    target_bug = json.loads(lines[-1])
                    crash_trace = target_bug.get("exception", str(target_bug))
                    hypothesis = await self.engine.generate_crash_hypothesis(crash_trace)
                    print(f"Generated CRASH FIX Hypothesis: [{hypothesis.id}] {hypothesis.rationale}")
            except Exception as e:
                print(f"Failed to generate crash hypothesis, falling back to levers. Error: {e}")
                
        if not hypothesis:
            hypothesis = await self.engine.generate_hypothesis(analysis_result.dict(), history)
            print(f"Generated LEVER Hypothesis: [{hypothesis.id}] {hypothesis.rationale}")
            
        print(f"Confidence: {hypothesis.confidence_score} | Expected Delta: {hypothesis.expected_impact}")

        print("\n=== Step 3: Branch and Patch ===")
        branch_name = self.git_mgr.create_experiment_branch(hypothesis.id)
        applied = self.git_mgr.apply_hypothesis_patch(hypothesis.dict())
        
        if not applied:
            print("Failed to apply patch. Reverting.")
            self.git_mgr.revert_to_original()
            return

        print("\n=== Step 4: Validation Pipeline ===")
        # Phase 4a: Debug Agent / Syntax Gate
        if not self.validator.run_syntax_debug():
            print("Syntax/Tests failed. Reverting experimental branch.")
            # In a real system, would use a 'Debug Agent' LLM to try a fix
            self.git_mgr.revert_to_original()
            return

        # Phase 4b: A/B Oracle
        print("Running Control Benchmark (Main)...")
        self.git_mgr.revert_to_original()
        control_benchmark = self.oracle.run_benchmark(["巴黎天气", "巴黎服装建议"], n_trials=5)
        
        print(f"Running Experimental Benchmark ({branch_name})...")
        # Go back to experiment branch 
        # (GitManager needs a way to checkout existing branch but for demo just checkout experiment)
        import git
        repo = git.Repo(self.root_dir)
        repo.heads[branch_name].checkout()
        
        experimental_benchmark = self.oracle.run_benchmark(["巴黎天气", "巴黎服装建议"], n_trials=5)

        # Mandatory Step 4.1: Live Hardening
        live_run_passed = self.validator.run_live_hardening()
        if not live_run_passed:
            print("❌ LIVE HARDENING FAILED. Reverting experimental branch despite benchmark results.")
            self.git_mgr.revert_to_original()
            return

        print("\n=== Step 5: Decision ===")
        comparison = self.oracle.calculate_improvement(control_benchmark, experimental_benchmark)
        print(f"Comparison: {json.dumps(comparison, indent=2)}")

        print("\n=== Step 6: Finalize ===")
        if comparison["is_significant"]:
            print("IMPROVEMENT CONFIRMED & LIVE RUN PASSED. Merging to main.")
            self.git_mgr.merge_success(branch_name)
        else:
            print("NO SIGNIFICANT IMPROVEMENT. Reverting.")
            self.git_mgr.revert_to_original()
        
        # Record everything in Knowledge Store to compound learning
        self.knowledge_store.log_trial(hypothesis, comparison)
        print("Loop complete. Result recorded in Knowledge Store.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AHO Orchestrator")
    parser.add_argument("--target-crash", action="store_true", help="Attempt to fix the latest bug in bug_tracker.jsonl")
    args = parser.parse_args()
    
    root = Path("/home/stephen/Scripts/Harness-Redo")
    orchestrator = AHOOrchestrator(root)
    asyncio.run(orchestrator.run_loop(target_crash=args.target_crash))
