import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Set, List

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def get_allowed_harness_keys() -> Set[str]:
    init_path = _REPO_ROOT / "src" / "smallctl" / "harness" / "initialization.py"
    if not init_path.exists():
        print(f"Error: initialization.py not found at {init_path}", file=sys.stderr)
        return set()
        
    with open(init_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
        
    keys = {"endpoint", "model", "run_logger", "strategy_prompt", "strategy"}
    
    class ParamKeyExtractor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "params":
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    keys.add(node.slice.value)
            self.generic_visit(node)
            
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "params" and node.func.attr == "get":
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        keys.add(node.args[0].value)
            self.generic_visit(node)
            
    extractor = ParamKeyExtractor()
    extractor.visit(tree)
    return keys

def check_harness_config(config_dict: dict[str, Any], allowed_keys: Set[str]) -> List[str]:
    warnings = []
    aho_keys = {
        "strategy_id", "version", "n_trials", "trial_timeout_sec", "task", "scoring", "auto_batch",
        "context_optimization", "harness_evolution_enabled", "adaptive_compression", "adaptive_steps", "max_loops",
        "distilled_mode", "strategy", "inline_tool_results", "auto_batch_enabled", "mock_llm", "generate_test_on_crash",
        "n_trials", "trial_timeout_sec", "tool_guard_enabled",
    }
    
    for key in config_dict:
        if key in aho_keys:
            continue
        if key not in allowed_keys:
            warnings.append(f"Config property '{key}' is defined but is not supported by Harness initialization kwargs.")
    return warnings

def check_dead_wired_config_fields(allowed_keys: Set[str]) -> List[str]:
    try:
        from src.smallctl.config import SmallctlConfig
        from dataclasses import fields
    except ImportError as e:
        return [f"Could not import SmallctlConfig: {e}"]
        
    dead_wired = []
    skipped = {
        "api_key",
        "cleanup",
        "compatibility_warnings",
        "config_path",
        "debug",
        "graph_thread_id",
        "log_file",
        "preset",
        "restore_graph_state",
        "staged_reasoning",
        "task",
        "tui",
    }
    for field in fields(SmallctlConfig):
        if field.name in skipped:
            continue
        if field.name not in allowed_keys:
            dead_wired.append(field.name)
            
    return dead_wired

def main():
    print("Running AHO Configuration Parity Verifier...")
    allowed_keys = get_allowed_harness_keys()
    if not allowed_keys:
        sys.exit(1)
        
    print(f"Loaded {len(allowed_keys)} allowed harness parameters.")
    
    # Verify harness_config.json
    config_path = _REPO_ROOT / "aho" / "harness_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config_data = json.load(f)
                warnings = check_harness_config(config_data, allowed_keys)
                if warnings:
                    print("\n[WARNING] Configuration properties not present in Harness initialization:")
                    for w in warnings:
                        print(f"  - {w}")
                else:
                    print("\n[SUCCESS] All keys in harness_config.json map to valid parameters.")
            except Exception as e:
                print(f"Error parsing harness_config.json: {e}", file=sys.stderr)
    else:
        print(f"Warning: harness_config.json not found at {config_path}")
        
    # Check for dead-wired fields in SmallctlConfig
    dead_wired = check_dead_wired_config_fields(allowed_keys)
    if dead_wired:
        print("\n[ALERT] Dead-Wired Config Fields Detected (defined in SmallctlConfig but missing from Harness constructor):")
        for dw in sorted(dead_wired):
            print(f"  - {dw}")
    else:
        print("\n[SUCCESS] No dead-wired fields detected in SmallctlConfig.")

if __name__ == "__main__":
    main()
