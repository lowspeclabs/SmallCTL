# Agentic Harness Optimizer (AHO) 🌌

A self-modifying adaptation system for the `smallctl` harness that implements a rigorous scientific method loop.

## 🔬 The Core Loop

1. **Phase 1: Static Analysis** — The `HarnessStaticAnalyzer` uses Python AST to extract "levers" (constants, prompt templates, and config) from the `smallctl` source.
2. **Phase 2: Hypothesis Engine** — Based on the structural model and past performance, an LLM proposes the highest-confidence experiment. Each hypothesis is **scoped** and **falsifiable**, predicting exactly how much metrics like `success_rate` or `token_cost` will change.
3. **Phase 3: Branch & Patch** — The `GitManager` creates an isolated experiment branch and applies a minimal patch.
4. **Phase 4: Validation Pipeline** — 
    - **Debug Agent**: Performs a "syntax-gate" check (linter + unit tests). 
    - **A/B Oracle**: Runs a representative task distribution against both `main` and the branch.
    - **Live Hardening**: Executes the actual `smallctl` harness on a variety of real terminal tasks to confirm stability. **[MANDATORY]**
5. **Phase 5: Decision** — Compares metrics against a strict significance threshold. Marginal gains are rejected to prevent system drift.
6. **Phase 6: Finalize & Compound** — Merges changes on success; reverts on failure. The `KnowledgeStore` records every experiment to avoid re-proposing failed ideas.

## 🛠️ Components

- `aho/static_analysis/analyzer.py`: AST parsing logic.
- `aho/hypothesis/engine.py`: Structured experiment generation.
- `aho/git_manager/vcs.py`: Automated branching and patching.
- `aho/validation/pipeline.py`: Benchmark execution and significance calculation.
- `aho/knowledge_store/base.py`: History and learning log.
- `aho/main.py`: The orchestrator loop.

## 🚀 Getting Started

1. Set up the local environment:
   ```bash
   cd aho
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. Run the optimizer loop:
   ```bash
   python main.py
   ```

## ⚠️ Safety Protocols

- **Isolated Venv**: Runs in its own environment to avoid dependency pollution.
- **Git Sandboxing**: Every experiment happens on a branch. `main` is only touched after passing the A/B Oracle.
- **Minimal Diffs**: One conceptual change per branch for clear signal.
- **Syntax Gate**: Automatic fixes are limited to syntax/lint level.
- **Mandatory Live Run**: No changes are merged until a live harness execution (Phase 4.1) confirms expected behavior on a real task.
