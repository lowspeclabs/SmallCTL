# SmallCTL

SmallCTL is an experimental agent harness for small local or self-hosted language models. It wraps an OpenAI-compatible model with staged task flow, evidence tracking, context compression, tool safety, and recovery logic so the model can handle real technical work with fewer runaway loops and less guesswork.

It is aimed at coding-agent and diagnostic workflows where a small model is useful, but needs structure around planning, tool use, verification, and memory.

## What It Helps With

- Exploring repositories and logs before changing anything
- Turning observations into bounded plans
- Running tool-assisted coding or diagnosis loops
- Keeping evidence, decisions, claims, and artifacts attached to the task
- Compressing long sessions into task-relevant memory
- Recovering from failed writes, bad tool calls, repeated actions, or verifier failures

SmallCTL is not a claim that a 4B-9B model is a fully autonomous senior engineer. It is a runtime for making those models more inspectable and less fragile.

## Install

Requires Python 3.10+.

```bash
git clone <repo-url>
cd Harness-Redo
./install.sh
source .venv/bin/activate
smallctl --help
```

Optional local config:

```bash
cp .smallctl.yaml.example .smallctl.yaml
```

Set the model endpoint with CLI flags, `.smallctl.yaml`, `.env`, or environment variables:

```bash
smallctl --endpoint http://localhost:8000/v1 --model qwen3.5:4b --task "Inspect this repo"
```

Common provider profiles include `generic`, `openai`, `ollama`, `vllm`, `lmstudio`, `openrouter`, and `llamacpp`.

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

Run a one-off task:

```bash
smallctl --task "Find where tool dispatch is implemented"
```

Use a local coding preset:

```bash
smallctl --preset coding-local --task "Investigate the failing tests"
```

Launch the Textual UI:

```bash
smallctl --tui
```

Enable the newer ToolPlan runtime for read-only evidence planning:

```bash
smallctl --run-mode tool_plan --task "Find the risk policy flow"
```

## Core Features

### Staged Workflows

SmallCTL separates a task into phases: `explore`, `plan`, `author`, `execute`, `verify`, and `repair`. Each phase has a contract that changes prompt focus and blocks inappropriate tools. For example, exploration and planning block mutation tools, while verification blocks file writes.

Research lineage: staged reasoning and plan-act-verify agent patterns; intended as a safer alternative to an unstructured ReAct loop for long tasks.

### Evidence-First State

The harness keeps structured records for evidence, decisions, claims, plans, context briefs, write sessions, artifacts, and verifier results. This lets later turns reason over what was observed, what was inferred, and what is still unproven.

Research lineage: evidence-grounded reasoning, retrieval-augmented context, and agent memory systems.

### Context Compression

SmallCTL does not rely only on the latest chat transcript. It compiles prompt state from recent messages, run briefs, working memory, observation packets, turn bundles, episodic summaries, artifact snippets, and warm experience memory.

Research lineage: RAG-style retrieval, hierarchical memory, and long-context compaction.

### ToolPlan and ReWOO-Style Evidence Gathering

The ToolPlan runtime can ask the model to produce a bounded read-only evidence plan, execute those reads, then hand compressed observations to a solver. Optional ReWOO lane frames separate planner, solver, and refiner context into plan, evidence, decision, and experience lanes.

Research lineage: ReWOO-style planner/worker/solver separation.

### Parallel Read-Only Tool DAGs

Independent ToolPlan steps can be grouped into dependency batches and dispatched concurrently when they use approved read-only tools such as file reads, grep, directory listing, artifact reads, web fetches, git status/diff, and log reads.

Research lineage: LLMCompiler-style parallel tool execution for independent steps.

### Reflexion-Style Repair

Failure events can become compact reflections and recovery hints. The solver-refine path critiques draft answers against observations and can revise or block unsupported completions.

Research lineage: Reflexion-style verbal feedback and self-correction.

### FAMA Mitigation Capsules

The `fama` package detects failure modes such as early stopping, looping, tool-output misreads, remote/local confusion, write-session stalls, and context drift. It injects short mitigation capsules into the prompt when those failures are active.

Research lineage: failure-aware prompting, metacognitive control, and adaptive agent guardrails.

### Risk and Approval Gates

Tools carry risk labels. Shell, SSH, network, and file mutation flows pass through policy checks, approval surfaces, phase gates, and write-session guards before execution.

Research lineage: tool-use safety, sandboxing, and human-in-the-loop approval.

## Configuration

SmallCTL reads configuration in this order:

1. user config path from `--config` or `SMALLCTL_CONFIG`
2. local `.smallctl.yaml`
3. `.env` and `SMALLCTL_*` environment variables
4. CLI flags

Useful flags:

- `--endpoint`, `--model`, `--provider-profile`
- `--preset safe-small-model|coding-local|lmstudio-small-model`
- `--run-mode auto|chat|loop|planning|indexer|tool_plan`
- `--staged-reasoning`
- `--staged-execution`
- `--checkpoint-on-exit` and `--resume`
- `--tool-profiles core,data,network,mutate,indexer`

## Project Status

SmallCTL is a practical research harness, not a polished product. The codebase includes tests and eval tasks for staged execution, ToolPlan behavior, risk policy, FAMA, Reflexion, compaction, write recovery, web tools, git tools, UI paths, and provider compatibility.

Best fit today:

- local-model coding experiments
- repo and log investigation
- guarded sysadmin diagnosis
- research on small-model tool use, memory, staged reasoning, and recovery

Avoid using it as an unbounded autonomous executor unless your environment has strong approvals, verifiers, and rollback controls.
