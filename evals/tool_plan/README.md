# ToolPlan Eval Scaffold

This folder contains lightweight prompts for comparing normal loop mode with
the ReWOO-style `tool_plan` runtime.

Tasks can be defined as individual `.yaml` files, a `.jsonl` file, or a directory
containing any mix of supported formats. The runner deduplicates by task `id`.

Example dry run (single YAML task):

```bash
python scripts/tool_plan_eval.py --tasks evals/tool_plan/multi_file_bughunt_001.yaml --dry-run
```

Example dry run (directory — loads all `.yaml` and `.jsonl` files):

```bash
python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --dry-run
```

Example live run:

```bash
python scripts/tool_plan_eval.py \
  --tasks evals/tool_plan/tasks.jsonl \
  --output .smallctl/artifacts/tool_plan_eval_results.jsonl
```

Example ReWOO lane-frame rollout check:

```bash
python scripts/tool_plan_eval.py \
  --tasks evals/tool_plan/ \
  --mode both \
  --rewoo-frames \
  --output .smallctl/artifacts/tool_plan_rewoo_eval_results.jsonl \
  --report .smallctl/artifacts/tool_plan_rewoo_eval_report.json
```

Repeat the full suite for stochastic sampling and emit a markdown decision
report next to the JSON report:

```bash
python scripts/tool_plan_eval.py \
  --tasks evals/tool_plan/ \
  --mode both \
  --repeat 3 \
  --rewoo-frames \
  --output .smallctl/artifacts/tool_plan_rewoo_eval_results.jsonl \
  --report .smallctl/artifacts/tool_plan_rewoo_eval_report.json \
  --markdown-report .smallctl/artifacts/tool_plan_rewoo_eval_report.md
```

Each task is run once with `--run-mode loop` and once with
`--run-mode tool_plan`. The runner records duration, exit code, stdout/stderr,
generic token usage when exposed by the final JSON payload, selected latency
metrics, ToolPlan recovery metrics, prompt-shape checks, and the last JSON
object printed by `smallctl` when one can be parsed.

The comparison report includes summary, by-tag, and per-task sections. Token
and latency deltas stay `null` when either side is missing measurable data,
instead of treating missing usage as zero. ToolPlan-specific planner, worker,
solver-grounding, fallback, abort, loop-guard, and model-stream-halt fields are
kept as diagnostics.

When `--mode both` writes a report, the command exits nonzero if the decision is
not `continue` or if wrong-path fallback behavior regresses. Acceptance gates
cover planner validity, worker success, solver grounding, abort parity, latency
parity, and at least 20% token savings on measurable `repo_analysis` and
`log_investigation` comparisons.

Fixtures may optionally include an `expectations:` block with fields such as
`required_files`, `required_terms`, `grounding_required`, `success_statuses`,
and `unsafe_fallback_expected`. The runner preserves those fields in the JSON
results and uses them when scoring grounding and fallback behavior.

Current fixture coverage is intentionally tag-driven: coding/read-path,
repo-spelunking, config/debug, log triage, wrong-path safety, and web research
tasks are kept separate so regressions can be localized.

Use `--rewoo-frames` when evaluating the lane-frame rollout. It only adds
`--rewoo-planner-frame`, `--rewoo-solver-frame`, and `--rewoo-refiner-frame`
to the `tool_plan` command, leaving the loop baseline unchanged.
