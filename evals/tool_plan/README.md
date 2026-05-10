# ToolPlan Eval Scaffold

This folder contains lightweight prompts for comparing normal loop mode with
the ReWOO-style `tool_plan` runtime.

Example dry run:

```bash
python scripts/tool_plan_eval.py --tasks evals/tool_plan/tasks.jsonl --dry-run
```

Example live run:

```bash
python scripts/tool_plan_eval.py \
  --tasks evals/tool_plan/tasks.jsonl \
  --output .smallctl/artifacts/tool_plan_eval_results.jsonl
```

Each task is run once with `--run-mode loop` and once with
`--run-mode tool_plan`. The runner records duration, exit code, stdout/stderr,
and the last JSON object printed by `smallctl` when one can be parsed.

