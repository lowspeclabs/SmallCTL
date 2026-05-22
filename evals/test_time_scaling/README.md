# Test-Time Scaling Eval Scaffold

This folder documents the staged execution test-time scaling comparison path.
It reuses `scripts/tool_plan_eval.py` so the existing result JSONL and markdown
report flow stays in one place.

Run a dry check:

```bash
python scripts/tool_plan_eval.py \
  --comparison test_time_scaling \
  --tasks evals/test_time_scaling/ \
  --mode both \
  --dry-run
```

Run a live Pass@1 vs scaled Pass@N comparison:

```bash
python scripts/tool_plan_eval.py \
  --comparison test_time_scaling \
  --tasks evals/test_time_scaling/ \
  --mode both \
  --timeout-sec 300 \
  --repeat 3 \
  --output .smallctl/artifacts/test_time_scaling_eval_results.jsonl \
  --report .smallctl/artifacts/test_time_scaling_eval_report.json \
  --markdown-report .smallctl/artifacts/test_time_scaling_eval_report.md
```

The runner default is longer, but use at least `--timeout-sec 300` for live
smoke runs through OpenRouter or other provider-backed models. `180` seconds can
be tight for the baseline path when the run has already reached verification but
is still waiting on provider/runtime latency.

By default, this directory runs only fixtures enabled for the live smoke gate.
The source-inspection probes are preserved as exploratory fixtures and can be
run explicitly:

```bash
python scripts/tool_plan_eval.py \
  --comparison test_time_scaling \
  --tasks evals/test_time_scaling/ \
  --mode both \
  --task-id staged_read_path_probe
```

Run the local mutation isolation probe explicitly:

```bash
python scripts/tool_plan_eval.py \
  --comparison test_time_scaling \
  --tasks evals/test_time_scaling/ \
  --mode both \
  --timeout-sec 300 \
  --task-id staged_explicit_hard_file_mutation
```

or all at once:

```bash
python scripts/tool_plan_eval.py \
  --comparison test_time_scaling \
  --tasks evals/test_time_scaling/ \
  --mode both \
  --include-disabled-tasks
```

The runner uses two pseudo-modes:

- `staged_baseline`: runs planning with staged execution enabled and
  `SMALLCTL_TEST_TIME_SCALING_ENABLED=false`.
- `staged_scaled`: runs the same command with
  `SMALLCTL_TEST_TIME_SCALING_ENABLED=true`.

The report compares baseline Pass@1 to scaled Pass@N, plus token and duration
deltas, scaling attempts, candidate counts, selected candidate metadata, and
abort parity. A result of `needs harder fixtures` means the suite completed but
did not actually trigger scaling, so the fixtures should include retrying or
explicitly hard staged steps.

`staged_explicit_hard_loop_status.yaml` is the smallest smoke fixture: it keeps
the approved staged plan to one explicit hard read-only step so scaled runs
should exercise `maybe_scale_step` and emit scaling metrics quickly.
`staged_explicit_hard_file_read.yaml` extends that gate to a single file-read
step with an explicit `file_read` allowlist. The broader source inspection
fixtures and `staged_explicit_hard_file_mutation.yaml` are useful follow-up
coverage, but they are disabled for default runs because they can be slower and
more model-behavior dependent. The mutation fixture writes only under
`.smallctl/artifacts` and uses `sequential_branch` so scaled runs exercise the
isolated local mutation branch before replaying the passing candidate into the
real workspace.

Fixtures may include a `test_time_scaling:` block to make eval behavior
deterministic without changing product defaults:

```yaml
test_time_scaling:
  trigger: any
  policy: proposal_then_execute
  max_candidates: 2
  min_candidates: 2
```

## Deferred integration

This eval path currently targets staged execution only. ToolPlan/ReWOO
test-time scaling should stay deferred until the staged path has stable
fixtures, UI surfacing, and branch isolation for mutating tool candidates.
