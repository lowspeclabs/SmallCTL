# ReWOO Harness Measurement Plan

## Goal

Validate whether SmallCTL should keep investing in the ReWOO-style
Planner -> Worker -> Solver path for local tasks.

The practical claim is still the same: separating reasoning from tool
observations should improve small-model robustness and reduce prompt/token
pressure, especially when tools fail or when the task is mostly evidence
gathering.

Primary decision question:

> Should SmallCTL keep expanding `tool_plan`, narrow it to specific read-heavy
> task classes, or pause further architecture work until measurement improves?

## Current Codebase Shape

The codebase has moved beyond an initial scaffold. The current implementation
already has a working ToolPlan runtime plus several related runtime systems.

### Runtime And Routing

- `src/smallctl/graph/runtime_tool_plan.py` owns the ToolPlan graph:
  planner prompt, planner model call, plan parse/validation, read-only worker
  dispatch, observation compression, solver prompt/model call, optional solver
  refine, fallback-to-loop, and trajectory recording.
- `src/smallctl/graph/runtime_auto.py` can route to explicit or auto-selected
  runtimes, including `ToolPlanRuntime` when configured.
- `src/smallctl/graph/runtime_specialized.py` exports the specialized runtimes.
- `src/smallctl/harness/run_mode.py` contains the run-mode decision service and
  follow-up/approval routing helpers.
- `src/smallctl/config.py`, `src/smallctl/config_support.py`,
  `src/smallctl/harness/initialization.py`, and `src/smallctl/main.py` all know
  about ToolPlan/ReWOO flags.

Relevant config fields now include:

- `run_mode`
- `tool_plan_runtime_enabled`
- `tool_plan_auto_select`
- `tool_plan_readonly_only`
- `tool_plan_max_steps`
- `tool_plan_max_repair_attempts`
- `tool_plan_observation_token_limit`
- `tool_plan_max_observation_chars_per_step`
- `tool_plan_solver_fresh_output_limit`
- `tool_plan_allow_web`
- `tool_plan_allow_artifact_read`
- `tool_plan_fallback_to_loop_on_invalid_plan`
- `tool_dag_enabled`
- `tool_dag_max_parallel`
- `tool_dag_timeout_sec`
- `tool_dag_preserve_result_order`
- `solver_refine_enabled`
- `solver_refine_max_passes`
- `solver_refine_*`
- `rewoo_lane_frames_enabled`
- `rewoo_planner_frame_enabled`
- `rewoo_solver_frame_enabled`
- `rewoo_refiner_frame_enabled`
- `rewoo_frame_token_budget`

### Plan Parsing, Safety, Dispatch, And DAG

- `src/smallctl/graph/tool_plan_schema.py` defines the ToolPlan schema and
  read-only/mutating tool sets.
- `src/smallctl/graph/tool_plan_parser.py` parses bounded ToolPlan JSON.
- `src/smallctl/graph/tool_plan_safety.py` validates read-only, workspace-safe
  steps.
- `src/smallctl/graph/tool_plan_executor.py` converts accepted steps into
  pending tool calls.
- `src/smallctl/graph/tool_dag.py`,
  `src/smallctl/graph/tool_dag_executor.py`, and
  `src/smallctl/graph/tool_dag_safety.py` provide optional parallel execution
  for independent read-only steps.
- `src/smallctl/graph/tool_execution_nodes.py`,
  `src/smallctl/harness/tool_dispatch.py`, and persistence helpers remain the
  underlying execution/persistence path for both normal loop and ToolPlan work.

### Observations, Evidence, And ReWOO Lanes

- `src/smallctl/graph/tool_plan_observations.py` turns worker execution records
  into compact `ToolPlanObservation` objects, renders the solver observation
  block, dedupes repeated reads, preserves artifact/operation ids, records
  missing execution records, and attaches observations to the reasoning graph.
- `src/smallctl/context/rewoo_lanes.py` compiles role-specific ReWOO context
  frames for planner, solver, and refiner.
- `src/smallctl/graph/tool_plan_prompts.py` builds planner and solver prompt
  suffixes.
- `src/smallctl/harness/trajectory_recorder.py` records ToolPlan trajectories
  under `.smallctl/traces`, including plan steps, rendered observations, final
  answer, success, and core ToolPlan metrics.
- FAMA now consumes some ToolPlan metrics through
  `src/smallctl/fama/detectors.py` and related tests.

### Metrics Already Recorded

The runtime already records these recovery metrics:

- `tool_plan_invocations`
- `tool_plan_parse_failures`
- `tool_plan_steps_requested`
- `tool_plan_steps_executed`
- `tool_plan_step_failures`
- `tool_plan_unsafe_steps_blocked`
- `tool_plan_wrong_path_count`
- `tool_plan_repeated_read_count`
- `tool_plan_fallback_count`
- `tool_plan_evidence_before_patch_count`
- `tool_plan_observation_tokens`
- `tool_plan_planner_tokens`
- `tool_plan_solver_tokens`
- `tool_plan_total_tokens`

Planner/solver token accounting is ToolPlan-specific and currently accumulates
into `tool_plan_total_tokens`. The eval runner now uses generic final-result
`token_usage` for loop-vs-ToolPlan comparisons and keeps ToolPlan token counters
as diagnostics.

### Eval Runner And Fixtures

- `scripts/tool_plan_eval.py` runs task pairs in `loop` and `tool_plan` mode,
  accepts a task file or directory, supports `.jsonl`, `.yaml`, and `.yml`,
  dedupes tasks by `id`, tails child logs, records stdout/stderr tails, parses
  the last JSON object, extracts the ToolPlan recovery metrics above, checks
  ReWOO prompt-shape markers, and writes JSONL results plus a comparison report.
- `tests/test_tool_plan_eval_runner.py` covers task loading, command
  construction, metric extraction, success classification, prompt-shape checks,
  wrong-path report behavior, and report exit code.
- `evals/tool_plan/README.md` documents dry-run/live usage.
- `evals/tool_plan/tasks.jsonl` currently contains repo-analysis, FAMA,
  web-surface, bughunt, web-research, log-investigation, and wrong-path tasks.
- Additional YAML fixtures currently present:
  `multi_file_bughunt_001.yaml`, `log_investigation_001.yaml`,
  `web_research_001.yaml`, and `wrong_path_recovery_001.yaml`.

Important working-tree note: `evals/tool_plan/README.md` and
`evals/tool_plan/tasks.jsonl` already have local modifications. Treat them as
user-owned unless a task explicitly asks to edit them.

## Current Status

The implementation plan is now complete enough to make a measurement decision
from local runs.

Completed measurement work:

- Generic token usage is extracted from final JSON for both loop and ToolPlan
  runs. Missing usage is reported as `null`, and token deltas are computed only
  when both sides are measurable.
- ToolPlan-specific planner, solver, and observation token counters remain as
  diagnostics rather than loop-comparison inputs.
- The runtime records ToolPlan phase timers for planner, worker/evidence
  dispatch, and solver model calls. The eval runner flattens those selected
  `latency_metrics` plus wall-clock duration deltas.
- Planner validity, repair attempts, unsafe/wrong-path counts, accepted step
  count, and accepted tools are exported into report comparisons.
- Worker aggregate stats are exported from observation building: requested and
  executed steps, failures, success rate, missing records, duplicate reads,
  artifact yield, and failure classes.
- Solver grounding is scored deterministically from fixture expectations,
  ToolPlan evidence ids, artifact refs, required files, required terms, refine
  verdict, and evidence-before-mutation metrics.
- Robustness fields are normalized into comparisons: timeout, return code,
  fallback count, abort count, loop-guard count, repeated reads, planner repair
  loops, and model-stream halt count.
- The runner supports `--repeat`, per-task filtering, JSON and markdown reports,
  by-tag summaries, acceptance gates, and nonzero exit when the report decision
  is not `continue`.

Known limitations:

- Solver grounding is deterministic and intentionally conservative. It is not a
  judge-model evaluation, so it can miss nuanced unsupported claims.
- Token-savings gates apply only when both loop and ToolPlan usage are present.
- Web research fixtures remain separated by tag because they are inherently
  time- and network-sensitive.

## Fixture Schema

The runner preserves arbitrary task keys and scores optional `expectations`
blocks while remaining backward-compatible with fixtures that do not define
expectations.

Example:

```yaml
id: repo_runtime_routing
task: >-
  Read through the repo and find where runtime mode routing happens.
tags:
  - repo_spelunking
expectations:
  required_files:
    - src/smallctl/main.py
    - src/smallctl/graph/runtime_auto.py
    - src/smallctl/graph/runtime_tool_plan.py
    - src/smallctl/harness/run_mode.py
  required_terms:
    - ToolPlanRuntime
    - run_mode
    - tool_plan_runtime_enabled
  grounding_required: true
  success_statuses:
    - completed
  unsafe_fallback_expected: false
```

Expectation scoring tolerates missing `expectations`.

## Fixture Coverage Target

Keep the suite balanced and tag-driven so regressions can be localized.

### Coding

Purpose: measure whether ToolPlan gathers enough evidence before proposing or
making code changes.

Current seed:

- `multi_file_bughunt_001`

Covered fixtures:

- `coding_symlink_read_bug`
- `coding_patch_existing_bug`
- `coding_test_failure_root_cause`

Expectations:

- Required files/functions are mentioned.
- Solver cites worker evidence ids or artifact ids where grounding is required.
- No mutation happens before evidence unless the task explicitly asks for an
  immediate edit.

### Repo Spelunking

Purpose: measure read-only architecture discovery.

Current seeds:

- `runtime_seams`
- `tool_dispatch_seams`
- `fama_capsule_path`
- `web_tool_surface`

Covered fixtures:

- `repo_runtime_routing`
- `repo_tool_dispatch_persistence`
- `repo_rewoo_lane_frames`
- `repo_tool_dag_execution`

Expectations:

- Required files/functions are identified.
- ToolPlan uses bounded grep/read steps.
- Solver output is grounded in direct observations.

### Config And Debug

Purpose: measure config/env flag discovery and failure-route robustness.

Current seed:

- `wrong_path_recovery_001`

Covered fixtures:

- `config_rewoo_flags`
- `config_tool_dag_flags`
- `debug_wrong_path_fallback`
- `debug_provider_timeout`

Expectations:

- Workspace boundaries are respected.
- Config source chain is reported.
- Unsafe-plan fallback is success for wrong-path fixtures.

### Log Triage

Purpose: measure noisy observation handling and failure-class extraction.

Current seed:

- `log_investigation_001`

Covered fixtures:

- `log_error_frequency`
- `log_model_stream_halt`
- `log_tool_loop_guard`

Expectations:

- If logs are present, identify the most frequent class.
- If logs are absent, say so and inspect code paths instead.
- Do not fabricate counts.

### Web Research

Purpose: keep network/time-sensitive behavior visible but separate from the
core local benchmark.

Current seed:

- `web_research_001`

Expectations:

- Tag web tasks separately.
- Prefer authoritative sources.
- Do not let web instability dominate local ToolPlan acceptance gates.

## Desired Report Format

The report should evolve from a flat comparison object into a summary plus
tag-level and per-task details.

Target shape:

```json
{
  "summary": {
    "total_comparisons": 0,
    "tool_plan_wins": 0,
    "loop_wins": 0,
    "both_pass": 0,
    "both_fail": 0,
    "token_delta_total": null,
    "token_delta_pct_mean": null,
    "latency_delta_total_sec": 0.0,
    "latency_delta_pct_mean": 0.0,
    "planner_valid_rate": 0.0,
    "worker_success_rate_mean": 0.0,
    "solver_grounded_rate": 0.0,
    "abort_loop_rate": 0.0
  },
  "by_tag": {},
  "prompt_shape_failures": [],
  "comparisons": []
}
```

Each comparison should include:

- Task metadata: `task_id`, `tags`, `task`, optional `expectations`.
- Success: loop/tool-plan final success and expectation score.
- Planner: validity, repair count, unsafe count, accepted step count, tools.
- Worker: requested/executed/success rate/failure classes/artifact yield.
- Solver: grounded flag, evidence refs, artifact refs, unbacked-claim count.
- Efficiency: loop/tool-plan tokens, token delta, loop/tool-plan duration,
  latency delta.
- Robustness: timeout, fallback, abort, loop-guard counts.
- Prompt shape: existing ReWOO role-frame assertions.

## Acceptance Gates

Use these gates before treating ReWOO as broadly justified:

- Planner validity >= 90% across non-adversarial fixtures.
- Worker success rate >= 85% across fixtures, excluding intentionally unsafe
  tasks.
- Solver grounded rate >= 85% where `grounding_required: true`.
- ToolPlan abort/loop rate is not worse than loop baseline.
- ToolPlan total tokens improve by at least 20% on repo-spelunking and
  log-triage tasks where loop usage is measurable.
- Latency is no worse than 1.25x loop median unless success rate or token use
  clearly improves.
- Wrong-path/config-debug fixtures show equal or better safety behavior than
  loop.

If these fail, the next architecture move should be targeted: planner validity,
worker observation quality, solver grounding, or routing/gating.

## Implementation Phases Status

### Phase 1: Make Eval Tokens And Latency Honest - Complete

Files:

- `scripts/tool_plan_eval.py`
- `tests/test_tool_plan_eval_runner.py`
- possibly final-result payload assembly if generic usage is missing

Tasks:

1. Extract generic token usage for both loop and ToolPlan runs.
2. Stop treating missing loop tokens as `0`.
3. Extract selected `latency_metrics` from final JSON.
4. Add token and latency deltas to rows and comparisons.
5. Add tests for missing/null usage and measurable loop usage.

Delivered:

- The report no longer claims token savings from missing loop data.
- ToolPlan planner, worker, and solver phase timers are emitted in
  `latency_metrics` when the runtime takes that path.

### Phase 2: Add Stage Scores - Complete

Files:

- `src/smallctl/graph/runtime_tool_plan.py`
- `src/smallctl/graph/tool_plan_observations.py`
- `scripts/tool_plan_eval.py`
- `tests/test_tool_plan_eval_runner.py`
- `tests/test_runtime_tool_plan.py`
- `tests/test_tool_plan_observations.py`

Tasks:

1. Export planner validity, repair attempts, accepted tools, and step count.
2. Export worker success stats, missing records, artifact yield, and failure
   classes.
3. Add deterministic solver-grounding scorer in the eval runner.
4. Aggregate planner valid rate, worker success rate, and solver grounded rate.
5. Add focused unit tests for each scoring path.

Delivered:

- The report says which ToolPlan stage worked or failed.

### Phase 3: Add Expectations And Fixture Coverage - Complete

Files:

- `evals/tool_plan/*.yaml`
- `evals/tool_plan/tasks.jsonl`
- `evals/tool_plan/README.md`
- `scripts/tool_plan_eval.py`
- `tests/test_tool_plan_eval_runner.py`

Tasks:

1. Add optional `expectations` fields to fixtures.
2. Score required files, required terms, grounding markers, and expected unsafe
   fallback.
3. Expand fixtures across coding, repo spelunking, config/debug, log triage,
   and web research.
4. Preserve backward compatibility for existing simple fixtures.
5. Update README commands after report format changes.

Delivered:

- A local benchmark suite that represents SmallCTL's actual workload.

### Phase 4: Robustness Sampling - Complete

Files:

- `scripts/tool_plan_eval.py`
- optional `evals/tool_plan/faults/` fixtures

Tasks:

1. Add intentionally bad-path and missing-file tasks.
2. Add tasks where one planned read fails but another evidence path exists.
3. Track fallback, abort, loop-guard, and model-stream halt rates by tag.
4. Add `--repeat N` for small-model stochasticity.
5. Aggregate mean, median, and worst-case metrics.

Delivered:

- Evidence about ToolPlan under tool failure, not just happy-path completion.

### Phase 5: Human Decision Report - Complete

Files:

- `scripts/tool_plan_eval.py`
- `evals/tool_plan/README.md`
- optional `rewoo-harness-results.md`

Tasks:

1. Write a markdown summary next to the JSON report.
2. Include pass/fail gates, per-tag regressions, and top failure examples.
3. Emit a recommended decision: continue, narrow rollout, or pause.

Delivered:

- A report that is readable before deciding what to build next.

## Suggested Commands

Dry-run all fixtures:

```bash
python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --mode both --rewoo-frames --dry-run
```

Run the ReWOO comparison:

```bash
PYTHONPATH=src python scripts/tool_plan_eval.py \
  --tasks evals/tool_plan/ \
  --mode both \
  --rewoo-frames \
  --output .smallctl/artifacts/tool_plan_rewoo_eval_results.jsonl \
  --report .smallctl/artifacts/tool_plan_rewoo_eval_report.json
```

Run the focused tests:

```bash
pytest tests/test_tool_plan_eval_runner.py tests/test_runtime_tool_plan.py tests/test_tool_plan_observations.py tests/test_rewoo_lanes.py tests/test_tool_dag.py tests/test_tool_dag_executor.py
```

## Risks And Controls

- Provider usage may be missing or provider-specific. Treat missing token data
  as `null`, not zero.
- Solver grounding can be gamed by printing evidence ids. Combine evidence-id
  checks with required files/terms and fixture-specific expectations.
- Log-triage tasks are environment-sensitive. Fixtures should define expected
  behavior when logs are absent.
- Web tasks are time-sensitive and network-sensitive. Keep them tagged apart
  from core local acceptance gates.
- Repeated runs may be expensive. Default to one run and make `--repeat`
  explicit.
- ToolPlan DAG can improve latency but creates different failure modes. Keep
  DAG metrics separate enough to tell serial ToolPlan from parallel ToolPlan.

## Next Patch Checklist

The original implementation checklist is complete. The next useful patches are
measurement refinements rather than plan-completion work:

1. Add more adversarial fixtures where one planned read fails but another path
   can still recover useful evidence.
2. Tighten deterministic grounding beyond required file/term and evidence-id
   checks, especially for unsupported causal claims.
3. Decide whether to add an optional judge-model lane after the deterministic
   suite has stable baselines.
4. Run repeated live comparisons and save the resulting markdown report as a
   tracked release artifact if the team wants historical trend data.
