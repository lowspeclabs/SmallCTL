# LLMCompiler-lite Plan: Parallel Read-Only ToolPlan Execution

## 1. Current Codebase Shape

SmallCTL already has most of the runtime mechanics needed for LLMCompiler-style
parallel evidence gathering. The remaining work is mostly about tightening the
safety boundary, teaching the planner to create useful dependency graphs, and
adding observability.

| Component | Current status | Location |
|-----------|----------------|----------|
| ToolPlan step schema with `depends_on` | Exists | `src/smallctl/graph/tool_plan_schema.py` |
| Topological batch builder | Exists | `src/smallctl/graph/tool_dag.py` |
| Async DAG dispatcher with `asyncio.gather` and semaphore | Exists | `src/smallctl/graph/tool_dag_executor.py` |
| DAG safety guard | Exists, but blocklist-only | `src/smallctl/graph/tool_dag_safety.py` |
| ToolPlan validation allowlist | Exists as `READONLY_TOOL_PLAN_TOOLS` | `src/smallctl/graph/tool_plan_safety.py` |
| Observation compression, dedupe, evidence attachment | Exists | `src/smallctl/graph/tool_plan_observations.py` |
| ReWOO lane frames around planner and solver | Exists | `src/smallctl/graph/runtime_tool_plan.py`, `src/smallctl/context/rewoo_lanes.py` |
| ToolPlan runtime metrics | Partially exists | `src/smallctl/graph/runtime_tool_plan.py` |
| Eval scaffold for ToolPlan tasks | Exists and has DAG-oriented fixtures | `evals/tool_plan/`, `scripts/tool_plan_eval.py` |
| Config flags for DAG execution | Exist in config, missing from example | `src/smallctl/config.py`, `.smallctl.yaml.example` |

Key current defaults:

```python
tool_plan_runtime_enabled = False
tool_dag_enabled = False
tool_dag_max_parallel = 4
tool_dag_timeout_sec = 30
tool_dag_preserve_result_order = True
tool_plan_observation_token_limit = 900
tool_plan_max_observation_chars_per_step = 600
```

## 2. Gaps To Close

1. `tool_dag_enabled` exists but defaults to `False`, so the DAG executor is
   currently opt-in.
2. The planner prompt does not explain `depends_on`, so models have little
   incentive to omit false dependencies between independent reads.
3. DAG safety currently rejects only known mutating tools. For parallel dispatch,
   the safer boundary is a positive allowlist.
4. `ssh_file_read` is registered and read-only, but it is not allowed in
   ToolPlan, so remote read-only evidence plans cannot use it.
5. DAG execution does not record batch shape, wall time, estimated serial time,
   speedup, or fallback count.
6. `.smallctl.yaml.example` documents ToolPlan but not the DAG flags.
7. Git/log read helpers (`git_status`, `git_diff`, `read_log`) are still absent
   from the registry and should remain a later phase.

## 3. Design Decisions

- Keep ToolPlan read-only. Do not admit write, shell, control, memory, or human
  interaction tools into ToolPlan.
- Use one explicit allowlist for tools that can appear in ToolPlan and run in
  DAG batches. Today the best name is `PARALLELIZABLE_TOOL_PLAN_TOOLS`, with
  `READONLY_TOOL_PLAN_TOOLS` kept as a compatibility alias if needed.
- Treat `depends_on` as planner-owned. The executor should only respect declared
  edges and should not infer hidden dependencies.
- Keep `ssh_exec` blocked. It can run arbitrary remote commands even when a
  prompt asks for a read.
- Preserve result order by default. The current observation builder maps records
  back to step ids, but stable order makes debugging and tests simpler.
- Roll out by first making the implementation safe and observable, then flip the
  default once tests and evals are green.

## 4. Phase 1: Positive Allowlist And Remote Reads

Goal: make the set of parallelizable ToolPlan tools auditable and include
remote read-only file reads.

### 4.1 Schema

Update `src/smallctl/graph/tool_plan_schema.py`:

```python
PARALLELIZABLE_TOOL_PLAN_TOOLS = frozenset({
    "file_read",
    "dir_list",
    "grep",
    "find_files",
    "artifact_read",
    "artifact_grep",
    "web_search",
    "web_fetch",
    "ssh_file_read",
})

READONLY_TOOL_PLAN_TOOLS = set(PARALLELIZABLE_TOOL_PLAN_TOOLS)
```

Keep `MUTATING_TOOL_PLAN_BLOCKLIST` for validation messages and defense in
depth. Add aliases only if the parser or prompt actually needs them.

### 4.2 Safety

Replace the DAG blocklist check with a positive allowlist check in
`src/smallctl/graph/tool_dag_safety.py`:

```python
from .tool_plan_schema import PARALLELIZABLE_TOOL_PLAN_TOOLS, ToolPlanStep


class NonParallelizableStepInDAGError(Exception):
    """Raised when a DAG batch contains a tool outside the parallel allowlist."""


def assert_parallelizable_steps(batches: list[list[ToolPlanStep]]) -> None:
    for batch in batches:
        for step in batch:
            if step.tool not in PARALLELIZABLE_TOOL_PLAN_TOOLS:
                raise NonParallelizableStepInDAGError(
                    f"DAG batch contains non-parallelizable tool "
                    f"'{step.tool}' in step '{step.id}'"
                )
```

Compatibility option: keep `MutatingStepInDAGError` as an alias or subclass
during the migration if existing tests or callers still import it.

Then update `src/smallctl/graph/runtime_tool_plan.py` to call
`assert_parallelizable_steps(batches)`.

### 4.3 ToolPlan Validation

Update `src/smallctl/graph/tool_plan_safety.py`:

- Include `ssh_file_read` in the allowed ToolPlan set through the schema alias.
- Add `ssh_file_read` to a remote path validation bucket rather than
  `_LOCAL_PATH_TOOLS`.
- Validate required remote-read args according to the registered tool shape in
  `src/smallctl/tools/register_operational.py`.
- Keep local path validation unchanged for `file_read`, `dir_list`, `grep`, and
  `find_files`.

### 4.4 Tests

Update or add:

| Test | File |
|------|------|
| allowlisted tools pass DAG safety | `tests/test_tool_dag_safety.py` |
| unknown read-ish tool is blocked by DAG safety | `tests/test_tool_dag_safety.py` |
| `file_write` and `shell_exec` remain blocked | `tests/test_tool_dag_safety.py` |
| `ssh_file_read` validates in ToolPlan when registered | `tests/test_tool_plan_safety.py` |
| local path checks do not run against remote paths | `tests/test_tool_plan_safety.py` |

## 5. Phase 2: Planner Prompt For Parallel Thinking

Goal: make independent reads naturally appear in the same DAG batch.

Update `build_tool_plan_planner_prompt` in
`src/smallctl/graph/tool_plan_prompts.py`.

Current signature:

```python
def build_tool_plan_planner_prompt(*, task: str, max_steps: int, context_frame: str = "") -> str:
```

Recommended signature:

```python
def build_tool_plan_planner_prompt(
    *,
    task: str,
    max_steps: int,
    max_parallel: int = 4,
    context_frame: str = "",
) -> str:
```

Then pass `tool_dag_max_parallel` from `_prepare_planner_prompt` in
`runtime_tool_plan.py`.

Prompt requirements:

- Show `depends_on` as optional in the JSON shape.
- State that omitted or empty `depends_on` means the step can run in parallel.
- Instruct the planner to use dependencies only when a step needs a value found
  by an earlier step.
- Include common false-dependency examples.
- Include the current `TOOL_PLAN_TOOL_LIST`, including `ssh_file_read` after
  Phase 1.
- Keep the existing source-code guidance and context-frame insertion.

Suggested prompt block:

```text
Dependency rules:
- Omit depends_on or set it to [] when a step does not need output from another step.
- Add depends_on only when this step literally needs a path, id, URL, query term,
  or other value discovered by an earlier step.
- Do not make file reads depend on searches unless the search result determines
  the path to read.
- Do not make web_search depend on file_read unless the query comes from that file.
- Independent steps may run concurrently. Max parallel width is {max_parallel}.
```

Add a focused prompt test in `tests/test_tool_plan_prompts.py` or the existing
prompt/eval test file.

## 6. Phase 3: DAG Metrics And Fallback Accounting

Goal: know whether DAG execution is actually helping.

### 6.1 Runtime Metrics

In `_dispatch_tools_node`, record:

| Metric | Type | Description |
|--------|------|-------------|
| `tool_plan_dag_batch_count` | gauge | Number of non-empty topological batches dispatched |
| `tool_plan_dag_max_batch_size` | gauge | Largest batch width |
| `tool_plan_dag_step_count` | gauge | Number of pending calls dispatched through DAG |
| `tool_plan_dag_actual_ms` | timer/gauge | Wall time around `dispatch_tool_dag` |
| `tool_plan_dag_estimated_serial_ms` | timer/gauge | Sum of per-call dispatch durations |
| `tool_plan_dag_speedup` | float | `estimated_serial_ms / actual_ms`, when both are positive |
| `tool_plan_dag_fallback_count` | counter | DAG safety or dispatch setup failed and serial dispatch was used |
| `tool_plan_dag_dispatch_error_count` | counter | Individual DAG call exceptions/timeouts |

Store these through `recovery_metrics(harness.state)`, matching the surrounding
ToolPlan metric style.

### 6.2 Per-Step Timing

Modify `dispatch_tool_dag` or `_dispatch_single_tool` to attach timing metadata
to each `ToolExecutionRecord` result:

```python
metadata["dag_latency_ms"] = elapsed_ms
metadata["dag_batch_index"] = batch_index
metadata["dag_batch_size"] = len(batch)
```

Avoid changing the public return shape unless needed. Metadata is enough for
speedup math and downstream diagnostics.

### 6.3 Fallback Behavior

Current behavior falls through to serial dispatch only for
`MutatingStepInDAGError`. Expand this to cover the new safety exception and DAG
setup failures. Keep individual tool failures inside the DAG as normal
`ToolEnvelope(success=False, ...)` records.

Do not add an automatic kill-switch yet. A per-plan failure cascade toggle is
only useful after the metrics show repeated DAG setup failures in real runs.

## 7. Phase 4: Observation Budget Tuning

Goal: keep solver context stable when the planner fans out.

The current observation path already:

- maps records back to ToolPlan step ids,
- dedupes identical tool and arg pairs before final token fitting,
- truncates summaries/excerpts to `tool_plan_max_observation_chars_per_step`,
- attaches observations as evidence records.

Add only a small batch-aware budget adjustment after Phase 3 metrics exist:

```python
token_limit = int(_tool_plan_config(self.deps, "tool_plan_observation_token_limit", 900) or 900)
max_chars = int(_tool_plan_config(self.deps, "tool_plan_max_observation_chars_per_step", 600) or 600)

metrics = recovery_metrics(self.deps.harness.state)
max_batch = int(metrics.get("tool_plan_dag_max_batch_size", 1) or 1)
if max_batch >= 3:
    token_limit = int(token_limit * 0.85)
    max_chars = int(max_chars * 0.80)
```

Keep this heuristic conservative and covered by tests that assert duplicate
reads still dedupe before token fitting.

## 8. Phase 5: Config And Rollout

### 8.1 Example Config

Add the existing DAG flags to `.smallctl.yaml.example`:

```yaml
tool_dag_enabled: false
tool_dag_max_parallel: 4
tool_dag_timeout_sec: 30
tool_dag_preserve_result_order: true
```

Keep `tool_dag_enabled: false` until Phases 1-4 are merged and evals are green.
After that, flip the dataclass default in `src/smallctl/config.py` to `True`
and update the example to match.

### 8.2 Eval Pass

Use the existing ToolPlan eval scaffold:

```bash
PYTHONPATH=src python scripts/tool_plan_eval.py
```

Pay special attention to:

- `evals/tool_plan/repo_tool_dag_execution.yaml`
- `evals/tool_plan/config_tool_dag_flags.yaml`
- `evals/tool_plan/repo_tool_dispatch_persistence.yaml`
- `evals/tool_plan/wrong_path_recovery_001.yaml`

Useful eval expectations:

- independent read-only steps have omitted or empty `depends_on`,
- dependent reads preserve real dependencies,
- DAG flags appear in config-oriented tasks,
- wrong-path recovery still rejects unsafe local paths.

### 8.3 Default-On Gate

Make `tool_dag_enabled` default to `True` only after:

- unit tests pass for schema, safety, prompt, executor, observations, and runtime,
- ToolPlan evals do not regress,
- fallback and dispatch error metrics are visible,
- at least one integration test proves a multi-step independent plan executes as
  one DAG batch.

## 9. Phase 6: Optional Git And Log Read Tools

These are useful but not required for the initial latency win. Keep them out of
Phases 1-5 unless the evals show a strong need.

### 9.1 `git_status`

- Args: `path` optional workspace-relative cwd, `short` bool default `true`.
- Runs `git -C <path> status --short` or `git -C <path> status`.
- Returns stdout plus parsed clean/dirty status when practical.

### 9.2 `git_diff`

- Args: `path` optional workspace-relative cwd, `cached` bool default `false`,
  `target` optional workspace-relative file path.
- Runs `git -C <path> diff [--cached] [-- <target>]`.
- Returns truncated diff text with metadata.

### 9.3 `read_log`

- Args: `path` workspace-relative log file, `lines` default `100`, optional
  `offset`.
- Reuse local path safety from `tool_plan_safety.py`.
- Prefer tail-style bounded reads over full-file reads.

### 9.4 Registration

Register git/log tools in either `src/smallctl/tools/register_filesystem.py` or
a new focused registration module. Add them to
`PARALLELIZABLE_TOOL_PLAN_TOOLS` only after they have path safety tests.

## 10. File-Level Change List

| File | Planned change |
|------|----------------|
| `src/smallctl/graph/tool_plan_schema.py` | Add positive parallel allowlist; include `ssh_file_read`; keep read-only alias |
| `src/smallctl/graph/tool_plan_safety.py` | Validate `ssh_file_read`; separate local and remote path rules |
| `src/smallctl/graph/tool_dag_safety.py` | Replace blocklist-only guard with allowlist guard |
| `src/smallctl/graph/runtime_tool_plan.py` | Pass `max_parallel` into prompt; use new safety guard; record DAG metrics; tune observation budget |
| `src/smallctl/graph/tool_dag_executor.py` | Attach per-call DAG timing metadata |
| `src/smallctl/graph/tool_plan_prompts.py` | Teach `depends_on` and independent-step parallelism |
| `.smallctl.yaml.example` | Document existing DAG flags |
| `tests/test_tool_dag_safety.py` | Update for positive allowlist |
| `tests/test_tool_plan_safety.py` | Add `ssh_file_read` ToolPlan validation coverage |
| `tests/test_tool_plan_prompts.py` or existing prompt tests | Assert dependency guidance appears |
| `tests/test_runtime_tool_plan.py` | Cover one-batch independent plan and fallback metrics |
| `tests/test_tool_plan_observations.py` | Cover batch-aware budget behavior if added |
| `evals/tool_plan/*.yaml` | Keep DAG eval expectations aligned with new prompt and metrics |

## 11. Expected Impact

With DAG execution enabled and the planner omitting false dependencies, a typical
four-step evidence plan can move from roughly serial latency to the latency of
the slowest independent step plus small scheduling overhead.

Safety should improve rather than loosen: ToolPlan remains read-only, mutating
tools stay blocked, and DAG dispatch now requires an explicit positive allowlist
instead of relying only on a mutating-tool blocklist.
