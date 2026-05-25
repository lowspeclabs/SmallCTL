# Tool-Integrated Test-Time Scaling for Hard Steps — Refined Implementation Plan

## Goal

Add runtime-only **test-time scaling** to SmallCTL for hard execution steps: when a staged plan step is likely to fail or has already failed once, generate a small set of candidate continuations, run them with tools/tests, score them with existing verification machinery, and commit only the best candidate.

Primary decision question:

> Does spending 2-5x model/tool effort on only hard steps improve end-to-end task success enough to justify the extra tokens, wall time, and implementation complexity?

Research framing:
- **T1**: small models benefit when verification-heavy work is offloaded to tools.
- **S\***: hybrid parallel/sequential test-time scaling improves code generation when selection is execution-grounded.
- **First Finish Search**: a cheap policy for local deployments when the first verified candidate is good enough.

## Current Codebase Shape

The current repo already has most of the primitives, but the previous plan overstated how directly they fit together.

### Runtime Shape

| Area | Current code | Implication for this plan |
|---|---|---|
| Staged execution graph | `src/smallctl/graph/runtime_staged.py` | Best v1 integration point. It already activates one `PlanStep`, prepares a step sandbox prompt, performs model/tool loops, then verifies completion. |
| Plan lifecycle | `src/smallctl/graph/plan_execution.py` | `PlanExecutionEngine` owns `active_step_id`, `active_step_run_id`, retry/block transitions, file baselines, and evidence commit. Candidate work should cooperate with this instead of replacing it. |
| Step schema | `src/smallctl/state_schema.py`, `src/smallctl/state_records.py` | `PlanStep` already has `tool_allowlist`, `prompt_token_budget`, `verifiers`, `outputs_expected`, `max_retries`, `retry_count`, and `failure_reasons`. Add only minimal metadata such as `difficulty`, if needed. |
| Loop state persistence | `src/smallctl/state.py`, `src/smallctl/state_coercion.py`, `src/smallctl/state_records.py` | Any durable candidate fields need serialization/coercion. Prefer scratchpad/transient runtime state in v1 to avoid schema churn. |
| Step sandbox prompt | `src/smallctl/context/step_sandbox.py`, `src/smallctl/graph/lifecycle_prompt.py` | `prepare_staged_prompt()` builds messages for the active step. Candidate generation can vary these messages with small suffixes. |
| Tool exposure | `src/smallctl/graph/lifecycle_prompt.py::select_staged_tools` | Staged tools are limited to `step_complete`, `step_fail`, `loop_status`, `ask_human`, plus the step allowlist. Candidate branches must preserve that rule. |
| Tool dispatch | `src/smallctl/graph/tool_execution_nodes.py` | Normal staged execution dispatches model tool calls sequentially through validation, recovery nudges, hidden-tool checks, write-session handling, and UI events. Mutating candidate execution should reuse this path. |
| Tool result persistence | `src/smallctl/graph/tool_execution_persistence.py`, `src/smallctl/graph/tool_execution_support.py` | Tool records already include `plan_id`, `step_id`, `step_run_id`, and `step_attempt`. Candidate isolation can be keyed by distinct `step_run_id`s. |
| Step verification | `src/smallctl/graph/plan_verification.py` | `StepCompletionGate.verify_step()` is binary but returns verifier details. Add scoring beside it rather than rewriting it. |
| Evidence commit | `compact_step_evidence()` + `PlanExecutionEngine.complete_step()` | Only the winning branch should become official `step_evidence`. |
| Read-only DAG execution | `src/smallctl/graph/tool_dag_executor.py`, `src/smallctl/graph/runtime_tool_plan.py` | Useful for preflight evidence and future ToolPlan scaling. It is read-only and ToolPlan-focused today; it is not the right v1 executor for mutating candidates. |
| ToolPlan runtime | `src/smallctl/graph/runtime_tool_plan.py`, `src/smallctl/graph/tool_plan_schema.py` | Mature read-only planner/solver pipeline with DAG batches, observations, ReWOO lane frames, and metrics. Treat as Phase 6+ integration. |
| Config plumbing | `src/smallctl/config.py`, `src/smallctl/config_support.py`, `.smallctl.yaml.example` | New flags must be added to dataclass defaults, env/dotenv mapping, YAML/CLI normalization, integer parsing, and the example file. |
| Recovery signals | `src/smallctl/fama/*`, `src/smallctl/recovery_metrics.py`, `src/smallctl/harness/reflexion_service.py` | Good hard-step signals exist, but v1 should use cheap heuristics first and only record scaling outcomes as metrics. |

### What Is Actually Missing

| Missing piece | Current gap |
|---|---|
| Hard-step decision | No `PlanStep.difficulty`, no detector, and no config that decides when staged execution should branch. |
| Candidate branch model | No transient object for candidate id, generated tool calls/text, distinct `step_run_id`, branch records, score, and selection status. |
| Safe branch isolation | The repo can tag records by `step_run_id`, but global state also includes recent messages, artifacts, scratchpad, write sessions, files, and real workspace mutations. This is the hardest part. |
| Candidate model calls | `model_call()` assumes one graph state and records assistant messages/tool calls into the shared harness. Need a branch-aware wrapper or a cheaper v1 that branches at retry boundaries. |
| Candidate scoring | `StepCompletionGate` produces pass/fail plus details, not numeric scores or tie-breaks. |
| Commit/discard lifecycle | There is no primitive for merging one branch and hiding or discarding losers. |
| UI/metrics | No visibility for "scaled N candidates, winner X" and no eval metrics for Pass@N/cost/latency. |

## Recommended Shape

### v1 Principle

Start with **staged execution only** and branch at the safest boundary:

1. Build candidate continuations from the active step prompt.
2. Execute candidates with distinct `step_run_id`s.
3. Score each branch through `StepCompletionGate`.
4. Commit the winner through the existing `PlanExecutionEngine.complete_step()`.
5. Keep the feature disabled by default.

Avoid making ToolPlan/DAG the v1 executor for mutating candidates. The DAG executor is currently for read-only ToolPlan batches and intentionally bypasses much of the normal staged write/recovery path.

### High-Level Flow

```text
StagedExecutionRuntime
  -> activate_or_finalize_step
  -> maybe_scale_step
       disabled / not hard:
         existing prepare_prompt -> model_call -> dispatch_tools -> persist -> apply -> verify

       hard:
         build_candidate_prompts
         generate_candidate_outputs
         execute_candidate_branches
         score_candidate_branches
         commit_winner_or_fail_step
  -> activate_or_finalize_step
```

### Candidate Branch Object

Keep this transient in v1, preferably in a new module rather than the durable `LoopState` schema.

```python
@dataclass(slots=True)
class CandidateBranch:
    branch_id: str
    candidate_idx: int
    step_id: str
    step_run_id: str
    prompt_variant: str
    assistant_text: str = ""
    thinking_text: str = ""
    pending_tool_calls: list[PendingToolCall] = field(default_factory=list)
    tool_results: list[ToolExecutionRecord] = field(default_factory=list)
    verification: StepVerificationResult | None = None
    score: "CandidateScore | None" = None
    error: str = ""
```

```python
@dataclass(slots=True)
class CandidateScore:
    passed: bool
    score: float
    failed_criteria: list[str] = field(default_factory=list)
    verifier_breakdown: list[dict[str, Any]] = field(default_factory=list)
    token_cost: int = 0
    latency_ms: float = 0.0
```

If durable debugging is needed later, store compact summaries in `scratchpad["_test_time_scaling_last"]` or recovery metrics, not full branch transcripts.

## Branch Isolation Strategy

This needs to be more conservative than the original draft.

### v1 Safe Scope

For the first implementation, allow scaling only when either:

- The step is explicitly marked `difficulty="hard"` and has a verifier strong enough to determine success, or
- The step has already failed at least once and is about to retry.

For mutating tools, v1 should **not** run branches truly in parallel against the same workspace unless the tools are known safe to isolate. Most file and shell operations mutate shared state, so naive parallel execution can interleave writes and make verification meaningless.

Recommended v1 policy:

| Candidate type | Execution policy |
|---|---|
| Read-only candidates | May run in parallel with cloned branch state and first-finish cancellation. |
| Local file-edit candidates | Run sequentially with snapshot/restore, or only branch on model proposal before applying one winner. |
| Shell/SSH/service candidates | Branch-execute only when classified read-only; mutating shell/SSH candidates are skipped and fall back to normal retry/fail policy. |

### Practical v1 Option: Proposal Scaling First

The lowest-risk implementation is **proposal scaling**:

1. Generate N candidate tool-call proposals from the active step prompt.
2. Score proposals cheaply for parse validity, tool availability, step allowlist, and risk.
3. Execute only the best proposal through the normal staged path.
4. If it fails verification, optionally execute the next candidate.

This gives much of the test-time benefit while avoiding concurrent mutations and branch rollback complexity.

### Full Branch Execution Option

If full execution is implemented, add a `CandidateStateGuard` that snapshots and restores:

- `active_step_run_id`
- `step_sandbox_history`
- `tool_execution_records`
- `recent_messages` and `transcript_messages` additions
- `scratchpad` keys touched by staged execution
- `files_changed_this_cycle`
- `write_session`
- generated artifacts and artifact metadata
- `pending_interrupt`

For local file mutations, also require either:

- A git/worktree-backed sandbox per candidate, or
- A file snapshot/restore layer for paths declared by the step verifiers/outputs and candidate local file mutation tool arguments.

Without one of those, losers cannot be safely discarded.

## Hard-Step Detection

New module: `src/smallctl/graph/hard_step_detector.py`

v1 detector should be simple and deterministic:

```python
class HardStepDetector:
    def should_scale(self, *, step: PlanStep, state: LoopState, config: SmallctlConfig) -> bool:
        ...
```

Triggers:

- Feature disabled -> always false.
- `step.difficulty == "hard"` -> true.
- `step.retry_count > 0` -> true.
- Required verifier exists and prior failure reason mentions verifier/tool failure -> true.
- Tool allowlist contains high-risk tools and config permits risk-based scaling.

High-risk allowlist examples:

```python
{"file_write", "file_append", "file_patch", "ast_patch", "file_delete", "shell_exec", "ssh_exec", "ssh_file_write", "ssh_file_patch"}
```

Use FAMA/reflexion signals only after v1:

- Recent `failure_events`
- repeated-tool-loop metrics
- schema validation repair attempts
- backend/stream halt metrics

## Candidate Generation

New module: `src/smallctl/graph/candidate_scaling.py` or separate:

- `candidate_generation.py`
- `candidate_execution.py`
- `candidate_selection.py`

Recommended first file:

`src/smallctl/graph/test_time_scaling.py`

Keep the first implementation cohesive; split once the object boundaries settle.

Generation strategies:

| Strategy | v1? | Notes |
|---|---:|---|
| Diverse prompt nudges | Yes | Best default. Reuse `build_step_sandbox_prompt()` output and append a short hidden variant instruction. |
| Temperature sweep | Maybe | Only if client/provider path can safely override temperature per call without global config mutation. |
| Plan perturbation | Later | More relevant to ToolPlan/ReWOO planning than staged execution. |

Prompt variants:

1. Standard staged prompt.
2. Conservative: verify/read before mutation, prefer smallest change.
3. Alternative: solve with a different path/tool sequence.
4. Debug-first: explain likely failure mode, then act.
5. Minimal-risk: avoid shell/SSH unless necessary.

Important: each variant must preserve staged step controls and not expose tools outside `select_staged_tools()`.

## Candidate Scoring

Extend verification without breaking existing callers.

New method in `src/smallctl/graph/plan_verification.py`:

```python
class StepCompletionGate:
    async def verify_step(...): ...

    async def score_step(self, harness: Any, step: PlanStep) -> CandidateScore:
        result = await self.verify_step(harness, step)
        ...
```

Default scoring:

- `1.0` if all required outputs and verifiers pass.
- `0.7-0.9` if required verifiers pass but optional verifiers warn.
- `0.3-0.6` for partial progress with concrete evidence.
- `0.0` for safety violation, blocked tool, unknown tool, missing required output, or failed required verifier.

Tie-breaks:

1. Required verification passed.
2. Fewer failed criteria.
3. Lower risk tool set.
4. Lower token cost.
5. Earlier completion.

Use existing `tool_execution_records` by `step_run_id`; do not score against unrelated records.

## Runtime Integration

### Minimal Graph Change

Add one new node after `activate_or_finalize_step`:

```text
activate_or_finalize_step
  -> maybe_scale_step
      -> prepare_prompt          # normal path
      -> activate_or_finalize_step / interrupt / END
```

Files:

- `src/smallctl/graph/runtime_staged.py`
- `src/smallctl/graph/runtime_base.py` only if a shared route helper is genuinely useful
- `src/smallctl/graph/test_time_scaling.py`

`maybe_scale_step` should:

1. Load plan and active step.
2. Call `HardStepDetector`.
3. If false, route to existing `prepare_prompt`.
4. If true, run candidate scaling.
5. On pass: compact evidence and call `PlanExecutionEngine.complete_step()`.
6. On fail: call `PlanExecutionEngine.fail_step()` and let existing retry/block behavior handle it.

### Avoid `nodes.py` Growth

The repo already uses dedicated modules for graph behavior. Add a small wrapper method on `StagedExecutionRuntime` rather than stuffing candidate logic into `src/smallctl/graph/nodes.py`.

## Config

Add defaults to `SmallctlConfig`:

```python
test_time_scaling_enabled: bool = False
test_time_scaling_runtimes: list[str] = field(default_factory=lambda: ["staged_execution"])
test_time_scaling_trigger: str = "retry_or_explicit"  # retry_or_explicit | explicit | heuristic | any
test_time_scaling_max_candidates: int = 3
test_time_scaling_min_candidates: int = 2
test_time_scaling_policy: str = "proposal_then_execute"  # proposal_then_execute | first_pass | sequential_branch
test_time_scaling_strategy: str = "diverse_nudges"
test_time_scaling_score_threshold: float = 0.85
test_time_scaling_parallel_max: int = 1
test_time_scaling_timeout_sec: int = 120
test_time_scaling_mutating_parallel_enabled: bool = False
test_time_scaling_all_fail_action: str = "fallback_normal_retry"  # fallback_normal_retry | fail_step
```

Wire config in:

- `src/smallctl/config.py`
- `src/smallctl/config_support.py`
- `.smallctl.yaml.example`
- CLI args in `src/smallctl/main.py` only if this repo currently exposes comparable runtime toggles there.

Policy meanings:

- `proposal_then_execute`: generate N tool-call proposals, score them cheaply, and dispatch only the selected proposal through the normal staged path.
- `first_pass`: proposal scaling that stops as soon as a clean candidate clears `test_time_scaling_score_threshold`; with parallel proposal generation, unfinished proposal calls are cancelled after the first passing result.
- `sequential_branch`: execute candidate proposals through staged dispatch one at a time, score with `StepCompletionGate`, commit the first branch that clears the threshold, and restore state/files for losing branches.
- `test_time_scaling_all_fail_action`: `fallback_normal_retry` restores branch state/files and routes to the normal staged retry path; `fail_step` calls `PlanExecutionEngine.fail_step()` immediately after all candidate branches fail.

Default `parallel_max=1` is intentional for branch execution. Proposal generation may use higher `parallel_max`; mutating branch execution remains sequential unless stronger isolation is added.

## Schema Changes

Keep schema changes minimal for v1.

Recommended:

- Add `PlanStep.difficulty: str = ""`.
- Coerce it in `src/smallctl/state_records.py::_coerce_plan_step`.
- Ensure `json_safe_value()` and existing dataclass filtering handle it naturally.
- Add tests for round-trip state serialization if needed.

Avoid adding durable `CandidateBranch` to `LoopState` in v1. Branches are execution internals, and durable branch transcripts would increase checkpoint size and prompt pollution risk.

## Implementation Phases

### Phase 0: Baseline Tests and Plan Contracts

Purpose: lock down current behavior before touching runtime.

Touch:

- `tests/test_plan_execution_engine.py`
- `tests/test_step_verification.py`
- `tests/test_state_schema_versioning.py` if `PlanStep.difficulty` is added

Acceptance:

- Existing staged execution tests still pass.
- `PlanStep.difficulty` round-trips and defaults to empty.

### Phase 1: Config and Detector

Touch:

- `src/smallctl/config.py`
- `src/smallctl/config_support.py`
- `.smallctl.yaml.example`
- `src/smallctl/graph/hard_step_detector.py`
- `tests/test_hard_step_detector.py`

Acceptance:

- Feature is off by default.
- Disabled detector always returns false.
- Explicit hard and retry triggers work.
- Context/token pressure can suppress scaling if configured.

### Phase 2: Proposal Scaling

Touch:

- `src/smallctl/graph/test_time_scaling.py`
- `src/smallctl/graph/runtime_staged.py`
- `tests/test_runtime_staged_scaling.py`

Implement:

- Build N prompt variants from `build_step_sandbox_prompt()`.
- Make N model calls, parse candidate tool calls using existing parser.
- Rank proposals by parse success, allowlist compliance, risk, and optional model text quality.
- Execute only the selected proposal through the normal staged `dispatch_tools -> persist -> apply -> verify` path, or inject the selected branch into `GraphRunState.pending_tool_calls`.

Acceptance:

- Mocked model returns three candidates.
- Invalid/unavailable tool proposal loses.
- Selected proposal enters existing dispatch path.
- Existing non-scaling staged flow is unchanged.

### Phase 3: Verification Scoring

Touch:

- `src/smallctl/graph/plan_verification.py`
- `tests/test_candidate_scoring.py`

Implement:

- `CandidateScore`
- `StepCompletionGate.score_step()`
- score from required/optional verifier results
- tie-break helper

Acceptance:

- Binary `verify_step()` behavior remains unchanged.
- `score_step()` gives deterministic scores.
- Required verifier failure cannot win over a passing branch.

### Phase 4: Sequential Full-Branch Execution

Touch:

- `src/smallctl/graph/test_time_scaling.py`
- `src/smallctl/graph/runtime_staged.py`
- `tests/test_runtime_staged_scaling.py`

Implement:

- `CandidateStateGuard` for in-memory state restore.
- Distinct `step_run_id` per candidate.
- Sequential candidate execution.
- Winner merge: restore baseline, replay/keep winner state, compact evidence, complete step.

Acceptance:

- Loser tool records and sandbox messages do not appear in official step evidence.
- Winner records retain `step_run_id`.
- Failure of all candidates calls `fail_step()` and respects existing retry/block rules.

### Phase 5: Safe File Mutation Isolation

Touch:

- `src/smallctl/graph/test_time_scaling.py`
- possibly `src/smallctl/tools/fs_*` helpers only if shared snapshot support belongs there
- tests with `tmp_path`

Implement one:

- File snapshot/restore for declared verifier/output paths.
- Or git worktree-backed candidate execution when repo is clean enough.

Acceptance:

- Candidate A edits a file incorrectly, candidate B edits it correctly.
- Final workspace contains only B's content.
- Dirty unrelated files are not reverted.

### Phase 6: Read-Only Parallel Candidate Execution

Status: implemented for staged `sequential_branch` candidates whose pending tool calls are classified read-only. Candidate proposal generation can also run in parallel. Mutating branches remain sequential.

Touch:

- `src/smallctl/graph/tool_dag_executor.py` only if semaphore/cancellation hooks are needed
- `src/smallctl/graph/runtime_tool_plan.py` later
- `tests/test_tool_dag_executor.py`

Implement:

- Parallelize only read-only candidate branches. Done for staged branch execution.
- Use `tool_dag_max_parallel` or a separate scaling semaphore. Done via `test_time_scaling_parallel_max`.
- Add first-pass/first-finish cancellation for candidates that reach score threshold. Done for read-only branch execution.

Acceptance:

- Read-only branches can execute concurrently.
- Mutating branches remain sequential unless `test_time_scaling_mutating_parallel_enabled` is explicitly true and isolation exists.

### Phase 7: Metrics, UI, and Evals

Touch:

- `src/smallctl/recovery_metrics.py` or existing metric helpers
- `src/smallctl/ui/statusbar.py` if TUI visibility is needed
- `evals/test_time_scaling/README.md`
- `evals/tool_plan/*.yaml` only if reusing fixtures is appropriate
- `scripts/tool_plan_eval.py` only if eval runner is extended

Metrics:

- scaling attempts
- candidate count
- winner score
- pass/fail
- token cost
- latency
- trigger reason
- policy used

Acceptance:

- Eval report compares baseline Pass@1 vs scaled Pass@N.
- Cost and latency are reported beside success rate.

## Files to Touch

Likely v1:

| File | Change |
|---|---|
| `src/smallctl/config.py` | Add config defaults and normalization lists. |
| `src/smallctl/config_support.py` | Add env/dotenv keys and type coercion. |
| `.smallctl.yaml.example` | Document disabled-by-default flags. |
| `src/smallctl/state_schema.py` | Add `PlanStep.difficulty` only. |
| `src/smallctl/state_records.py` | Coerce `difficulty`. |
| `src/smallctl/graph/hard_step_detector.py` | New detector. |
| `src/smallctl/graph/test_time_scaling.py` | New candidate orchestration. |
| `src/smallctl/graph/runtime_staged.py` | Add `maybe_scale_step` node/route. |
| `src/smallctl/graph/plan_verification.py` | Add scoring beside existing verification. |
| `tests/test_hard_step_detector.py` | New tests. |
| `tests/test_candidate_scoring.py` | New tests. |
| `tests/test_runtime_staged_scaling.py` | New integration-style tests. |

Later:

| File | Change |
|---|---|
| `src/smallctl/graph/tool_dag_executor.py` | Read-only candidate parallelism/cancellation hooks. |
| `src/smallctl/graph/runtime_tool_plan.py` | ToolPlan/ReWOO integration. |
| `src/smallctl/ui/statusbar.py` | Candidate status display. |
| `evals/test_time_scaling/README.md` | Eval protocol. |

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Workspace corruption from parallel mutating candidates | Do not parallelize mutating branches in v1. Add snapshot/worktree isolation before enabling it. |
| Prompt/transcript pollution from loser branches | Keep branches transient; tag by `step_run_id`; restore message and scratchpad deltas. |
| Token cost explosion | Feature off by default; cap candidates; suppress under context pressure; prefer retry/explicit triggers. |
| Backend overload | Default `parallel_max=1`; only read-only branch execution and proposal generation may raise it. |
| Verification too weak to select a winner | Scale only steps with required verifiers or explicit hard labels in v1. |
| Runtime graph complexity | Add one staged node and keep orchestration in a dedicated module. |
| Dirty worktree surprises | Never rely on destructive git resets. If using git isolation later, use separate worktrees and preserve user changes. |

## Overall Acceptance Criteria

1. With `test_time_scaling_enabled = False`, all existing tests and staged execution behavior are unchanged.
2. Config is documented and parsed from YAML/env with safe defaults.
3. Detector only triggers on explicit/retry/hard cases in v1.
4. Proposal scaling can pick the best candidate and execute it through the existing staged dispatch path.
5. Scoring is deterministic and grounded in `StepCompletionGate` verifier results.
6. Full branch execution, if enabled, proves loser branches do not pollute official evidence or final workspace state.
7. Evals report success, token cost, and latency against baseline.

## Open Questions

1. Should v1 stop at proposal scaling?
   - Recommendation: yes. It is much safer and will reveal whether candidate diversity helps before building rollback machinery.

2. Should `PlanStep.difficulty` be planner-authored or only human-authored?
   - Recommendation: allow both, but only treat exact `hard` as a trigger.

3. What is the minimum verifier strength needed for scaling?
   - Recommendation: require at least one required verifier or required output, unless the step is retrying after a concrete failure.

4. Should ToolPlan get scaling at the same time?
   - Recommendation: no. ToolPlan already has read-only DAG execution and observations; integrate after staged execution proves ROI.

5. Should FAMA learn from loser branches?
   - Recommendation: eventually yes, but v1 should only record compact metrics and avoid adding loser content to prompt-visible memory.
