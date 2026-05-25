# Found Bugs

Scope: static pass over `src/smallctl`, targeted reads of runtime/tool/config paths, plus a bounded full test run with `timeout 120s venv/bin/python -m pytest -q`.

Sorting: first by estimated fix difficulty, then by priority inside each tier.

## Easy

### P1 - `file_delete` reports success without deleting the target during an active write session

- Location: `src/smallctl/tools/fs_mutations.py:151`
- Symptom: If `file_delete()` targets the same path as an active write session, it unlinks the staging file, archives the session, clears `state.write_session`, and returns `ok("deleted")` before reaching `target.unlink()`.
- Impact: Callers receive a successful delete result while the actual target file remains on disk. This can cause false task completion, stale files, and misleading verification metadata.
- Suggested fix: After aborting the write session, continue into the normal delete path or explicitly delete `target` before returning success. If the intended behavior is only "abort session", return a distinct non-delete status.

### P1 - Repair-cycle read gate is bypassed by historical file-read evidence

- Location: `src/smallctl/tools/fs_sessions.py:260`
- Failing tests:
  - `tests/test_plan_playbook.py::test_repair_cycle_requires_read_before_patch`
  - `tests/test_plan_playbook.py::test_repair_cycle_requires_new_read_after_failed_file_patch_on_same_path`
- Symptom: `_repair_cycle_allows_patch()` falls back to `_latest_path_evidence_allows_repair_patch()` when `_repair_cycle_reads` is empty. That lets old `file_read` records satisfy a new `repair_cycle_id`, and in the tested path a repeat patch succeeds before the required fresh read.
- Impact: Repair mode can patch against stale disk state, which defeats the whole repair-cycle safety guard and can compound failed edits.
- Suggested fix: Treat a non-empty `repair_cycle_id` as requiring `_repair_cycle_reads` for that cycle. Historical tool records should only satisfy the gate if they are explicitly stamped with the current repair cycle.

### P1 - One-shot repeated-tool allowance is consumed, but the next identical call is still not blocked

- Location: `src/smallctl/graph/tool_loop_guards.py:727`
- Failing test: `tests/test_plan_playbook.py::test_one_shot_repeat_guard_allows_scheduled_recovery_file_read_once`
- Symptom: `allow_repeated_tool_call_once()` correctly removes `_repeat_guard_one_shot_fingerprints` on the first `_detect_repeated_tool_loop()` call, but the immediate second identical call still returns `None` instead of the expected repeated-loop error.
- Impact: Recovery nudges can accidentally reopen an unbounded repeated `file_read` loop after the one permitted retry.
- Suggested fix: After the one-shot is consumed, the normal repeated-loop path needs to see the existing repeated history. Check the ordering of file-read progress exemptions versus the exact/semantic fingerprint checks.

### P1 - `SMALLCTL_TOOL_PLAN_ALLOW_GIT` is never read

- Location: `src/smallctl/config_support.py:124`
- Symptom: `SmallctlConfig` has `tool_plan_allow_git`, `resolve_config()` normalizes it, and ToolPlan safety reads it, but `_env_config()` never includes `SMALLCTL_TOOL_PLAN_ALLOW_GIT`.
- Impact: Users cannot enable Git evidence steps through the environment, even though the rest of the configuration path implies support.
- Suggested fix: Add `"tool_plan_allow_git": env_or_dotenv(f"{ENV_PREFIX}TOOL_PLAN_ALLOW_GIT")` and include it in the boolean normalization list in `_env_config()`.

### P2 - Several advertised config fields are absent from env parsing

- Location: `src/smallctl/config_support.py:92`
- Symptom: An AST comparison found these `SmallctlConfig` fields missing from `_env_config()` raw parsing: `needs_human_timeout_sec`, `solver_refine_enabled`, `solver_refine_max_passes`, `solver_refine_on_final_answer`, `solver_refine_on_patch_plan`, `solver_refine_on_task_complete`, `solver_refine_token_budget`, `cleanup`, and `runtime_context_probe`.
- Impact: Environment-based configuration silently ignores these settings. This is especially visible for solver-refine and interrupt timeout behavior.
- Suggested fix: Add the missing environment keys and extend the existing bool/int/float normalization lists.

### P2 - Search providers can emit empty/invalid URL results

- Location: `src/smallctl/search_server/providers.py:175`
- Symptom: `_normalize_items()` normalizes `raw_url` but never skips empty or non-http URLs before creating `WebSearchResult`.
- Impact: Bad provider entries can become result IDs with empty canonical URLs/domains. Later `web_fetch(result_id=...)` resolves them into unusable fetch requests or confusing unknown-domain artifacts.
- Suggested fix: Drop items where normalized URL is empty or fails public web URL validation/canonicalization.

## Medium

### P1 - ToolPlan Git allowance is not wired into the harness runtime

- Locations:
  - `src/smallctl/harness/initialization.py:130`
  - `src/smallctl/harness/bootstrap_support.py:120`
  - `src/smallctl/graph/runtime_tool_plan.py:492`
- Symptom: `runtime_tool_plan` reads `tool_plan_allow_git`, but `initialize_harness()` does not accept or store it, and `build_harness_kwargs()` does not include it. Even config-file or CLI injection cannot reach runtime config.
- Impact: ToolPlan Git evidence remains effectively disabled because `_tool_plan_config(..., "tool_plan_allow_git", False)` always falls back to `False`.
- Suggested fix: Thread `tool_plan_allow_git` through `resolve_config()` -> `main.py` harness construction -> `initialize_harness()` -> `build_harness_kwargs()`.

### P1 - Solver-refine config is dead-wired and cannot enable ToolPlan refinement

- Locations:
  - `src/smallctl/harness/initialization.py:130`
  - `src/smallctl/harness/bootstrap_support.py:120`
  - `src/smallctl/graph/runtime_tool_plan.py:679`
- Symptom: `runtime_tool_plan` checks `solver_refine_enabled` and related limits, but those values are not accepted by `initialize_harness()` or included in `build_harness_kwargs()`. They are also absent from env parsing.
- Impact: The solver-refine path is effectively unreachable through normal configuration, so final ToolPlan drafts are not critiqued even when configured.
- Suggested fix: Thread the solver-refine fields through all config and harness construction paths, and add regression coverage that `config.solver_refine_enabled=True` reaches `harness.config`.

### P1 - TUI mode drops staged execution and test-time-scaling settings

- Locations:
  - TUI harness kwargs: `src/smallctl/main.py:327`
  - Non-TUI harness kwargs: `src/smallctl/main.py:463`
- Symptom: The non-TUI `Harness(...)` construction passes `staged_execution_enabled`, `staged_step_prompt_tokens`, and all `test_time_scaling_*` options. The TUI `harness_kwargs` block omits them.
- Impact: The same CLI/config flags behave differently in TUI versus non-TUI. Users launching `smallctl --tui --staged-execution` do not get the staged runtime settings they requested.
- Suggested fix: Keep TUI and non-TUI harness kwargs in one shared builder, or add the missing fields to the TUI block with tests for parity.

### P1 - `file_download` can write outside the workspace without the filesystem write guards

- Locations:
  - Handler: `src/smallctl/tools/http.py:66`
  - Registration: `src/smallctl/tools/register_operational.py:276`
- Symptom: `file_download()` resolves `output_path` directly, creates parents, and writes bytes. It does not accept `cwd`, does not use the filesystem write-session/risk helpers, and does not constrain paths to the workspace.
- Impact: A network tool can overwrite arbitrary local paths that the process can write, bypassing the more careful local file mutation flow.
- Suggested fix: Inject `cwd`, resolve relative paths against `state.cwd`, reject paths outside the workspace unless explicitly allowed, and run the same risk/write-session policy checks used by file mutation tools.

### P2 - Web fetch character budget is checked only after the network fetch has completed

- Locations:
  - Fetch count increment: `src/smallctl/tools/web.py:579`
  - Character budget check: `src/smallctl/tools/web.py:173`
- Symptom: `web_fetch()` increments the fetch count and performs `runtime.fetch()` before checking whether `full_text` fits the remaining total character budget.
- Impact: An over-budget fetch still consumes a fetch slot, performs network work, and may cache the body in the search daemon before returning a failure and no artifact.
- Suggested fix: Pre-check `bounded_max_chars` against remaining char budget before the network fetch, and charge actual fetched chars only after successful artifact persistence.

## Hard

### P1 - Configuration plumbing is duplicated enough that feature parity keeps regressing

- Locations:
  - `src/smallctl/config.py`
  - `src/smallctl/config_support.py`
  - `src/smallctl/main.py`
  - `src/smallctl/harness/initialization.py`
  - `src/smallctl/harness/bootstrap_support.py`
- Symptom: Config fields are declared, parsed, normalized, passed to CLI harness construction, passed to TUI harness construction, accepted by harness initialization, and stored into bootstrap kwargs in separate hand-maintained lists.
- Impact: Multiple active bugs above come from the same structural issue: fields exist in one layer but silently disappear in another. Any new runtime flag is likely to regress in at least one entry point.
- Suggested fix: Introduce a single typed config-to-harness projection, use it for TUI and non-TUI, and add an automated parity test that every runtime-consumed config key is either intentionally local-only or reaches `harness.config`.

## Verification Notes

- `venv/bin/python -m compileall -q src tests scripts`: passed.
- `timeout 120s venv/bin/python -m pytest -q`: failed with 3 known failures listed above; summary was `3 failed, 1636 passed, 1 skipped`.
- `venv/bin/python -m ruff check src tests scripts`: not run because `ruff` is not installed in `venv`.
- `venv/bin/python -m mypy src`: not run because `mypy` is not installed in `venv`.
