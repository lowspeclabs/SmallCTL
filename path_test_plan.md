# Gemma 4 IT `file_patch` Test Plan

## Goal

Verify that `file_patch` works end-to-end with small Gemma-4 IT checkpoints (`gemma-4-e2b`, `gemma-4-e4b`) served by a local llama.cpp backend, root-cause any harness failures, apply the narrowest possible fixes, and stress-test edge cases.

## Live Run Command

```bash
.venv/bin/python -m smallctl.main \
  --run-mode loop \
  --provider-profile llamacpp \
  --endpoint http://192.168.1.9:8080 \
  --model "Gemma 4 e4b" \
  --max-prompt-tokens 64976 \
  --fama-enabled \
  --task "<TASK_TEXT>"
```

For batch/reproducible runs always use `--run-mode loop`. The TUI (`--tui`) cannot be driven from a non-interactive shell.

## Completed Work (Run / RCA / Fix / Re-run)

### 1. No native tool calls from Gemma 4 e4b

- **Run:** Simple `file_patch` task against `./temp/hello.py`.
- **RCA:** The harness forced `reasoning_mode = "tags"` for exact small Gemma-4 IT names and instructed the model to wrap reasoning in explicit `<think>` tags. This caused the model to emit malformed JSON text instead of native `tool_calls` deltas.
- **Fix:** In `src/smallctl/harness/initialization.py`, select `reasoning_mode = "field"` with `_thinking_tags_disabled = True` for exact small Gemma-4 IT checkpoints (`gemma-4-e2b`, `gemma-4-e4b`). Larger/generic Gemma-4 IT still uses `"tags"` when explicitly classified.
- **Re-run:** Simple `file_patch` succeeded and `task_complete` reported `deliverable_verified: true`.

### 2. `task_complete` blocked on multi-change patch

- **Run:** Multi-change `file_patch` against `./temp/config.py`.
- **RCA:** The task text mentioned `TypeError`, which `runtime_error_repair.py` matched as a user-reported runtime error. That recorded a `_reported_runtime_error` and the verification gate blocked `task_complete` until the "error" was fixed.
- **Fix:** Remove bare exception class names (`TypeError`, `ModuleNotFoundError`, etc.) from `_RUNTIME_ERROR_HINT_RE` in `src/smallctl/runtime_error_repair.py`. Real reported errors are still caught by `error:`, `exception`, `traceback`, etc.
- **Re-run:** Multi-change `file_patch` completed and `task_complete` succeeded.

### 3. Complex multi-line insert/delete patch

- **Run:** One-at-a-time patches against `./temp/utils.py` (remove `old_helper`, insert `validate_email`, add type check to `add`).
- **RCA:** First attempts produced structurally broken files because `Gemma 4 e4b` constructed incorrect `target_text` / `replacement_text` when asked to perform multiple edits in a single patch.
- **Fix:** Prompting change only — instruct the model to apply exactly one patch at a time and re-read the file between patches.
- **Re-run:** Complex patch succeeded with `deliverable_verified: true`.

## Remaining Edge Cases to Test

1. **Gemma 4 e2b checkpoint**
   - Swap `--model "Gemma 4 e2b"` and re-run the simple + multi-change patch tasks.
   - Confirm the same reasoning-mode fix applies and native tool calls still work.

2. **ToolPlan runtime with Gemma 4 e4b/e2b**
   - Run `python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --mode both`.
   - Watch for planner JSON parse failures, worker observation compression issues, or solver fallback loops.

3. **Patch with ambiguous / repeated target text**
   - Task: patch a function body that appears twice in the file.
   - Expect `file_patch` to fail or warn; verify the harness reports a useful error and does not silently corrupt the file.

4. **Patch at file boundaries**
   - Insert at top of file (before first line) and delete the last function.
   - Verify line offsets and trailing newlines remain sane.

5. **Patch with Unicode / non-ASCII content**
   - Target text containing emoji or non-Latin characters.
   - Verify the replacement is byte-identical to what was requested.

6. **Patch under write session**
   - Open a `patch_existing` write session, apply `file_patch` inside it, then close the session.
   - Verify `write_session_id` and `section_name` enforcement do not block legitimate patch-based sessions.

7. **Patch failure recovery**
   - Give a `target_text` that does not exist.
   - Verify the harness records the failure, surfaces it to the model, and does not cache the file read as still-matching.

8. **Concurrent patch + read consistency**
   - `file_patch` followed immediately by `file_read` in the same turn.
   - Confirm the read returns the post-patch content, not a stale cached version.

9. **Reasoning tag regression for other Gemma variants**
   - Run the existing `tests/test_harness_gemma_reasoning_mode.py` suite after any future reasoning-mode changes.

10. **Runtime-error false-positive regression**
    - Run `tests/test_runtime_error_repair.py` and add new exception-class edge cases if more are discovered.

## Verification Commands

```bash
# Targeted tests for the fixes already landed
.venv/bin/pytest -q tests/test_harness_gemma_reasoning_mode.py tests/test_runtime_error_repair.py

# Full regression suite
.venv/bin/pytest -q

# ToolPlan eval dry-run (no backend calls)
.venv/bin/python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --dry-run

# ToolPlan eval against the live backend (costly)
.venv/bin/python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --mode both
```

## Critical Context

- Backend context limit is 32768; `--max-prompt-tokens 64976` is capped to ~26624 effective tokens.
- Native `tool_calls` deltas now appear for `Gemma 4 e4b` after the reasoning-mode fix.
- `verified_after_last_change: false` can still block `task_complete` when only `file_read` is used as verification; `shell_exec` / `ssh_exec` commands are recorded as verifiers.
- Complex multi-line insert/delete currently requires explicit one-at-a-time instructions for `Gemma 4 e4b`; this is a model-capability limitation rather than a harness bug.

## Files Involved

- `src/smallctl/harness/initialization.py` — reasoning-mode selection for Gemma models.
- `src/smallctl/runtime_error_repair.py` — `_RUNTIME_ERROR_HINT_RE` false-positive fix.
- `tests/test_harness_gemma_reasoning_mode.py` — updated Gemma reasoning-mode tests.
- `tests/test_runtime_error_repair.py` — added false-positive tests.
