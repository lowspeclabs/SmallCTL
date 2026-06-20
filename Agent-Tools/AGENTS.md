# Agent-Tools

This directory contains Python debugging tools for the SmallCTL harness. They
are designed for agentic use: each tool gives you structured context quickly
so you spend less time grepping logs and more time fixing the right code.

All tools are dependency-light and run with Python 3.10+. Most use only the
standard library. `tui_screenshot.py` requires the project virtualenv because
it imports SmallCTL and Textual and uses `cairosvg` for SVG-to-PNG conversion.

## Quick start

```bash
python3 Agent-Tools/logwatch.py latest
python3 Agent-Tools/run_diagnose.py latest
python3 Agent-Tools/trace_call.py --last-error
python3 Agent-Tools/runscan.py --last 20
```

Most tools also support `latest-N` (e.g. `latest-1` is the second-most-recent
run) and can be run directly because they have executable shebangs:

```bash
Agent-Tools/logwatch.py latest-1
Agent-Tools/runscan.py --last 50 --failures-only
```

## Tools

### `logwatch.py` — run health check

Summarizes a single run directory. Prints session/task status, interesting
event counts, tool dispatch summary, and recent error/warning records.

```bash
python3 Agent-Tools/logwatch.py [latest|latest-N|RUN_ID|RUN_DIR]
python3 Agent-Tools/logwatch.py 6cf0f870
python3 Agent-Tools/logwatch.py latest-1
python3 Agent-Tools/logwatch.py latest --errors      # print every error record
python3 Agent-Tools/logwatch.py latest --warnings    # print every warning record
python3 Agent-Tools/logwatch.py latest --tail 20     # last 20 harness records
python3 Agent-Tools/logwatch.py latest --save        # write logwatch-summary.json
```

Use this first when you need to know whether a run succeeded, why it failed,
and how many error/FAMA/recovery events occurred.

The summary now includes a **Primary blockers** section that surfaces
environmental blockers such as `connection refused` separately from harness/model
symptoms, and it warns when `deliverable_verified=True` looks weak (no code
changes, verifier ran before the last change, or a trivial verifier command).

**Error vs warning records:** Some SmallCTL records such as `fama_signal_detected`,
`reflexion_created`, and `same_scope_iteration_recorded` carry `failure_class`
metadata. They are counted as warnings, not errors, so you can distinguish
signals the harness noticed from actual failures.

### `run_diagnose.py` — structured failure diagnosis

Produces a narrative diagnosis of a run, classifies the failure type, and
suggests concrete next steps.

```bash
python3 Agent-Tools/run_diagnose.py [latest|latest-N|RUN_ID|RUN_DIR]
python3 Agent-Tools/run_diagnose.py latest --save    # write diagnosis.json
python3 Agent-Tools/run_diagnose.py latest --json    # structured JSON output
```

Use this when `logwatch.py` shows a non-trivial failure and you need a
starting hypothesis before opening source files. The report now separates
environmental primary blockers from model/harness failure modes.

### `runscan.py` — batch scan recent runs

When you have dozens of logs, `runscan.py` prints a compact table of recent
runs with status, duration, error/warning counts, and a failure classification.

```bash
python3 Agent-Tools/runscan.py
python3 Agent-Tools/runscan.py --last 50
python3 Agent-Tools/runscan.py --last 100 --failures-only
python3 Agent-Tools/runscan.py --last 20 --json
python3 Agent-Tools/runscan.py --same-objective "vikunja" --last 50
```

Use this to spot regressions across a sweep, find runs that need triage, or
verify that a fix reduced failures over the last N runs. The `--same-objective`
filter makes it easier to compare runs that targeted the same file or task.

### `trace_call.py` — follow a `trace_id` across channels

Every model call, tool dispatch, and chat request shares a `trace_id` like
`<session>:<task>:step-<n>:call-<m>`. This tool joins `harness.jsonl`,
`tools.jsonl`, `chat.jsonl`, and `model_output.jsonl` for that trace.

```bash
python3 Agent-Tools/trace_call.py TRACE_ID
python3 Agent-Tools/trace_call.py --run 6cf0f870 6cf0f870:task-0001:step-1:call-1
python3 Agent-Tools/trace_call.py --run latest step-1:call-1
python3 Agent-Tools/trace_call.py --run latest-1 step-1:call-1
python3 Agent-Tools/trace_call.py --last-error      # trace the most recent error
python3 Agent-Tools/trace_call.py TRACE_ID --json
python3 Agent-Tools/trace_call.py TRACE_ID --compact # collapse token/chunk records
```

Use this when you need to see exactly what the model emitted, which tool it
called, and what the tool returned for a specific step. The `--compact` flag
is useful when a trace contains hundreds of `model_token` or `chunk` records;
it also surfaces the repeated phrase when the backend halted for a repetition
loop. If the resolved run does not match the trace id prefix, the tool prints a
warning so you do not chase records from the wrong session.

### `symbolmap.py` — find definitions in source code

Maps functions, classes, event strings, and tool registrations to files in
`src/smallctl/`. It builds a cached AST index on first run.

```bash
python3 Agent-Tools/symbolmap.py dispatch_tools
python3 Agent-Tools/symbolmap.py --event action_stall
python3 Agent-Tools/symbolmap.py --tool file_patch
python3 Agent-Tools/symbolmap.py --class LoopState
python3 Agent-Tools/symbolmap.py --rebuild
```

Use this when an event, tool name, or function appears in a log and you need
to find where it is defined or emitted.

### `tui_screenshot.py` — capture the SmallCTL TUI as PNG

Launches `SmallctlApp` headlessly via Textual's pilot harness, drives a task
to completion (or a fixed wait), saves the terminal screenshot as SVG, and
converts it to PNG.

```bash
# Requires the project venv so cairosvg and textual are available
.venv/bin/python Agent-Tools/tui_screenshot.py --task "what is 2+2"
.venv/bin/python Agent-Tools/tui_screenshot.py --task "list files in src/smallctl" --timeout 60
.venv/bin/python Agent-Tools/tui_screenshot.py --task "read temp/vikunja-9b.py" --width 120 --height 40 --name vikunja

# JSON output for agents
.venv/bin/python Agent-Tools/tui_screenshot.py --task "hello" --json

# Keep the intermediate SVG
.venv/bin/python Agent-Tools/tui_screenshot.py --task "hello" --keep-svg
```

Use this when you need to visually verify what the TUI rendered after a task,
or when you want to share the exact UI state with a model for debugging.

**Requirements:**
- Run with the project virtualenv (`.venv/bin/python`) so `textual` and
  `cairosvg` are importable.
- If `cairosvg` is missing, the tool falls back to `rsvg-convert`, `inkscape`,
  or ImageMagick `convert`.
- A model endpoint must be configured (`.smallctl.yaml`, `.env`, or CLI flags).

The tool prints the PNG path on success. With `--json` it also returns the
run log directory for follow-up analysis with `logwatch.py` or `run_diagnose.py`.

### `rundiff.py` — compare two runs

Compares two run directories on outcome, tool-call volume, errors, and
event-count deltas. Useful for before/after regression checks.

```bash
python3 Agent-Tools/rundiff.py RUN_A RUN_B
python3 Agent-Tools/rundiff.py latest-1 latest
python3 Agent-Tools/rundiff.py 6d6c87f1 57e619aa
python3 Agent-Tools/rundiff.py latest-1 latest --events action_stall no_tool_recovery
python3 Agent-Tools/rundiff.py latest-1 latest --same-objective "vikunja"
```

Use this after making a code change to verify the failure rate or event
profile improved. The `--same-objective` filter ensures both runs targeted the
same task/file.

### `fama_inspector.py` — FAMA activity inspector

```bash
python3 Agent-Tools/fama_inspector.py latest
python3 Agent-Tools/fama_inspector.py 4b54c65e --signals
python3 Agent-Tools/fama_inspector.py 4b54c65e --mitigations --json
python3 Agent-Tools/fama_inspector.py 4b54c65e --exposure
```

Inspects FAMA signals, mitigations, and tool-exposure decisions. Use when
`logwatch.py` shows FAMA warnings and you need to see which detectors fired
and which mitigations were activated.

### `promptdiff.py` — compare prompt-state frame snapshots

```bash
python3 Agent-Tools/promptdiff.py latest-1 latest
python3 Agent-Tools/promptdiff.py 4b54c65e 2e2d6b5f --step 5
python3 Agent-Tools/promptdiff.py latest-1 latest --lane turn_bundles
python3 Agent-Tools/promptdiff.py latest --step 3 --step 10
```

Compares `prompt_state_frame_compiled` records between two runs or steps.
Catches stale-artifact and context-invalidation regressions.

### `model_output_lint.py` — scan model output for issues

```bash
python3 Agent-Tools/model_output_lint.py latest
python3 Agent-Tools/model_output_lint.py 4b54c65e --repeated-phrases
python3 Agent-Tools/model_output_lint.py latest --json
```

Counts degenerate loops, thinking-tag leakage, empty outputs, and repeated
assistant texts.

### `checkpoint_browser.py` — browse and diff checkpoints

```bash
python3 Agent-Tools/checkpoint_browser.py latest
python3 Agent-Tools/checkpoint_browser.py latest --diff 0 1
python3 Agent-Tools/checkpoint_browser.py latest --json
```

Lists checkpoint records (JSON files or harness records) and diffs two
snapshots.

### `session_replay.py` — replay tool-call sequence

```bash
python3 Agent-Tools/session_replay.py latest
python3 Agent-Tools/session_replay.py 4b54c65e --failures-only
python3 Agent-Tools/session_replay.py latest --policy-check
python3 Agent-Tools/session_replay.py latest --json
```

Pairs `dispatch_start`/`dispatch_complete` by trace_id, annotating each with
phase, run mode, and profiles. `--failures-only` filters to failed dispatches.

## Shared library

`agent_tools_lib.py` contains shared parsing helpers. Do not invoke it
directly; it is imported by the other scripts.

## Tool tracking rules

Every agent that uses these tools must do two things before finishing:

1. **If the tool exposed a bug or produced a misleading result, file it in
   `tools_bugtracker.md`.**
   - Add a new `BUG-NNN` entry with the tool, severity, description, repro
     command, and status.
   - This includes crashes, wrong classifications, missing records, stale
     cached indexes, bad file paths, or recommendations that do not match the
     data.

2. **If you notice a way the tool could be better, file it in
   `tools_improvements.md`.**
   - Add a new `IMP-NNN` entry with the tool, priority, proposed change,
     motivation, and status.
   - Even small ideas (a new flag, better column alignment, clearer error
     message) belong here so they are not lost.

These files are the long-term memory of the tooling. Do not rely on the next
agent re-discovering the same limitation. A short, concrete entry is enough.

## Agentic debug loop

When a SmallCTL run fails, follow this loop instead of opening files blindly:

1. **Orient** with `logwatch.py latest` or `runscan.py --last N`
   - Identify final status, failure classification, and top error/warning events.
2. **Diagnose** with `run_diagnose.py latest`
   - Get a failure-class label and recommended next steps.
3. **Trace** the failure
   - Use the trace_id from the diagnosis or `trace_call.py --last-error`.
   - Read the model output and tool result for that step.
   - Use `--compact` if the trace is dominated by token/chunk noise.
4. **Map to source** with `symbolmap.py`
   - Search for the failing event name, tool name, or function.
5. **Fix and verify**
   - Make a targeted change.
   - Run the relevant test or reproduce the task.
   - Use `rundiff.py` to compare the failing run with the new run.
   - Use `runscan.py --last N --failures-only` to confirm the failure pattern
     dropped across recent runs.
   - Optionally use `tui_screenshot.py` to capture the TUI state for visual
     verification or to share with a model.

After any tool use, return to the **Tool tracking rules** above and record a
bug or improvement if one was observed.

### Example session

```bash
# 1. See what went wrong in the latest run
python3 Agent-Tools/logwatch.py latest

# 2. Get a diagnosis
python3 Agent-Tools/run_diagnose.py latest

# Suppose it reports model_degeneration at trace 6cf0f870:task-0001:step-5:call-5
# 3. Trace that exact step (compact view)
python3 Agent-Tools/trace_call.py 6cf0f870:task-0001:step-5:call-5 --compact

# 4. Find where model-output degeneration is detected
python3 Agent-Tools/symbolmap.py --event model_output_degenerate_loop_exhausted

# 5. After fixing, compare runs and scan recent history
python3 Agent-Tools/rundiff.py latest-1 latest --events model_output_degenerate_loop_exhausted action_stall
python3 Agent-Tools/runscan.py --last 20 --failures-only
```

After step 1, if `logwatch.py` misclassified a FAMA signal as an error, you
would add a bug to `tools_bugtracker.md`. After step 3, if you wish
`trace_call.py` had a `--no-model-output` flag, you would add an improvement
to `tools_improvements.md`.

## Conventions

- Run directories live under `logs/` and are named `<uuidhex>-<YYYYMMDD-HHMMSS>`.
- All tools accept a run id prefix (e.g. `6cf0f870`) and resolve it to the
  matching directory.
- All tools accept `latest` (most recent run) and `latest-N` (N-th most recent
  run, zero-indexed). For example, `latest-1` is the second-most-recent run.
- Tools prefer the structured `.jsonl` log files and fall back to the text
  `.log` files only when JSONL is missing.
- When stdout is not a TTY, ANSI color codes are disabled automatically.
- Use `python3` for the standard-library tools. Use `.venv/bin/python` for
  `tui_screenshot.py` because it needs project dependencies.

## Adding a new tool

1. Place the script in `Agent-Tools/`.
2. Add `#!/usr/bin/env python3` as the first line and `chmod +x` it.
3. Use `agent_tools_lib.py` for run discovery, JSONL parsing, and record
   filtering so behavior stays consistent.
4. Keep dependencies standard-library only, unless the tool intentionally
   wraps SmallCTL/Textual like `tui_screenshot.py`.
5. Add a section to this file explaining the tool and when to use it.
6. Run `python3 -m py_compile Agent-Tools/your_tool.py` to verify syntax.
7. If the new tool exposes any bug or limitation during your testing, add it
   to `tools_bugtracker.md`. If you think of an improvement while building it,
   add it to `tools_improvements.md`.

## Tracking files

- `tools_bugtracker.md` — bugs and incorrect behavior in Agent-Tools.
- `tools_improvements.md` — feature requests and quality-of-life improvements.

Always keep these files up to date. They are the tooling equivalent of FAMA
reflexion memory: without them, the next agent will relearn the same problems.
