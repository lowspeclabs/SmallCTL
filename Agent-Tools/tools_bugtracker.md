# Agent-Tools Bug Tracker

Track bugs discovered while using the SmallCTL Agent-Tools.

When you finish using any tool, if it produced an incorrect result, crashed,
misclassified a record, gave a misleading recommendation, or otherwise
behaved in a way that could trip up the next agent, add an entry here before
moving on.

## Bug Format

| ID | Tool | Severity | Description | Repro | Status |
|----|------|----------|-------------|-------|--------|

- **ID**: `BUG-NNN` (increment the highest existing number)
- **Tool**: which script (`logwatch.py`, `run_diagnose.py`, etc.)
- **Severity**: `critical` (wrong conclusion / crash), `major` (missing important data), `minor` (cosmetic / noisy)
- **Description**: what went wrong and what the correct behavior should be
- **Repro**: a concrete command and run id (or input) that demonstrates the bug
- **Status**: `open`, `fixed`, `wontfix`

## Bugs

| ID | Tool | Severity | Description | Repro | Status |
|----|------|----------|-------------|-------|--------|
| BUG-014 | model_output_lint.py | minor | Reports `missing_tool_calls` for assistant turns that use native function calling (`tool_calls` in the API response). The `assistant_text` contains explanatory prose while the actual tool calls live in separate `tool_calls` fields, so the heuristic that looks for inline `tool_name(` patterns is noisy for native-tooling runs. | `python3 Agent-Tools/model_output_lint.py e19cdf4c --json` shows 4 `missing_tool_calls` entries even though each turn emitted valid `tool_calls`. | open |
| BUG-015 | tui_screenshot.py | major | `pilot.exit(0)` triggers an `asyncio` `RecursionError` (`Task.task_wakeup` -> `child.cancel` repeated ~990 times). The screenshot and PNG are produced, but the exception indicates a shutdown/cancellation lifecycle bug in the UI/harness boundary that may also affect real TUI sessions. | `.venv/bin/python Agent-Tools/tui_screenshot.py --task "what is 2+2" --timeout 30 --json` or any task that completes. Observed on runs `55f2c1f0-20260625-172955` and `92eaaaf3-20260625-173043`. | open |

## Fixed Bugs

Move resolved bugs here and keep the original ID for reference.

| ID | Tool | Fix Summary | Fixed Date |
|----|------|-------------|------------|
| BUG-013 | run_diagnose.py, runscan.py | Moved `policy_block` and `fama_block` classification checks ahead of `model_tool_loop_stall` so done-gate / not-exposed tool blocks are surfaced before generic stall labels. Added regression tests. | 2026-06-22 |
| BUG-012 | model_output_lint.py | Added `thinking_text` to the thinking-tag leak scan so literal `<think>`/`<thinking>` tags inside recorded reasoning are surfaced. | 2026-06-21 |
| BUG-011 | run_diagnose.py | Detect `stderr_signature_circuit_breaker` trips and classify as `harness_circuit_breaker_false_positive` instead of `model_degeneration`. | 2026-06-19 |
| BUG-010 | trace_call.py | Warn when the resolved run directory does not match a full trace id prefix. | 2026-06-19 |
| BUG-009 | run_diagnose.py | Distinguish chat-mode terminal-only repetition loops from ask_human resume terminal stalls by focusing on the final task and terminal-only tool exposure. | 2026-06-19 |
| BUG-008 | run_diagnose.py | Classify repeated patch-first `file_write` blocks as `patch_first_policy_loop` before falling back to `model_degeneration`. | 2026-06-19 |
| BUG-005 | run_diagnose.py | Report environmental objective blockers (e.g. connection refused) as `environment_blocker` and surface them in a dedicated primary-blockers section. | 2026-06-19 |
| BUG-007 | trace_call.py | Expanded `step-N:call-M` suffixes even though they contain a colon. | 2026-06-19 |
| BUG-006 | run_diagnose.py, runscan.py | Classify completed sessions with no incomplete tasks as `success`/`success_with_errors` before unrecovered failure-mode labels. | 2026-06-19 |
| BUG-002 | run_diagnose.py, runscan.py | Added `write_session_overwrite_guard_loop` classification from failed dispatch payloads. | 2026-06-19 |
| BUG-003 | run_diagnose.py, runscan.py | Added `ask_human_resume_terminal_tool_stall` classification for affirmative interrupt resumes that fall into terminal-only tool exposure and stall. | 2026-06-19 |
| BUG-004 | runscan.py | Added the missing `chat_failure` label so emitted classifications are represented in the scanner label set. | 2026-06-19 |

## Notes

- Do not delete open bugs unless they are duplicates.
- If a bug is caused by a missing harness log field, note the field and
  consider whether the tool should degrade gracefully instead of failing.
- When fixing a bug, update the Status and add the fix to the Fixed Bugs
  table.
