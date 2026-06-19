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
| BUG-005 | run_diagnose.py | major | Diagnosis over-emphasized model degeneration for run ba7cdbd6 even though the run objective was already blocked by repeated localhost:3456 connection refusals. It should report primary objective blockers separately from secondary harness/model failures. | python3 Agent-Tools/run_diagnose.py ba7cdbd6 --json | open |
| BUG-006 | run_diagnose.py | major | Classifies a successful run as `model_degeneration` when the session completed and was verified but an earlier `model_output_degenerate_loop_exhausted` error exists. Success with recovered errors should be classified as `success` or `success_with_errors`, not a failure mode. | python3 Agent-Tools/run_diagnose.py 317775e8 --json | open |
| BUG-008 | run_diagnose.py | major | Classifies run 97e42939 as `model_degeneration` even though the task failed because `file_write` was blocked by patch-first policy and subsequent `file_patch` calls were malformed/no-ops. The primary failure mode is a `policy/write_guard` loop, not model degeneration. | python3 Agent-Tools/run_diagnose.py 97e42939 --json | open |

## Fixed Bugs

Move resolved bugs here and keep the original ID for reference.

| ID | Tool | Fix Summary | Fixed Date |
|----|------|-------------|------------|
| BUG-007 | trace_call.py | Expanded `step-N:call-M` suffixes even though they contain a colon. | 2026-06-19 |
| BUG-002 | run_diagnose.py, runscan.py | Added `write_session_overwrite_guard_loop` classification from failed dispatch payloads. | 2026-06-19 |
| BUG-003 | run_diagnose.py, runscan.py | Added `ask_human_resume_terminal_tool_stall` classification for affirmative interrupt resumes that fall into terminal-only tool exposure and stall. | 2026-06-19 |
| BUG-004 | runscan.py | Added the missing `chat_failure` label so emitted classifications are represented in the scanner label set. | 2026-06-19 |

## Notes

- Do not delete open bugs unless they are duplicates.
- If a bug is caused by a missing harness log field, note the field and
  consider whether the tool should degrade gracefully instead of failing.
- When fixing a bug, update the Status and add the fix to the Fixed Bugs
  table.
