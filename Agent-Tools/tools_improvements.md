# Agent-Tools Improvement Tracker

Track improvements, feature requests, and quality-of-life ideas for the
SmallCTL Agent-Tools.

When you finish using any tool, if you notice a way it could be more useful,
faster, easier to integrate into an agentic workflow, or more accurate, add
an entry here before moving on. Even small ideas belong here so they are not
lost.

## Improvement Format

| ID | Tool | Priority | Description | Motivation | Status |
|----|------|----------|-------------|------------|--------|

- **ID**: `IMP-NNN` (increment the highest existing number)
- **Tool**: which script, or `general` if it affects multiple tools
- **Priority**: `high` (big agent-time win), `medium` (nice to have), `low` (polish)
- **Description**: the proposed change
- **Motivation**: what agent workflow becomes better as a result
- **Status**: `open`, `done`, `rejected`

## Open Improvements

| ID | Tool | Priority | Description | Motivation | Status |
|----|------|----------|-------------|------------|--------|

## Completed Improvements

Move implemented improvements here and keep the original ID for reference.

| ID | Tool | Summary | Done Date |
|----|------|---------|-----------|
| IMP-005 | runscan.py, rundiff.py | Added `--same-objective TEXT` filter to restrict comparisons to runs whose objective contains the given text. | 2026-06-19 |
| IMP-004 | trace_call.py | Show a preview of blocked `file_write` content when a write dispatch fails. | 2026-06-19 |
| IMP-003 | logwatch.py | Surface weak/unchecked verification (no code changes, verified before last change, weak verifier command) when `deliverable_verified=True`. | 2026-06-19 |
| IMP-002 | logwatch.py, run_diagnose.py | Added a primary-blocker section that ranks environmental blockers (connection refused, service not listening, etc.) separately from model/harness symptoms. | 2026-06-19 |
| IMP-001 | trace_call.py | In compact mode, include the repeated phrase sample from `model_output_degenerate_loop_exhausted` records. | 2026-06-19 |

## Rejected Ideas

If an idea is considered and rejected, move it here with a short reason so it
is not re-proposed.

| ID | Tool | Reason | Date |
|----|------|--------|------|

## Notes

- Prefer narrow, actionable improvements over broad redesigns.
- If an improvement would require a new dependency, note it and consider
  whether the value is worth breaking the standard-library-only rule.
- Improvements that touch multiple tools usually belong in `agent_tools_lib.py`
  rather than being duplicated across scripts.
