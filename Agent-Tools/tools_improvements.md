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
| IMP-001 | trace_call.py | medium | Include a short degenerate-stream sample or repeated phrase in compact traces when the backend halts output for repetition. | Agents can distinguish an actual blank/thinking disappearance from a repetition guard placeholder without opening raw model_output logs. | open |
| IMP-002 | logwatch.py, run_diagnose.py | high | Add a primary-blocker section that ranks environmental blockers such as connection refused or service not listening separately from model/harness symptoms. | RCA reports should keep run objective blockers separate from secondary harness/model failures. | open |
| IMP-003 | logwatch.py | medium | Surface whether the final `deliverable_verified=True` was backed by an actual code change or only a weak verifier (e.g. `--help` on an unchanged file). | Prevents false confidence in runs where verification passed but the mutation never happened. | open |
| IMP-004 | trace_call.py | medium | Show the original blocked `file_write` content preview when a write is rejected by policy, not just the error string. | Helps distinguish a blocked full rewrite from a narrow patch attempt. | open |
| IMP-005 | runscan.py, rundiff.py | low | Add a `--same-objective` filter to compare runs that targeted the same file/task, making success-factor comparisons easier. | Speeds up identifying which model/config/context differences turn failures into successes. | open |

## Completed Improvements

Move implemented improvements here and keep the original ID for reference.

| ID | Tool | Summary | Done Date |
|----|------|---------|-----------|

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
