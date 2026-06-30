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
| IMP-012 | trace_call.py | medium | When a partial trace id (e.g. `step-1:call-1` or `call-1`) matches no records, print a warning that includes the run's trace-id prefix and the expected `<session>:<task>:step-N:call-M` format. | The docs show `trace_call.py --run latest step-1:call-1`, but trace records are keyed with a task id (e.g. `task-0002`), so the partial form silently returns an empty trace. A warning would save agents from chasing non-existent records. | open |

## Completed Improvements

Move implemented improvements here and keep the original ID for reference.

| ID | Tool | Summary | Done Date |
|----|------|---------|-----------|
| IMP-011 | trace_call.py | In `--compact` mode, consecutive `ui_event` records are collapsed into a single summary line that preserves per-kind counts and first/last timestamps. Collapsed lines use a wider max width so the summary is not truncated. | 2026-06-28 |
| IMP-010 | session_replay.py, trace_call.py | `session_replay.py` now pairs multiple dispatches per trace_id and prints a `call-N:dispatch-M` sub-trace id plus `ui_event` kinds. `trace_call.py` renders all dispatches for a trace with their own tool/args/output blocks and ui_event kind summary. | 2026-06-25 |
| IMP-009 | model_output_lint.py, run_diagnose.py | Detect control-token fragments and reasoning-channel tool-call wrappers without matching dispatches. Added `tool_call_protocol_mismatch` classification in `run_diagnose.py`/`runscan.py` and a dedicated lint category. | 2026-06-25 |
| IMP-008 | run_diagnose.py, runscan.py | Added `continue_prompt_budget_loop` classification when user "continue"/"proceed" messages after a `task_fail` are followed by repeated `PROMPT BUDGET OVERFLOW` errors. | 2026-06-25 |
| IMP-007 | model_output_lint.py | Accepted `latest-N` run references via the shared `agent_tools_lib.resolve_run_dir` resolver; help text already listed `latest-N`. | 2026-06-20 |
| IMP-006 | run_diagnose.py, runscan.py | Added `detect_apt_deb822_guard_misfire` helper and `guard_misfire` classification; surfaced misfires in diagnosis narrative, JSON, and recommended next steps. | 2026-06-20 |
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
