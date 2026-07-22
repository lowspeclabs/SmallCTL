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
| BUG-031 | run_diagnose.py, logwatch.py, model_output_lint.py | major | A semantic diagnostic loop with 14 SSH probes, consecutive identical successful health checks, no mutation, and eventual user cancellation is classified as generic `recovery_failure`; model-output lint reports no loops because token text itself is not repetitive. The tools should detect repeated successful diagnostic commands and excessive read-only SSH activity without an intervening mutation. | `python3 Agent-Tools/run_diagnose.py 33020b24 --json`; `python3 Agent-Tools/model_output_lint.py 33020b24 --json`; `python3 Agent-Tools/session_replay.py 33020b24 --policy-check` | open |
| BUG-030 | run_diagnose.py, logwatch.py | major | A run with repeated semantically failed SSH shell commands masked by `|| true`, `2>&1 | head`, or semicolon chaining is reported with zero tool failures and classified as generic `recovery_failure`. The tools should surface masked command failures and identify the destructive `docker compose down` followed by an interrupted/piped `up` as the primary blocker. | `python3 Agent-Tools/run_diagnose.py 16c99e84 --json`; `python3 Agent-Tools/logwatch.py 16c99e84 --errors --warnings` | open |
| BUG-029 | run_diagnose.py, logwatch.py | major | A run where a successful SSH call is followed by `redacted_password_provided` is classified as generic `recovery_failure` and reports no primary blocker. The tools should identify SSH credential continuity/redaction failure by correlating the successful prior dispatch, the redacted-password error, and the same host/user. | `python3 Agent-Tools/run_diagnose.py 16c77a05 --json`; `python3 Agent-Tools/logwatch.py 16c77a05 --errors --warnings` | open |
| BUG-028 | run_diagnose.py | major | A run blocked because a required credential was scrubbed and `ask_human` was unavailable is classified as generic `model_tool_loop_stall`; the reported primary blocker is also rendered as `None [command: ...]`. Diagnosis should identify credential-handoff/tool-exposure failure and omit null blocker text. | `python3 Agent-Tools/run_diagnose.py 57c9949e --json` | open |
| BUG-027 | run_diagnose.py | major | A genuine SSH password-authentication rejection is classified as `ssh_transport_circuit_breaker_false_positive` whenever the FAMA SSH circuit-breaker pattern appears. The diagnosis should distinguish a real `Permission denied (publickey,password)` dispatch failure from an exit-255 remote-command false positive. | `python3 Agent-Tools/run_diagnose.py e6492f7c` | open |
| BUG-026 | trace_call.py | minor | Supplying a `step-N:call-M` suffix with `--run` resolves it against the first task in the run, even when the requested call belongs to `task-0002`; the command returns an empty trace as `45b24bdd:task:step-N:call-M`. The documented suffix form should locate the matching task trace or report that the suffix is ambiguous. | `python3 Agent-Tools/trace_call.py --run 45b24bdd step-12:call-12 --compact` | open |
| BUG-025 | logwatch.py, run_diagnose.py, session_replay.py, fama_inspector.py | major | A run directory containing records from multiple session-id prefixes is aggregated as one run without separating session epochs. This merges reused task IDs, produces conflicting dispatch totals (31 in replay vs 37 in diagnosis), and can attribute an earlier session's cancellation/FAMA state to the later session named by the directory. Tools should detect and partition by `session_id`, or require an explicit session selector when more than one is present. | `python3 Agent-Tools/run_diagnose.py 23791799 --json`; `python3 Agent-Tools/session_replay.py 23791799 --policy-check`; inspect session IDs `1e818cfa` and `23791799` in `logs/23791799-20260720-232426/harness.jsonl`. | open |
| BUG-017 | run_diagnose.py, runscan.py | major | Runs that fail with `PROMPT BUDGET OVERFLOW` are classified as generic `incomplete_unverified` instead of a prompt-budget failure class. `run_diagnose.py` repeats the postmortem but does not surface a concrete next-step label, and `runscan.py` does not include a `prompt_budget_overflow` classification even though the postmortem_summary clearly states it. | `python3 Agent-Tools/run_diagnose.py b2818de8` and `python3 Agent-Tools/runscan.py --last 20 --failures-only` (run `b2818de8-20260706-002838`). | open |
| BUG-022 | run_diagnose.py | major | Classifies a stale FAMA done-gate as generic `recovery_failure` even when a later successful verifier is recorded and completion tools remain hidden. It should identify the stale mitigation terminal-stall pattern. | `python3 Agent-Tools/run_diagnose.py 9a0507bf --json` | open |

## Fixed Bugs

Move resolved bugs here and keep the original ID for reference.

| ID | Tool | Fix Summary | Fixed Date |
|----|------|-------------|------------|
| BUG-023 | logwatch.py, run_diagnose.py | Added `harness_reported_blocker()` to agent_tools_lib.py; both tools now surface the harness-computed `primary_blocker` from session/task summaries as a non-environmental blocker when their own environmental scan finds none. Regression tests added. | 2026-07-17 |
| BUG-024 | run_diagnose.py, runscan.py | Added `cancelled_remote_verification_loop` classification before generic recovery failures, based on a cancelled/interrupted terminal state and failed `task_complete` dispatches blocked for remote verification. Regression tests added. | 2026-07-20 |
| BUG-021 | symbolmap.py | Added source fingerprint metadata to the AST cache and automatic rebuild when source file count/mtime or source directory changes. | 2026-07-14 |
| BUG-020 | run_diagnose.py, runscan.py | Added FAMA SSH transport circuit-breaker detection and classification before model degeneration. | 2026-07-14 |
| BUG-019 | run_diagnose.py, logwatch.py, session_replay.py, runscan.py | Added detection for background state-changing shell commands, surfaced them as primary blockers/classification, and annotated session replay warnings. | 2026-07-14 |
| BUG-018 | run_diagnose.py, logwatch.py | Removed broad bare `timeout` environmental blocker matching so CLI usage text like `--timeout TIMEOUT` is not treated as a timeout blocker. | 2026-07-14 |
| BUG-017 | run_diagnose.py, runscan.py, logwatch.py | Added prompt-budget overflow detection/classification and concrete next-step guidance. | 2026-07-14 |
| BUG-016 | run_diagnose.py, runscan.py | Added `model_stream_stall` classification for `reasoning_only_stream_exhausted`/`model_stream_halt_exhausted` events before generic incomplete/recovery labels. | 2026-06-29 |
| BUG-015 | tui_screenshot.py | Removed the explicit `pilot.exit(0)` call, added graceful cancellation of the active harness task before Textual tears the app down, and added a logging filter that suppresses the benign asyncio `Task.task_wakeup` RecursionError logged during shutdown. | 2026-06-25 |
| BUG-014 | model_output_lint.py | Skip the `missing_tool_calls` heuristic when the record (or matching tools-channel dispatch) shows native tool_calls. Added explicit detection for control-token fragments and reasoning-channel tool-call wrappers. | 2026-06-25 |
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
