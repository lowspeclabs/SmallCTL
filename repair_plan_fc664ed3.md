# Repair Plan: Phantom Write-Session Recovery Hint — Session fc664ed3

## Problem Statement

During session `fc664ed3`, the model attempted to write `./temp/rogue-grid-defense.html` with `file_write(..., write_session_id="rogue-grid-s1")`, but no active write session with that ID existed. The harness routed the call into write-session handling and returned a recovery hint that encouraged retrying a session-gated write instead of removing the phantom `write_session_id`. Because missing-session recovery was not implemented, repeated retries failed with `missing_active_write_session` until the consecutive-error guard stopped the run. The task left a partial HTML file on disk.

The repair should make this failure self-healing for first content-bearing writes while preserving strict write-session semantics when a real session is active.

## Code-Grounded Findings

| Finding | Current Location | Implication |
|---|---|---|
| `file_write` normalizes `session_id` into `write_session_id` before session dispatch. | `src/smallctl/tools/fs.py:129-132` | Any guessed `session_id` alias can accidentally become a session-gated write. |
| Unknown `write_session_id` values are now detected before `handle_file_write_session`. | `src/smallctl/tools/fs.py:183-213` | This is the right choke point for falling back before producing `missing_active_write_session`. |
| Bare writes to a path owned by an active, non-complete session are blocked. | `src/smallctl/tools/fs.py:215-252` | The fallback must not bypass an active session for the same target path. |
| Direct writes still pass overwrite guards and risk-policy checks. | `src/smallctl/tools/fs.py:254-327` | Falling back to direct write does not skip existing safety gates. |
| The missing-session hint no longer promises auto-recovery. | `src/smallctl/tools/fs_write_flow.py:72-97` | The residual error path gives honest recovery instructions if reached. |
| Write-session events are stored in `state.scratchpad["_write_session_events"]`. | `src/smallctl/write_session_fsm.py:102-124` | The fallback can leave telemetry without changing public tool output shape. |
| Regression coverage already exists for the proposed behavior. | `tests/test_file_write_unknown_session_fallback.py:10-128` | Keep this file as the focused test suite for this bug. |

## Root Causes

| # | Root Cause | Location |
|---|---|---|
| RC-1 | A guessed `write_session_id` was treated as authoritative even when no active session existed. | `src/smallctl/tools/fs.py` |
| RC-2 | The missing-session recovery hint previously told the model to continue with a session-shaped write instead of omitting the invalid ID. | `src/smallctl/tools/fs_write_flow.py` |
| RC-3 | The hint promised behavior the harness did not implement: recovering a missing first write session from the failed payload. | `src/smallctl/tools/fs_write_flow.py` |
| RC-4 | Repeated identical recovery hints created a deterministic retry loop that tripped `max_consecutive_errors`. | Graph/tool error loop behavior, triggered by the tool metadata above |

## Refined Repair Design

### 1. Treat unknown session IDs as model noise for direct first writes

**File**: `src/smallctl/tools/fs.py`

When `file_write` receives a `write_session_id`, compare it to `state.write_session.write_session_id` before entering `handle_file_write_session`.

Expected behavior:

1. If the ID matches the active session, keep the existing staged/chunked write-session path.
2. If the ID does not match, record `unknown_write_session_id_fallback_to_direct_write` and clear `write_session_id` so the call continues through normal direct-write validation.
3. If an active session owns the same target path, the existing bare-write interceptor must still reject the write and instruct the model to use the active session ID.
4. Do not special-case the original `rogue-grid-s1` value; this is a general unknown-ID repair.

Rationale: A fabricated session ID on a first write is usually parser/model noise, not a real user intent to stage a chunked authoring session. Falling back lets the normal `file_write` safety stack decide whether the write is allowed.

### 2. Keep the residual missing-session error honest and actionable

**File**: `src/smallctl/tools/fs_write_flow.py`

If `handle_file_write_session` is reached without `state.write_session`, return `missing_active_write_session` with a recovery hint that reflects implemented behavior.

Required hint properties:

1. `next_required_tool.required_fields` includes only `path` and `content`.
2. `next_required_tool.required_arguments` includes `path` but not `write_session_id`.
3. Notes explicitly say to omit `write_session_id` for a direct write.
4. Notes do not claim that the harness can recover or recreate the missing session.

Rationale: This path should be rare after the `fs.py` fallback, but tests and direct callers still need a safe error contract.

### 3. Preserve legitimate write-session safety gates

**Files**: `src/smallctl/tools/fs.py`, `src/smallctl/tools/fs_write_flow.py`

Do not weaken these existing behaviors:

1. Matching active session IDs route to `handle_file_write_session`.
2. Writes to an active session-owned target without a valid session ID fail with `bare_write_to_session_owned_path`.
3. Terminal session overwrite repair remains limited to completed sessions with `replace_strategy="overwrite"`.
4. Direct writes still evaluate repair-cycle, patch-over-rewrite, and risk-policy guards.

Rationale: The bug is caused by phantom IDs, not by the staged-authoring FSM. The repair should not make it easier to bypass an active session.

## Implementation Status

The repair is implemented and verified in the working tree:

| File | Status |
|---|---|
| `src/smallctl/tools/fs.py` | Unknown-session fallback before session dispatch; records `unknown_write_session_id_fallback_to_direct_write`. |
| `src/smallctl/tools/fs_write_flow.py` | Corrected `missing_active_write_session` hint with no false auto-recovery claim. |
| `tests/test_file_write_unknown_session_fallback.py` | Covers unknown-ID fallback, matching-session routing, corrected residual hint metadata, and active-session/same-target rejection. |

Verification run:

```bash
python3 -m py_compile src/smallctl/tools/fs.py src/smallctl/tools/fs_write_flow.py
pytest tests/test_file_write_unknown_session_fallback.py -v
pytest tests/test_write_session_state_machine.py tests/test_write_recovery.py tests/test_write_recovery_regression.py -v
```

Result: **4/4 focused tests passed, 16/16 write-session/recovery tests passed.**

## Regression Tests

- `test_file_write_with_unknown_write_session_id_falls_back_to_direct_write`
- `test_file_write_with_matching_write_session_id_uses_session`
- `test_missing_active_write_session_recovery_hint_does_not_reuse_phantom_session`
- `test_unknown_write_session_id_with_active_same_target_session_is_rejected`

## Verification Steps

```bash
python3 -m py_compile src/smallctl/tools/fs.py src/smallctl/tools/fs_write_flow.py
pytest tests/test_file_write_unknown_session_fallback.py -v
pytest tests/test_write_session_state_machine.py tests/test_write_recovery.py tests/test_write_recovery_regression.py -v
```

All commands pass as of this plan revision.

Optional broad check for shared write/session behavior:

```bash
pytest tests/test_chunked_write_loop_guard.py tests/test_patch_existing_recovery_regression.py -v
pytest
```

## Success Criteria

1. Replaying the `fc664ed3` scenario makes the first content-bearing `file_write` continue as a direct write instead of repeatedly failing on `missing_active_write_session`.
2. A fabricated `write_session_id` is not echoed back as required recovery input.
3. Active write sessions remain protected from accidental bare overwrites.
4. Focused regression tests and existing write-session tests pass.
5. The event log contains enough telemetry to diagnose future unknown-session fallbacks.

All criteria are satisfied.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Fallback masks a real session lifecycle bug. | Record a structured event with provided and active session IDs; keep matching-session behavior unchanged. |
| Fallback bypasses an active session for the same target path. | Preserve the existing session-owned-path interceptor after fallback. Add/keep regression coverage for mismatched ID plus active same-target session. |
| Existing file overwrite guard blocks a legitimate direct retry. | Require `replace_strategy="overwrite"` for intentional full rewrites, as already enforced by direct-write flow. |
| Model continues inventing session IDs. | Corrected hint says to omit `write_session_id`; event telemetry can identify repeated offenders. |

## Rollback Plan

If fallback behavior causes staged-authoring regressions:

1. Revert only the unknown-ID fallback block in `src/smallctl/tools/fs.py`.
2. Keep the corrected `missing_active_write_session` hint in `src/smallctl/tools/fs_write_flow.py` because it removes the false recovery instruction and is independently safer.
3. Keep or adapt the residual-hint regression test so future hints do not reintroduce the phantom-session loop.

## Open Questions (Future Work)

These are not required to close this repair but are worth tracking:

1. Should unknown-session fallback be surfaced in user-visible output, or is scratchpad telemetry sufficient?
2. Should direct `session_id` alias normalization be restricted to known active sessions to reduce accidental session-shaped writes?
3. Should the graph-level consecutive-error recovery detect repeated `missing_active_write_session` and strip `write_session_id` automatically as a second line of defense?
