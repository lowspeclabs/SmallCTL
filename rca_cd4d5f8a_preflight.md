# RCA: Session `cd4d5f8a` — Preflight Checks Hindering Interactive Remote Install

**Date:** 2026-05-22  
**Session:** `cd4d5f8a`  
**Model:** Qwen3.5-9b via OpenRouter/DeepInfra  
**Task:** Research FOG PXE server, SSH into `192.168.1.89`, clean up failed install, attempt fresh install  
**Status:** Interrupted/cancelled after 18 steps (no completion)

---

## 1. Executive Summary

**CONFIRMED: Preflight checks are hindering, not helping.**

Out of **23 sessions** that triggered `"Remote installer preflight required"`, **only 1** (`cd4d5f8a`) ever reached `"remote_installer_preflight_cleared"` — and it **still failed immediately afterward** because the preflight key mutated between attempts. The other **22 sessions** (96%) were permanently stuck with no resolution. Preflight checks consumed turns, triggered useless automatic web searches for internal error strings, and never actually enabled successful installer execution.

---

## 2. Timeline of Failure (Session `cd4d5f8a`)

| Trace | Action | Outcome |
|-------|--------|---------|
| 14 | `ssh_exec`: `git clone … fogproject.git` | ✅ Success |
| 15 | `ssh_exec`: `cd /root/fogproject && ./bin/installer.sh` | ❌ **Preflight blocked**. Required checks included `test -x ./bin/installfog.sh` (wrong script name — hardcoded regex mismatch) |
| 16 | `ssh_exec`: runs all required checks including `test -x ./bin/installfog.sh` | ✅ Checks pass; preflight marked `"clean"` for key `192.168.1.89|root|/root/fogproject|` (empty script path) |
| 17 | `ssh_exec`: `cd /root/fogproject && ./bin/installfog.sh` | ❌ **Preflight blocked AGAIN**. Key now includes script path (`/root/fogproject/bin/installfog.sh`), which is not in the clean list. |
| 17 | Automatic `web_search` triggered for repeated identical error | 🔄 Searches DuckDuckGo for an internal `smallctl` error message; results are irrelevant |
| 18 | Model concludes `"installfog.sh doesn't exist"` and tries to explore `bin/` | ❌ `dispatch_tools_error`; FAMA blocks `task_complete`; session interrupted |

**Total turns wasted on preflight loop:** 4+ (traces 15–18)  
**Total turns to actual install:** 0 (never reached)

---

## 3. Root Cause Analysis

### 3.1 Bug A: Script extraction regex is hardcoded to `installfog.sh`

**Location:** `src/smallctl/tools/shell_support.py` — `_remote_installer_cwd_and_script()`

```python
script_match = re.search(
    r"(?:^|\s)(?:bash|sh)?\s*(\.?/[\w./-]*installfog\.sh|[\w./-]*installfog\.sh)\b",
    raw
)
```

This regex **only** matches `installfog.sh`. When the model sent `./bin/installer.sh` (a legitimate install script), the regex failed to match. The function returned `script_path=""`, causing:
- The preflight key to omit the script component: `host|user|cwd|`
- The fallback check to target `./bin/installfog.sh` instead of `./bin/installer.sh`
- A **false preflight requirement** for a file the model never intended to run

**Irony:** `_looks_like_remote_installer_mutation()` correctly detects generic install scripts (`*.sh` containing `"install"`), but `_remote_installer_cwd_and_script()` ignores that detection and uses a hardcoded regex.

### 3.2 Bug B: Preflight key is unstable

**Location:** `src/smallctl/tools/shell_support.py` — `_remote_installer_preflight_guard()`

The preflight key is built as:
```python
key = "|".join([host, user, cwd, script_path])
```

- **Trace 15:** `script_path=""` → key = `192.168.1.89|root|/root/fogproject|`
- **Trace 17:** `script_path="./bin/installfog.sh"` (normalized to `/root/fogproject/bin/installfog.sh`) → key = `192.168.1.89|root|/root/fogproject|/root/fogproject/bin/installfog.sh`

These are **different keys**. The "clean" status from trace 16 does not apply to trace 17, so the model is blocked again.

### 3.3 Bug C: Preflight clearing logic matches substring against combined commands

**Location:** `src/smallctl/harness/tool_result_artifact_updates.py` — `_observe_remote_installer_preflight()`

```python
if command == check or check in command:
    completed.append(check)
```

The model ran a **single combined command** (`pwd && cd … && test -x ./bin/installfog.sh`). All four checks were substring-matched against this one command. This is fragile:
- It doesn't verify each check **individually** succeeded with exit code 0.
- It allowed `test -x ./bin/installfog.sh` to be marked "completed" even though the model's *actual* target was `./bin/installer.sh`.

### 3.4 Bug D: Automatic web search fires on internal error strings

**Location:** Harness repeated-error handler

When the model hit the same preflight error twice, the harness scheduled an automatic `web_search` for:
> `"error: Remote installer preflight required before running this high-risk installer mutation…"`

This is an **internal `smallctl` error message**. DuckDuckGo returned generic Windows installer documentation. The search consumed a turn, added noise to context, and provided zero actionable information.

### 3.5 Design Issue: `ssh_exec` is the wrong tool for interactive installers

The FOG installer (`installfog.sh`) is **interactive** — it prompts for MySQL passwords, interface selection, etc. The model attempted to use `ssh_exec`, which:
- Has no PTY by default
- Cannot respond to prompts
- Is guarded by preflight, yes-pipe blocks, and foreground-command blocks

The harness provides `ssh_session_start` / `ssh_session_send` / `ssh_session_read` for interactive sessions, but **the model never used them**. The error metadata includes `suggested_tool_after_preflight: "ssh_session_start"`, but this is buried in JSON metadata and never surfaced in the plain-text error the model reads.

---

## 4. Cross-Session Validation

```
Sessions with "Remote installer preflight required":     23
Sessions with "remote_installer_preflight_cleared":       1  (cd4d5f8a)
Sessions that succeeded after clearing:                   0
Sessions permanently stuck:                              22  (96%)
```

**Other affected sessions:**
- `f86007f2` — FOG install, preflight required, never cleared, auto web search fired
- `8ca5eb94` — FOG install, preflight required, never cleared

**Pattern:** Every FOG installation attempt hits preflight. None succeed.

---

## 5. Proposed Fixes (No Code Changes Until Approved)

### 5.1 Critical Fix: Unify script detection (1–2 lines)

In `_remote_installer_cwd_and_script()`, replace the hardcoded `installfog.sh` regex with the **same logic** already proven in `_looks_like_remote_installer_mutation()`:

```python
# Instead of hardcoded regex, walk command segments and extract
# the script path using the same heuristic as the mutation detector.
```

This ensures the preflight checks target the **actual script the model asked to run**.

### 5.2 Critical Fix: Stabilize the preflight key

Option A — **Remove script_path from the key** (recommended):
```python
key = "|".join([host, user, cwd])
```
The preflight is about verifying the *environment* (cwd/repo integrity), not the specific script file. If `test -x <script>` is part of the checks, it should use the script from the **current command**, not a cached one.

Option B — Normalize script_path to basename:
```python
key = "|".join([host, user, cwd, Path(script_path).name])
```

### 5.3 Fix: Verify checks individually, not via substring matching

Change `_observe_remote_installer_preflight()` to require **individual `ssh_exec` calls** for each check (or at minimum, parse the combined command to verify each sub-command exited 0). Current substring matching is trivially bypassed.

### 5.4 Fix: Suppress auto web search for internal error signatures

Add the preflight error signature to a **blocklist** for `repeated_error_web_search_scheduled`. Internal guard errors should never be searched.

### 5.5 Improvement: Surface `ssh_session_start` in plain text

When preflight (or yes-pipe/foreground) guards block an `ssh_exec` installer attempt, the error message should explicitly say:

> "This looks like an interactive installer. Use `ssh_session_start` instead of `ssh_exec` so you can send answers to prompts with `ssh_session_send`."

### 5.6 Improvement: Provide installer-specific guidance

For known interactive installers (FOG, etc.), inject a **memory fact** or **system hint**:
- FOG supports `--autoaccept` or a `.fogsettings` preseed file
- Example: `printf 'Y\nY\nY\n' | ssh_session_send …`

### 5.7 Alternative: Replace preflight with approval + verification

**Recommendation: Remove the preflight gate entirely** and replace with:

1. **Pre-execution:** A single user approval for high-risk remote mutations (already exists via `evaluate_risk_policy`).
2. **Post-execution:** Automatic verification that the installer exited cleanly, with rollback guidance if it failed.

Rationale:
- Preflight checks guess what the model *should* have done. They are brittle.
- Risk policy + approval already handles the "are you sure?" safety need.
- Post- verification is more valuable because it catches actual failures, not hypothetical ones.

---

## 6. Recommendations Summary

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P0 | Fix regex in `_remote_installer_cwd_and_script` to match generic install scripts | 15 min | High |
| P0 | Stabilize preflight key (remove script_path) | 15 min | High |
| P1 | Block auto-web-search for internal preflight signatures | 30 min | Medium |
| P1 | Surface `ssh_session_start` suggestion in error text | 30 min | High |
| P2 | Refactor preflight clearing to verify individual commands | 2 hr | Medium |
| P2 | Add FOG/non-interactive install hints to memory/context | 1 hr | Medium |
| P3 | **Remove preflight gate**; rely on risk approval + post-verification | 4 hr | High (long-term) |

---

## 7. Conclusion

**Preflight checks hurt model performance in this domain.** They are buggy (hardcoded regex, unstable key), confusing (auto web search for internal errors), and redundant (risk approval already exists). The model's actual failure mode was not "running an installer in a bad directory" — it was **being unable to run the installer at all** because the guard created an unresolvable loop.

The fastest path to value is:
1. Fix the regex and key stability (P0).
2. Remove or replace preflight with post-verification (P3).
3. Guide models toward `ssh_session_start` for interactive remote work.

---

*Report generated from analysis of session logs and source code. No code changes were made.*
