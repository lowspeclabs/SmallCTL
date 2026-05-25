# RCA: Three Consecutive TUI Sessions (8bdb0d26, 3c840685, dcab9db1)

**Date:** 2026-05-23  
**Command:** `python3 -m smallctl.main --provider-profile llamacpp --tui --endpoint http://192.168.1.9:8080 --model "qwen3.5-9b" --max-prompt-tokens 64976`  
**Outcome:** All three sessions required manual `Ctrl+C` termination. Two tasks completed; one failed after multiple internal errors.

---

## 1. Executive Summary

| Session | Task | Result | Why Ctrl+C was needed |
|---------|------|--------|----------------------|
| **8bdb0d26** | SSH cleanup of `/tmp` & `/var/tmp` | ✅ Completed | TUI remained open after `task_complete` |
| **3c840685** | Package-change impact RCA | ✅ Completed (wrong path) | TUI remained open after `task_complete` |
| **dcab9db1** | "save to local host" (follow-up) | ❌ Failed | TUI remained open after `task_fail` |

The root cause of the *user-visible symptom* (needing Ctrl+C) is that **TUI mode without `--task` is a persistent chat session**. However, beneath that surface are **four concrete bugs/failure modes** that degraded task quality and caused the final session to fail.

---

## 2. Failure Mode A: `escalate_to_bigger_model` Is Structurally Broken

### Evidence
```
escalate_to_bigger_model failed: Unknown escalation verdict: next_action
```
This exact error appears **12 times** in session `dcab9db1` (notepad count at `15:58:20.852`).

### Root Cause
**Schema / validator mismatch.**

- `tools/register_escalation.py` (line 38) advertises `"next_action"` as a valid `requested_output` enum value.
- `harness/escalation_response.py` (lines 12-20) defines `ALLOWED_VERDICTS` **without** `"next_action"`:
  ```python
  ALLOWED_VERDICTS = {
      "continue",
      "need_more_evidence",
      "reject_current_plan",
      "propose_patch",
      "final_answer_ok",
      "ask_human",
      "abort",
  }
  ```

When the model invokes `escalate_to_bigger_model(requested_output="next_action")`, the escalation service calls the bigger model, which obediently returns `"verdict": "next_action"`. The validator then rejects it, returning a tool failure that pollutes the context window and wastes a model call (~59 sec in session 3).

### Impact
- Wastes model calls and latency.
- Pollutes the failure / reflexion memory with false positives.
- In session 3, this failure became one of three blockers that forced the subtask into `blocked` status.

---

## 3. Failure Mode B: Path Confusion — `./temp/` vs `/tmp/`

### Evidence
- **Session 3c840685**: User asked for `./temp/package_change_impact_rca.txt`. Model wrote to `/tmp/package_change_impact_rca.txt` (remote host) and later tried to read it back locally, causing `file_read failed: File does not exist`.
- **Session dcab9db1 task-0001**: User asked for `./temp/service_config_validation.txt`. Model saved it to the **remote** host path instead of local workspace. User follow-up "file isn't on disk" triggered task-0002, which still failed to place the file locally.

### Root Cause
The model conflates:
1. **Local workspace-relative paths** (`./temp/...`) intended for the orchestrator filesystem.
2. **Remote absolute paths** (`/tmp/...`) on the SSH target.
3. The prompt does not contain a strong, unambiguous directive that "write the RCA to `./temp/...`" means **local** file write, not remote.

When the task narrative mixes SSH operations with local file saves, the model defaults to writing evidence on the remote host (where it has just been working) and then cannot retrieve it locally.

### Impact
- Deliverables land in the wrong filesystem.
- Wastes user follow-up turns (task-0002 and task-0003 in session 3 were entirely consumed trying to recover a misplaced file).

---

## 4. Failure Mode C: FAMA Tool-Exposure Narrowing Hides Required Tools

### Evidence
Session `dcab9db1`, step 6:
```json
{"hidden_tools": ["ssh_session_close", "ssh_session_read", "ssh_session_send", "ssh_session_start"], "mode": "loop"}
```

Immediately after, the model attempts to copy a remote file locally and discovers:
> "SCP command is blocked (SSH/scp/sftp tools not available)"

### Root Cause
FAMA's `tool_exposure_narrowing` mitigation activates when `loop_guard` detects `no_actionable_progress=3`. It hides SSH session tools to prevent looping. However, in this case the *next* required action was exactly an SSH file transfer (`scp` / `ssh_file_read`). The mitigation counter-productively removed the only tools capable of solving the problem.

### Impact
- Model is forced to call `task_fail` due to "tool limitations" that were artificially imposed by its own safety system.
- Task fails unnecessarily.

---

## 5. Failure Mode D: TUI Does Not Auto-Exit After Single Task

### Evidence
All three sessions end with `task_finalize` (completed or failed) and then **no further log events** until the terminal emits the Ctrl+C shutdown JSON.

### Root Cause
When `--tui` is used **without** `--task`, `SmallctlApp` runs as an interactive chat. After `_run_harness_task` finishes, it simply renders the result and clears the activity spinner. There is no "single-shot" mode that exits the TUI after the user-typed task finishes.

### Impact
- User must manually `Ctrl+C` to return to shell.
- In a headless / scripted mental model, this feels like a hang even though the app is idle and responsive.

---

## 6. Proposed Fixes & Improvements

### 6.1 Fix `escalate_to_bigger_model` schema/validator mismatch
**File:** `src/smallctl/harness/escalation_response.py`
**Change:** Add `"next_action"` to `ALLOWED_VERDICTS` (or remove it from the tool schema if it is intentionally unsupported).

```python
ALLOWED_VERDICTS = {
    "continue",
    "need_more_evidence",
    "reject_current_plan",
    "propose_patch",
    "final_answer_ok",
    "ask_human",
    "abort",
    "next_action",   # <-- add this
}
```
**Rationale:** The tool advertises it; the validator must accept it. If the intent is that `"next_action"` should map to `"continue"`, then the escalation prompt or response parser should normalize it, not fail the tool call.

### 6.2 Add path-disambiguation guidance for mixed local/remote tasks
**File:** `src/smallctl/prompts.py` (or wherever task preamble is built)
**Change:** When the task contains both SSH targets and local relative paths (`./temp/...`), inject a system reminder:
> "Paths starting with `./` refer to the **local orchestrator filesystem**. Paths on the remote host must use absolute paths (`/tmp/...`) or be explicitly prefixed with the remote host."

**Rationale:** Prevents the model from defaulting to remote `/tmp/` when the user explicitly asked for local `./temp/`.

### 6.3 Make FAMA `tool_exposure_narrowing` aware of task intent
**File:** `src/smallctl/fama/detectors.py` or `src/smallctl/fama/migrations.py`
**Change:** Before hiding `network*` / `ssh*` tools, check whether the current `active_intent` (e.g. `requested_ssh_exec`, `requested_ssh_file_read`) or the raw task text explicitly requires network operations. If so, skip tool-exposure narrowing for that category.

**Rationale:** Hiding the very tools needed to satisfy the active intent guarantees failure.

### 6.4 Add `--single-task` / auto-exit flag for TUI
**File:** `src/smallctl/main.py` + `src/smallctl/ui/app_flow.py`
**Change:**
1. Add CLI flag `--single-task` (or reuse `--task` semantics).
2. In `_run_harness_task`, after `result` is returned and rendered, if `single_task_mode` is enabled, call `self.exit()` to close the TUI gracefully.

**Rationale:** Users who type one task and expect the app to return to the shell should not need to `Ctrl+C`.

### 6.5 Improve subtask blocker messaging
**File:** `src/smallctl/harness/subtask_ledger_service.py`
**Change:** When a subtask transitions to `blocked`, surface the **actionable** blocker first. In session 3, the blocker list was:
1. `verifier_failed: task_complete rejected ... scp ...`
2. `verifier_failed: file_read failed ...`
3. `verifier_failed: escalate_to_bigger_model failed ...`

The escalation failure (#3) was a red herring caused by the validator bug. Better ordering or deduplication would help the model focus on the real issue (missing local file).

### 6.6 Add TUI idle-timeout or "Press Enter to exit" prompt
**File:** `src/smallctl/ui/app_flow.py`
**Change:** After a task completes in TUI mode, if the input pane remains empty for >N seconds, print a status hint:
> "Task finished. Type a new message or press Ctrl+C to exit."

**Rationale:** Reduces the perception that the app is hung.

---

## 7. Metrics

| Metric | Session 1 | Session 2 | Session 3 |
|--------|-----------|-----------|-----------|
| Duration | ~5 min | ~6 min | ~12 min |
| Model calls | 10 steps | 9 steps | 6 steps (task 3) |
| `escalate_to_bigger_model` failures | 0 | 0 | 12 |
| FAMA loop signals | 0 | 0 | 3 (escalated) |
| Tool dispatch max latency | ~1.8s | ~0.3s | **59s** (escalation) |
| Final status | completed | completed | **failed** |

---

## 8. Conclusion

The Ctrl+C terminations are **symptoms**, not root causes. The underlying issues are:

1. **A real code bug** (`ALLOWED_VERDICTS` missing `"next_action"`) that breaks the escalation tool.
2. **A prompt/context bug** that lets the model confuse local and remote paths.
3. **A FAMA policy bug** that hides tools required by the active intent.
4. **A UX gap** where TUI mode has no graceful single-task exit path.

Fixing #1 is the highest-impact, lowest-effort change. Fixing #2 and #3 will prevent the cascading failures seen in session `dcab9db1`. Fixing #4 will eliminate the user's primary complaint (having to Ctrl+C).
