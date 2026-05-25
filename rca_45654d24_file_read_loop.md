# RCA: Session `45654d24` — Model Ignored Harness Warnings in File-Read Loop Until Human Intervention

**Date:** 2026-05-23  
**Session:** `45654d24`  
**Model:** qwen3.5-9b / qwen36-9b-mtp via llamacpp  
**Task:** SSH to `192.168.1.89`, inspect DNS resolver config, write findings to `./temp/dns_audit.txt`, read file and post summary  
**Status:** Completed after human resteer (initial task aborted due to loop)

---

## 1. Executive Summary

The model entered a **6-turn cyclic loop** of `file_read` failures on a non-existent local file (`./temp/dns_audit.txt`). Despite two harness interventions — a **recovery nudge** (trace 12) and a **repeated-tool-loop interrupt** (trace 13) — the model persisted in calling `file_read` with identical arguments. The human-interrupt mechanism failed immediately (0.014 s), forcing a manual human resteer. After resteer, the model completed the task in 3 turns.

**Root cause**: The model suffered from a **local/remote file tool confusion** and a **say/do mismatch** (text claimed `file_write`, actual tool call was `file_read`). The harness's cyclic-loop guard detected the pattern but could not self-recover because the model ignored the injected warning.

---

## 2. Detailed Timeline

| Time (UTC) | Trace | Model Text Claim | Actual Tool Call | Result | Harness Action |
|------------|-------|------------------|------------------|--------|----------------|
| 15:04:41 | 1-3 | SSH inspection steps | `ssh_exec` ×3 | ✅ Success | — |
| 15:05:48 | 4 | "Let me read the file" | `file_read(./temp/dns_audit.txt)` | ❌ File not found | Notepad fail entry |
| 15:06:03 | 5 | "Write findings to file" | `ssh_file_write(/home/stephen/…/temp/dns_audit.txt)` | ✅ Written to **remote** host | None |
| 15:06:14 | 6 | "Read the file" | `file_read(./temp/dns_audit.txt)` | ❌ File not found | None |
| 15:06:31 | 7 | "Check if temp exists" | `dir_list(./temp)` | ✅ Listed | None |
| 15:06:44 | 8 | "Read the file" | `file_read(./temp/dns_audit.txt)` | ❌ File not found | None |
| 15:06:59 | 9 | "Read the file" | `file_read(./temp/dns_audit.txt)` | ❌ File not found | None |
| 15:07:12 | 10 | "Read the file" | `file_read(./temp/dns_audit.txt)` | ❌ File not found | None |
| 15:07:25 | 11 | "Read the file" | `file_read(./temp/dns_audit.txt)` | ❌ File not found | None |
| 15:07:38 | 12 | **"Now I'll write … using `file_write`"** | **`file_read(./temp/dns_audit.txt)`** | ⛔ Blocked by guard | `file_read_recovery_nudge` injected |
| 15:07:51 | 13 | **"Now I'll write … using `file_write`"** | **`file_read(./temp/dns_audit.txt)`** | ⛔ Blocked by guard | `repeated_tool_loop_interrupt` + `interrupt_for_human_start` |
| 15:07:51 | 13 | — | — | — | `interrupt_for_human_error` (0.014 s) |
| 15:08:09 | — | — | — | — | `recovery_human_resteer_recorded` |
| 15:08:37 | 1 (reset) | "Writing file now" | `file_write(/home/stephen/…/temp/dns_audit.txt)` | ✅ Success | — |
| 15:08:46 | 2 (reset) | "Reading file" | `file_read(/home/stephen/…/temp/dns_audit.txt)` | ✅ Success | — |
| 15:08:55 | 3 (reset) | "Task complete" | `task_complete` | ✅ Success | — |

---

## 3. Root Cause Analysis

### 3.1 Primary: Local/Remote File Tool Confusion (Trace 5)
The model used `ssh_file_write` with an **absolute local path** (`/home/stephen/Scripts/Harness-Redo/temp/dns_audit.txt`). This wrote the audit content to the **remote host's** filesystem at that path, not the local machine. The model then repeatedly tried to `file_read` the **local** path `./temp/dns_audit.txt`, which did not exist because the file was never written locally.

### 3.2 Secondary: Say/Do Mismatch (Traces 12-13)
At traces 12 and 13, the model's **visible text output** explicitly stated it would use `file_write`:
> "Now I'll write the detailed findings to the LOCAL ./temp/dns_audit.txt file using `file_write`:"

However, the **parsed tool call** was still `file_read` with the same failing path. The harness correctly detected the mismatch and blocked the call, but the model never actually emitted a `file_write` tool call until after human resteer.

### 3.3 Tertiary: Recovery Nudge Ignored
The harness injected a strong guard message at trace 12:
> "Guard tripped: cyclic multi-tool loop … Choose exactly one: A. Explain the blocker and stop. B. Try a different specific fix. C. Ask for missing information."

The model's next turn (trace 13) still attempted the exact same `file_read` call, demonstrating that the guard text was either:
- Not prominent enough in the prompt context (buried under retrieval knowledge and observations)
- Not interpreted by the model as a hard stop
- Overridden by the model's stronger internal plan to "read the file"

### 3.4 Quaternary: Human Interrupt Mechanism Failed
The `interrupt_for_human` graph node failed after only **0.014 seconds**, meaning the session could not cleanly pause for human input. Instead, the task had to be manually resteered from outside the harness graph.

---

## 4. What Else Went Wrong

1. **Missing directory auto-creation**: The `./temp` directory did not exist locally. Neither `file_read` nor the harness auto-created it or suggested `mkdir`.
2. **No tool-call vs text-output consistency check**: The harness blocks repeated tool loops, but does not validate that the model's natural-language plan matches the actual JSON tool call being emitted.
3. **Interrupt node fragility**: The 0.014-second failure suggests a runtime exception in the interrupt node (possibly missing UI handle, timeout socket error, or graph state inconsistency).
4. **Task summary inaccuracy**: `session_summary.json` reports `final_task_status: "completed"` and `guard_trips: 0`, even though the original task was **aborted** and required human intervention. The summary reflects the reset task, masking the failure.

---

## 5. Proposed Fixes (Awaiting Approval)

| Priority | Component | Fix Description |
|----------|-----------|-----------------|
| **P0** | Tool-call validator | Add a **say/do guard** that compares the model's text plan against the parsed tool call. If the text mentions tool X but the call is tool Y (especially after a recovery nudge), block and restate the instruction. |
| **P0** | Recovery nudge | Harden the recovery nudge so it is **injected as a system-level message** (or top of prompt) rather than being mixed into normalized observations. Ensure it uses imperative language: **"STOP. Do not call file_read again."** |
| **P1** | `file_write` / `ssh_file_write` docs | Clarify in tool descriptions that `file_write` is **local**, `ssh_file_write` is **remote**, and `ssh_file_write` should use **remote-relative paths** unless the remote filesystem matches the local one. |
| **P1** | Missing directory handling | When `file_read` fails with "File does not exist" and the parent directory also does not exist, append a hint: `"Directory ./temp does not exist. Use file_write to create the file (directories are auto-created), or use dir_list to verify the path."` |
| **P1** | Interrupt node | Debug the `interrupt_for_human` node 0.014-second failure. Add exception handling and fallback to a synchronous `ask_human` tool injection if the interrupt channel fails. |
| **P2** | Session summary | Update post-run summary logic to record `guard_trips > 0` and `final_task_status: "human_resteer_required"` when `recovery_human_resteer_recorded` or `repeated_tool_loop_interrupt` events occur. |

---

## 6. Confirm/Deny: Did the Harness Help or Hinder?

**Mixed verdict.**
- **Helped**: The cyclic-loop guard correctly identified the repeating failure pattern and prevented additional wasted turns.
- **Hurt**: The recovery nudge was ignored, the interrupt node crashed, and the model was not given a clear forced-choice alternative. The harness delayed completion without providing an actionable escape path.

**Recommendation**: The guard logic is sound, but the **intervention delivery mechanism** needs to be more forceful (hard stop + forced tool substitution) rather than a textual suggestion.
