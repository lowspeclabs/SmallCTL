# RCA: Session `a707ce3f` — Chunked Write Collapse on `network_allocator.py`

**Date:** 2026-05-22  
**Model:** Qwen3.5-9b via OpenRouter (DeepInfra / SiliconFlow)  
**Task:** Build `./temp/network_allocator.py` (self-contained IP allocator with unittests)  
**Session mode:** `loop` / `local_execute` / `execute` phase  
**Active profiles:** `core`, `network`, `network_read`  
**Write session:** `ws_a22af5` (intent: `replace_file`)

---

## Executive Summary

The task failed with a **LoopGuard hard-abort** after four successive blocked writes to section `types/interfaces`. The root cause is not a single bug, but a **cascade of four interacting failures**:

1. **Tool-call stream stall** (`empty_payload`) triggered a no-tools fallback path.
2. **Model response leakage** into `assistant.content` — Qwen3.5-9b emitted malformed JSON (`\`\`\`json\n{"name":`) instead of a proper `tool_calls` block.
3. **Fallback parser mis-classified the leakage** as a synthetic `file_write`, creating a zero/empty-content write attempt.
4. **LoopGuard correctly blocked the synthetic write** (section already checkpointed), but because no staged-file state was injected into the prompt after outline confirmation, the model had no legal path forward and the escalation threshold was reached.

Additionally, the **plan/approval mechanism was structurally bypassed**: the model proposed an outline via `ask_human`, but `draft_plan` was never populated, so the user’s `"continue"` reply was handled as a generic LoopGuard resume rather than a plan step advance.

---

## 1. Failure Sequence (Chronological)

| Step | Artifact | Event | Consequence |
|------|----------|-------|-------------|
| A0001 | `memory_update` | Session opens; model records intent to continue `ws_a22af5`. | Working memory seeded. |
| A0002 | `file_write` | `imports` section written successfully. | Staged file created with 4 lines. Next section set to `types/interfaces`. |
| A0003 | `file_write` | Model rewrites `imports` verbatim. | **LoopGuard blocks** (`repeated_tool_loop`). Model told: "read before rewrite." |
| A0004 | `file_write` | Model attempts full-file append of `types/interfaces`. | **LoopGuard blocks again** (`read_required_retry`). Stagnation score rises to 2. |
| A0005 | `read_log` | Model calls `read_log(cwd=...)` with invalid `cwd` arg. | Tool error: unexpected keyword argument. No log read. |
| A0006 | `ask_human` | Model proposes 5-section outline via `ask_human`. | Harness enters `chunked_write_loop_guard_outline` interrupt. `draft_plan` remains `null`. |
| — | User reply | `"continue"` | Harness clears outline mode, resets step count, resumes loop. |
| A0007 | `file_write` | **Tool-call stream stalls** (`empty_payload`). Model emits `\`\`\`json\n{"name":` into `content`. | Fallback parser synthesizes a `file_write` with empty/malformed content. LoopGuard blocks (`read_required_retry`). Stagnation = 3 → **hard-abort**. |

---

## 2. Detailed Failure Modes

### 2.1 Model Response Leakage into Chat (`content`)

**Evidence (chat.log, chunk index 48→44 of final stream):**

```
... delta {"content": "```", ...}
... delta {"content": "json", ...}
... delta {"content": "\n", ...}
... delta {"content": "{\"", ...}
... delta {"content": "name", ...}
... delta {"content": "\":", ...}
```

After a stream stall where `tool_calls` arguments were partially received (`{"path": "...", "write_session_id": "..."}`) but **no `content` key was provided**, the model fallback path emitted raw JSON into the assistant `content` field. The harness’s `stream_text_write_fallback` detected fenced code and attempted recovery.

**Why it happened:**
- The `payload_preflight_repaired` log shows **all tools were stripped from the request** because previous turns contained unavailable tools (`memory_update`, `file_write`, `artifact_read`, `read_log`, `ask_human`).
- With **zero tools in the schema**, the model could not emit a proper `tool_calls` block.
- The model instead "reasoned out loud" (reasoning field) then emitted a fenced code block containing JSON-like text.
- The fallback parser saw `\`\`\`json` + partial object and classified it as `assistant_unclosed_fenced_code`, then synthesized a `file_write` call.

**Impact:** The synthesized write had no valid `content` payload, so it was treated as an empty write to `types/interfaces`. LoopGuard blocked it because the section had not been read since the last checkpoint.

---

### 2.2 Plan Proposal Never Loaded into `draft_plan`

**Evidence (chat state `a707ce3f.json` at A0006):**

```json
"active_plan": null,
"plan_execution_mode": false,
"draft_plan": null,
"acceptance_criteria": [],
"acceptance_ledger": {}
```

The model used `ask_human` to present an outline:

> "(1) types/interfaces — Protocol class … (2) constants/globals … (3) helpers/network_info … (4) main_logic … (5) file_close … Reply with which section to complete now, or reply 'continue' to proceed with `types/interfaces` by default."

However, this was emitted as a **generic `ask_human` tool call**, not as a formal `plan_set` / `plan_request_execution` pair. The harness has a dedicated plan/approval subsystem (`draft_plan`, `acceptance_ledger`, `plan_execution_mode`), but none of these state fields were populated.

**Why it happened:**
- The model was not instructed (or did not infer) that outline proposals must be registered via the plan subsystem.
- The harness’s `chunked_write_loop_guard_outline` interrupt handler treated the `ask_human` result as a human-input gate, not as a plan draft.

**Impact:** When the user replied `"continue"`, the harness had no plan step to advance. It simply cleared the outline flag and resumed the loop. The model was then free to attempt raw `file_write` calls again — which immediately re-triggered LoopGuard because the staged file state was not re-injected into the prompt.

---

### 2.3 Approval "continue" Not Routed to Plan Acceptance

**Evidence (harness.log):**

```
chunked_write_loop_guard_outline_resume cleared loop guard outline mode after human confirmation
step_count_reset resetting step count for continuation
```

The user said `"continue"` **twice** (once after A0006, once implicitly after the stall). Both times the harness:
1. Cleared the LoopGuard outline flag.
2. Reset the step counter.
3. Resumed the standard `execute` phase loop.

There was **no `acceptance_ledger` update**, no `plan_execution_mode = true`, and no transition to a structured step dispatcher.

**Why it happened:** Because `draft_plan` was `null`, the approval bridge had nothing to bridge. The `"continue"` reply was handled by the generic `interrupt_for_human` resume path, not the plan-acceptance path.

**Impact:** The model re-entered the loop with no structural guardrails. It attempted to write `types/interfaces` again — but since the prompt did not contain the staged file content, the model’s context was stale, and its write attempt was blocked.

---

### 2.4 LoopGuard Hard-Abort After Escalation

**Evidence (harness.log):**

```
chunked_write_loop_guard_hard_abort aborted task after repeated chunked write loop recovery failures
  path: /home/stephen/Scripts/Harness-Redo/temp/network_allocator.py
  section_name: types/interfaces
  attempts: 4
  trigger_kind: read_required_retry
  stagnation_score: 3
```

The LoopGuard correctly enforced its policy:
- A0003: rewrite of `imports` → blocked (`repeated_tool_loop`)
- A0004: append without read → blocked (`read_required_retry`)
- A0007: synthetic empty write → blocked (`read_required_retry`)

After 4 attempts with no forward progress, the harness escalated to **level-4 hard-abort** and failed the task.

**Why it happened:** The model was never given a legal path to advance. After outline confirmation, the prompt should have included:
1. The **staged file content** so the model could read it.
2. A **plan step directive** telling it to write `types/interfaces` next.
3. An **explicit read-then-write gate reset** so LoopGuard would allow the new section.

Instead, the model was dropped back into a generic `execute` prompt with no tools available (all stripped due to prior unavailable-tool history), and it hallucinated a JSON block.

---

### 2.5 Ancillary Tool Misuse

| Tool | Issue | Evidence |
|------|-------|----------|
| `read_log` | Called with invalid `cwd` argument | `read_log failed: read_log() got an unexpected keyword argument 'cwd'` (A0005) |
| `artifact_read` | Hallucinated / unavailable tool | `message[8]:tool_call[0]:unavailable_tool:artifact_read` (payload preflight) |

These did not directly cause the abort, but they contributed to the **tool-stripping cascade** that left the model with zero available tools in the final request.

---

## 3. Why the Proposed Writing Plan Failed to Execute

The model *proposed* a writing plan (the 5-section outline), but the harness never *loaded* it. The harness has a two-part plan system:

1. **Drafting:** `plan_set` registers sections/criteria.
2. **Execution:** `plan_request_execution` transitions to `plan_execution_mode = true`.

In this session, the model used `ask_human` instead of `plan_set`. The harness's `chunked_write_loop_guard_outline` interrupt is designed as a **human gating mechanism**, not a plan loader. Therefore:

- No `draft_plan` was created.
- No `acceptance_criteria` were registered.
- The `acceptance_ledger` remained empty.
- The user's `"continue"` could not be mapped to a plan step.

The outline confirmation became a **no-op resume** rather than a structured plan launch.

---

## 4. Root Cause Diagram

```
User asks for network_allocator.py
        │
        ▼
Model opens write session ws_a22af5
        │
        ▼
Writes imports (success) ─────────────────────┐
        │                                      │
Rewrites imports ──► LoopGuard blocks ◄────────┘
        │
Attempts append without read ──► LoopGuard blocks again
        │
Calls read_log(cwd=...) ──► Tool error
        │
Calls ask_human with outline ──► Harness enters outline interrupt
        │                              draft_plan = null
        │
User says "continue" ──► Harness clears outline flag
        │                              (no plan step advanced)
        │
Model tries file_write again ──► Tool schema stripped (prior unavailable tools)
        │
Model emits ```json {"name":...} into content
        │
Fallback parser synthesizes empty file_write
        │
LoopGuard blocks (read_required_retry)
        │
Stagnation score = 3 ──► HARD ABORT
```

---

## 5. Recommended Fixes

### 5.1 Harden Tool-Call Stream Parsing for Qwen3.5-9b

**Problem:** When `tool_calls` stream stalls and tools are later stripped, Qwen3.5-9b leaks JSON into `content`.

**Fix:**
- In `stream_text_write_fallback`, detect the pattern `\`\`\`json\n{"name":` (or similar partial JSON) and **reject it** rather than synthesizing a tool call.
- If `content` contains fenced JSON but no complete `arguments` object, flag it as `model_malformed_output` and surface a system-level retry, not a synthetic write.
- Add a guard: if `estimated_payload_tokens` shows zero tools in schema (`tool_count: 0`), disable the `assistant_unclosed_fenced_code` recovery path entirely — the model cannot legally emit a tool call, so any fenced block is suspect.

---

### 5.2 Require `plan_set` for Multi-Section Outlines

**Problem:** `ask_human` was used for outline proposal, bypassing the plan subsystem.

**Fix:**
- When the model proposes a chunked-write outline during a `write_session_stall` or `chunked_write_loop_guard_outline` state, **require** it to call `plan_set` with the section list before (or instead of) `ask_human`.
- Alternatively, teach the prompt: *"If you propose a multi-section writing plan, you must use `plan_set` to register it. Use `ask_human` only for questions, not for plan proposals."*
- The harness should auto-detect outline-like `ask_human` content and either:
  - Reject it with `plan_not_registered`, or
  - Auto-translate it into a `draft_plan` if section keywords are detected.

---

### 5.3 Bridge User "continue" into Plan Acceptance

**Problem:** `"continue"` was treated as a generic resume because no plan existed.

**Fix:**
- When `chunked_write_loop_guard_outline` is active and the user replies `"continue"`, if a `draft_plan` exists, advance the `acceptance_ledger` and set `plan_execution_mode = true`.
- If no `draft_plan` exists but the interrupt carries an implicit outline (as it did here), **create a synthetic `draft_plan`** from the outline sections at resume time, then advance to step 1.
- Update the prompt after outline confirmation to explicitly state: *"Plan step 1/5: write section `types/interfaces`. The staged file content is shown below. You may now append this section."*

---

### 5.4 Re-Inject Staged File State After Outline Confirmation

**Problem:** After the outline was confirmed, the model’s prompt did not include the staged file, so LoopGuard blocked the next write.

**Fix:**
- In the `chunked_write_loop_guard_outline_resume` handler, before resuming the loop, **attach the staged artifact snippet** (`ws_a22af5__stage`) to the prompt as an `active_artifact_read`.
- This satisfies LoopGuard’s read-before-write gate and gives the model the legal context to append the next section.

---

### 5.5 Fix `read_log` Schema and Add `artifact_read` Guard

**Problem:** Model passed `cwd` to `read_log`; `artifact_read` was hallucinated.

**Fix:**
- Remove `cwd` from the `read_log` schema if it is not supported, or update the tool implementation to accept it.
- Add `artifact_read` to the unavailable-tool guard so that if the model hallucinates it, the preflight repair strips it cleanly without poisoning the tool schema for subsequent turns.

---

### 5.6 Prevent Tool-Schema Stripping Cascade

**Problem:** Prior unavailable-tool calls caused **all tools** to be stripped in the final request.

**Evidence:**
```
payload_preflight_repaired ... issues: [
  "message[2]:tool_call[0]:unavailable_tool:memory_update",
  "message[4]:tool_call[0]:unavailable_tool:file_write",
  ...
], tool_count: 0
```

**Fix:**
- In `payload_preflight_repaired`, when unavailable tools are detected in *history*, do not strip the **current** tool schema. Only strip unavailable tools from the *current* request payload.
- The preflight should be idempotent: historical tool errors should not disable future tool availability.

---

## 6. Data Sources

| Source | Path | Lines | Key Evidence |
|--------|------|-------|--------------|
| Chat state | `.smallctl/chat_states/a707ce3f.json` | ~400 | `active_plan: null`, `draft_plan: null`, `acceptance_ledger: {}` |
| Harness log | `logs/a707ce3f-20260522-165417/harness.log` | 434 | LoopGuard blocks, hard-abort, outline resume, payload preflight |
| Chat stream | `logs/a707ce3f-20260522-165417/chat.log` | 655 | Stream stall, `\`\`\`json` leakage, tool stripping |
| Task summary | `tasks/task-0001/task_summary.json` | ~60 | Task status: `failed`, error: `guard` |
| Staged file | `.smallctl/write_sessions/ws_a22af5__network_allocator__stage.py` | 4 | Only imports checkpointed |

---

## 7. Classification

| Attribute | Value |
|-----------|-------|
| Primary failure mode | **Model output leakage + fallback misrecovery** |
| Secondary failure mode | **Plan subsystem bypassed by `ask_human`** |
| Tertiary failure mode | **Tool-schema stripping cascade** |
| Severity | **High** — task failed; no file produced |
| Escalation trigger | LoopGuard stagnation score ≥ 3 |
| Recovery possible? | Yes — re-run with staged file injected and plan loaded |
| Regression risk | Medium — Qwen3.5-9b behavior may repeat without parser hardening |

---

*RCA compiled from full log traces. All times UTC.*
