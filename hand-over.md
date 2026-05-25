## Handover Prompt — Fix Stagnation Guard & File-Mutation Progress Detection

**Target:** `Scripts/Harness-Redo/`  
**Session:** `8f76b57f` (cancelled after correct patches applied)  
**Priority:** P0 — fixes false-positive guard trips on every local file-mutation task

---

### 1. Background

Session `8f76b57f` produced correct, compiling code (two `file_patch` calls + `py_compile` exit 0) but was **cancelled by the user** because the agent looped for ~3 minutes without calling `task_complete`.

**Root cause chain identified:**
1. **`file_patch`, `file_write`, and `file_append` do not set `changed=True` in metadata.**  
   The stagnation guard (`_turn_has_actionable_progress` in `progress_guard.py`) requires `metadata.changed is True` to count a mutation as progress. Because local file tools never set this key, **every successful patch/write is scored as non-progress**.
2. **Paginated `artifact_read` consumed stagnation slots.**  
   A 275-line file triggered `artifact_read_truncated`, forcing two `artifact_read` calls. Both were scored as non-progress (the first because history was empty post-task-boundary-reset, the second for reasons that are moot once Fix A is in place).
3. **Shell approval gate added ~66 s of dead time.**  
   `python3 -m py_compile ./temp/pong.py` required manual approval.
4. **No completion nudge after successful compile.**  
   After `py_compile` exit 0, the model re-read the file twice instead of calling `task_complete`.

---

### 2. Required Changes

#### **Fix A — CRITICAL: Add `changed=True` to local mutation-tool metadata**

**File:** `src/smallctl/tools/fs.py`  
**Line:** ~246 (inside `file_write`)  
**Current:**
```python
return ok("written", metadata={"path": str(target), "bytes": len(content.encode(encoding))})
```
**Change to:**
```python
return ok("written", metadata={"path": str(target), "bytes": len(content.encode(encoding)), "changed": True})
```

**File:** `src/smallctl/tools/fs_mutations.py`  
**Line:** ~102 (inside `file_append`)  
**Current:**
```python
return ok("appended", metadata={"path": str(target)})
```
**Change to:**
```python
return ok("appended", metadata={"path": str(target), "changed": True})
```

**File:** `src/smallctl/tools/fs_patching.py`  
**Function:** `_build_patch_metadata` (~line 389)  
**Current dict construction:**
```python
metadata: dict[str, Any] = {
    "path": str(path),
    "requested_path": requested_path,
    ...
}
```
**Add inside that dict:**
```python
    "changed": True,
```
*(Only when the patch actually modified the file — do NOT set it for `dry_run=True`.)*

> **Why this is critical:** `_turn_has_actionable_progress` checks `(record.result.metadata or {}).get("changed") is True`. Remote tools (`ssh_file_patch`, `ast_patch`) already set this. Local tools do not, so any task that patches a local file will always trip the stagnation guard.

---

#### **Fix B — Exempt advancing `artifact_read` pages from stagnation counting**

**File:** `src/smallctl/graph/progress_guard.py`  
**Function:** `_turn_has_actionable_progress` (~line 135)  
**Current logic for `artifact_read`:**
```python
        if record.tool_name == "artifact_read" and record.result.success:
            if _artifact_read_is_past_eof(harness, record):
                return False
            if _artifact_read_result_is_new_range(harness, record):
                return True
```
**Add immediately before the `if _artifact_read_result_is_new_range(...)` check:**
```python
            # If this read is continuing pagination on the same artifact,
            # treat it as progress so the guard doesn't fire mid-read.
            if _artifact_read_is_continuation_page(harness, record):
                return True
```

**Implement `_artifact_read_is_continuation_page` in the same file:**
```python
def _artifact_read_is_continuation_page(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    artifact_id = _requested_artifact_read_target(args)
    if not artifact_id:
        return False
    start_line, _ = _requested_file_read_range(args)
    if start_line is None or start_line <= 1:
        return False
    coverage = _artifact_coverage_entry(harness, artifact_id)
    if coverage is None:
        return False
    # True if we have already read some prefix of this artifact
    normalized = _normalize_line_ranges(coverage.get("ranges", []))
    return bool(normalized and normalized[-1][1] >= start_line - 1)
```
*(Reuse existing `_artifact_coverage_entry` and `_normalize_line_ranges` helpers already in the file.)*

---

#### **Fix C — Auto-approve safe, read-only compile/lint commands**

**File:** `src/smallctl/harness/approvals.py` (or wherever `shell_exec` risk/approval logic lives)  
**Requirement:** Add a configurable allow-list of commands that are:
- Read-only (no file writes)
- No network side effects
- Short-running

**Initial allow-list:**
```python
_SAFE_COMPILE_LINT_COMMANDS = {
    "python3 -m py_compile",
    "flake8",
    "mypy",
    "ruff check",
    "shellcheck",
}
```

**Logic:** Before prompting human approval for `shell_exec`, normalize the command string. If it starts with any of the above prefixes, auto-approve (or set risk to `low` and skip human gate).

> **Warning:** Ensure the normalization does not accidentally match substrings (e.g., `python3 -m py_compile ./temp/pong.py && rm -rf /`). Use a strict prefix match or parse the first token(s).

---

#### **Fix D — Inject completion nudge after successful verifier on mutated file**

**File:** `src/smallctl/harness/tool_result_flow.py` or `src/smallctl/graph/progress_guard.py`  
**Trigger condition:**  
- Tool is `shell_exec` or `ssh_exec`  
- Exit code 0  
- Command matches verifier patterns (`py_compile`, `pytest`, `mypy`, etc.)  
- The most recently mutated file (from `files_changed_this_cycle` or last `file_patch` metadata) matches the verifier target path

**Action:** Append a system message to `state.recent_messages`:
> *"The verification command succeeded with exit code 0. If the task requirements are satisfied, call `task_complete(message=...)` now instead of performing additional reads."*

**Acceptance:** After this nudge, the model should call `task_complete` within the next turn with >80% probability on similar tasks.

---

#### **Fix E — Expand snippet size for the most-recently-mutated artifact**

**File:** `src/smallctl/context/frame_compiler.py` (or prompt assembler lane)  
**Requirement:** In the `artifact_snippets` lane, detect the artifact that corresponds to the file most recently touched by `file_patch` / `file_write` / `ast_patch`. For that artifact, bypass the default token cap and include a minimum of **200 tokens** (or the full artifact if it is < 300 lines).

**Why:** In session `8f76b57f`, the snippet for the patched artifact was only 53 tokens, forcing the model to issue redundant `artifact_read` calls to reconstruct the full context.

---

### 3. Acceptance Criteria

Run the following validation **before** submitting:

| # | Test | Expected Result |
|---|------|-----------------|
| 1 | Unit test: simulate `file_patch` success, then call `_turn_has_actionable_progress`. | Returns `True`. |
| 2 | Unit test: simulate `file_write` success, then call `_turn_has_actionable_progress`. | Returns `True`. |
| 3 | Unit test: simulate `artifact_read(A, 1, 150)` then `artifact_read(A, 150, 275)`. | Second call returns `True` from `_turn_has_actionable_progress`. |
| 4 | Regression: run `pytest tests/test_progress_guard_ssh_exec.py` (or equivalent). | All existing tests pass. |
| 5 | End-to-end: run a harness session with task *"read ./temp/pong.py and add a start menu"*. | Task completes without stagnation guard trip; both patches applied; `task_complete` called within 2 turns of compile success. |

---

### 4. Out-of-Scope / Do Not Change

- Do **not** lower the global `loop_guard_stagnation_threshold` (default 3/5).
- Do **not** change the FAMA signal routing or mitigation capsules.
- Do **not** modify the `artifact_read` tool truncation logic itself (Fix B handles the guard side).
- Do **not** commit to git unless explicitly instructed.

---

### 5. One-Line Summary

> **Add `changed=True` to local file-mutation metadata, exempt paginated artifact reads from stagnation, auto-approve safe compile commands, and nudge completion after verifier success.**
