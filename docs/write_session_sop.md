# SOP: Staged Authoring & Write Sessions

To ensure reliable file modifications and avoid harness-level stagnation, all staged authoring workflows must adhere to the following protocol.

## 1. Explicit First-Write Choice in `patch_existing` Sessions
When a `patch_existing` session has **no committed sections yet**, the first same-target mutation must be explicit:
- Use `file_write(..., replace_strategy='overwrite')` to replace the entire staged file.
- Use `file_patch` for a narrow exact edit inside the staged copy.
- Use `ast_patch` for a narrow structural edit.

A prior `file_read` is **not strictly required** if you already know the staged content and choose an explicit repair shape such as `replace_strategy='overwrite'`. However, if earlier chunks are not fully visible in your local context, call `file_read(path=...)` first to recover the current staged content before choosing your repair shape.

## 2. Explicit Mutation Strategies
Once the staged content is in context, use one of the following **explicit** strategies:

| Strategy | Tool | Use Case |
| :--- | :--- | :--- |
| **`file_patch`** | `file_patch` | Exact string replacement for narrow edits. |
| **`ast_patch`** | `ast_patch` | Structural Python edits (requires valid AST). |
| **`overwrite`** | `file_write` | Complete replacement of the staged file. |

## 3. Prohibited Actions
- **DO NOT** use `replace_strategy='auto'` (default) in a `patch_existing` session.
- **DO NOT** use `replace_strategy='append'` unless specifically intended for log-style files.
- **DO NOT** attempt a second implicit `file_write` if the first one was rejected; you must switch to an explicit strategy.

## 4. Recovery from Rejection
If the harness rejects a write with a "Patch-existing write sessions need an explicit first-chunk choice" error:
1. Choose exactly one explicit repair shape: `file_patch`, `ast_patch`, or `file_write` with `replace_strategy='overwrite'`.
2. If you are uncertain about the current staged content, call `file_read` on the target path first.
3. Apply the change using the chosen explicit strategy.

## 5. Bare `file_write` Is Blocked While a Session Is Open

If a Write Session is active for a given path, **all** `file_write` calls to that path **must** include both `write_session_id` and `section_name`. A bare `file_write` that omits `write_session_id` will be **rejected by the harness** with an `error_kind: bare_write_to_session_owned_path` error — the file will **not** be written, and the session FSM will **not** advance.

**Recovery steps:**
1. Call `loop_status` to retrieve the current `write_session_id` and `write_next_section`.
2. Retry the `file_write` with `write_session_id=<id>` and `section_name=<next_section>` added.
3. If no further sections remain, call `finalize_write_session` to promote the staged file.

> **Why this matters:** Omitting `write_session_id` previously caused a silent bypass — the file was physically written but the session FSM stayed at "Start", making `task_complete` permanently blocked. The harness now blocks the write early so you receive the error while you still have context to fix it.
