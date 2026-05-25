# Handover: Continuing `smallctl` UI Troubleshooting

## Context

The `smallctl` TUI (built on [Textual](https://textual.textualize.io/) 8.1.1) had reports of "not starting" or appearing to hang when launched via `smallctl --tui`. A recent round of fixes addressed the most obvious root cause, but there may be additional edge cases, threading deadlocks, or initialization failures still hiding in the startup path.

---

## Environment

- **Python**: 3.12.3
- **Textual**: 8.1.1
- **Working directory**: `Scripts/Harness-Redo/`
- **Entry points**:
  - `.venv/bin/smallctl --tui`
  - `python -m smallctl --tui` (now works; `__main__.py` was added)
- **Key modules**:
  - `src/smallctl/main.py` — CLI dispatch
  - `src/smallctl/ui/app.py` — `SmallctlApp` (Textual App)
  - `src/smallctl/ui/app_flow.py` — Harness lifecycle & bridge management
  - `src/smallctl/ui/harness_bridge.py` — Background thread running `Harness` in its own asyncio loop
  - `src/smallctl/harness/__init__.py` / `initialization.py` — `Harness` constructor (~549 lines of setup)

---

## What Was Already Fixed

### 1. Blocking call in async `on_mount()` (CRITICAL FIX)
**Problem**: `HarnessBridge.start()` calls `threading.Event().wait()`, blocking the Textual event loop during mount. Under thread contention or slow initialization, the UI never rendered.

**Fix**: `_create_harness()` is now `async def` and offloads the blocking `start()` to `asyncio.to_thread()`.
- `src/smallctl/ui/app_flow.py` — `_create_harness()` now async
- `src/smallctl/ui/app.py` — `await self._create_harness()` in `on_mount()`
- `src/smallctl/ui/app_actions.py` — `await self._create_harness()` in restore/new-conversation actions

### 2. Infinite hang if background thread crashes
**Problem**: If `HarnessBridge._thread_main()` crashed before `self._ready.set()`, `start()` hung forever.

**Fix**: `self._ready.wait(timeout=30.0)` with explicit `RuntimeError` in `src/smallctl/ui/harness_bridge.py`.

### 3. Non-TTY guard & terminal cleanup
**Problem**: Running `--tui` headless (piped, CI, SSH without `-t`) silently entered the alternate screen buffer with nowhere to render.

**Fix**: `main.py` emits a JSON warning to stderr when `sys.stdin.isatty()` is false, but still allows startup so tests aren't broken. Also added `\033[?1049l` (exit alternate screen buffer) to crash/finally handlers so a fatal error doesn't leave the terminal unusable.

---

## Changes Made (Round 2026-05-25)

### ✅ Issue F — `RunLogger.set_session_id` deadlock — FIXED
**Problem**: `cli()` sets `run_logger._finalize_listener` to a callback that calls `run_logger.log()` inside it. When `Harness.__init__` calls `run_logger.set_session_id()`, it acquires `RunLogger._lock` (a non-reentrant `threading.Lock`), then invokes `_finalize_listener`. The listener calls `run_logger.log()`, which tries to acquire the **same** lock again, causing a **deadlock**. The app appears to hang immediately after `build_registry: starting registration` because the UI thread is blocked forever.

**Fix**:
- `src/smallctl/logging_utils.py` (`RunLogger.set_session_id`):
  - Moved `_finalize_listener` invocation **outside** the `with self._lock:` block so the lock is released before calling the callback.

**Verification**:
```bash
script -q -c 'timeout 3 .venv/bin/smallctl --tui' /dev/null
# → Full UI rendered (~17 KB of terminal output), no hang
```

### ✅ Issue A — `Harness.__init__` failure before UI mount — FIXED
**Problem**: `Harness(**kwargs)` inside `_create_harness()` could raise exceptions (missing `endpoint`, `build_registry()` failure, `OpenAICompatClient` init failure). Textual swallowed the exception or failed to render anything, leaving a blank screen.

**Fix**:
- `src/smallctl/ui/app_flow.py` (`_create_harness()`):
  - Added `try/except Exception` around `Harness(**self.harness_kwargs)`.
  - On failure, logs traceback via `self._app_logger.exception("harness_init_failed")`.
  - Surfaces error to user with `await self._append_system_line(f"Failed to initialize harness: {exc}", force=True)`.
  - Sets status bar to `"error"` and returns early with `self.harness = None`.
  - Added defensive cleanup: shuts down any existing `_harness_bridge` or `harness` before creating new ones.
- `src/smallctl/ui/app.py` (`on_mount()`):
  - Added early return if `self.harness is None` after `_create_harness()`.
  - UI now mounts successfully even when harness init fails, so the user can read the error message and exit cleanly.

**Verification**:
```bash
cd Scripts/Harness-Redo
. .venv/bin/activate
python -c "
import asyncio
from smallctl.ui.app import SmallctlApp
app = SmallctlApp(harness_kwargs={'model': 'test', 'phase': 'explore', 'provider_profile': 'generic'})
asyncio.run(app._create_harness())
print('harness is None:', app.harness is None)
"
# → harness is None: True (KeyError for missing 'endpoint' caught and surfaced)
```

### ✅ Issue D — Memory / leak on repeated `_create_harness` — FIXED
**Problem**: When switching models via the fallback path in `_switch_model()` (when the bridge has no `switch_model` callable), a new `Harness` and `HarnessBridge` were created without shutting down the old background thread.

**Fix**:
- `src/smallctl/ui/app_flow.py` (`_create_harness()`):
  - Added defensive shutdown of existing `old_bridge` or `harness` at the top of the method. Any caller is now protected from orphan threads.
- `src/smallctl/ui/app_flow.py` (`_switch_model()`):
  - In the `else` fallback path (no `bridge.switch_model` and no `harness.switch_model`), added explicit shutdown of `old_bridge` or `harness` before calling `_create_harness()`.

**Verification**:
```bash
python -c "
import asyncio, threading, time
from smallctl.ui.app import SmallctlApp
app = SmallctlApp(harness_kwargs={'endpoint': 'http://localhost:8000', 'model': 'test', 'phase': 'explore', 'provider_profile': 'generic'})
asyncio.run(app._create_harness())
print('After create:', [t.name for t in threading.enumerate() if 'harness' in t.name.lower()])
asyncio.run(app._switch_model('another-model'))
time.sleep(0.5)
print('After switch:', [t.name for t in threading.enumerate() if 'harness' in t.name.lower()])
"
# → Only one 'smallctl-harness' thread at each stage
```

---

## Known Open Issues / Next Investigation Targets

### 🟡 Issue B: `_run_harness_task` swallows `CancelledError` silently — PARTIALLY ADDRESSED
The current code already appends `"Task cancelled."` to the console inside the `except asyncio.CancelledError` block and sets the status bar to `"cancelled"`. The Stop button (`Ctrl+K`) calls `bridge.abort()` then `self.active_task.cancel()`, which propagates correctly.

**Remaining risk**: If the harness coroutine running inside the background loop swallows `CancelledError` internally, the bridge future may never complete and the UI task could hang. To verify:
1. Start a long-running task.
2. Press Stop.
3. Check if the status bar returns to idle and the input widget becomes focusable again.

### 🟡 Issue C: `HarnessBridge` thread safety with `post_ui_event`
`_post_harness_bridge_event` uses Textual's `self.post_message()`, which is documented as thread-safe. The `is_running` guard may drop events during the startup burst (between bridge thread start and Textual's `is_running` becoming `True`).

**How to investigate**:
1. Add a counter in `_post_harness_bridge_event` to count events received vs. events processed by `_drain_harness_events`.
2. Run a fast task and compare counts.

### 🟡 Issue E: CSS / compose errors on startup
CSS parses successfully (`styles.tcss` is valid). If a widget in `compose()` raises, Textual may exit before the alternate screen buffer is visible.

**How to investigate**:
1. Run with `TEXTUAL_DEBUG=1` or textual's dev console.
2. Temporarily introduce a bad CSS rule or a widget that raises in `compose()` and observe behavior.

### 🔴 Issue F — `RunLogger.set_session_id` deadlock — FIXED
**Problem**: `cli()` sets `run_logger._finalize_listener` to a callback that calls `run_logger.log()` inside it. When `Harness.__init__` calls `run_logger.set_session_id()`, it acquires `RunLogger._lock` (a non-reentrant `threading.Lock`), then invokes `_finalize_listener`. The listener calls `run_logger.log()`, which tries to acquire the **same** lock again, causing a **deadlock**. The app appears to hang immediately after `build_registry: starting registration` because the UI thread is blocked forever.

**Fix**:
- `src/smallctl/logging_utils.py` (`RunLogger.set_session_id`):
  - Moved `_finalize_listener` invocation **outside** the `with self._lock:` block so the lock is released before calling the callback.
  - Stored `finalized_run_dir = str(self.run_dir)` while holding the lock, then passed the snapshot to the listener after release.

**Verification**:
```bash
cd Scripts/Harness-Redo
. .venv/bin/activate
# Reproduction of deadlock (should complete instantly, not hang)
python -c "
from smallctl.logging_utils import create_run_logger
run_logger = create_run_logger('logs')
def _emit(final_dir):
    print('finalize listener called:', final_dir)
    run_logger.log('harness', 'log_dir_finalized', run_log_dir=final_dir)
run_logger._finalize_listener = _emit
run_logger.set_session_id('test-session')
print('SUCCESS: no deadlock')
"
# TUI startup with pseudo-tty (should render UI frames, not hang)
script -q -c 'timeout 3 .venv/bin/smallctl --tui' /dev/null
# → Full UI rendered, ~17 KB of terminal output
```

### 🟡 Issue G — `Harness.__init__` is still synchronous and runs on the UI thread
Even though `_create_harness()` is `async def`, the line `self.harness = Harness(**self.harness_kwargs)` is a **synchronous constructor call** that runs ~549 lines of setup directly on Textual's event loop. Under slow disk I/O, heavy `build_registry()`, or slow `OpenAICompatClient` initialization, this blocks the UI for hundreds of milliseconds (or longer), making the app appear to "hang" before the first frame renders.

**Recommended fix for next round**:
Offload the entire `Harness(**kwargs)` constructor to a worker thread:
```python
async def _create_harness(self) -> None:
    # ... shutdown old bridge ...
    try:
        self.harness = await asyncio.to_thread(Harness, **self.harness_kwargs)
    except Exception as exc:
        # ... existing error handling ...
```

**Caveat**: Verify that `Harness.__init__` is thread-safe (it does not interact with asyncio loops or Textual state). If it is not thread-safe, consider deferring heavy initialization to a lazy `Harness.initialize()` async method.

---

## Quick Diagnostic Commands

```bash
# 1. Verify compilation of all UI modules
cd Scripts/Harness-Redo
source .venv/bin/activate
python -m py_compile src/smallctl/ui/*.py src/smallctl/main.py

# 2. Run UI-specific tests
python -m pytest tests/test_chat_selector.py tests/test_model_selector.py \
  tests/test_ui_harness_bridge.py tests/test_shutdown_cleanup.py \
  tests/test_tool_plan_run_mode.py -v

# 3. Test headless (should emit warning, not hang)
echo "test" | timeout 2 python -m smallctl --tui 2>&1 | head

# 4. Test interactive (requires real TTY; use script if over SSH)
script -q -c "timeout 3 python -m smallctl --tui" /dev/null

# 5. Thread leak check after model switches
python -c "
import threading, time, asyncio
from smallctl.ui.app import SmallctlApp
app = SmallctlApp(harness_kwargs={'endpoint': 'http://localhost:8000', 'model': 'test', 'phase': 'explore', 'provider_profile': 'generic'})
asyncio.run(app._create_harness())
print('After create:', [t.name for t in threading.enumerate() if 'harness' in t.name.lower()])
asyncio.run(app._switch_model('another-model'))
time.sleep(0.5)
print('After switch:', [t.name for t in threading.enumerate() if 'harness' in t.name.lower()])
"

# 6. Verify harness-init failure surfaces error without crashing UI
python -c "
import asyncio
from smallctl.ui.app import SmallctlApp
app = SmallctlApp(harness_kwargs={'model': 'test', 'phase': 'explore'})
asyncio.run(app._create_harness())
print('harness is None:', app.harness is None)
"
```

---

## Files Most Likely to Need Changes

| File | Purpose |
|------|---------|
| `src/smallctl/ui/app_flow.py` | `_create_harness`, `_run_harness_task`, model switching |
| `src/smallctl/ui/app.py` | `on_mount`, `compose`, event dispatch |
| `src/smallctl/ui/harness_bridge.py` | Thread lifecycle, event forwarding |
| `src/smallctl/main.py` | TTY guard, terminal reset, CLI argument plumbing |
| `src/smallctl/harness/initialization.py` | `Harness` constructor — failure root cause |

---

## Test Notes

- **Mock `_create_harness` must be async**: Any test that monkeypatches `SmallctlApp._create_harness` must use `async def _fake_create_harness(self) -> None:`. Existing tests in `test_chat_selector.py`, `test_model_selector.py`, and `test_shutdown_cleanup.py` have already been updated.
- **Non-TTY tests**: Tests that call `cli(["--tui"])` without a TTY will now see a JSON warning on stderr but still execute. Do **not** change the TTY guard to a hard error without also updating those tests.

---

## Exit Criteria for Next Round

1. `smallctl --tui` starts reliably in an interactive terminal with valid config.
2. `smallctl --tui` with **invalid** config (bad endpoint, missing model) surfaces a clear error instead of hanging or showing a blank screen.
3. Thread count does not grow after model switches / new conversations.
4. All UI-related tests pass (`test_chat_selector`, `test_model_selector`, `test_ui_harness_bridge`, `test_shutdown_cleanup`, `test_tool_plan_run_mode`).
5. Terminal is never left in an unusable state after TUI exit or crash.
6. `Harness` initialization does not block the Textual event loop — the UI renders within ~100 ms of launch even on slow systems.
7. `RunLogger.set_session_id()` does not deadlock when `_finalize_listener` calls back into `run_logger.log()`. UI tests and logging tests pass.
