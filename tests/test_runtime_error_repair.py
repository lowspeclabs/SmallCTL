from __future__ import annotations

from smallctl.runtime_error_repair import (
    _diff_covers_line,
    maybe_record_reported_runtime_error,
    runtime_error_ask_human_block,
    runtime_error_completion_block,
    runtime_error_task_fail_block,
    runtime_error_verifier_passes,
)
from smallctl.state import LoopState


TRACEBACK = """pygame 2.6.1
Traceback (most recent call last):
  File "/tmp/project/app.py", line 14, in <module>
    from widget_core import main
ModuleNotFoundError: No module named 'widget_core'
"""


def test_records_reported_runtime_error_and_repetition() -> None:
    state = LoopState()

    first = maybe_record_reported_runtime_error(state, TRACEBACK)
    second = maybe_record_reported_runtime_error(state, TRACEBACK)

    assert first is not None
    assert second is not None
    report = state.scratchpad["_reported_runtime_error"]
    assert report["kind"] == "ModuleNotFoundError"
    assert report["module"] == "widget_core"
    assert report["entrypoint"] == "/tmp/project/app.py"
    assert report["repeated_count"] == 2
    assert state.active_intent == "reported_runtime_error_repair"


def test_completion_block_requires_relevant_passing_verifier() -> None:
    state = LoopState()
    maybe_record_reported_runtime_error(state, TRACEBACK)

    irrelevant_pass = {
        "verdict": "pass",
        "command": "python -m py_compile /tmp/project/other.py",
        "key_stdout": "OK",
    }
    assert runtime_error_completion_block(state, verifier_verdict=irrelevant_pass) is not None

    relevant_pass = {
        "verdict": "pass",
        "command": "cd /tmp/project && python -c 'import widget_core'",
        "key_stdout": "OK",
        "key_stderr": "",
    }
    assert runtime_error_verifier_passes(state.scratchpad["_reported_runtime_error"], relevant_pass)
    assert runtime_error_completion_block(state, verifier_verdict=relevant_pass) is None


def test_task_fail_block_rejects_unsupported_unrelated_cause() -> None:
    state = LoopState()
    maybe_record_reported_runtime_error(state, TRACEBACK)

    block = runtime_error_task_fail_block(
        state,
        message="pytest changed into the venv and could not find tests",
        verifier_verdict={
            "verdict": "fail",
            "command": "pytest tests/test_other.py",
            "key_stderr": "file not found",
        },
    )

    assert block is not None
    assert block["reason"] == "reported_runtime_error_failure_irrelevant_verifier"


def test_ask_human_block_rejects_retry_prompt() -> None:
    state = LoopState()
    maybe_record_reported_runtime_error(state, TRACEBACK)

    block = runtime_error_ask_human_block(
        state,
        question="Please run it again and let me know if you see any errors.",
    )

    assert block is not None
    assert block["reason"] == "reported_runtime_error_ask_human_before_repair"


def test_ask_human_allows_missing_information_question() -> None:
    state = LoopState()
    maybe_record_reported_runtime_error(state, TRACEBACK)

    block = runtime_error_ask_human_block(
        state,
        question="Which command produced this traceback?",
    )

    assert block is None


def test_diff_covers_line_parses_hunk_headers() -> None:
    diff = """--- a/file.py
+++ b/file.py
@@ -245,7 +245,7 @@
         current_key = pygame.KEYDOWN
-        clock.get_scrolling()
+
         time_key = current_key
@@ -260,3 +260,3 @@
     pygame.display.quit()
"""
    assert _diff_covers_line(diff, 245) is True
    assert _diff_covers_line(diff, 248) is True
    assert _diff_covers_line(diff, 251) is True
    assert _diff_covers_line(diff, 244) is False
    assert _diff_covers_line(diff, 252) is False
    assert _diff_covers_line(diff, 260) is True
    assert _diff_covers_line(diff, 259) is False


def test_patch_acts_as_verifier_for_exact_line_error() -> None:
    traceback = """Traceback (most recent call last):
  File "/tmp/project/app.py", line 248, in main
    clock.get_scrolling()
    ^^^^^^^^^^^^^^^^^^^
AttributeError: 'pygame.time.Clock' object has no attribute 'get_scrolling'
"""
    state = LoopState()
    maybe_record_reported_runtime_error(state, traceback)

    # No patch yet -> blocked
    assert runtime_error_completion_block(state, verifier_verdict=None) is not None

    # Inject a successful file_patch artifact on the exact file/line
    state.artifacts["A0001"] = {
        "kind": "file_patch",
        "metadata": {
            "path": "/tmp/project/app.py",
            "success": True,
            "changed": True,
            "diff": "@@ -245,7 +245,7 @@\n         current_key = pygame.KEYDOWN\n-        clock.get_scrolling()\n+\n         time_key = current_key",
        },
    }

    # Patch covers line 248 -> allowed
    assert runtime_error_completion_block(state, verifier_verdict=None) is None
    assert state.scratchpad["_reported_runtime_error"]["status"] == "verified_fixed"


def test_patch_acts_as_verifier_rejects_wrong_file() -> None:
    traceback = """Traceback (most recent call last):
  File "/tmp/project/app.py", line 14, in <module>
    from widget_core import main
ModuleNotFoundError: No module named 'widget_core'
"""
    state = LoopState()
    maybe_record_reported_runtime_error(state, traceback)

    state.artifacts["A0001"] = {
        "kind": "file_patch",
        "metadata": {
            "path": "/tmp/project/other.py",
            "success": True,
            "changed": True,
            "diff": "@@ -1,1 +1,1 @@\n-old\n+new",
        },
    }

    assert runtime_error_completion_block(state, verifier_verdict=None) is not None


def test_patch_acts_as_verifier_rejects_unchanged_patch() -> None:
    traceback = """Traceback (most recent call last):
  File "/tmp/project/app.py", line 248, in main
    clock.get_scrolling()
AttributeError: 'pygame.time.Clock' object has no attribute 'get_scrolling'
"""
    state = LoopState()
    maybe_record_reported_runtime_error(state, traceback)

    state.artifacts["A0001"] = {
        "kind": "file_patch",
        "metadata": {
            "path": "/tmp/project/app.py",
            "success": True,
            "changed": False,
            "diff": "@@ -245,7 +245,7 @@\n-no change",
        },
    }

    assert runtime_error_completion_block(state, verifier_verdict=None) is not None
