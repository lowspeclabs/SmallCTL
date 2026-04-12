from __future__ import annotations

import json
from types import SimpleNamespace

from smallctl.graph.write_recovery import (
    build_synthetic_write_args,
    maybe_finalize_recovered_assistant_write,
    recover_write_intent,
)
from smallctl.write_session_fsm import new_write_session


class _FakeHarness:
    def __init__(self, session: object) -> None:
        self.state = SimpleNamespace(
            write_session=session,
            cwd="/tmp/workspace",
            scratchpad={},
        )
        self.config = SimpleNamespace(
            enable_write_intent_recovery=True,
            write_recovery_allow_raw_text_targets=True,
            enable_assistant_code_write_recovery=True,
            write_recovery_min_confidence="high",
        )


def _assistant_inline_write(
    *,
    path: str,
    write_session_id: str,
    section_name: str,
    content: str,
    next_section_name: str = "",
) -> str:
    payload = {
        "path": path,
        "write_session_id": write_session_id,
        "section_name": section_name,
        "content": content,
    }
    if next_section_name:
        payload["next_section_name"] = next_section_name
    return f"[file_write] {json.dumps(payload, ensure_ascii=True)}"


def test_recovered_complete_python_file_clears_session_default_next_section() -> None:
    session = new_write_session(
        session_id="ws_test",
        target_path="./temp/task_queue.py",
        intent="replace_file",
        suggested_sections=["imports", "types_interfaces", "constants_globals"],
        next_section="imports",
    )
    session.write_current_section = "imports"
    harness = _FakeHarness(session)
    assistant_text = _assistant_inline_write(
        path="./temp/task_queue.py",
        write_session_id="ws_test",
        section_name="imports",
        content=(
            '"""Priority Task Queue Implementation."""\n'
            "import heapq\n"
            "import unittest\n\n"
            "class Task:\n"
            "    pass\n\n"
            "def enqueue() -> None:\n"
            "    pass\n\n"
            "class TestTaskQueue(unittest.TestCase):\n"
            "    def test_enqueue(self) -> None:\n"
            "        self.assertTrue(True)\n\n"
            'if __name__ == "__main__":\n'
            "    unittest.main()\n"
        ),
    )
    partial_tool_calls = [
        {
            "id": "partial-1",
            "type": "function",
            "function": {"name": "file_write", "arguments": '{"path":"./temp/task_queue.py"}'},
        }
    ]

    intent = recover_write_intent(
        harness=harness,
        pending=None,
        assistant_text=assistant_text,
        partial_tool_calls=partial_tool_calls,
    )

    assert intent is not None
    assert intent.next_section_name == "types_interfaces"
    assert intent.next_section_name_origin == "session_default"

    changed = maybe_finalize_recovered_assistant_write(intent)

    assert changed is True
    assert intent.next_section_name == ""
    assert intent.next_section_name_origin == ""
    assert "cleared_session_default_next_section_name" in intent.evidence


def test_recovered_small_chunk_keeps_session_default_next_section() -> None:
    session = new_write_session(
        session_id="ws_test",
        target_path="./temp/task_queue.py",
        intent="replace_file",
        suggested_sections=["imports", "types_interfaces", "constants_globals"],
        next_section="imports",
    )
    session.write_current_section = "imports"
    harness = _FakeHarness(session)
    assistant_text = _assistant_inline_write(
        path="./temp/task_queue.py",
        write_session_id="ws_test",
        section_name="imports",
        content="import heapq\nimport logging\n",
    )
    partial_tool_calls = [
        {
            "id": "partial-1",
            "type": "function",
            "function": {"name": "file_write", "arguments": '{"path":"./temp/task_queue.py"}'},
        }
    ]

    intent = recover_write_intent(
        harness=harness,
        pending=None,
        assistant_text=assistant_text,
        partial_tool_calls=partial_tool_calls,
    )

    assert intent is not None
    assert intent.next_section_name == "types_interfaces"
    assert intent.next_section_name_origin == "session_default"
    assert maybe_finalize_recovered_assistant_write(intent) is False
    assert intent.next_section_name == "types_interfaces"


def test_recovered_write_intent_rebinds_mismatched_active_write_session_id() -> None:
    session = new_write_session(
        session_id="ws_active",
        target_path="./temp/task_queue.py",
        intent="replace_file",
        suggested_sections=["imports", "types_interfaces"],
        next_section="imports",
    )
    session.write_current_section = "imports"
    harness = _FakeHarness(session)
    assistant_text = _assistant_inline_write(
        path="./temp/task_queue.py",
        write_session_id="ws_model_guess",
        section_name="imports",
        content="import heapq\n",
    )
    partial_tool_calls = [
        {
            "id": "partial-1",
            "type": "function",
            "function": {
                "name": "file_append",
                "arguments": '{"path":"./temp/task_queue.py","write_session_id":"ws_model_guess"}',
            },
        }
    ]

    intent = recover_write_intent(
        harness=harness,
        pending=None,
        assistant_text=assistant_text,
        partial_tool_calls=partial_tool_calls,
    )

    assert intent is not None
    assert intent.write_session_id == "ws_active"
    assert "active_write_session_id_rebound" in intent.evidence
    assert build_synthetic_write_args(intent)["write_session_id"] == "ws_active"
