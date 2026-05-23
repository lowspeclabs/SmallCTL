from __future__ import annotations

from smallctl.state import WriteSession
from smallctl.tools.fs import file_write, promote_write_session_target
import asyncio


class MockState:
    def __init__(self, cwd: str):
        self.cwd = cwd
        self.write_session = None
        self.scratchpad = {}
        self.current_phase = "explore"
        self.repair_cycle_id = ""
        self.stagnation_counters = {}
        self.last_failure_class = ""
        self.files_changed_this_cycle = []
        self.recent_messages = []

    def touch(self):
        pass

    def _runlog(self, *args, **kwargs):
        pass

    def append_message(self, message):
        self.recent_messages.append(message)


def test_repro_chunked_write_loss(tmp_path):
    cwd = str(tmp_path)
    state = MockState(cwd)
    path = "lost_imports.py"
    target_file = tmp_path / path
    session_id = "ws_repro"

    state.write_session = WriteSession(
        write_session_id=session_id,
        write_target_path=path,
        write_session_intent="replace_file",
        write_session_mode="chunked_author",
        status="open",
        suggested_sections=["imports", "main_logic"],
        write_next_section="imports",
    )

    imports_content = "import os\nimport sys\n"

    result = asyncio.run(
        file_write(
            path=path,
            content=imports_content,
            section_name="imports",
            next_section_name="",
            write_session_id=session_id,
            state=state,
            cwd=cwd,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["write_session_final_chunk"] is False
    assert result["metadata"]["write_next_section_inferred"] is True
    assert result["metadata"]["write_next_section"] == "main_logic"
    assert state.write_session.write_next_section == "main_logic"

    logic_content = "def main():\n    print('hello')\n"

    result2 = asyncio.run(
        file_write(
            path=path,
            content=logic_content,
            section_name="main_logic",
            next_section_name="",
            write_session_id=session_id,
            state=state,
            cwd=cwd,
        )
    )

    assert result2["success"] is True
    assert result2["metadata"]["write_session_final_chunk"] is True

    promoted2, detail2 = promote_write_session_target(state.write_session, cwd=cwd)
    assert promoted2 is True

    final_content = target_file.read_text()
    assert final_content == imports_content + logic_content
    assert "import os" in final_content
    assert "def main" in final_content
