from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from smallctl.state import LoopState, WriteSession
from smallctl.tools.fs import file_write, promote_write_session_target
from smallctl.write_session_fsm import transition_write_session
import pytest
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
    
    # Session ID
    session_id = "ws_repro"
    
    # 1. Start session and write first chunk (imports)
    # We'll seed the session in state
    state.write_session = WriteSession(
        write_session_id=session_id,
        write_target_path=path,
        write_session_intent="replace_file",
        write_session_mode="chunked_author",
        status="open"
    )
    
    imports_content = "import os\nimport sys\n"
    
    # Model omits next_section_name, triggering finalization in tool_outcomes.py 
    # but let's see how file_write handles it.
    result = asyncio.run(file_write(
        path=path,
        content=imports_content,
        section_name="imports",
        next_section_name="", # Final chunk candidate
        write_session_id=session_id,
        state=state,
        cwd=cwd
    ))
    
    assert result["success"] is True
    assert result["metadata"]["write_session_final_chunk"] is True
    
    # In tool_outcomes.py, if final_chunk is True, promote_write_session_target is called
    promoted, detail = promote_write_session_target(state.write_session, cwd=cwd)
    assert promoted is True
    assert target_file.exists()
    assert target_file.read_text() == imports_content
    
    # After promotion, update status to complete (simulating tool_outcomes.py)
    transition_write_session(state.write_session, next_status="complete")
    
    # 2. BUG: Now write second chunk using same session
    logic_content = "def main():\n    print('hello')\n"
    
    # Model realizes it needs more logic
    result2 = asyncio.run(file_write(
        path=path,
        content=logic_content,
        section_name="logic",
        next_section_name="",
        write_session_id=session_id,
        state=state,
        cwd=cwd
    ))
    
    assert result2["success"] is True
    
    # Now Promote again
    promoted2, detail2 = promote_write_session_target(state.write_session, cwd=cwd)
    assert promoted2 is True
    
    # VERIFY: Does the file have both chunks?
    final_content = target_file.read_text()
    
    print(f"\nFinal Content:\n{final_content}")
    
    # If the bug exists, final_content will ONLY be logic_content
    # because the staging file was emptied.
    assert "import os" in final_content, "Imports were lost!"
    assert "def main" in final_content, "Logic was not added!"
