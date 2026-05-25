from __future__ import annotations

from smallctl.harness.task_classifier import task_is_local_coding_target
from smallctl.harness.tool_visibility import filter_tools_for_runtime_state, _append_retry_tool_exposures
from smallctl.harness.run_mode import ensure_remote_tool_profile
from smallctl.state import LoopState


def test_task_is_local_coding_target_detects_py_script() -> None:
    task = "Build a self-contained Python script at `./temp/foo.py` that includes built-in unittest."
    assert task_is_local_coding_target(task) is True


def test_task_is_local_coding_target_rejects_remote_host() -> None:
    task = "SSH to host 192.168.1.10 and build a self-contained Python script at `./temp/foo.py`."
    assert task_is_local_coding_target(task) is False


def test_filter_tools_removes_ssh_for_local_coding() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/foo.py`."
    tools = [
        {"function": {"name": "file_write"}},
        {"function": {"name": "shell_exec"}},
        {"function": {"name": "ssh_exec"}},
        {"function": {"name": "ssh_file_read"}},
    ]
    filtered = filter_tools_for_runtime_state(tools, state=state, mode="loop")
    names = {t["function"]["name"] for t in filtered}
    assert "file_write" in names
    assert "shell_exec" in names
    assert "ssh_exec" not in names
    assert "ssh_file_read" not in names


def test_append_retry_tool_exposures_skips_ssh_for_local_coding() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/foo.py`."
    state.scratchpad["_retry_tool_exposures"] = [
        {"mode": "loop", "tool_name": "ssh_exec"},
        {"mode": "loop", "tool_name": "file_patch"},
    ]
    class MockSpec:
        def openai_schema(self):
            return {"function": {"name": "file_patch"}}
    class MockRegistry:
        def names(self):
            return set()
        def get(self, name):
            if name == "file_patch":
                return MockSpec()
            return None
    harness = type("Harness", (), {"state": state, "registry": MockRegistry()})()
    schemas = [{"function": {"name": "file_write"}}]
    merged = _append_retry_tool_exposures(harness, schemas, mode="loop")
    names = {t["function"]["name"] for t in merged}
    assert "file_write" in names
    assert "file_patch" in names
    assert "ssh_exec" not in names


def test_ensure_remote_tool_profile_skips_network_for_local_coding() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/foo.py`."
    state.active_tool_profiles = ["core"]
    harness = type("Harness", (), {"state": state})()
    ensure_remote_tool_profile(harness)
    assert "network" not in state.active_tool_profiles
