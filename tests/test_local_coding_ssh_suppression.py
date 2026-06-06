from __future__ import annotations

from smallctl.harness.task_classifier import classify_task_mode, task_is_local_coding_target
from smallctl.harness.task_classifier_support import has_remote_execution_target
from smallctl.harness.tool_visibility import filter_tools_for_runtime_state, _append_retry_tool_exposures
from smallctl.harness.run_mode import ensure_remote_tool_profile
from smallctl.remote_scope import remote_scope_is_active
from smallctl.state import LoopState


def test_task_is_local_coding_target_detects_py_script() -> None:
    task = "Build a self-contained Python script at `./temp/foo.py` that includes built-in unittest."
    assert task_is_local_coding_target(task) is True


def test_task_is_local_coding_target_detects_py_script_without_temp_prefix() -> None:
    task = "Create a Python script named system_triage_report.py that generates an HTML report at ./temp/system_triage_report.html."
    assert task_is_local_coding_target(task) is True


def test_task_is_local_coding_target_rejects_remote_host() -> None:
    task = "SSH to host 192.168.1.10 and build a self-contained Python script at `./temp/foo.py`."
    assert task_is_local_coding_target(task) is False


def test_remote_sysadmin_task_with_ssh_config_safety_constraint_keeps_remote_target() -> None:
    task = """Sysadmin Challenge: Disk, Process, Network, and Configuration RCA

Target host: root@192.168.1.89

Task:
Connect to the remote Linux host over SSH and perform a root-cause style health investigation.

8. Create a report at:

/root/rca-health-investigation-report.txt

Constraints:
- Do not reboot the host.
- Do not modify SSH configuration.
- Do not modify files except creating or overwriting /root/rca-health-investigation-report.txt.
- Prefer read-only commands.
"""
    assert has_remote_execution_target(task) is True
    assert classify_task_mode(task) == "remote_execute"


def test_remote_sysadmin_root_report_path_keeps_remote_scope_active() -> None:
    state = LoopState()
    state.task_mode = "remote_execute"
    state.run_brief.original_task = """Sysadmin Challenge: Disk, Process, Network, and Configuration RCA

Target host: root@192.168.1.89
Connect to the remote Linux host over SSH.
Create a report at:

/root/rca-health-investigation-report.txt
"""

    assert remote_scope_is_active(state) is True


def test_explicit_no_ssh_constraint_still_suppresses_remote_target() -> None:
    task = "Do not SSH to 192.168.1.10; write a local summary from the provided notes."
    assert has_remote_execution_target(task) is False


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
