from __future__ import annotations

import pytest

from smallctl.context.frame_run_rendering import render_run_brief
from smallctl.harness.task_classifier import classify_task_mode
from smallctl.harness.task_intent import infer_requested_tool_name
from smallctl.harness.task_classifier_support import (
    task_is_local_ssh_file_target,
    task_is_local_system_target,
)
from smallctl.prompts import build_system_prompt
from smallctl.state import LoopState
from smallctl.tools.profiles import classify_tool_profiles, NETWORK_PROFILE, NETWORK_READ_PROFILE


LOCAL_KNOWN_HOSTS_TASKS = [
    "remove entries related to 192.168.1.16 from the known_hosts file of the current user",
    "delete any lines matching 10.0.0.5* from ~/.ssh/known_hosts",
    "clean up authorized_keys for this user on this host",
    "remove 192.168.1.16 from my known hosts file",
    "update this user's ~/.ssh/authorized_keys to remove the old key",
]

REMOTE_SSH_TASKS = [
    "ssh to 192.168.1.16 and remove old entries from known_hosts",
    "connect to remote host 192.168.1.16 and clean up authorized_keys",
    "over ssh, update known_hosts on 192.168.1.16",
]


@pytest.mark.parametrize("task", LOCAL_KNOWN_HOSTS_TASKS)
def test_local_ssh_file_task_classifies_as_local_execute(task: str) -> None:
    assert classify_task_mode(task) == "local_execute"


@pytest.mark.parametrize("task", LOCAL_KNOWN_HOSTS_TASKS)
def test_local_ssh_file_task_is_detected_as_local_ssh_file_target(task: str) -> None:
    assert task_is_local_ssh_file_target(task) is True
    assert task_is_local_system_target(task) is True


@pytest.mark.parametrize("task", REMOTE_SSH_TASKS)
def test_explicit_remote_ssh_task_classifies_as_remote_execute(task: str) -> None:
    assert classify_task_mode(task) == "remote_execute"
    assert task_is_local_ssh_file_target(task) is False


def test_known_hosts_removal_intent_is_shell_exec_not_ssh_exec() -> None:
    task = "remove entries related to 192.168.1.16 from the known_hosts file of the current user"

    class _Harness:
        pass

    assert infer_requested_tool_name(_Harness(), task) == "shell_exec"


def test_known_hosts_read_intent_is_read_file() -> None:
    task = "find any lines matching 192.168.1.16 in the current user's known_hosts file"

    class _Harness:
        pass

    assert infer_requested_tool_name(_Harness(), task) == "read_file"


def test_local_ssh_file_profiles_exclude_network() -> None:
    task = "remove entries related to 192.168.1.16 from the known_hosts file of the current user"
    profiles = classify_tool_profiles(task)

    assert NETWORK_PROFILE not in profiles
    assert NETWORK_READ_PROFILE not in profiles
    assert "mutate" in profiles or "core" in profiles


def test_remote_ssh_profiles_include_network() -> None:
    task = "ssh to 192.168.1.16 and remove old entries from known_hosts"
    profiles = classify_tool_profiles(task)

    assert NETWORK_PROFILE in profiles


def test_run_brief_includes_local_scope_line() -> None:
    state = LoopState()
    state.run_brief.original_task = (
        "remove entries related to 192.168.1.16 from the known_hosts file of the current user"
    )

    text = render_run_brief(state)

    assert "Scope: local user task" in text
    assert "prefer local file tools and shell_exec over ssh_exec" in text


def test_system_prompt_includes_local_scope_preference() -> None:
    state = LoopState()
    state.run_brief.original_task = (
        "remove entries related to 192.168.1.16 from the known_hosts file of the current user"
    )

    prompt = build_system_prompt(state, "execute")

    assert "prefer local file tools and shell_exec over ssh_exec" in prompt
