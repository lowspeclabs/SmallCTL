from __future__ import annotations

from smallctl.context.frame_run_rendering import continuation_anchor_lines, render_task_ground_truth
from smallctl.state import LoopState


def test_continuation_anchor_lines_shows_approval_denied_note() -> None:
    """When last_failed_tool has approval_denied=True, a grounding note is rendered."""
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_boundary_previous_task"] = "prior"
    state.scratchpad["_last_task_handoff"] = {
        "last_failed_tool": {
            "tool_name": "shell_exec",
            "error": "User denied approval",
            "approval_denied": True,
        }
    }
    lines = continuation_anchor_lines(state)
    assert any("last_failed_tool=shell_exec" in line for line in lines)
    assert any("denied by user approval" in line for line in lines)
    assert any("re-execute the same call" in line for line in lines)


def test_continuation_anchor_lines_shows_ssh_target_reminder() -> None:
    """When ssh_target is present, a host constraint reminder is rendered."""
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_boundary_previous_task"] = "prior"
    state.scratchpad["_last_task_handoff"] = {
        "ssh_target": {"host": "192.168.1.4", "user": "root"},
    }
    lines = continuation_anchor_lines(state)
    assert any("ssh_target=root@192.168.1.4" in line for line in lines)
    assert any("Remote target is 192.168.1.4" in line for line in lines)
    assert any("use ssh_exec with host=192.168.1.4" in line for line in lines)
    assert any("Do not use any other host" in line for line in lines)


def test_render_task_ground_truth_counts_ssh_exec_mutation() -> None:
    state = LoopState(cwd="/tmp")
    state.tool_history.append(
        'ssh_exec|{"command": "/root/backup.sh", "host": "192.168.1.64", "user": "root"}|success'
    )
    text = render_task_ground_truth(state)
    assert "Mutating operations performed" in text
    assert "ssh_exec" in text


def test_render_task_ground_truth_ignores_ssh_exec_read() -> None:
    state = LoopState(cwd="/tmp")
    state.tool_history.append(
        'ssh_exec|{"command": "cat /root/backup.sh", "host": "192.168.1.64", "user": "root"}|success'
    )
    text = render_task_ground_truth(state)
    assert "Mutating operations performed" not in text
    assert "Only read/observation tools" in text


def test_render_task_ground_truth_counts_shell_exec_mutation() -> None:
    state = LoopState(cwd="/tmp")
    state.tool_history.append(
        'shell_exec|{"command": "systemctl restart nginx"}|success'
    )
    text = render_task_ground_truth(state)
    assert "Mutating operations performed" in text
    assert "shell_exec" in text
