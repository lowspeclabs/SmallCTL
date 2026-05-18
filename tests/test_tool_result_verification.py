from __future__ import annotations

from smallctl.harness.tool_result_verification import _store_verifier_verdict
from smallctl.graph.progress_guard import _build_progress_stagnation_nudge
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def _cleanup_state() -> LoopState:
    state = LoopState()
    state.run_brief.original_task = "Clean up and remove FOG from the remote server."
    return state


def test_removal_grep_no_matches_is_verifier_pass() -> None:
    state = _cleanup_state()
    result = ToolEnvelope(
        success=False,
        output={"exit_code": 1, "stdout": "", "stderr": ""},
        error="",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.63", "command": "systemctl list-units --all | grep -i fog"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict["verifier_kind"] == "removal_absence_probe"
    assert verdict["failure_mode"] == ""
    assert state.repair_cycle_id == ""


def test_removal_ls_no_such_file_is_verifier_pass() -> None:
    state = _cleanup_state()
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 2,
            "stdout": "",
            "stderr": "ls: cannot access '/etc/nfs.conf.d/fog-nfs.conf': No such file or directory",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.63", "command": "ls -l /etc/nfs.conf.d/fog-nfs.conf"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert "ls reported" in verdict["absence_probe_reason"]


def test_removal_find_matches_are_verifier_failure() -> None:
    state = _cleanup_state()
    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "/opt/fog\n/var/www/fog\n", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.63", "command": "find /opt /var/www -iname '*fog*' -print"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict["failure_mode"] == "removal_residue"
    assert verdict["verifier_kind"] == "removal_absence_probe"


def test_non_removal_grep_exit_one_still_fails() -> None:
    state = LoopState()
    state.run_brief.original_task = "Check whether FOG is installed."
    result = ToolEnvelope(
        success=False,
        output={"exit_code": 1, "stdout": "", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.63", "command": "systemctl list-units --all | grep -i fog"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert "verifier_kind" not in verdict


def test_diagnostic_status_probe_exit_one_is_pass() -> None:
    """systemctl status returning exit 1 with 'not found' is informational."""
    state = LoopState()
    state.run_brief.original_task = "Check the status of the fog pxe install."
    result = ToolEnvelope(
        success=False,
        output={"exit_code": 1, "stdout": "", "stderr": "Unit fog-pxe.service could not be found."},
        error="Unit fog-pxe.service could not be found.",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.89", "command": "systemctl status fog-pxe || service fog-pxe status"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert state.repair_cycle_id == ""


def test_diagnostic_apt_probe_exit_one_is_pass() -> None:
    """apt list returning exit 1 with empty output (no matches) is informational."""
    state = LoopState()
    state.run_brief.original_task = "Check whether FOG is installed."
    result = ToolEnvelope(
        success=False,
        output={"exit_code": 1, "stdout": "", "stderr": ""},
        error="Remote SSH command exited with code 1",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.89", "command": "apt list --installed 2>/dev/null | grep -i fog"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert state.repair_cycle_id == ""


def test_nested_raw_ssh_failure_does_not_record_auth_recovery_state() -> None:
    state = LoopState()
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 255,
            "stdout": "",
            "stderr": "root@192.168.1.89: Permission denied (publickey,password).",
        },
        error="root@192.168.1.89: Permission denied (publickey,password).",
        metadata={"ssh_auth_mode": "password", "ssh_auth_transport": "sshpass_env"},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={
            "host": "192.168.1.89",
            "user": "root",
            "password": "Temp@Pass",
            "command": "ssh root@192.168.1.89 whoami",
        },
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert "_ssh_auth_recovery_state" not in state.scratchpad


def test_latest_blocker_tracks_fogproject_account_over_interactive_prompt() -> None:
    state = LoopState()
    state.thread_id = "fog-run"
    interactive = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": "What version of Linux would you like to run the installation for?\n * Are you sure you wish to continue (Y/N)\n * Sorry, answer not recognized\n",
            "stderr": "",
        },
        error="Remote SSH command exited with code 1",
    )
    first = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=interactive,
        arguments={"host": "192.168.1.89", "command": "cd /opt/fogproject/bin && bash installfog.sh"},
    )
    assert first is not None
    assert first["latest_blocker"]["is_interactive_prompt"] is True
    state.stagnation_counters["no_progress"] = 2
    state.stagnation_counters["repeat_command"] = 3

    account_exists = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": (
                ("Installing package output...\n" * 40)
                +
                'The account "fogproject" already exists and has been used to\n'
                "log in to this server.\n"
                'Please remove the account "fogproject" manually before running\n'
                "the installer again, or set the system username yourself.\n"
            ),
            "stderr": "",
        },
        error="Remote SSH command exited with code 1",
    )
    second = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=account_exists,
        arguments={"host": "192.168.1.89", "command": "cd /opt/fogproject/bin && bash installfog.sh -Y"},
    )

    assert second is not None
    blocker = state.scratchpad["_latest_execution_blocker"]
    assert blocker["blocker_class"] == "fogproject_account_exists"
    assert blocker["is_interactive_prompt"] is False
    assert "fogproject" in blocker["salient_error"]
    assert state.stagnation_counters["no_progress"] == 0
    assert state.stagnation_counters["repeat_command"] == 0


def test_stagnation_nudge_uses_latest_noninteractive_blocker() -> None:
    state = LoopState()
    state.run_brief.original_task = "fix the service files, and resume the fog install"
    state.scratchpad["_latest_execution_blocker"] = {
        "command": "cd /opt/fogproject/bin && bash installfog.sh -Y",
        "blocker_class": "fogproject_account_exists",
        "salient_error": 'The account "fogproject" already exists. Please remove the account "fogproject".',
        "is_interactive_prompt": False,
    }
    harness = type("Harness", (), {"state": state})()

    nudge = _build_progress_stagnation_nudge(harness)

    assert 'The account "fogproject" already exists' in nudge
    assert "Do not keep applying stale interactive-prompt/stdin advice" in nudge
