from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context.artifacts import ArtifactStore
from smallctl.harness.tool_result_verification import _store_verifier_verdict
from smallctl.harness.tool_results import ToolResultService
from smallctl.challenge_progress import record_verifier_result, initialize_challenge_progress_from_task
from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.graph.progress_guard import _build_progress_stagnation_nudge
from smallctl.graph.tool_outcome_resolution import maybe_apply_terminal_tool_outcome
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def _cleanup_state() -> LoopState:
    state = LoopState()
    state.run_brief.original_task = "Clean up and remove FOG from the remote server."
    return state


def test_passing_verifier_clears_latest_execution_blocker() -> None:
    state = _cleanup_state()
    state.scratchpad["_latest_execution_blocker"] = {
        "tool": "ssh_exec",
        "command": "cat /tmp/webmin-install.log",
        "salient_error": "No such file or directory",
    }
    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "webmin-2.641-1.noarch", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.161", "command": "dnf list installed webmin"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert "_latest_execution_blocker" not in state.scratchpad


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


def test_removal_docker_probe_no_matches_is_verifier_pass() -> None:
    state = LoopState()
    state.run_brief.original_task = "remove the vikunja container from the remote host"
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
        },
        error="",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.89", "command": "docker ps -a | grep vikunja"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict["verifier_kind"] == "removal_absence_probe"
    assert "grep absence probe returned no matches" in verdict["absence_probe_reason"]


def test_removal_docker_rm_failure_is_not_verifier_pass() -> None:
    state = LoopState()
    state.run_brief.original_task = "remove the vikunja container from the remote host"
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": "",
            "stderr": "Error response from daemon: No such container: vikunja\n",
        },
        error="Error response from daemon: No such container: vikunja",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.89", "command": "docker stop vikunja && docker rm vikunja"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict.get("verifier_kind") != "removal_absence_probe"
    assert verdict.get("failure_mode") != "removal_residue"


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


def test_removal_task_heredoc_write_is_not_absence_probe() -> None:
    state = _cleanup_state()
    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "Evidence saved to incident_triage.txt\n", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={
            "host": "192.168.1.89",
            "command": (
                "cat >> /home/stephen/Scripts/Harness-Redo/temp/incident_triage.txt << 'EOF'\n"
                "cleanup/removal notes: find showed no stale fog resources\n"
                "EOF"
            ),
        },
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict.get("verifier_kind") != "removal_absence_probe"
    assert verdict.get("failure_mode") != "removal_residue"


def test_failed_ssh_transport_mutation_is_not_removal_absence_probe() -> None:
    state = LoopState()
    state.run_brief.original_task = (
        "remove entries related to 192.168.1.16 from the known_hosts file of the current user"
    )
    result = ToolEnvelope(
        success=False,
        output={"exit_code": 255, "stdout": "", "stderr": "ssh: connect to host 192.168.1.16 port 22: No route to host"},
        error="Remote SSH command exited with code 255",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={
            "host": "192.168.1.16",
            "user": "root",
            "command": "grep -v '192.168.1.16' /root/.ssh/known_hosts > /tmp/known_hosts && mv /tmp/known_hosts /root/.ssh/known_hosts",
        },
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict.get("verifier_kind") != "removal_absence_probe"
    assert verdict.get("failure_mode") == "environment"


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


def test_docker_inventory_with_non_swarm_stderr_is_partial_diagnostic_pass() -> None:
    state = LoopState()
    state.run_brief.original_task = "Inspect Docker state on the remote host."
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": (
                "CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES\n"
                "abc123         nginx     nginx     1 hour    Up        80/tcp    web\n"
                "DRIVER    VOLUME NAME\n"
                "local     portainer_data\n"
            ),
            "stderr": (
                "Error response from daemon: This node is not a swarm manager. "
                "Use \"docker swarm init\" or \"docker swarm join\" to connect this node to swarm and try again.\n"
            ),
        },
        error="Error response from daemon: This node is not a swarm manager.",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={
            "host": "192.168.1.89",
            "command": "docker ps -a && docker volume ls && docker config ls",
        },
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict["verifier_kind"] == "partial_docker_diagnostic"
    assert verdict["partial_diagnostic"] is True
    assert state.repair_cycle_id == ""


def test_docker_swarm_status_help_with_inventory_is_partial_diagnostic_pass() -> None:
    state = LoopState()
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": (
                "CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES\n"
                "DRIVER    VOLUME NAME\n"
                "NETWORK ID     NAME      DRIVER    SCOPE\n"
                "REPOSITORY     TAG       IMAGE ID  CREATED   SIZE\n"
            ),
            "stderr": (
                "Usage:  docker swarm COMMAND\n\n"
                "Manage Swarm\n\n"
                "Run 'docker swarm COMMAND --help' for more information on a command.\n"
            ),
        },
        error="Usage:  docker swarm COMMAND",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={
            "host": "192.168.1.89",
            "command": "docker ps -a; docker volume ls; docker network ls; docker image ls; docker swarm status",
        },
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict["verifier_kind"] == "partial_docker_diagnostic"


def test_mutating_docker_failure_with_stdout_still_fails() -> None:
    state = LoopState()
    result = ToolEnvelope(
        success=False,
        output={
            "exit_code": 1,
            "stdout": "CONTAINER ID   IMAGE\nabc123 nginx\n",
            "stderr": "Error response from daemon: No such container: abc123\n",
        },
        error="Error response from daemon: No such container: abc123",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.89", "command": "docker rm abc123 && docker ps -a"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict.get("verifier_kind") != "removal_absence_probe"
    assert verdict.get("partial_diagnostic") is not True


def test_zero_unittest_cases_is_verifier_failure_even_with_exit_zero() -> None:
    state = LoopState()
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "",
            "stderr": "----------------------------------------------------------------------\nRan 0 tests in 0.000s\n\nNO TESTS RAN\n",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=result,
        arguments={"command": "python3 ./temp/text_chunker.py"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert "Ran 0 tests" in verdict["acceptance_delta"]["notes"][0] or "NO TESTS RAN" in verdict["acceptance_delta"]["notes"][0]
    assert "verifier_kind" not in verdict


def test_unittest_failure_text_is_failure_even_when_pipeline_exit_is_zero() -> None:
    state = LoopState()
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": (
                "test_exact_size_chunks (text_chunker.TestChunker.test_exact_size_chunks) ... FAIL\n"
                "======================================================================\n"
                "FAIL: test_exact_size_chunks (text_chunker.TestChunker.test_exact_size_chunks)\n"
                "FAILED (failures=4)\n"
            ),
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=result,
        arguments={"command": "python3 -m unittest text_chunker -v 2>&1 | head -50"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict["failure_mode"] == "test"
    assert "FAILED (failures=4)" in verdict["acceptance_delta"]["notes"][0]


def test_py_compile_pass_does_not_clear_prior_failed_runtime_verifier() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build ./temp/text_chunker.py. Run Script when done to verify functionality, fix until complete."
    state.scratchpad["_last_failed_verifier"] = {
        "tool_name": "shell_exec",
        "command": "cd /repo && python3 ./temp/text_chunker.py",
        "summary": ["FAILED (failures=4)"],
        "raw_output": "FAILED (failures=4)",
    }
    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=result,
        arguments={"command": "cd /repo && python3 -m py_compile ./temp/text_chunker.py"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict["failure_mode"] == "insufficient_verifier"
    assert verdict["insufficient_verifier"] is True
    assert "prior failed verifier" in verdict["acceptance_delta"]["notes"][0]


def test_diagnostic_pass_does_not_overwrite_install_service_status_failure() -> None:
    """Regression for 35b29086: journalctl success must not clear a failed systemctl status."""
    state = LoopState()
    state.run_brief.original_task = "ssh root@192.168.1.89 and try to get docker running"
    state.scratchpad["_last_failed_verifier"] = {
        "tool_name": "ssh_exec",
        "command": "systemctl status docker.service",
        "summary": ["docker.service failed with exit code"],
        "raw_output": "docker.service: failed",
    }
    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "dockerd logs...", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.89", "command": "journalctl -xeu docker.service"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict["failure_mode"] == "insufficient_verifier"
    assert verdict["insufficient_verifier"] is True
    assert "systemctl status docker.service" in verdict["acceptance_delta"]["notes"][0]
    assert "journalctl" in verdict["acceptance_delta"]["notes"][0]


def test_equal_strength_diagnostic_pass_does_not_overwrite_service_status_failure() -> None:
    """A read-only diagnostic must not clear a functional service-status failure of equal strength."""
    state = LoopState()
    state.run_brief.original_task = "ssh root@192.168.1.89 and try to get docker running"
    state.scratchpad["_last_failed_verifier"] = {
        "tool_name": "ssh_exec",
        "command": "systemctl start docker.service",
        "summary": ["docker.service failed to start"],
        "raw_output": "docker.service: failed",
    }
    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.89", "command": "cat /etc/docker/daemon.json"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict["failure_mode"] == "insufficient_verifier"
    assert verdict["insufficient_verifier"] is True
    assert "daemon.json" in verdict["acceptance_delta"]["notes"][0]


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
        metadata={"ssh_auth_mode": "password", "ssh_auth_transport": "sshpass_file"},
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


def test_approval_denied_verifier_is_needs_human_and_does_not_start_repair() -> None:
    state = LoopState()
    result = ToolEnvelope(
        success=False,
        output={"exit_code": None, "stdout": "", "stderr": ""},
        error="Shell execution denied by user.",
        metadata={"approval_denied": True},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=result,
        arguments={"command": "cd /repo && timeout 3 python pong.py || true"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "needs_human"
    assert verdict["failure_mode"] == "approval_denied"
    assert verdict["approval_denied"] is True
    assert verdict["acceptance_delta"]["status"] == "pending"
    assert "denied by user" in verdict["acceptance_delta"]["notes"][0].lower()
    assert state.repair_cycle_id == ""


def test_approval_denied_tool_outcome_stops_generation() -> None:
    state = LoopState()
    graph_state = GraphRunState(loop_state=state, thread_id="approval-denied", run_mode="loop")
    record = ToolExecutionRecord(
        operation_id="op-denied",
        tool_name="shell_exec",
        args={"command": "python3 test.py"},
        tool_call_id="call-denied",
        result=ToolEnvelope(
            success=False,
            error="Shell execution denied by user.",
            metadata={"approval_denied": True, "command": "python3 test.py"},
        ),
    )
    harness = SimpleNamespace(state=state)
    deps = SimpleNamespace(harness=harness, event_handler=None)

    handled = asyncio.run(
        maybe_apply_terminal_tool_outcome(
            graph_state,
            deps,
            record,
            chat_mode=False,
        )
    )

    assert handled is True
    assert graph_state.final_result["status"] == "denied"
    assert graph_state.final_result["message"] == "Shell execution denied by user."
    assert graph_state.interrupt_payload is None


def test_tool_outcome_ignores_non_dict_metadata() -> None:
    state = LoopState()
    graph_state = GraphRunState(loop_state=state, thread_id="none-metadata", run_mode="loop")
    record = ToolExecutionRecord(
        operation_id="op-none-metadata",
        tool_name="shell_exec",
        args={"command": "python3 test.py"},
        tool_call_id="call-none-metadata",
        result=ToolEnvelope(
            success=False,
            error="command failed",
            metadata=None,
        ),
    )
    harness = SimpleNamespace(state=state)
    deps = SimpleNamespace(harness=harness, event_handler=None)

    handled = asyncio.run(
        maybe_apply_terminal_tool_outcome(
            graph_state,
            deps,
            record,
            chat_mode=False,
        )
    )

    assert handled is False
    assert graph_state.final_result is None


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


def test_diagnostic_health_analysis_task_is_classified_as_diagnostic() -> None:
    from smallctl.diagnostic_tasks import diagnostic_failure_task

    state = LoopState()
    state.run_brief.original_task = (
        "ssh root@192.168.1.89 and perform Task 1: System Health & Resource Analysis. "
        "The server is reported as slow. Determine current CPU, memory, and disk utilization."
    )

    assert diagnostic_failure_task(state) is True


def test_diagnostic_task_failed_probe_does_not_enter_repair_phase() -> None:
    """Pure diagnostic tasks should not enter repair phase when a probe fails."""
    state = LoopState()
    state.run_brief.original_task = (
        "System Health & Resource Analysis: determine CPU, memory, and disk utilization on 192.168.1.89"
    )
    failed = ToolEnvelope(
        success=False,
        output={"exit_code": 2, "stdout": "", "stderr": "bash: syntax error"},
        error="Remote SSH command exited with code 2",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=failed,
        arguments={"host": "192.168.1.89", "command": "uptime && vmstat 1 5"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert state.repair_cycle_id == ""
    assert state.scratchpad.get("_contract_phase") != "repair"


def test_report_back_task_is_classified_as_diagnostic() -> None:
    from smallctl.diagnostic_tasks import diagnostic_failure_task

    state = LoopState()
    state.run_brief.original_task = (
        "read /home/stephen/Scripts/Harness-Redo/temp/proxmox-manager/AGENTS.md, "
        "then load the proxmox cli and report back running lxc and vms"
    )

    assert diagnostic_failure_task(state) is True


def test_report_back_task_failed_probe_does_not_enter_repair_phase() -> None:
    """A pure 'report back' observation task should not enter repair when a probe fails."""
    state = LoopState()
    state.run_brief.original_task = (
        "read /home/stephen/Scripts/Harness-Redo/temp/proxmox-manager/AGENTS.md, "
        "then load the proxmox cli and report back running lxc and vms"
    )
    failed = ToolEnvelope(
        success=False,
        output={"exit_code": 4, "stdout": "", "stderr": "hostname lookup 'pve1' failed"},
        error="hostname lookup 'pve1' failed",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=failed,
        arguments={"command": "python scripts/Proxmox-cli.py lxcs list --node pve1"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert state.repair_cycle_id == ""
    assert state.scratchpad.get("_contract_phase") != "repair"


def test_report_back_task_corrected_probe_overwrites_failed_probe() -> None:
    """A different successful probe for a 'report back' task must not be rejected as insufficient."""
    state = LoopState()
    state.run_brief.original_task = (
        "read /home/stephen/Scripts/Harness-Redo/temp/proxmox-manager/AGENTS.md, "
        "then load the proxmox cli and report back running lxc and vms"
    )
    failed = ToolEnvelope(
        success=False,
        output={"exit_code": 4, "stdout": "", "stderr": "hostname lookup 'pve1' failed"},
        error="hostname lookup 'pve1' failed",
    )
    _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=failed,
        arguments={"command": "python scripts/Proxmox-cli.py lxcs list --node pve1"},
    )

    success = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "## LXCs on pve\n\n- 0: ...", "stderr": ""},
    )
    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=success,
        arguments={"command": "python scripts/Proxmox-cli.py lxcs list --node pve"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict.get("insufficient_verifier") is not True
    assert state.repair_cycle_id == ""


def test_stronger_verifier_overwrites_prior_failed_verifier() -> None:
    """A strictly stronger passing verifier should overwrite a prior weaker failure."""
    state = LoopState()
    state.run_brief.original_task = "Build and run the script."
    state.scratchpad["_last_failed_verifier"] = {
        "tool_name": "shell_exec",
        "command": "cat /tmp/debug.log",
        "summary": ["missing log"],
        "raw_output": "missing log",
    }
    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "script output ok", "stderr": ""},
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=result,
        arguments={"command": "cd /repo && python3 ./temp/script.py"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict.get("insufficient_verifier") is not True


def test_diagnostic_task_later_probe_overwrites_failed_diagnostic_verifier() -> None:
    """A different successful diagnostic probe should not be rejected as 'insufficient'."""
    state = LoopState()
    state.run_brief.original_task = (
        "System Health & Resource Analysis: determine CPU, memory, and disk utilization on 192.168.1.89"
    )
    failed = ToolEnvelope(
        success=False,
        output={"exit_code": 2, "stdout": "", "stderr": "bash: syntax error"},
        error="Remote SSH command exited with code 2",
    )
    _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=failed,
        arguments={"host": "192.168.1.89", "command": "uptime && vmstat 1 5"},
    )

    success = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": " 20:14:55 up 51 days\n", "stderr": ""},
    )
    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=success,
        arguments={"host": "192.168.1.89", "command": "free -h"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert verdict.get("insufficient_verifier") is not True
    assert state.repair_cycle_id == ""


def _make_harness(tmp_path) -> SimpleNamespace:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-1"
    state.run_brief.original_task = "Inspect the workspace"
    state.artifacts = {}
    state.retrieval_cache = []
    state.scratchpad = {}
    return SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path, "run-1"),
        context_policy=SimpleNamespace(tool_result_inline_token_limit=800, artifact_summarization_threshold=999999),
        summarizer_client=None,
        summarizer=None,
        _current_user_task=lambda: state.run_brief.original_task,
        _runlog=lambda *args, **kwargs: None,
    )


def test_failed_tool_result_records_one_recent_error(tmp_path) -> None:
    """A single failed tool dispatch must create exactly one recent_errors entry."""
    harness = _make_harness(tmp_path)
    service = ToolResultService(harness)
    result = ToolEnvelope(
        success=False,
        error="command not found",
        metadata={"exit_code": 127, "stderr": "bash: not_found: command not found"},
    )

    asyncio.run(
        service.record_result(
            tool_name="shell_exec",
            tool_call_id="call-fail-1",
            result=result,
            arguments={"command": "not_found"},
            operation_id="op-fail-1",
        )
    )

    assert len(harness.state.recent_errors) == 1
    assert "command not found" in harness.state.recent_errors[0]


def _make_vm_creation_state(vm_name: str) -> LoopState:
    state = LoopState()
    state.run_brief.original_task = f"Create a Debian VM named '{vm_name}' on Proxmox"
    initialize_challenge_progress_from_task(state, state.run_brief.original_task)
    return state


def test_generic_inventory_command_cannot_verify_vm_creation() -> None:
    """Commands like `nodes list` must not count as verification for a VM creation objective."""
    vm_name = "3.6-35-a3b-tester"
    state = _make_vm_creation_state(vm_name)

    record_verifier_result(
        state,
        tool_name="ssh_exec",
        command="python scripts/Proxmox-cli.py nodes list",
        verifier_kind="diagnostic",
        verdict="pass",
        exit_code=0,
        stdout="",
        stderr="",
    )

    assert state.challenge_progress.verified_after_last_change is False


def test_vm_status_command_with_resource_name_verifies_vm_creation() -> None:
    """A status/readback command that names the requested VM must count as verification."""
    vm_name = "3.6-35-a3b-tester"
    state = _make_vm_creation_state(vm_name)

    record_verifier_result(
        state,
        tool_name="ssh_exec",
        command=f"qm status {vm_name}",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
        stdout=f"VM {vm_name} status: running\n",
        stderr="",
    )

    assert state.challenge_progress.verified_after_last_change is True


def test_mixed_diagnostic_then_fix_task_is_not_classified_as_diagnostic() -> None:
    """Broad diagnostic markers must not override mutation/remediation intent."""
    from smallctl.diagnostic_tasks import diagnostic_failure_task

    state = LoopState()
    state.run_brief.original_task = (
        "show me the latest failures in the log and then fix the broken service"
    )

    assert diagnostic_failure_task(state) is False


def test_pure_diagnostic_task_still_classified_as_diagnostic() -> None:
    """Pure report-only tasks should remain classified as diagnostic."""
    from smallctl.diagnostic_tasks import diagnostic_failure_task

    state = LoopState()
    state.run_brief.original_task = (
        "show me the latest failures in the log and report back what you find"
    )

    assert diagnostic_failure_task(state) is True
