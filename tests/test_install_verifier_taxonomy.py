from __future__ import annotations

from types import SimpleNamespace

from smallctl.challenge_progress import _verifier_matches_user_objective
from smallctl.harness.core_facade import _run_metric_flags
from smallctl.harness.tool_result_verification_helpers import (
    command_is_install_absence_probe,
    install_absence_probe_confirmed,
    verifier_kind_for_command,
    verifier_strength,
)
from smallctl.harness.tool_result_verification_semantic import _install_task_requires_strong_verifier
from smallctl.harness.tool_result_verification_store import (
    _store_verifier_verdict,
    _verifier_path_failure_is_false_negative,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.prompts import build_system_prompt
from smallctl.state import LoopState


def test_install_verifier_taxonomy_recognizes_strong_verifiers() -> None:
    assert verifier_kind_for_command("systemctl status fogservice") == "install_service_status"
    assert verifier_kind_for_command("dpkg -l | grep -i fog") == "install_package_status"
    assert verifier_kind_for_command("ss -tlnp | grep :80") == "install_port_listener"
    assert verifier_kind_for_command("fog --version") == "install_version_command"
    assert verifier_kind_for_command("which fog") == "install_version_command"
    assert verifier_strength("install_service_status") > verifier_strength("diagnostic")

    state = LoopState()
    state.run_brief.original_task = "Install FOG on Debian"
    weak, reason = _install_task_requires_strong_verifier(state, command="ls -la /opt/fog")
    assert weak is True
    strong, _ = _install_task_requires_strong_verifier(state, command="systemctl status fogproject")
    assert strong is False


def test_install_os_release_diagnostic_is_not_rejected_as_weak_verifier() -> None:
    state = LoopState()
    state.run_brief.original_task = "Install Vikunja on the remote host"
    result = ToolEnvelope(
        success=True,
        status="success",
        output={
            "exit_code": 0,
            "stdout": 'NAME="CentOS Stream"\nVERSION="9"\nID="centos"\n',
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.161", "command": "cat /etc/os-release"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"
    assert not verdict.get("insufficient_verifier")


def test_failed_ssh_exec_stores_generic_latest_blocker() -> None:
    state = LoopState()
    state.run_brief.original_task = "Install Vikunja on the remote host"
    result = ToolEnvelope(
        success=False,
        status="failed",
        output={
            "exit_code": 1,
            "stdout": "Vikunja 18 kB/s | 27 kB\n",
            "stderr": "Errors during downloading metadata for repository 'vikunja': repomd.xml 404\n",
        },
        error="Errors during downloading metadata for repository 'vikunja': repomd.xml 404",
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.161", "command": "dnf install -y vikunja-server"},
    )

    assert verdict is not None
    blocker = verdict.get("latest_blocker")
    assert isinstance(blocker, dict)
    assert "repomd.xml 404" in blocker["salient_error"]
    assert state.scratchpad["_latest_execution_blocker"]["salient_error"] == blocker["salient_error"]


def test_evidence_anchored_diagnosis_rule_in_install_prompt() -> None:
    state = LoopState()
    state.run_brief.original_task = "Install FOG on Debian"
    state.scratchpad["_model_name"] = "qwen3-14b"
    prompt = build_system_prompt(state, "execute", available_tool_names=["shell_exec", "ssh_exec", "web_search"])
    assert "DIAGNOSIS RULE" in prompt
    assert "observed line" in prompt
    assert "next differentiating action" in prompt


def test_install_absence_probe_exit_one_with_no_matches_is_pass() -> None:
    assert command_is_install_absence_probe("dpkg -l | grep -i fog") is True
    assert command_is_install_absence_probe("echo hello | grep world") is False
    assert install_absence_probe_confirmed(
        command="dpkg -l | grep -i fog",
        exit_code=1,
        stdout="",
        stderr="",
    ) is True
    assert install_absence_probe_confirmed(
        command="dpkg -l | grep -i fog",
        exit_code=0,
        stdout="ii  fog-project ...",
        stderr="",
    ) is False

    state = LoopState()
    state.run_brief.original_task = "Install FOG on Debian"
    result = ToolEnvelope(
        success=True,
        status="success",
        output={"exit_code": 1, "stdout": "", "stderr": ""},
        metadata={"command": "dpkg -l | grep -i fog"},
    )
    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.10", "command": "dpkg -l | grep -i fog"},
    )
    assert verdict is not None
    assert verdict["verdict"] == "pass"


def test_verifier_fails_ssh_exec_with_interactive_prompt() -> None:
    """An ssh_exec installer command that returns a (y/N) prompt must fail verification."""
    state = LoopState()
    state.run_brief.original_task = "Install Webmin on the remote host."
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "Setting up webmin...\n(y/N) ? ",
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.162", "command": "apt-get install -y webmin"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert "interactive" in str(verdict.get("failure_mode", "")).lower()


def test_verifier_passes_install_followed_by_verification_command() -> None:
    """A successful install followed by dpkg -l verification passes."""
    state = LoopState()
    state.run_brief.original_task = "Install Webmin on the remote host."
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "ii  webmin  2.000-1  all  web-based administration interface\n",
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.162", "command": "dpkg -l | grep -w webmin"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"


def test_non_install_ssh_tasks_unaffected_by_interactive_prompt_check() -> None:
    """Non-install SSH commands are not blocked by interactive prompt detection."""
    state = LoopState()
    state.run_brief.original_task = "Check disk space on remote host."
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       20G   10G   10G  50% /\n",
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.162", "command": "df -h /"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"


def test_verifier_path_false_negative_when_recent_ssh_file_write() -> None:
    state = LoopState()
    state.recent_messages.append(
        SimpleNamespace(
            role="tool",
            name="ssh_file_write",
            metadata={
                "success": True,
                "host": "pi.hole",
                "path": "/tmp/pihole-install.sh",
            },
        )
    )
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="",
        stderr="bash: /tmp/pihole-install.sh: No such file or directory",
    ) is True


def test_verifier_path_false_negative_no_recent_write() -> None:
    state = LoopState()
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="",
        stderr="bash: /tmp/pihole-install.sh: No such file or directory",
    ) is False


def test_verifier_path_false_negative_different_host() -> None:
    state = LoopState()
    state.recent_messages.append(
        SimpleNamespace(
            role="tool",
            name="ssh_file_write",
            metadata={
                "success": True,
                "host": "other.host",
                "path": "/tmp/pihole-install.sh",
            },
        )
    )
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="",
        stderr="bash: /tmp/pihole-install.sh: No such file or directory",
    ) is False


def test_verifier_path_false_negative_no_path_in_output() -> None:
    state = LoopState()
    state.recent_messages.append(
        SimpleNamespace(
            role="tool",
            name="ssh_file_write",
            metadata={
                "success": True,
                "host": "pi.hole",
                "path": "/tmp/pihole-install.sh",
            },
        )
    )
    assert _verifier_path_failure_is_false_negative(
        state,
        host="pi.hole",
        command="bash /tmp/pihole-install.sh",
        stdout="some other error",
        stderr="permission denied",
    ) is False


def test_deliverable_verified_false_on_cancel_without_passing_verifier() -> None:
    state = LoopState()
    challenge_progress = {"verified_after_last_change": True}
    flags = _run_metric_flags(state, challenge_progress, status="cancelled")
    assert flags["deliverable_verified"] is False


def test_deliverable_verified_true_on_cancel_with_passing_verifier() -> None:
    state = LoopState()
    state.last_verifier_verdict = {"verdict": "pass", "command": "pihole status"}
    challenge_progress = {"verified_after_last_change": True}
    flags = _run_metric_flags(state, challenge_progress, status="cancelled")
    assert flags["deliverable_verified"] is True


def test_deliverable_verified_preserved_on_non_cancelled() -> None:
    state = LoopState()
    challenge_progress = {"verified_after_last_change": True}
    flags = _run_metric_flags(state, challenge_progress, status="completed")
    assert flags["deliverable_verified"] is True


def test_deliverable_verified_false_when_not_verified() -> None:
    state = LoopState()
    challenge_progress = {"verified_after_last_change": False}
    flags = _run_metric_flags(state, challenge_progress, status="completed")
    assert flags["deliverable_verified"] is False


def test_completed_test_only_coding_followup_is_not_diagnostic() -> None:
    state = LoopState()
    challenge_progress = {
        "task_category": "coding",
        "code_change_count": 0,
        "verified_after_last_change": False,
        "last_verifier_verdict": "pass",
        "last_verifier_kind": "test_suite",
    }

    flags = _run_metric_flags(state, challenge_progress, status="completed")

    assert flags["deliverable_verified"] is True
    assert flags["diagnostic_only"] is False


def test_completed_diagnostic_run_target_without_changes_stays_diagnostic() -> None:
    state = LoopState()
    challenge_progress = {
        "task_category": "coding",
        "code_change_count": 0,
        "verified_after_last_change": False,
        "last_verifier_verdict": "pass",
        "last_verifier_kind": "run_target",
    }

    flags = _run_metric_flags(state, challenge_progress, status="completed")

    assert flags["deliverable_verified"] is False
    assert flags["diagnostic_only"] is True


def test_verifier_matches_user_objective_allows_strong_verifier_for_install() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Install Pi-hole")
    assert _verifier_matches_user_objective(state, "pihole status") is True
    assert _verifier_matches_user_objective(state, "pihole-FTL --version") is True


def test_verifier_matches_user_objective_blocks_weak_verifier_for_install() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Install Pi-hole")
    assert _verifier_matches_user_objective(state, "ls -la /tmp/install.sh") is False
    assert _verifier_matches_user_objective(state, "test -f /tmp/install.sh") is False
    assert _verifier_matches_user_objective(state, "cat /tmp/install.sh") is False


def test_verifier_matches_user_objective_allows_any_verifier_for_non_install() -> None:
    state = LoopState()
    state.run_brief = SimpleNamespace(original_task="Fix the bug in utils.py")
    assert _verifier_matches_user_objective(state, "ls -la /tmp/test.py") is True
