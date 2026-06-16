from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context.retrieval import LexicalRetriever
from smallctl.graph.state import PendingToolCall
from smallctl.graph.tool_write_session_support import _build_schema_repair_message
from smallctl.harness.core_facade import _ssh_host_key_failure_causal_chain
from smallctl.prompts import build_system_prompt
from smallctl.state import ExperienceMemory, LoopState
from smallctl.tools.dispatcher_shell_guards import raw_ssh_shell_block_envelope
from smallctl.tools.fs import file_write
from smallctl.tools.network_ssh_helpers import (
    ssh_command_is_package_manager_install,
    ssh_timeout_suggests_verify,
)


def test_ssh_command_is_package_manager_install_detects_dnf_install() -> None:
    assert ssh_command_is_package_manager_install("dnf install -y webmin") is True
    assert ssh_command_is_package_manager_install("DEBIAN_FRONTEND=noninteractive apt install -y nginx") is True
    assert ssh_command_is_package_manager_install("yum list installed webmin") is False
    assert ssh_command_is_package_manager_install("systemctl status webmin") is False


def test_ssh_timeout_suggests_verify_for_installer() -> None:
    hint = ssh_timeout_suggests_verify("dnf install -y webmin")
    assert "verify the current state" in hint
    assert "systemctl status" in hint


def test_ssh_timeout_suggests_verify_empty_for_non_installer() -> None:
    assert ssh_timeout_suggests_verify("systemctl status webmin") == ""


def test_raw_ssh_keygen_non_removal_guides_to_known_hosts_removal() -> None:
    envelope = raw_ssh_shell_block_envelope("ssh-keygen -t rsa -f /tmp/key", ssh_available=True)

    assert envelope.success is False
    assert envelope.metadata["reason"] == "raw_ssh_shell_blocked"
    assert "ssh-keygen -R <host> -f ~/.ssh/known_hosts" in envelope.error
    assert "ssh_file_read" in envelope.metadata["next_required_tool"]["notes"][1]


def test_system_prompt_warns_known_hosts_is_local() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Remove 192.168.1.161 from the current user's ~/.ssh/known_hosts"

    prompt = build_system_prompt(state, "execute", available_tool_names=["ssh_file_read", "file_read", "shell_exec"])

    assert "known_hosts" in prompt
    assert "local to the harness machine" in prompt
    assert "ssh_file_read" in prompt


def test_file_write_blocks_known_hosts_overwrite(tmp_path) -> None:
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    known_hosts = ssh_dir / "known_hosts"
    known_hosts.write_text("192.168.1.161 ssh-ed25519 AAAAold\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))

    result = asyncio.run(
        file_write(
            path=str(known_hosts),
            content="# replacement\n",
            cwd=str(tmp_path),
            state=state,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "sensitive_known_hosts_overwrite_blocked"
    assert known_hosts.read_text(encoding="utf-8") == "192.168.1.161 ssh-ed25519 AAAAold\n"


def test_schema_repair_message_names_empty_replacement_text() -> None:
    pending = PendingToolCall(tool_name="file_patch", args={"path": "a.txt", "replacement_text": ""})
    harness = SimpleNamespace(registry=None, state=LoopState(cwd="/tmp"))

    message = _build_schema_repair_message(harness, pending, ["target_text", "replacement_text"])

    assert "empty `replacement_text`" in message
    assert "not allowed" in message
    assert "Missing required fields" not in message


def test_ssh_host_key_failure_causal_chain_mentions_blocked_recovery() -> None:
    state = LoopState(cwd="/tmp")
    state.recent_errors = ["Guard tripped: max_consecutive_errors (5)"]
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "root@192.168.1.161": {
            "last_error_class": "host_key_verification",
            "last_error": "Host key verification failed. Offending key in /home/u/.ssh/known_hosts:24",
            "last_command": "ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts",
        }
    }

    chain = _ssh_host_key_failure_causal_chain(
        state,
        {"reason": "raw_ssh_shell_blocked", "message": "Guard tripped: max_consecutive_errors"},
    )

    assert "ssh_exec failed due to host-key mismatch" in chain
    assert "ssh-keygen" in chain
    assert "guard trip" in chain


def test_retrieval_suppresses_irrelevant_memories_during_host_key_recovery() -> None:
    state = LoopState(cwd="/tmp")
    state.last_failure_class = "ssh_host_key_verification"
    state.run_brief.original_task = "Fix SSH host key verification failed for 192.168.1.161"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="irrelevant",
            tool_name="ssh_exec",
            failure_mode="auth_failed",
            notes="Old SSH password failed; ask for a different credential.",
            confidence=0.9,
        ),
        ExperienceMemory(
            memory_id="relevant",
            tool_name="ssh_exec",
            failure_mode="ssh_host_key_verification",
            notes="Use ssh-keygen -R host -f ~/.ssh/known_hosts after approval.",
            confidence=0.9,
        ),
    ]

    bundle = LexicalRetriever().retrieve_bundle(state=state, query="host key verification known_hosts")

    assert [memory.memory_id for memory in bundle.experiences] == ["relevant"]
