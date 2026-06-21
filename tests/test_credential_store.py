from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

from smallctl.harness.credential_store import CredentialStore
from smallctl.harness.core_facade import _write_checkpoint_file
from smallctl.harness.tool_result_ssh_memory import _remember_session_ssh_target
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.network_ssh_helpers import build_ssh_command
from smallctl.tools.shell_foreground import _feed_sudo_password_to_process


def test_credential_store_fingerprint_is_stable() -> None:
    store = CredentialStore()
    fp1 = store.fingerprint("secret")
    fp2 = store.fingerprint("secret")
    fp3 = store.fingerprint("other")
    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 64


def test_credential_store_round_trip_ssh_password() -> None:
    store = CredentialStore()
    fp = store.set_ssh_password("Example.Test", "root", "secret")
    assert store.get_ssh_password("example.test", "root") == "secret"
    assert store.get_ssh_password_by_fingerprint(fp) == "secret"
    assert store.get_ssh_password("other.test", "root") is None


def test_credential_store_sudo_password_not_serialized() -> None:
    store = CredentialStore()
    store.set_sudo_password("sudo-secret")
    assert store.get_sudo_password() == "sudo-secret"
    assert not hasattr(store, "to_dict")


def test_build_ssh_command_uses_file_not_env_for_password() -> None:
    cmd, env, password_file_path = build_ssh_command(
        host="example.test",
        command="whoami",
        user="root",
        port=22,
        identity_file=None,
        password="secret",
    )
    assert env is None
    assert password_file_path is not None
    assert os.path.exists(password_file_path)
    try:
        assert cmd.startswith("sshpass -f")
        assert "SSHPASS" not in cmd
        mode = os.stat(password_file_path).st_mode
        assert mode & 0o777 == 0o600
        content = open(password_file_path).read()
        assert content == "secret"
    finally:
        os.unlink(password_file_path)


def test_build_ssh_command_no_password_creates_no_file() -> None:
    cmd, env, password_file_path = build_ssh_command(
        host="example.test",
        command="whoami",
        user="root",
        port=22,
        identity_file=None,
        password=None,
    )
    assert password_file_path is None
    assert env is None
    assert cmd.startswith("ssh")


def test_to_dict_redacts_plaintext_passwords() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_session_ssh_targets"] = {
        "example.test": {
            "host": "example.test",
            "user": "root",
            "password": "secret",
        }
    }
    state.tool_execution_records["op-1"] = {
        "tool_name": "ssh_exec",
        "args": {"host": "example.test", "user": "root", "password": "secret"},
    }
    payload = state.to_dict()
    target = payload["scratchpad"]["_session_ssh_targets"]["example.test"]
    assert target["password"] != "secret"
    assert target["password"].startswith("[REDACTED]")
    args = payload["tool_execution_records"]["op-1"]["args"]
    assert args["password"] != "secret"
    assert args["password"].startswith("[REDACTED]")


def test_checkpoint_write_does_not_persist_plaintext_passwords(tmp_path) -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_session_ssh_targets"] = {
        "example.test": {"host": "example.test", "user": "root", "password": "secret"}
    }
    state.tool_execution_records["op-1"] = {
        "tool_name": "ssh_exec",
        "args": {"host": "example.test", "user": "root", "password": "secret"},
    }
    path = tmp_path / "checkpoint.json"
    _write_checkpoint_file(path, {}, state.to_dict())
    text = path.read_text(encoding="utf-8")
    assert "secret" not in text


def test_remember_session_ssh_target_stores_fingerprint_not_password() -> None:
    state = LoopState(cwd="/tmp")
    store = CredentialStore()
    harness = SimpleNamespace(state=state, credential_store=store)
    service = SimpleNamespace(harness=harness)
    _remember_session_ssh_target(
        service,
        tool_name="ssh_exec",
        result=ToolEnvelope(
            success=True,
            output={"stdout": "ok\n", "stderr": "", "exit_code": 0},
            metadata={},
        ),
        arguments={"host": "example.test", "user": "root", "password": "secret"},
    )
    target = state.scratchpad["_session_ssh_targets"]["example.test"]
    assert "password" not in target
    assert "password_fingerprint" in target
    assert store.get_ssh_password("example.test", "root") == "secret"


def test_feed_sudo_password_refuses_when_not_under_pty_or_sudo_child() -> None:
    async def _run() -> None:
        proc = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda _data: None),
            child_command_basename="cat",
        )
        store = CredentialStore()
        store.set_sudo_password("secret")
        harness = SimpleNamespace(credential_store=store)
        result = await _feed_sudo_password_to_process(proc, "sudo ls /root", harness)
        assert result is not None
        assert result["success"] is False
        assert result["metadata"]["reason"] == "sudo_prompt_unexpected"

    asyncio.run(_run())


def test_feed_sudo_password_accepts_when_under_pty() -> None:
    async def _run() -> None:
        written: list[bytes] = []
        proc = SimpleNamespace(
            is_running_under_pty=True,
            stdin=SimpleNamespace(
                write=lambda data: written.append(data),
                drain=AsyncMock(),
                close=lambda: None,
            ),
        )
        store = CredentialStore()
        store.set_sudo_password("secret")
        harness = SimpleNamespace(credential_store=store)
        result = await _feed_sudo_password_to_process(proc, "sudo ls /root", harness)
        assert result is None
        assert written == [b"secret\n"]

    asyncio.run(_run())


def test_feed_sudo_password_accepts_when_sudo_child() -> None:
    async def _run() -> None:
        written: list[bytes] = []
        proc = SimpleNamespace(
            child_command_basename="sudo",
            stdin=SimpleNamespace(
                write=lambda data: written.append(data),
                drain=AsyncMock(),
                close=lambda: None,
            ),
        )
        store = CredentialStore()
        store.set_sudo_password("secret")
        harness = SimpleNamespace(credential_store=store)
        result = await _feed_sudo_password_to_process(proc, "sudo ls /root", harness)
        assert result is None
        assert written == [b"secret\n"]

    asyncio.run(_run())


def test_feed_sudo_password_refuses_non_sudo_command() -> None:
    async def _run() -> None:
        proc = SimpleNamespace(stdin=SimpleNamespace(write=lambda _data: None))
        harness = SimpleNamespace(
            credential_store=CredentialStore(),
            get_sudo_password=lambda **_: "secret",
        )
        result = await _feed_sudo_password_to_process(proc, "cat /etc/passwd", harness)
        assert result is not None
        assert result["success"] is False
        assert result["metadata"]["reason"] == "sudo_prompt_unexpected"

    asyncio.run(_run())


def test_feed_sudo_password_accepts_sudo_command() -> None:
    async def _run() -> None:
        written: list[bytes] = []
        proc = SimpleNamespace(
            stdin=SimpleNamespace(
                write=lambda data: written.append(data),
                drain=AsyncMock(),
                close=lambda: None,
            )
        )
        store = CredentialStore()
        store.set_sudo_password("secret")
        harness = SimpleNamespace(credential_store=store)
        result = await _feed_sudo_password_to_process(proc, "sudo ls /root", harness)
        assert result is None
        assert written == [b"secret\n"]

    asyncio.run(_run())


def test_feed_sudo_password_refuses_when_no_store() -> None:
    async def _run() -> None:
        proc = SimpleNamespace(stdin=SimpleNamespace(write=lambda _data: None))
        harness = SimpleNamespace(get_sudo_password=lambda **_: None)
        result = await _feed_sudo_password_to_process(proc, "sudo ls", harness)
        assert result is not None
        assert result["status"] == "needs_human"

    asyncio.run(_run())
