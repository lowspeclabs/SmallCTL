from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallctl.state import LoopState
from smallctl.ui.app import SmallctlApp
from smallctl.ui.console import ConsolePane
from smallctl.ui.harness_bridge import HarnessBridge


# --- L27: auxiliary background tasks are retained and exceptions observed ------


def test_l27_console_background_task_exception_retrieved_and_logged(caplog) -> None:
    pane = ConsolePane()

    async def _run():
        loop = asyncio.get_running_loop()
        loop_errors: list[dict] = []
        loop.set_exception_handler(lambda _loop, ctx: loop_errors.append(ctx))

        async def _boom() -> None:
            raise RuntimeError("boom-l27")

        task = pane._spawn_background_task(_boom(), name="boom_task")
        assert task in pane._background_tasks
        await asyncio.sleep(0.05)
        gc.collect()
        await asyncio.sleep(0)
        return task, loop_errors

    with caplog.at_level(logging.WARNING, logger="smallctl.ui.console"):
        task, loop_errors = asyncio.run(_run())

    assert task.done()
    assert not task.cancelled()
    assert isinstance(task.exception(), RuntimeError)
    assert task not in pane._background_tasks
    assert any("boom_task" in record.getMessage() for record in caplog.records)
    assert not any("never retrieved" in str(ctx.get("message", "")) for ctx in loop_errors)


def test_l27_app_background_task_cancelled_on_drain() -> None:
    app = SmallctlApp(harness_kwargs={"endpoint": "http://localhost:1/v1", "model": "m"})

    async def _run():
        started = asyncio.Event()

        async def _linger() -> None:
            started.set()
            await asyncio.Event().wait()

        task = app._spawn_background_task(_linger(), name="linger_task")
        await started.wait()
        assert task in app._background_tasks
        await app._cancel_background_tasks()
        return task

    task = asyncio.run(_run())

    assert task.cancelled()


# --- L28: real dotenv reads are redacted; templates readable; perms warn -------


def test_l28_real_dotenv_read_redacted_with_permission_warning(tmp_path: Path) -> None:
    from smallctl.tools.fs_listing import file_read

    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROXMOX_TOKEN=live-secret-123\n# a comment\nEMPTY=\n",
        encoding="utf-8",
    )
    os.chmod(env_file, 0o644)

    result = asyncio.run(file_read(str(env_file)))

    assert result["success"] is True
    assert "live-secret-123" not in result["output"]
    assert "PROXMOX_TOKEN=[REDACTED]" in result["output"]
    assert "# a comment" in result["output"]
    assert result["metadata"]["dotenv_read_redacted"] is True
    warning = result["metadata"]["dotenv_permissions_warning"]
    assert "group/other" in warning
    assert "0o644" in warning
    assert "live-secret-123" not in warning

    os.chmod(env_file, 0o664)
    result = asyncio.run(file_read(str(env_file)))
    assert result["metadata"]["dotenv_permissions_warning"]
    assert "live-secret-123" not in result["metadata"]["dotenv_permissions_warning"]

    os.chmod(env_file, 0o600)
    result = asyncio.run(file_read(str(env_file)))
    assert result["metadata"]["dotenv_permissions_warning"] == ""


def test_l28_env_template_remains_readable(tmp_path: Path) -> None:
    from smallctl.tools.fs_listing import file_read

    template = tmp_path / ".env.example"
    template.write_text("PROXMOX_TOKEN=replace-me\n", encoding="utf-8")

    result = asyncio.run(file_read(str(template)))

    assert result["success"] is True
    assert "replace-me" in result["output"]
    assert result["metadata"]["dotenv_read_redacted"] is False


def test_l28_read_dotenv_warns_on_permissive_mode_without_values(tmp_path: Path, caplog) -> None:
    from smallctl.config_support import _read_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("SMALLCTL_API_KEY=topsecret-value\n", encoding="utf-8")
    os.chmod(env_file, 0o644)

    with caplog.at_level(logging.WARNING, logger="smallctl.config"):
        parsed = _read_dotenv(env_file)

    assert parsed["SMALLCTL_API_KEY"] == "topsecret-value"
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("group/other" in message for message in warnings)
    assert all("topsecret-value" not in message for message in warnings)

    caplog.clear()
    os.chmod(env_file, 0o600)
    with caplog.at_level(logging.WARNING, logger="smallctl.config"):
        _read_dotenv(env_file)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- L29: destructive delete guard bypasses ------------------------------------


def _guard(state: LoopState, command: str):
    from smallctl.tools.shell_support_delete_guards import (
        _shell_workspace_destructive_delete_guard,
    )

    return _shell_workspace_destructive_delete_guard(state, command)


def test_l29_out_of_workspace_delete_blocked(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))

    result = _guard(state, "rm -rf /etc/important-config")

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "workspace_destructive_delete_blocked"
    assert result["metadata"]["blocked_targets"][0]["reasons"] == ["outside_workspace"]


@pytest.mark.parametrize(
    "command,family",
    [
        ("git clean -fdx", "git_clean_force_include_ignored"),
        ("git reset --hard HEAD~1", "git_reset_hard"),
        ("shred -u secrets.txt", "shred"),
        ("dd if=/dev/zero of=/tmp/disk.img bs=1M count=1", "dd_write_capable"),
        ("dd if=/dev/zero oflag=direct of=/dev/sdz", "dd_write_capable"),
    ],
)
def test_l29_destructive_families_blocked(tmp_path: Path, command: str, family: str) -> None:
    state = LoopState(cwd=str(tmp_path))

    result = _guard(state, command)

    assert result is not None, command
    assert result["success"] is False
    assert result["metadata"]["destructive_family"] == family


def test_l29_explicitly_requested_destructive_family_allowed(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "Please run git clean -fdx to remove build artifacts."

    assert _guard(state, "git clean -fdx") is None


def test_l29_safe_disposable_cleanup_stays_allowed(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()

    assert _guard(state, f"rm -rf {cache_dir}") is None
    # dd without an output file/oflag/conv writes to stdout: not destructive.
    assert _guard(state, "dd if=/dev/zero bs=1M count=1") is None


# --- L30: trajectory recorder redacts secrets before serialization -------------


def test_l30_trajectory_payload_is_redacted(tmp_path: Path) -> None:
    from smallctl.harness.trajectory_recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(base_dir=tmp_path)
    harness = SimpleNamespace(
        state=SimpleNamespace(
            thread_id="t-secrets",
            run_brief=SimpleNamespace(
                original_task="deploy with Authorization: Bearer abc123secret and password=hunter2",
                effective_task="",
            ),
            scratchpad={
                "_tool_plan": {
                    "steps": [
                        {
                            "id": "E1",
                            "tool": "ssh_exec",
                            "args": {
                                "host": "1.2.3.4",
                                "password": "hunter2",
                                "command": "curl -H 'Authorization: Bearer abc123secret' https://user:secretpass@example.com",
                            },
                        }
                    ]
                },
                "_tool_plan_observations_text": "OPENROUTER_API_KEY=sk-live-999\nall good",
                "_last_solver_draft": "connect to https://user:secretpass@db.example.com",
            },
        ),
        conversation_id="t-secrets",
    )
    result = {
        "status": "completed",
        "reason": "finished using token=abc123secret",
        "latency_metrics": {},
    }

    out_path = recorder.record_tool_plan_trajectory(harness, result)

    assert out_path is not None
    raw = out_path.read_text(encoding="utf-8")
    for secret in ("abc123secret", "hunter2", "sk-live-999", "secretpass"):
        assert secret not in raw
    payload = json.loads(raw.strip())
    assert payload["success"] is True


# --- M11: cold-overflow artifact is registered in state.artifacts --------------


def test_m11_cold_overflow_artifact_registered() -> None:
    from smallctl.context.tiers import MessageTierManager
    from smallctl.state import ContextBrief

    def _brief(brief_id: str) -> ContextBrief:
        return ContextBrief(
            brief_id=brief_id,
            created_at="2026-07-16T00:00:00+00:00",
            tier="warm",
            step_range=(1, 2),
            task_goal="goal",
            current_phase="explore",
            key_discoveries=[f"fact-{brief_id}"],
            tools_tried=[],
            blockers=[],
            files_touched=[],
            artifact_ids=[],
            next_action_hint="",
            staleness_step=0,
            full_artifact_id=f"ART-FULL-{brief_id}",
        )

    persisted: list[dict] = []

    class _Store:
        def persist_thinking(self, *, raw_thinking: str, summary: str, source: str):
            persisted.append({"raw_thinking": raw_thinking, "summary": summary, "source": source})
            return SimpleNamespace(artifact_id="ART-OVERFLOW-1")

    manager = MessageTierManager()
    state = LoopState()
    state.context_briefs = [_brief(f"B{i}") for i in range(manager.warm_limit + 1)]
    state.working_memory.known_facts = [f"existing-fact-{i}" for i in range(12)]

    manager.promote_to_cold(state, artifact_store=_Store())

    assert persisted, "expected the cold overflow to be persisted"
    assert "ART-OVERFLOW-1" in state.artifacts


# --- M16: bridge shutdown is bounded when teardown wedges ----------------------


def test_m16_shutdown_returns_within_timeout_when_teardown_wedged(caplog) -> None:
    class _WedgedHarness:
        def __init__(self) -> None:
            self.teardown_started = asyncio.Event()

        async def teardown(self) -> None:
            self.teardown_started.set()
            await asyncio.Event().wait()

        def cancel(self, *_args) -> None:
            return None

    harness = _WedgedHarness()
    bridge = HarnessBridge(
        harness=harness,
        post_ui_event=lambda event: None,
        shutdown_timeout_sec=0.2,
    )
    bridge.start()

    async def _run() -> float:
        start = time.monotonic()
        await bridge.shutdown()
        return time.monotonic() - start

    with caplog.at_level(logging.WARNING, logger="smallctl.ui.harness_bridge"):
        elapsed = asyncio.run(_run())

    assert elapsed < 5.0
    assert bridge._thread is None or not bridge._thread.is_alive()
    assert any("forcing bridge loop stop" in record.getMessage() for record in caplog.records)


# --- M18: configurable StrictHostKeyChecking + first-trust visibility ----------


def test_m18_configured_mode_lands_in_ssh_argv() -> None:
    from smallctl.tools.network import _resolve_ssh_strict_host_key_mode
    from smallctl.tools.network_ssh_helpers import build_ssh_command

    command, _, password_file = build_ssh_command(
        host="example-host",
        command="ls",
        user="alice",
        port=22,
        identity_file=None,
        password=None,
        strict_host_key_checking="yes",
    )
    assert password_file is None
    assert "StrictHostKeyChecking=yes" in command

    default_command, _, _ = build_ssh_command(
        host="example-host",
        command="ls",
        user="alice",
        port=22,
        identity_file=None,
        password=None,
    )
    assert "StrictHostKeyChecking=accept-new" in default_command

    harness = SimpleNamespace(config=SimpleNamespace(ssh_strict_host_key_checking="yes"))
    assert _resolve_ssh_strict_host_key_mode(harness) == "yes"
    assert _resolve_ssh_strict_host_key_mode(SimpleNamespace()) == "accept-new"
    from smallctl.tools.network_ssh_helpers import SSHStrictHostKeyConfigError

    bogus = SimpleNamespace(config=SimpleNamespace(ssh_strict_host_key_checking="bogus"))
    with pytest.raises(SSHStrictHostKeyConfigError):
        _resolve_ssh_strict_host_key_mode(bogus)


def test_m18_first_host_key_trust_surfaces_warning(monkeypatch) -> None:
    from smallctl.tools import network

    runlog_events: list[tuple] = []

    class _FakeStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = list(chunks)

        async def read(self, _size: int) -> bytes:
            await asyncio.sleep(0)
            return self._chunks.pop(0) if self._chunks else b""

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = _FakeStream([b"ok\n"])
            self.stderr = _FakeStream([
                b"Warning: Permanently added 'example-host' (ED25519) to the list of known hosts.\n"
            ])
            self.returncode: int | None = None

        async def wait(self) -> int:
            await asyncio.sleep(0)
            self.returncode = 0
            return 0

    captured: dict[str, str] = {}

    async def _fake_create_process(*, command: str, **_kwargs):
        captured["command"] = command
        return _FakeProcess()

    monkeypatch.setattr(network, "create_process", _fake_create_process)

    harness = SimpleNamespace(
        config=SimpleNamespace(ssh_strict_host_key_checking="accept-new"),
        event_handler=None,
        _runlog=lambda event, message, **kwargs: runlog_events.append((event, message, kwargs)),
    )

    result = asyncio.run(
        network.run_ssh_command(
            host="example-host",
            command="ls",
            user="alice",
            harness=harness,
        )
    )

    assert "StrictHostKeyChecking=accept-new" in captured["command"]
    assert result["success"] is True
    output_metadata = (result.get("output") or {}).get("metadata") or {}
    assert output_metadata.get("ssh_known_hosts_added") is True
    warning = output_metadata.get("ssh_host_key_trust_warning", "")
    assert "first-seen host key" in warning
    trusted = [entry for entry in runlog_events if entry[0] == "ssh_host_key_trusted"]
    assert trusted, "expected a visible runlog event for the first-seen host key"
    assert trusted[0][2].get("level") == "warning"
