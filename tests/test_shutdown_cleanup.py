from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from smallctl import main as main_module
from smallctl.chat_sessions import load_chat_session_state
from smallctl.harness import runtime_facade


class _Approvals:
    def __init__(self) -> None:
        self.shell_reject_calls = 0
        self.sudo_reject_calls = 0

    def reject_pending_shell_approvals(self) -> None:
        self.shell_reject_calls += 1

    def reject_pending_sudo_password_prompts(self) -> None:
        self.sudo_reject_calls += 1


class _Transport:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _Pipe:
    def __init__(self) -> None:
        self.close_calls = 0
        self._transport = _Transport()

    def close(self) -> None:
        self.close_calls += 1


class _Writer:
    def __init__(self) -> None:
        self.close_calls = 0
        self.transport = _Transport()

    def close(self) -> None:
        self.close_calls += 1


class _FakeProcess:
    def __init__(self, *, wait_delay: float = 0.0) -> None:
        self.stdin = _Writer()
        self.stdout = _Pipe()
        self.stderr = _Pipe()
        self._transport = _Transport()
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self._wait_delay = wait_delay

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self._wait_delay:
            await asyncio.sleep(self._wait_delay)
        else:
            await asyncio.sleep(0)
        return int(self.returncode or 0)


def _make_harness(proc: _FakeProcess) -> tuple[SimpleNamespace, list[dict[str, str]]]:
    finalized: list[dict[str, str]] = []
    harness = SimpleNamespace(
        _pending_task_shutdown_reason="cancel_requested",
        approvals=_Approvals(),
        _active_processes={proc},
        _teardown_task=None,
        event_handler=object(),
        _finalize_task_scope=lambda **kwargs: finalized.append(kwargs),
    )
    return harness, finalized


def test_teardown_closes_process_transports_and_clears_event_handler(monkeypatch) -> None:
    proc = _FakeProcess()
    harness, finalized = _make_harness(proc)

    asyncio.run(runtime_facade.teardown(harness))

    assert finalized == [
        {
            "terminal_event": "task_interrupted",
            "status": "interrupted",
            "reason": "cancel_requested",
        }
    ]
    assert harness._pending_task_shutdown_reason == ""
    assert harness.event_handler is None
    assert harness._active_processes == set()
    assert harness.approvals.shell_reject_calls == 1
    assert harness.approvals.sudo_reject_calls == 1
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 0
    assert proc.wait_calls >= 1
    assert proc.stdin.close_calls == 1
    assert proc.stdin.transport.close_calls == 1
    assert proc.stdout.close_calls == 1
    assert proc.stdout._transport.close_calls == 1
    assert proc.stderr.close_calls == 1
    assert proc.stderr._transport.close_calls == 1
    assert proc._transport.close_calls == 1


def test_teardown_serializes_concurrent_cleanup(monkeypatch) -> None:
    proc = _FakeProcess(wait_delay=0.01)
    harness, finalized = _make_harness(proc)

    async def _run() -> None:
        await asyncio.gather(
            runtime_facade.teardown(harness),
            runtime_facade.teardown(harness),
        )

    asyncio.run(_run())

    assert len(finalized) == 1
    assert harness.approvals.shell_reject_calls == 1
    assert harness.approvals.sudo_reject_calls == 1
    assert proc.terminate_calls == 1


def test_teardown_autosaves_chat_session_state(tmp_path) -> None:
    proc = _FakeProcess()
    harness, _ = _make_harness(proc)
    harness.log = SimpleNamespace(debug=lambda *args, **kwargs: None)
    harness.client = SimpleNamespace(model="alpha-model")
    harness.conversation_id = "conversation-123"
    harness.state = SimpleNamespace(
        cwd=str(tmp_path),
        thread_id="thread-123",
        to_dict=lambda: {
            "thread_id": "thread-123",
            "created_at": "2026-04-23T12:00:00+00:00",
            "updated_at": "2026-04-23T12:05:00+00:00",
            "scratchpad": {"_model_name": "alpha-model"},
            "recent_messages": [{"role": "user", "content": "hello world"}],
        },
    )
    harness._autosave_chat_session_state = lambda: runtime_facade._autosave_chat_session_state(harness)

    asyncio.run(runtime_facade.teardown(harness))

    saved = load_chat_session_state(cwd=tmp_path, thread_id="thread-123")
    assert saved is not None
    assert saved["thread_id"] == "thread-123"
    assert saved["recent_messages"][0]["content"] == "hello world"


def test_teardown_waits_for_background_autosave_snapshot_work(tmp_path) -> None:
    proc = _FakeProcess()
    harness, _ = _make_harness(proc)
    harness.log = SimpleNamespace(debug=lambda *args, **kwargs: None)
    harness.client = SimpleNamespace(model="alpha-model")
    harness.conversation_id = "conversation-123"

    state_started = threading.Event()
    release_state = threading.Event()
    loop_thread_id: dict[str, int] = {}
    worker_thread_id: dict[str, int] = {}

    def _slow_state_payload() -> dict[str, object]:
        worker_thread_id["value"] = threading.get_ident()
        state_started.set()
        release_state.wait(timeout=1.0)
        return {
            "thread_id": "thread-123",
            "created_at": "2026-04-23T12:00:00+00:00",
            "updated_at": "2026-04-23T12:05:00+00:00",
            "scratchpad": {"_model_name": "alpha-model"},
            "recent_messages": [{"role": "user", "content": "hello world"}],
        }

    harness.state = SimpleNamespace(
        cwd=str(tmp_path),
        thread_id="thread-123",
        to_dict=_slow_state_payload,
    )
    harness._autosave_chat_session_state = lambda: runtime_facade._autosave_chat_session_state(harness)

    async def _run() -> None:
        loop_thread_id["value"] = threading.get_ident()
        teardown_task = asyncio.create_task(runtime_facade.teardown(harness))
        started = await asyncio.to_thread(state_started.wait, 1.0)
        assert started
        assert not teardown_task.done()
        release_state.set()
        await teardown_task

    asyncio.run(_run())

    assert worker_thread_id["value"] != loop_thread_id["value"]
    saved = load_chat_session_state(cwd=tmp_path, thread_id="thread-123")
    assert saved is not None
    assert saved["recent_messages"][0]["content"] == "hello world"


def test_tui_ctrl_c_exit_prints_shutdown_alert_with_session_id(monkeypatch) -> None:
    import smallctl.ui as ui_module

    alerts: list[str] = []

    class _FakeApp:
        def __init__(self, harness_kwargs: dict[str, object]) -> None:
            self.harness_kwargs = dict(harness_kwargs)
            self.closed_by_ctrl_c = True
            self.harness = SimpleNamespace(
                state=SimpleNamespace(thread_id="thread-123"),
                conversation_id="conversation-ignored",
            )

        def run(self) -> None:
            return None

    config = SimpleNamespace(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        phase="explore",
        provider_profile="generic",
        api_key=None,
        tool_profiles=None,
        reasoning_mode="auto",
        thinking_visibility=True,
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
        chat_endpoint="/chat/completions",
        checkpoint_on_exit=False,
        checkpoint_path=None,
        graph_checkpointer="memory",
        graph_checkpoint_path=None,
        fresh_run=False,
        fresh_run_turns=1,
        planning_mode=False,
        contract_flow_ui=False,
        staged_reasoning=False,
        restore_graph_state=False,
        graph_thread_id=None,
        context_limit=None,
        max_prompt_tokens=None,
        reserve_completion_tokens=1024,
        reserve_tool_tokens=512,
        first_token_timeout_sec=30,
        healthcheck_url=None,
        restart_command=None,
        startup_grace_period_sec=20,
        max_restarts_per_hour=2,
        backend_healthcheck_url=None,
        backend_restart_command=None,
        backend_unload_command=None,
        backend_healthcheck_timeout_sec=5,
        backend_restart_grace_sec=20,
        summarize_at_ratio=0.8,
        recent_message_limit=24,
        max_summary_items=3,
        max_artifact_snippets=4,
        artifact_snippet_token_limit=400,
        multi_file_artifact_snippet_limit=8,
        multi_file_primary_file_limit=3,
        remote_task_artifact_snippet_limit=8,
        remote_task_primary_file_limit=2,
        indexer=False,
        task=None,
        log_file=None,
        debug=False,
        compatibility_warnings=[],
    )

    monkeypatch.setattr(main_module, "resolve_config", lambda _args: config)
    monkeypatch.setattr(main_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "create_run_logger", lambda _path: SimpleNamespace(run_dir=Path(".")))
    monkeypatch.setattr(main_module, "log_kv", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "_print_shutdown_alert", lambda session_id: alerts.append(session_id))
    monkeypatch.setattr(ui_module, "SmallctlApp", _FakeApp)

    result = main_module.cli(["--tui"])

    assert result == 0
    assert alerts == ["thread-123"]
