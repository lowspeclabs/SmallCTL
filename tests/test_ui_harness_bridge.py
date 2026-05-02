from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from smallctl.models.events import UIEvent, UIEventType
from smallctl.ui.harness_bridge import HarnessBridge


class _FakeHarness:
    def __init__(self) -> None:
        self.run_thread_id: int | None = None
        self.resume_thread_id: int | None = None
        self.cancel_thread_id: int | None = None
        self.shell_resolution_thread_id: int | None = None
        self.sudo_resolution_thread_id: int | None = None
        self.shell_default_thread_id: int | None = None
        self.planning_mode_thread_id: int | None = None
        self.switch_model_thread_id: int | None = None
        self.teardown_thread_id: int | None = None
        self.shell_resolution: tuple[str, bool] | None = None
        self.sudo_resolution: tuple[str, str | None] | None = None
        self.shell_approval_session_default = False
        self.provider_profile = "generic"
        self.sync_run_logger_calls = 0
        self.state = SimpleNamespace(
            planning_mode_enabled=False,
            planner_resume_target_mode="loop",
            planner_requested_output_path="",
            planner_requested_output_format="",
            thread_id="thread-baseline",
            recent_messages=[
                SimpleNamespace(role="user", content="hello"),
                SimpleNamespace(role="assistant", content="hi"),
            ],
        )
        self._model = "baseline"

    async def run_auto_with_events(self, task: str, event_handler) -> dict[str, object]:
        self.run_thread_id = threading.get_ident()
        event_handler(UIEvent(event_type=UIEventType.SYSTEM, content=f"run:{task}"))
        await asyncio.sleep(0)
        return {"status": "ok", "task": task}

    async def resume_task_with_events(self, human_input: str, event_handler) -> dict[str, object]:
        self.resume_thread_id = threading.get_ident()
        event_handler(UIEvent(event_type=UIEventType.SYSTEM, content=f"resume:{human_input}"))
        await asyncio.sleep(0)
        return {"status": "resumed", "choice": human_input}

    def cancel(self) -> None:
        self.cancel_thread_id = threading.get_ident()

    def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
        self.shell_resolution_thread_id = threading.get_ident()
        self.shell_resolution = (approval_id, approved)

    def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
        self.sudo_resolution_thread_id = threading.get_ident()
        self.sudo_resolution = (prompt_id, password)

    def set_shell_approval_session_default(self, enabled: bool) -> None:
        self.shell_default_thread_id = threading.get_ident()
        self.shell_approval_session_default = bool(enabled)

    def set_planning_mode(self, enabled: bool) -> dict[str, object]:
        self.planning_mode_thread_id = threading.get_ident()
        self.state.planning_mode_enabled = bool(enabled)
        if enabled:
            self.state.planner_resume_target_mode = "loop"
        else:
            self.state.planner_requested_output_path = ""
            self.state.planner_requested_output_format = ""
        return self.build_status_snapshot()

    def switch_model(self, model: str) -> None:
        self.switch_model_thread_id = threading.get_ident()
        self._model = model
        self.provider_profile = "bridge-provider"

    def restore_graph_state(self, thread_id: str | None = None) -> bool:
        self.state.thread_id = str(thread_id or self.state.thread_id)
        return True

    def has_pending_interrupt(self) -> bool:
        return False

    def get_pending_interrupt(self) -> dict[str, object] | None:
        return None

    def _sync_run_logger_session_id(self) -> None:
        self.sync_run_logger_calls += 1

    def build_status_snapshot(self, *, activity: str = "", api_errors: int | None = None) -> dict[str, object]:
        return {
            "model": self._model,
            "phase": "explore",
            "step": 0,
            "mode": "planning" if self.state.planning_mode_enabled else "",
            "activity": activity,
            "api_errors": api_errors or 0,
        }

    async def teardown(self) -> None:
        self.teardown_thread_id = threading.get_ident()
        await asyncio.sleep(0)


def test_harness_bridge_runs_work_and_control_calls_on_background_loop() -> None:
    main_thread_id = threading.get_ident()
    posted: list[str] = []
    harness = _FakeHarness()
    bridge = HarnessBridge(
        harness=harness,
        post_ui_event=lambda event: posted.append(event.content),
    )

    async def _run() -> None:
        result = await bridge.run_auto("build")
        resumed = await bridge.resume("yes")
        switched = await bridge.switch_model("new-model", activity="switching", api_errors=2)
        planning_snapshot = await bridge.set_planning_mode(True)
        restored = await bridge.restore_graph_state("thread-restored")
        replaced = await bridge.replace_state_from_payload(
            {
                "thread_id": "thread-from-payload",
                "current_phase": "explore",
                "recent_messages": [
                    {"role": "user", "content": "from payload"},
                    {"role": "assistant", "content": "ack"},
                ],
            }
        )
        bridge.set_shell_approval_session_default(True)
        bridge.cancel()
        bridge.resolve_shell_approval("shell-1", True)
        bridge.resolve_sudo_password("sudo-1", "secret")
        await asyncio.sleep(0.05)
        await bridge.shutdown()

        assert result == {"status": "ok", "task": "build"}
        assert resumed == {"status": "resumed", "choice": "yes"}
        assert switched == {
            "provider_profile": "bridge-provider",
            "snapshot": {
                "model": "new-model",
                "phase": "explore",
                "step": 0,
                "mode": "",
                "activity": "switching",
                "api_errors": 2,
            },
        }
        assert planning_snapshot == {
            "model": "new-model",
            "phase": "explore",
            "step": 0,
            "mode": "planning",
            "activity": "",
            "api_errors": 0,
        }
        assert restored == {
            "restored": True,
            "thread_id": "thread-restored",
            "has_pending_interrupt": False,
            "snapshot": {
                "model": "new-model",
                "phase": "explore",
                "step": 0,
                "mode": "planning",
                "activity": "",
                "api_errors": 0,
            },
            "recent_messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        assert replaced["thread_id"] == "thread-from-payload"
        assert replaced["snapshot"] == {
            "model": "new-model",
            "phase": "explore",
            "step": 0,
            "mode": "",
            "activity": "",
            "api_errors": 0,
        }
        assert replaced["recent_messages"] == [
            {"role": "user", "content": "from payload"},
            {"role": "assistant", "content": "ack"},
        ]

    asyncio.run(_run())

    assert posted == ["run:build", "resume:yes"]
    assert harness.run_thread_id is not None and harness.run_thread_id != main_thread_id
    assert harness.resume_thread_id == harness.run_thread_id
    assert harness.switch_model_thread_id == harness.run_thread_id
    assert harness.planning_mode_thread_id == harness.run_thread_id
    assert harness.shell_default_thread_id == harness.run_thread_id
    assert harness.cancel_thread_id == harness.run_thread_id
    assert harness.shell_resolution_thread_id == harness.run_thread_id
    assert harness.sudo_resolution_thread_id == harness.run_thread_id
    assert harness.shell_resolution == ("shell-1", True)
    assert harness.sudo_resolution == ("sudo-1", "secret")
    assert harness.shell_approval_session_default is True
    assert harness.sync_run_logger_calls == 1
    assert harness.teardown_thread_id == harness.run_thread_id
