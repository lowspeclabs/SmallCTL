from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

from textual.css.query import NoMatches

from smallctl.chat_sessions import (
    load_chat_session_ui_transcript,
    persist_chat_session_state,
    persist_chat_session_ui_transcript,
    session_state_path,
    session_ui_transcript_path,
)
from smallctl.models.conversation import ConversationMessage
from smallctl.models.events import UIEvent, UIEventType
from smallctl.state import LoopState
from smallctl.ui.app_actions import SmallctlAppActionsMixin
from smallctl.ui.app_approvals import handle_approval_prompt, handle_sudo_password_prompt
from smallctl.ui.approval import ShellApprovalDecision
from smallctl.harness import HarnessConfig
from smallctl.ui.app import SmallctlApp
from smallctl.ui.app_flow import SmallctlAppFlowMixin
from smallctl.ui.display import compute_activity_for_event, format_test_time_scaling_event
from smallctl.ui.console import ConsolePane
from smallctl.ui.input import InputPane
from smallctl.ui.model_selector import ModelSelectButton
from smallctl.ui.statusbar import StatusBar


def test_get_console_handles_unmounted_view_without_name_error() -> None:
    seen: list[type[object]] = []

    class _Flow(SmallctlAppFlowMixin):
        def query_one(self, widget_type: type[object]) -> object:
            seen.append(widget_type)
            raise NoMatches("no console mounted")

    flow = _Flow()

    assert flow._get_console() is None
    assert seen
    assert seen[0].__name__ == "ConsolePane"


def test_on_harness_event_uses_lazy_harness_event_import_for_cross_thread_posting() -> None:
    posted: list[object] = []

    class _Flow(SmallctlAppFlowMixin):
        is_running = True

        def __init__(self) -> None:
            self._loop_thread_id = -1

        def post_message(self, message: object) -> None:
            posted.append(message)

        async def _handle_harness_event(self, event: UIEvent) -> None:
            raise AssertionError(f"should not handle directly: {event}")

    flow = _Flow()

    asyncio.run(
        flow.on_harness_event(
            UIEvent(event_type=UIEventType.SYSTEM, content="hello")
        )
    )

    assert len(posted) == 1
    assert posted[0].__class__.__name__ == "HarnessEvent"


def test_test_time_scaling_event_formats_statusbar_activity() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        content="Scaled 2 candidates; selected #1.",
        data={
            "kind": "test_time_scaling",
            "phase": "proposal_selected",
            "candidate_count": 2,
            "selected_candidate": 1,
            "selected_score": 1.0,
        },
    )

    assert compute_activity_for_event(event) == "scaling selected #1/2 score 1.0"


def test_test_time_scaling_event_formats_candidate_history_panel() -> None:
    event = UIEvent(
        event_type=UIEventType.SYSTEM,
        content="Scaled 2 branches; selected #2.",
        data={
            "kind": "test_time_scaling",
            "phase": "branch_selected",
            "policy": "sequential_branch",
            "candidate_count": 2,
            "selected_candidate": 2,
            "selected_score": 0.95,
            "read_only_branch_parallel_count": 1,
            "candidate_history": [
                {
                    "candidate": 1,
                    "score": 0.0,
                    "tools": ["ssh_file_write"],
                    "failed_criteria": ["unsafe_branch_tool:ssh_file_write"],
                    "prompt_variant": "minimal",
                    "read_only": False,
                },
                {
                    "candidate": 2,
                    "selected": True,
                    "score": 0.95,
                    "tools": ["file_read", "step_complete"],
                    "token_cost": 123,
                    "latency_ms": 45.5,
                    "prompt_variant": "verify-first",
                    "read_only": True,
                    "isolated": True,
                },
            ],
        },
    )

    text = format_test_time_scaling_event(event)

    assert "policy: sequential_branch" in text
    assert "parallel read-only branches: 1" in text
    assert "#2 selected | score 0.95 | 123 tokens | 45.5 ms | read-only | isolated" in text
    assert "tools: file_read, step_complete" in text
    assert "failed: unsafe_branch_tool:ssh_file_write" in text


def test_on_harness_event_queues_same_thread_work_before_rendering() -> None:
    seen: list[str] = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._pending_harness_events = []
            self._pending_status_event = None
            self._ui_event_drain_task = None

        async def _handle_harness_event(self, event: UIEvent) -> None:
            seen.append(event.content)

    flow = _Flow()

    async def _run() -> None:
        await flow.on_harness_event(UIEvent(event_type=UIEventType.SYSTEM, content="queued"))
        assert seen == []
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert seen == ["queued"]


def test_ui_transcript_does_not_persist_live_shell_stream() -> None:
    persist_calls = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._ui_transcript = []

        def _persist_ui_transcript(self) -> None:
            persist_calls.append(list(self._ui_transcript))

    flow = _Flow()

    flow._record_ui_transcript_event(UIEvent(UIEventType.SHELL_STREAM, "live output"))
    flow._record_ui_transcript_event(UIEvent(UIEventType.SYSTEM, "system note"))

    assert len(flow._ui_transcript) == 1
    assert flow._ui_transcript[0]["event_type"] == "system"
    assert flow._ui_transcript[0]["content"] == "system note"
    assert len(persist_calls) == 1


def test_ui_transcript_prunes_partial_assistant_on_model_output_loop() -> None:
    persist_calls = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._ui_transcript = []

        def _persist_ui_transcript(self) -> None:
            persist_calls.append(list(self._ui_transcript))

    flow = _Flow()

    flow._record_ui_transcript_event(UIEvent(UIEventType.SYSTEM, "ready"))
    flow._record_ui_transcript_event(UIEvent(UIEventType.ASSISTANT, "| CONTAINER ID | IMAGE |"))
    flow._record_ui_transcript_event(UIEvent(UIEventType.ASSISTANT, "\n| :--- | :--- |"))
    flow._record_ui_transcript_event(
        UIEvent(
            UIEventType.SYSTEM,
            "[harness] Context refreshed: dropped experience_memories (context_invalidated)",
            data={"ui_kind": "context_lane_dropped"},
        )
    )
    flow._record_ui_transcript_event(
        UIEvent(
            UIEventType.SYSTEM,
            "[harness] Model output loop detected repeating `|---`; recovery nudge injected",
            data={"ui_kind": "model_output_degenerate_loop_exhausted"},
        )
    )

    assert [item["event_type"] for item in flow._ui_transcript] == ["system", "system", "system"]
    assert all("CONTAINER ID" not in item["content"] for item in flow._ui_transcript)
    assert len(persist_calls) == 5


def test_ui_transcript_skips_loop_halt_placeholder() -> None:
    persist_calls = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._ui_transcript = []

        def _persist_ui_transcript(self) -> None:
            persist_calls.append(list(self._ui_transcript))

    flow = _Flow()

    flow._record_ui_transcript_event(UIEvent(UIEventType.SYSTEM, "ready"))
    flow._record_ui_transcript_event(
        UIEvent(
            UIEventType.ASSISTANT,
            "[Previous assistant output was halted because it entered a repetition loop.]",
            data={"kind": "replace"},
        )
    )

    assert flow._ui_transcript == [persist_calls[0][0]]
    assert flow._ui_transcript[0]["event_type"] == "system"
    assert len(persist_calls) == 1


def test_hidden_task_complete_does_not_promote_summary_after_visible_answer() -> None:
    rendered: list[UIEvent] = []

    class _Console:
        def get_active_assistant_text(self) -> str:
            return "Hello! How can I help you today?"

        async def append_event(self, event: UIEvent) -> None:
            rendered.append(event)

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._show_system_messages = False
            self._pending_user_echo = None
            self.active_task = None

        def _get_console(self) -> object:
            return _Console()

        def _set_activity(self, text: str | None) -> None:
            pass

        def _refresh_status(self) -> None:
            pass

        def _record_ui_transcript_event(self, event: UIEvent) -> None:
            rendered.append(event)

        def _should_render_event(self, event: UIEvent) -> bool:
            return True

    flow = _Flow()

    asyncio.run(
        flow._handle_harness_event(
            UIEvent(
                UIEventType.TOOL_CALL,
                "task_complete",
                data={"args": {"message": "Replied to the user's greeting."}},
            )
        )
    )

    assert rendered == []


def test_ui_transcript_coalesces_adjacent_stream_chunks() -> None:
    persist_calls: list[list[dict[str, Any]]] = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._ui_transcript = []
            self._ui_transcript_recent_fingerprints = []
            self._ui_transcript_persist_handle = None
            self._ui_transcript_debounce_seconds = 0.0

        def _persist_ui_transcript(self) -> None:
            persist_calls.append(list(self._ui_transcript))

    flow = _Flow()

    flow._record_ui_transcript_event(UIEvent(UIEventType.ASSISTANT, "Hello! How can I"))
    flow._record_ui_transcript_event(UIEvent(UIEventType.ASSISTANT, " help you today"))
    flow._record_ui_transcript_event(UIEvent(UIEventType.ASSISTANT, "?\n\n"))
    flow._record_ui_transcript_event(UIEvent(UIEventType.THINKING, "Plan: "))
    flow._record_ui_transcript_event(UIEvent(UIEventType.THINKING, "reply."))
    flow._record_ui_transcript_event(UIEvent(UIEventType.USER, "hi"))
    flow._record_ui_transcript_event(UIEvent(UIEventType.ASSISTANT, "ok"))
    flow._record_ui_transcript_event(
        UIEvent(UIEventType.ASSISTANT, "replaced", data={"kind": "replace"})
    )

    assistant_events = [e for e in flow._ui_transcript if e["event_type"] == "assistant"]
    thinking_events = [e for e in flow._ui_transcript if e["event_type"] == "thinking"]

    assert len(assistant_events) == 3
    assert assistant_events[0]["content"] == "Hello! How can I help you today?\n\n"
    assert assistant_events[1]["content"] == "ok"
    assert assistant_events[2]["content"] == "replaced"
    assert assistant_events[2]["data"].get("kind") == "replace"

    assert len(thinking_events) == 1
    assert thinking_events[0]["content"] == "Plan: reply."


def test_ui_transcript_persistence_can_be_debounced() -> None:
    persist_calls = []

    class _Handle:
        def __init__(self, callback) -> None:
            self.callback = callback
            self._cancelled = False

        def cancelled(self) -> bool:
            return self._cancelled

        def cancel(self) -> None:
            self._cancelled = True

    class _Loop:
        def __init__(self) -> None:
            self.handles: list[_Handle] = []

        def call_later(self, delay: float, callback):
            handle = _Handle(callback)
            self.handles.append(handle)
            return handle

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._ui_transcript = []
            self._ui_transcript_persist_handle = None
            self._ui_transcript_debounce_seconds = 0.25

        def _persist_ui_transcript(self) -> None:
            persist_calls.append(list(self._ui_transcript))

    loop = _Loop()
    original_get_running_loop = asyncio.get_running_loop
    asyncio.get_running_loop = lambda: loop  # type: ignore[assignment]
    try:
        flow = _Flow()
        flow._record_ui_transcript_event(UIEvent(UIEventType.SYSTEM, "one"))
        flow._record_ui_transcript_event(UIEvent(UIEventType.SYSTEM, "two"))
        assert persist_calls == []
        assert len(loop.handles) == 1
        loop.handles[0].callback()
    finally:
        asyncio.get_running_loop = original_get_running_loop  # type: ignore[assignment]

    assert len(persist_calls) == 1
    assert [item["content"] for item in persist_calls[0]] == ["one", "two"]


def test_render_restored_chat_hides_tool_output_when_tool_calls_hidden() -> None:
    events: list[UIEvent] = []

    class _Console:
        async def clear_bubbles(self) -> None:
            return None

        async def append_event(self, event: UIEvent) -> None:
            events.append(event)

    class _Flow(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = None
            self._show_tool_calls = False
            self._show_system_messages = False
            self._verbose = False
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()
    messages = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "ssh_exec", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "ssh_exec",
            "tool_call_id": "call-1",
            "content": "--- [FAILURE SUMMARY] ---\nRemote SSH command exited with code 127",
        },
        {"event_type": "tool_result", "content": "live tool output", "data": {"tool_name": "shell_exec"}},
        {"event_type": "shell_stream", "content": "live shell stream", "data": {"tool_name": "shell_exec"}},
        {"event_type": "system", "content": "normal hidden system note", "data": {}},
    ]

    asyncio.run(flow._render_restored_chat(messages=messages))

    assert [(event.event_type, event.content) for event in events] == [
        (UIEventType.USER, "run it"),
    ]


def test_handle_slash_command_routes_planning_mode_through_bridge() -> None:
    refresh_calls: list[dict[str, object]] = []
    events: list[UIEvent] = []
    lines: list[str] = []

    class _Bridge:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        async def set_planning_mode(self, enabled: bool) -> dict[str, object]:
            self.calls.append(enabled)
            return {"model": "bridge-model", "phase": "explore", "mode": "planning", "step": 0}

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = object()
            self._harness_bridge = _Bridge()
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = None

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            refresh_calls.append(dict(snapshot or {}))

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            lines.append(text)

        async def on_harness_event(self, event: UIEvent) -> None:
            events.append(event)

    flow = _Flow()

    result = asyncio.run(flow._handle_slash_command("/plan-mode"))

    assert result is True
    assert flow._harness_bridge.calls == [True]
    assert refresh_calls == [{"model": "bridge-model", "phase": "explore", "mode": "planning", "step": 0}]
    assert lines == ["Planning mode enabled."]
    assert len(events) == 1
    assert events[0].event_type == UIEventType.ALERT
    assert events[0].content == "Planning mode enabled."


def test_maybe_restore_harness_state_prefers_bridge_payload() -> None:
    class _Bridge:
        async def restore_graph_state(self, thread_id: str | None = None) -> dict[str, object]:
            assert thread_id == "thread-restore"
            return {
                "restored": True,
                "thread_id": "thread-restore",
                "has_pending_interrupt": True,
                "interrupt": {"kind": "plan_execute_approval"},
                "snapshot": {"model": "bridge-model", "phase": "explore", "step": 0},
            }

    class _ExplodingHarness:
        def restore_graph_state(self, thread_id: str | None = None) -> bool:
            raise AssertionError("restore should go through the bridge")

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.fresh_run = False
            self.restore_graph_state_on_startup = True
            self.restore_thread_id = "thread-restore"
            self.harness = _ExplodingHarness()
            self._harness_bridge = _Bridge()
            self._latest_status_snapshot = None

    flow = _Flow()

    result = asyncio.run(flow._maybe_restore_harness_state())

    assert result == {
        "status": "restored",
        "thread_id": "thread-restore",
        "has_pending_interrupt": True,
        "interrupt": {"kind": "plan_execute_approval"},
    }
    assert flow._latest_status_snapshot == {"model": "bridge-model", "phase": "explore", "step": 0}


def test_maybe_restore_harness_state_loads_ui_transcript_for_checkpoint_restore(tmp_path) -> None:
    persist_chat_session_ui_transcript(
        cwd=tmp_path,
        thread_id="thread-restore",
        ui_transcript=[{"event_type": "system", "content": "checkpoint UI note", "data": {}}],
    )

    class _Bridge:
        async def restore_graph_state(self, thread_id: str | None = None) -> dict[str, object]:
            return {
                "restored": True,
                "thread_id": "thread-restore",
                "has_pending_interrupt": False,
                "recent_messages": [{"role": "user", "content": "prompt history"}],
            }

    class _Harness:
        state = SimpleNamespace(cwd=str(tmp_path))

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.fresh_run = False
            self.restore_graph_state_on_startup = True
            self.restore_thread_id = "thread-restore"
            self.harness = _Harness()
            self._harness_bridge = _Bridge()
            self._latest_status_snapshot = None

    result = asyncio.run(_Flow()._maybe_restore_harness_state())

    assert result is not None
    assert result["ui_transcript"] == [
        {"event_type": "system", "content": "checkpoint UI note", "data": {}}
    ]


def test_maybe_restore_harness_state_falls_back_to_saved_chat_state(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-saved"
    state.append_message(ConversationMessage(role="user", content="saved question"))
    state.append_message(ConversationMessage(role="assistant", content="saved answer"))
    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-saved",
        state_payload=state.to_dict(),
        model="test-model",
    )
    persist_chat_session_ui_transcript(
        cwd=tmp_path,
        thread_id="thread-saved",
        ui_transcript=[{"event_type": "system", "content": "restored UI note", "data": {}}],
    )

    class _Bridge:
        async def restore_graph_state(self, thread_id: str | None = None) -> dict[str, object]:
            return {"restored": False, "thread_id": str(thread_id or "")}

        async def replace_state_from_payload(self, state_payload: dict[str, object]) -> dict[str, object]:
            restored = LoopState.from_dict(state_payload)
            return {
                "thread_id": restored.thread_id,
                "snapshot": {"model": "test-model", "phase": restored.current_phase, "step": restored.step_count},
                "recent_messages": [
                    {"role": message.role, "content": message.content or ""}
                    for message in restored.transcript_messages
                ],
            }

    class _Harness:
        state = SimpleNamespace(cwd=str(tmp_path))

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.fresh_run = False
            self.restore_graph_state_on_startup = True
            self.restore_thread_id = "thread-saved"
            self.harness = _Harness()
            self._harness_bridge = _Bridge()
            self._latest_status_snapshot = None

    flow = _Flow()

    result = asyncio.run(flow._maybe_restore_harness_state())

    assert result is not None
    assert result["status"] == "restored"
    assert result["thread_id"] == "thread-saved"
    assert result["recent_messages"] == [
        {"role": "user", "content": "saved question"},
        {"role": "assistant", "content": "saved answer"},
    ]
    assert result["ui_transcript"] == [
        {"event_type": "system", "content": "restored UI note", "data": {}}
    ]


def test_saved_chat_state_preserves_separate_ui_transcript(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-ui"
    state.append_message(ConversationMessage(role="user", content="prompt state user"))

    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-ui",
        state_payload=state.to_dict(),
        model="test-model",
    )
    persist_chat_session_ui_transcript(
        cwd=tmp_path,
        thread_id="thread-ui",
        ui_transcript=[{"event_type": "system", "content": "UI-only recovery note", "data": {}}],
    )
    runtime_payload = json.loads(session_state_path(tmp_path, "thread-ui").read_text(encoding="utf-8"))
    ui_payload = json.loads(session_ui_transcript_path(tmp_path, "thread-ui").read_text(encoding="utf-8"))
    assert "ui_transcript" not in runtime_payload
    assert ui_payload["ui_transcript"] == [
        {"event_type": "system", "content": "UI-only recovery note", "data": {}}
    ]
    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-ui",
        state_payload=state.to_dict(),
        model="test-model",
    )

    assert load_chat_session_ui_transcript(cwd=tmp_path, thread_id="thread-ui") == [
        {"event_type": "system", "content": "UI-only recovery note", "data": {}}
    ]


def test_load_chat_session_ui_transcript_reads_legacy_combined_file(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-legacy-ui"
    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-legacy-ui",
        state_payload=state.to_dict(),
        model="test-model",
    )
    path = session_state_path(tmp_path, "thread-legacy-ui")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ui_transcript"] = [
        {"event_type": "system", "content": "legacy embedded note", "data": {}}
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    assert not session_ui_transcript_path(tmp_path, "thread-legacy-ui").exists()
    assert load_chat_session_ui_transcript(cwd=tmp_path, thread_id="thread-legacy-ui") == [
        {"event_type": "system", "content": "legacy embedded note", "data": {}}
    ]


def test_persist_chat_session_state_migrates_legacy_embedded_transcript(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-legacy-migrate"
    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-legacy-migrate",
        state_payload=state.to_dict(),
        model="test-model",
    )
    runtime_path = session_state_path(tmp_path, "thread-legacy-migrate")
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["ui_transcript"] = [
        {"event_type": "system", "content": "legacy note", "data": {}}
    ]
    runtime_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-legacy-migrate",
        state_payload=state.to_dict(),
        model="test-model",
    )

    assert "ui_transcript" not in json.loads(runtime_path.read_text(encoding="utf-8"))
    assert load_chat_session_ui_transcript(cwd=tmp_path, thread_id="thread-legacy-migrate") == [
        {"event_type": "system", "content": "legacy note", "data": {}}
    ]


def test_append_system_line_records_ui_transcript(tmp_path) -> None:
    lines: list[tuple[str, str]] = []

    class _Console:
        async def append_line(self, text: str, *, kind: str = "system") -> None:
            lines.append((text, kind))

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._show_system_messages = True
            self.console = _Console()
            self.harness = SimpleNamespace(
                state=SimpleNamespace(cwd=str(tmp_path), thread_id="thread-ui-line")
            )

        def _get_console(self) -> _Console:
            return self.console

    asyncio.run(_Flow()._append_system_line("Persistent UI line", kind="warning"))

    assert lines == [("Persistent UI line", "warning")]
    transcript = load_chat_session_ui_transcript(cwd=tmp_path, thread_id="thread-ui-line")
    assert transcript[-1]["event_type"] == "system"
    assert transcript[-1]["content"] == "Persistent UI line"
    assert transcript[-1]["data"] == {"kind": "warning"}


def test_refresh_status_updates_model_button_and_status_details() -> None:
    class _Task:
        def done(self) -> bool:
            return False

    class _ModelButton:
        def __init__(self) -> None:
            self.model = ""
            self.busy = False

        def set_model(self, model: str) -> None:
            self.model = model

        def set_busy(self, busy: bool) -> None:
            self.busy = busy

    class _Status:
        def __init__(self) -> None:
            self.kwargs = {}

        def set_state(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = None
            self.harness_config = HarnessConfig(endpoint="http://test/v1", model="wrench-9b", phase="execute")
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = _Task()
            self.model_button = _ModelButton()
            self.status = _Status()

        def query_one(self, widget_type: type[object]) -> object:
            if widget_type is ModelSelectButton:
                return self.model_button
            if widget_type is StatusBar:
                return self.status
            raise NoMatches("unexpected query")

    flow = _Flow()

    flow._refresh_status()

    assert flow.model_button.model == "wrench-9b"
    assert flow.model_button.busy is True
    assert flow.status.kwargs["model"] == "wrench-9b"
    assert flow.status.kwargs["phase"] == "execute"
    assert flow.status.kwargs["activity"] == "thinking..."


def test_refresh_status_prefers_cached_snapshot_over_live_harness_reads() -> None:
    class _ExplodingHarness:
        @property
        def state(self) -> object:
            raise AssertionError("refresh path should not read harness state")

    class _ModelButton:
        def __init__(self) -> None:
            self.model = ""
            self.busy = False

        def set_model(self, model: str) -> None:
            self.model = model

        def set_busy(self, busy: bool) -> None:
            self.busy = busy

    class _Status:
        def __init__(self) -> None:
            self.kwargs = {}

        def set_state(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = _ExplodingHarness()
            self.harness_config = HarnessConfig(endpoint="http://test/v1", model="wrench-9b", phase="execute")
            self._status_activity = ""
            self._api_error_count = 0
            self._latest_status_snapshot = {
                "model": "snapshot-7b",
                "phase": "verify",
                "step": 4,
                "mode": "planning",
                "activity": "cached activity",
            }
            self.active_task = None
            self.model_button = _ModelButton()
            self.status = _Status()

        def query_one(self, widget_type: type[object]) -> object:
            if widget_type is ModelSelectButton:
                return self.model_button
            if widget_type is StatusBar:
                return self.status
            raise NoMatches("unexpected query")

    flow = _Flow()

    flow._refresh_status()

    assert flow.model_button.model == "snapshot-7b"
    assert flow.status.kwargs["phase"] == "verify"
    assert flow.status.kwargs["step"] == 4
    assert flow.status.kwargs["activity"] == "cached activity"


def test_refresh_status_without_cached_snapshot_uses_defaults_not_live_harness() -> None:
    class _ExplodingHarness:
        @property
        def state(self) -> object:
            raise AssertionError("refresh path should not read harness state")

    class _Task:
        def done(self) -> bool:
            return False

    class _ModelButton:
        def __init__(self) -> None:
            self.model = ""
            self.busy = False

        def set_model(self, model: str) -> None:
            self.model = model

        def set_busy(self, busy: bool) -> None:
            self.busy = busy

    class _Status:
        def __init__(self) -> None:
            self.kwargs = {}

        def set_state(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = _ExplodingHarness()
            self.harness_config = HarnessConfig(endpoint="http://test/v1", model="fallback-3b", phase="execute", contract_flow_ui=True)
            self._status_activity = ""
            self._api_error_count = 2
            self._latest_status_snapshot = None
            self.active_task = _Task()
            self.model_button = _ModelButton()
            self.status = _Status()

        def query_one(self, widget_type: type[object]) -> object:
            if widget_type is ModelSelectButton:
                return self.model_button
            if widget_type is StatusBar:
                return self.status
            raise NoMatches("unexpected query")

    flow = _Flow()

    flow._refresh_status()

    assert flow.model_button.model == "fallback-3b"
    assert flow.model_button.busy is True
    assert flow.status.kwargs["phase"] == "execute"
    assert flow.status.kwargs["activity"] == "thinking..."
    assert flow.status.kwargs["contract_flow_ui"] is True
    assert flow.status.kwargs["api_errors"] == 2


def test_model_bar_layout_toggle_hides_old_row_and_shows_sidebar() -> None:
    class _Widget:
        def __init__(self) -> None:
            self.classes: dict[str, bool] = {}
            self.label = ""

        def set_class(self, enabled: bool, class_name: str) -> None:
            self.classes[class_name] = enabled

    class _Actions(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self._model_bar_layout = "bottom"
            self.widgets = {
                "#status-row": _Widget(),
                "#model-sidebar": _Widget(),
                "#model-bar-toggle": _Widget(),
                "#model-bar-toggle-sidebar": _Widget(),
            }

        def query_one(self, selector: str, *_args) -> _Widget:
            return self.widgets[selector]

    actions = _Actions()

    actions.action_toggle_model_bar_layout()

    assert actions._model_bar_layout == "right"
    assert actions.widgets["#status-row"].classes["hidden"] is True
    assert actions.widgets["#model-sidebar"].classes["hidden"] is False

    actions.action_toggle_model_bar_layout()

    assert actions._model_bar_layout == "bottom"
    assert actions.widgets["#status-row"].classes["hidden"] is False
    assert actions.widgets["#model-sidebar"].classes["hidden"] is True


def test_smallctl_app_model_bar_toggle_binding_switches_layout(monkeypatch) -> None:
    async def _fake_create_harness(self: SmallctlApp) -> None:
        async def _teardown() -> None:
            return None

        self.harness = SimpleNamespace(
            state=SimpleNamespace(
                current_phase="explore",
                step_count=0,
                planning_mode_enabled=False,
                active_plan=None,
                draft_plan=None,
                scratchpad={},
                token_usage=0,
                contract_phase=lambda: "",
                acceptance_checklist=lambda: [],
                current_verifier_verdict=lambda: None,
            ),
            context_policy=SimpleNamespace(max_prompt_tokens=4096),
            server_context_limit=None,
            guards=SimpleNamespace(max_tokens=4096),
            set_interactive_shell_approval=lambda enabled: None,
            set_shell_approval_session_default=lambda enabled: None,
            teardown=_teardown,
        )

    monkeypatch.setattr(SmallctlApp, "_create_harness", _fake_create_harness)

    async def _run() -> tuple[bool, bool, bool, bool, bool]:
        app = SmallctlApp({"endpoint": "http://example.test/v1", "model": "alpha-model"})
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            status_row = app.query_one("#status-row")
            sidebar = app.query_one("#model-sidebar")
            initial = (status_row.has_class("hidden"), sidebar.has_class("hidden"))
            labels_synced = "alpha-model" in str(app.query_one("#model-button-sidebar").label)
            await pilot.press("ctrl+alt+b")
            await pilot.pause(0.2)
            toggled = (status_row.has_class("hidden"), sidebar.has_class("hidden"))
            return initial + toggled + (labels_synced,)

    assert asyncio.run(_run()) == (False, True, True, False, True)



def test_right_model_bar_clips_streamed_tool_output(monkeypatch) -> None:
    async def _fake_create_harness(self: SmallctlApp) -> None:
        async def _teardown() -> None:
            return None

        self.harness = SimpleNamespace(
            state=SimpleNamespace(
                current_phase="explore",
                step_count=0,
                planning_mode_enabled=False,
                active_plan=None,
                draft_plan=None,
                scratchpad={},
                token_usage=0,
                contract_phase=lambda: "",
                acceptance_checklist=lambda: [],
                current_verifier_verdict=lambda: None,
            ),
            context_policy=SimpleNamespace(max_prompt_tokens=4096),
            server_context_limit=None,
            guards=SimpleNamespace(max_tokens=4096),
            set_interactive_shell_approval=lambda enabled: None,
            set_shell_approval_session_default=lambda enabled: None,
            teardown=_teardown,
        )

    monkeypatch.setattr(SmallctlApp, "_create_harness", _fake_create_harness)

    async def _run() -> None:
        app = SmallctlApp({"endpoint": "http://example.test/v1", "model": "alpha-model"})
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            app.action_toggle_model_bar_layout()
            await pilot.pause(0.2)

            console = app.query_one(ConsolePane)
            await console.append_event(
                UIEvent(
                    UIEventType.TOOL_CALL,
                    "ssh_exec",
                    data={
                        "display_text": "ssh_exec(command=\"" + ("x" * 500) + "\")",
                        "tool_call_id": "tool-1",
                    },
                )
            )
            await console.append_event(
                UIEvent(
                    UIEventType.SHELL_STREAM,
                    "remote output " + ("y" * 800),
                    data={"tool_name": "ssh_exec", "tool_call_id": "tool-1"},
                )
            )
            await pilot.pause(0.2)

            console_wrap = app.query_one("#console-wrap")
            sidebar = app.query_one("#model-sidebar")
            assert not sidebar.has_class("hidden")
            assert console_wrap.region.right <= sidebar.region.x
            assert console.region.right <= sidebar.region.x

    asyncio.run(_run())

def test_subtask_checklist_updates_goal_bar_in_right_layout() -> None:
    appended: list[UIEvent] = []

    class _Toggle:
        def __init__(self) -> None:
            self.label = ""

    class _Details:
        def __init__(self) -> None:
            self.text = ""
            self.hidden = False

        def update(self, text: str) -> None:
            self.text = text

        def set_class(self, value: bool, name: str) -> None:
            if name == "hidden":
                self.hidden = value

    class _Console:
        async def append_event(self, event: UIEvent) -> None:
            appended.append(event)

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._model_bar_layout = "right"
            self._status_activity = ""
            self._show_system_messages = False
            self._show_tool_calls = True
            self.active_task = None
            self._goal_bar_expanded = True
            self._goal_bar_goal = "No active goal"
            self._goal_bar_tasks = []
            self.toggle = _Toggle()
            self.details = _Details()
            self.console = _Console()

        def query_one(self, selector: str, *_args):
            if selector == "#goal-bar-toggle":
                return self.toggle
            if selector == "#goal-bar-details":
                return self.details
            raise NoMatches("unexpected query")

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="Goal Objective: implement the right sidebar\n  ○ wire status",
        data={"ui_kind": "subtask_checklist"},
    )

    asyncio.run(flow._handle_harness_event(event))

    assert flow.toggle.label == "^ Goal: implement the right sidebar"
    assert "[bold #93c5fd]Goal[/] implement the right sidebar" in flow.details.text
    assert "[bold #bfdbfe]Tasks[/]" in flow.details.text
    assert "[#94a3b8]○ wire status[/]" in flow.details.text
    assert flow.details.hidden is False
    assert appended == []


def test_subtask_checklist_updates_goal_bar_in_bottom_layout() -> None:
    appended: list[UIEvent] = []

    class _Toggle:
        def __init__(self) -> None:
            self.label = ""

    class _Details:
        def __init__(self) -> None:
            self.text = ""
            self.hidden = False

        def update(self, text: str) -> None:
            self.text = text

        def set_class(self, value: bool, name: str) -> None:
            if name == "hidden":
                self.hidden = value

    class _Console:
        async def append_event(self, event: UIEvent) -> None:
            appended.append(event)

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._model_bar_layout = "bottom"
            self._status_activity = ""
            self._show_system_messages = False
            self._show_tool_calls = True
            self._verbose = False
            self.active_task = None
            self._goal_bar_expanded = False
            self._goal_bar_goal = "No active goal"
            self._goal_bar_tasks = []
            self.toggle = _Toggle()
            self.details = _Details()
            self.console = _Console()

        def query_one(self, selector: str, *_args):
            if selector == "#goal-bar-toggle":
                return self.toggle
            if selector == "#goal-bar-details":
                return self.details
            raise NoMatches("unexpected query")

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()
    event = UIEvent(
        event_type=UIEventType.ALERT,
        content="Goal Objective: keep bottom layout behavior",
        data={"ui_kind": "subtask_checklist"},
    )

    asyncio.run(flow._handle_harness_event(event))

    assert flow.toggle.label == "v Goal: keep bottom layout behavior"
    assert flow.details.text == ""
    assert flow.details.hidden is True
    assert appended == []


def test_status_events_are_coalesced_latest_wins() -> None:
    calls: list[dict[str, object]] = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._latest_status_snapshot = None
            self._status_refresh_pending = False
            self._pending_user_echo = None
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = None

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            calls.append(dict(snapshot or {}))

    flow = _Flow()

    async def _run() -> None:
        await flow._handle_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "first", "phase": "plan", "step": 1}},
            )
        )
        await flow._handle_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "second", "phase": "execute", "step": 2}},
            )
        )
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert calls == [{"model": "second", "phase": "execute", "step": 2}]


def test_set_shell_approval_session_default_prefers_bridge() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.values: list[bool] = []

        def set_shell_approval_session_default(self, enabled: bool) -> None:
            self.values.append(enabled)

    class _Harness:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def set_shell_approval_session_default(self, enabled: bool) -> None:
            self.calls.append(enabled)

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._shell_approval_session_default = False
            self.harness = _Harness()
            self._harness_bridge = _Bridge()

    flow = _Flow()

    flow._set_shell_approval_session_default(True)

    assert flow._shell_approval_session_default is True
    assert flow._harness_bridge.values == [True]
    assert flow.harness.calls == []


def test_render_restored_chat_uses_serialized_messages_without_harness_state_reads() -> None:
    rendered: list[tuple[UIEventType, str]] = []

    class _Console:
        async def clear_bubbles(self) -> None:
            return None

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content))

    class _ExplodingHarness:
        @property
        def state(self) -> object:
            raise AssertionError("serialized restore messages should avoid live harness reads")

    class _Flow(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = _ExplodingHarness()
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()

    asyncio.run(
        flow._render_restored_chat(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "system", "content": "note"},
            ]
        )
    )

    assert rendered == [
        (UIEventType.USER, "hello"),
        (UIEventType.ASSISTANT, "hi"),
    ]


def test_render_restored_chat_shows_recovery_system_messages() -> None:
    rendered: list[tuple[UIEventType, str]] = []

    class _Console:
        async def clear_bubbles(self) -> None:
            return None

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content))

    class _Flow(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = None
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()

    asyncio.run(
        flow._render_restored_chat(
            messages=[
                {
                    "role": "system",
                    "content": "Auto-continuing patch recovery for `x.py`.",
                    "metadata": {"is_recovery_nudge": True, "recovery_kind": "file_patch_read_autocontinue"},
                },
            ]
        )
    )

    assert rendered == [(UIEventType.SYSTEM, "Auto-continuing patch recovery for `x.py`.")]


def test_render_restored_chat_hides_internal_recovery_prompts() -> None:
    rendered: list[tuple[UIEventType, str]] = []

    class _Console:
        async def clear_bubbles(self) -> None:
            return None

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content))

    class _Flow(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = None
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()

    asyncio.run(
        flow._render_restored_chat(
            messages=[
                {
                    "role": "system",
                    "content": "The human answered affirmatively to the clarification request.",
                    "metadata": {
                        "is_recovery_nudge": True,
                        "recovery_kind": "ask_human_affirmative_resume",
                        "hidden_from_ui": True,
                    },
                },
            ]
        )
    )

    assert rendered == []


def test_render_restored_chat_shows_ui_transcript_system_messages() -> None:
    rendered: list[tuple[UIEventType, str]] = []

    class _Console:
        async def clear_bubbles(self) -> None:
            return None

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content))

    class _Flow(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = None
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

    asyncio.run(
        _Flow()._render_restored_chat(
            messages=[{"event_type": "system", "content": "UI-only status note", "data": {}}]
        )
    )

    assert rendered == [(UIEventType.SYSTEM, "UI-only status note")]


def test_submitted_paste_renders_full_user_bubble_and_runs_full_task() -> None:
    rendered: list[tuple[UIEventType, str]] = []
    started: list[str] = []
    recorded: list[str] = []

    class _Input:
        text = "[pasted ~541 chars]"

    class _Console:
        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content))

        async def begin_assistant_turn(self) -> None:
            return None

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.input = _Input()
            self.console = _Console()
            self.task_history = []
            self.history_index = 0
            self.active_task = None
            self._pending_user_echo = None
            self._status_activity = ""
            self._api_error_count = 0
            self._app_logger = logging.getLogger("test.ui_flow_regressions.submitted_paste")

        def query_one(self, widget_type: type[object]) -> object:
            return self.input

        def _get_console(self) -> _Console:
            return self.console

        def _record_chat_session_prompt(self, task: str) -> None:
            recorded.append(task)

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

        def _refresh_status(self, *args, **kwargs) -> None:
            return None

        async def _run_harness_task(self, task: str) -> None:
            started.append(task)

    full_text = "x" * 541
    flow = _Flow()

    async def _run() -> None:
        await flow.on_input_pane_submitted(InputPane.Submitted(full_text, display_value="[pasted ~541 chars]"))
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert flow.input.text == ""
    assert rendered == [(UIEventType.USER, full_text)]
    assert recorded == [full_text]
    assert started == [full_text]


def test_on_harness_event_coalesces_queued_status_events_latest_wins() -> None:
    calls: list[dict[str, object]] = []

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._latest_status_snapshot = None
            self._status_refresh_pending = False
            self._pending_harness_events = []
            self._pending_status_event = None
            self._ui_event_drain_task = None
            self._pending_user_echo = None
            self._status_activity = ""
            self._api_error_count = 0
            self.active_task = None

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            calls.append(dict(snapshot or {}))

    flow = _Flow()

    async def _run() -> None:
        await flow.on_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "first", "phase": "plan", "step": 1}},
            )
        )
        await flow.on_harness_event(
            UIEvent(
                event_type=UIEventType.STATUS,
                data={"snapshot": {"model": "second", "phase": "execute", "step": 2}},
            )
        )
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert calls == [{"model": "second", "phase": "execute", "step": 2}]


def test_run_harness_task_backfills_terminal_result_into_assistant_transcript() -> None:
    rendered: list[tuple[UIEventType, str, dict[str, object]]] = []
    system_lines: list[str] = []

    class _Console:
        def __init__(self) -> None:
            self._assistant_text = ""

        def get_active_assistant_text(self) -> str:
            return self._assistant_text

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content, dict(event.data)))
            if event.event_type == UIEventType.ASSISTANT:
                self._assistant_text += str(event.content)

    class _Bridge:
        async def run_auto(self, task: str) -> dict[str, object]:
            assert task == "weather"
            return {
                "status": "completed",
                "message": {"status": "complete", "message": "Jacksonville is 85F today."},
                "assistant": "",
            }

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = object()
            self._harness_bridge = _Bridge()
            self._status_activity = ""
            self._api_error_count = 0
            self._pending_user_echo = None
            self._show_system_messages = True
            self._show_tool_calls = True
            self.active_task = None
            self._task_start_time = None
            self._activity_timer = None
            self.console = _Console()
            self._app_logger = logging.getLogger("test.ui_flow_regressions.terminal_result")

        def _get_console(self) -> _Console:
            return self.console

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            return None

        def set_interval(self, interval: float, callback: object) -> object:
            class _Timer:
                def stop(self) -> None:
                    return None

            return _Timer()

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            system_lines.append(text)

    flow = _Flow()

    asyncio.run(flow._run_harness_task("weather"))

    assistant_events = [item for item in rendered if item[0] == UIEventType.ASSISTANT]
    assert assistant_events == [
        (
            UIEventType.ASSISTANT,
            "Jacksonville is 85F today.",
            {"promoted_from": "terminal_result"},
        )
    ]
    assert system_lines == [
        "Task completed. Type a new message or press Ctrl+C to exit."
    ]


def test_run_harness_task_forces_pre_stream_failure_visible_when_system_hidden() -> None:
    system_lines: list[tuple[str, bool]] = []
    status_steps: list[int | str | None] = []

    class _Console:
        pass

    class _Bridge:
        async def run_auto(self, task: str) -> dict[str, object]:
            assert task == "next task"
            raise RuntimeError("initialize_run failed before model_call")

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = object()
            self._harness_bridge = _Bridge()
            self._status_activity = ""
            self._api_error_count = 0
            self._pending_user_echo = "next task"
            self._show_system_messages = False
            self._show_tool_calls = False
            self.active_task = object()
            self._task_start_time = None
            self._activity_timer = None
            self.console = _Console()
            self._app_logger = logging.getLogger("test.ui_flow_regressions.pre_stream_failure")

        def _get_console(self) -> _Console:
            return self.console

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            status_steps.append(step_override)

        def set_interval(self, interval: float, callback: object) -> object:
            class _Timer:
                def stop(self) -> None:
                    return None

            return _Timer()

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            if not force and not self._show_system_messages:
                return
            system_lines.append((text, force))

    flow = _Flow()

    asyncio.run(flow._run_harness_task("next task"))

    assert system_lines == [("Task failed: initialize_run failed before model_call", True)]
    assert status_steps[-1] == "error"
    assert flow.active_task is None
    assert flow._pending_user_echo is None


def test_run_harness_task_shows_unverified_change_warning() -> None:
    system_lines: list[tuple[str, bool, str | None]] = []

    class _Console:
        def get_active_assistant_text(self) -> str:
            return ""

    class _Bridge:
        async def run_auto(self, task: str) -> dict[str, object]:
            assert task == "edit"
            return {
                "status": "cancelled",
                "reason": "cancel_requested",
                "unverified_change_warning": "Task cancelled after modifying files to temp/example.py. Changes were not verified.",
            }

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = object()
            self._harness_bridge = _Bridge()
            self._status_activity = ""
            self._api_error_count = 0
            self._pending_user_echo = None
            self._show_system_messages = False
            self._show_tool_calls = True
            self.active_task = None
            self._task_start_time = None
            self._activity_timer = None
            self.console = _Console()
            self._app_logger = logging.getLogger("test.ui_flow_regressions.unverified_warning")

        def _get_console(self) -> _Console:
            return self.console

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

        def _refresh_status(
            self,
            step_override: int | str | None = None,
            *,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            return None

        def set_interval(self, interval: float, callback: object) -> object:
            class _Timer:
                def stop(self) -> None:
                    return None

            return _Timer()

        async def _append_system_line(self, text: str, *, force: bool = False, kind: str | None = None) -> None:
            system_lines.append((text, force, kind))

    asyncio.run(_Flow()._run_harness_task("edit"))

    assert (
        "Task cancelled after modifying files to temp/example.py. Changes were not verified.",
        True,
        "warning",
    ) in system_lines


def test_task_complete_message_is_not_promoted_over_streamed_assistant_text() -> None:
    rendered: list[tuple[UIEventType, str, dict[str, object]]] = []

    class _Console:
        def get_active_assistant_text(self) -> str:
            return "Hello! How can I help you today?"

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content, dict(event.data)))

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self._pending_user_echo = None
            self._show_system_messages = False
            self._show_tool_calls = False
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

        def _update_activity_for_event(self, event: UIEvent) -> None:
            return None

        async def on_harness_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content, dict(event.data)))

    flow = _Flow()

    asyncio.run(
        flow._handle_harness_event(
            UIEvent(
                event_type=UIEventType.TOOL_CALL,
                content="task_complete",
                data={
                    "args": {"message": "Hello! I'm ready to help with whatever you need."},
                    "tool_name": "task_complete",
                },
            )
        )
    )

    assert rendered == []


def test_chat_completed_terminal_result_does_not_append_over_streamed_assistant_text() -> None:
    rendered: list[tuple[UIEventType, str, dict[str, object]]] = []

    class _Console:
        def get_active_assistant_text(self) -> str:
            return "Hello! How can I help you today?"

        async def append_event(self, event: UIEvent) -> None:
            rendered.append((event.event_type, event.content, dict(event.data)))

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.console = _Console()

        def _get_console(self) -> _Console:
            return self.console

    flow = _Flow()

    asyncio.run(
        flow._maybe_render_terminal_result(
            {
                "status": "chat_completed",
                "message": "Hello! I'm ready to help with whatever you need.",
                "assistant": "Hello! How can I help you today?",
            }
        )
    )

    assert rendered == []


def test_handle_approval_prompt_prefers_bridge_resolution_after_thread_split() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.resolutions: list[tuple[str, bool]] = []

        def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
            self.resolutions.append((approval_id, approved))

    class _Harness:
        def __init__(self) -> None:
            self.state = SimpleNamespace(cwd="/repo")

        def resolve_shell_approval(self, approval_id: str, approved: bool) -> None:
            raise AssertionError("bridge should own shell approval resolution")

    class _App:
        def __init__(self) -> None:
            self.harness = _Harness()
            self._harness_bridge = _Bridge()
            self._active_approval_prompt = None
            self._status_activity = ""
            self._shell_approval_session_default = False
            self._app_logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

        async def push_screen(self, prompt, callback) -> None:
            self._active_approval_prompt = prompt
            callback(ShellApprovalDecision(True, True))

        def _refresh_status(self) -> None:
            return None

        def _set_shell_approval_session_default(self, enabled: bool) -> None:
            self._shell_approval_session_default = bool(enabled)

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            return None

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

    app = _App()

    asyncio.run(
        handle_approval_prompt(
            app,
            UIEvent(
                event_type=UIEventType.ALERT,
                data={
                    "ui_kind": "approve_prompt",
                    "approval_id": "shell-123",
                    "command": "pytest",
                    "cwd": "/repo",
                    "timeout_sec": 30,
                    "proof_bundle": {"phase": "verify"},
                },
            ),
        )
    )

    assert app._harness_bridge.resolutions == [("shell-123", True)]
    assert app._shell_approval_session_default is True
    assert app._status_activity == "running shell..."
    assert app._active_approval_prompt is None


def test_handle_sudo_password_prompt_prefers_bridge_resolution_after_thread_split() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.resolutions: list[tuple[str, str | None]] = []

        def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
            self.resolutions.append((prompt_id, password))

    class _Harness:
        def __init__(self) -> None:
            self.state = SimpleNamespace(cwd="/repo")

        def resolve_sudo_password(self, prompt_id: str, password: str | None) -> None:
            raise AssertionError("bridge should own sudo prompt resolution")

    class _App:
        def __init__(self) -> None:
            self.harness = _Harness()
            self._harness_bridge = _Bridge()
            self._active_approval_prompt = None
            self._status_activity = ""
            self._app_logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

        async def push_screen(self, prompt, callback) -> None:
            self._active_approval_prompt = prompt
            callback("secret")

        def _refresh_status(self) -> None:
            return None

        def _set_activity(self, text: str | None) -> None:
            self._status_activity = str(text or "")

    app = _App()

    asyncio.run(
        handle_sudo_password_prompt(
            app,
            UIEvent(
                event_type=UIEventType.ALERT,
                data={
                    "ui_kind": "sudo_password_prompt",
                    "prompt_id": "sudo-123",
                    "command": "sudo ls",
                    "prompt_text": "Password please",
                },
            ),
        )
    )

    assert app._harness_bridge.resolutions == [("sudo-123", "secret")]
    assert app._status_activity == "running shell..."
    assert app._active_approval_prompt is None


def test_action_cancel_task_handles_race_where_active_task_becomes_none() -> None:
    cancelled: list[asyncio.Task[None]] = []

    class _Task:
        def __init__(self) -> None:
            self._done = False

        def done(self) -> bool:
            return self._done

        def cancel(self) -> None:
            cancelled.append(self)  # type: ignore[arg-type]

    class _Actions(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.harness = None
            self._harness_bridge = None
            self.active_task: _Task | None = _Task()
            self._app_logger = logging.getLogger("test.ui_flow_regressions.cancel_race")
            self._lines: list[str] = []

        def _dismiss_active_approval_prompt(self) -> None:
            return None

        def _get_console(self) -> None:
            return None

        async def _append_system_line(self, text: str, *, force: bool = False, kind: str | None = None) -> None:
            self._lines.append(text)

        def _refresh_status(self, step_override: int | str | None = None, *, snapshot: dict[str, object] | None = None) -> None:
            return None

    actions = _Actions()

    async def _run() -> None:
        # Simulate the race: after the cancel task starts checking, the task
        # finishes and clears self.active_task before the await resumes.
        task = actions.active_task
        assert task is not None

        async def _clear_active_task_after_yield() -> None:
            await asyncio.sleep(0)
            actions.active_task = None

        asyncio.create_task(_clear_active_task_after_yield())
        await actions.action_cancel_task()
        assert actions.active_task is None
        assert not cancelled

    asyncio.run(_run())
