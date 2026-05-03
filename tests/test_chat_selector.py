from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from textual.app import App, ComposeResult
from textual.widgets import Button

from smallctl.chat_sessions import persist_chat_session_state
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState
from smallctl.ui.app import SmallctlApp
from smallctl.ui.chat_selector import ChatMenuScreen, ChatSelectButton, ChatSessionSelectScreen
from smallctl.ui.chat_sessions import (
    ChatSessionSummary,
    format_relative_age,
    load_chat_session_summaries,
    record_chat_session_prompt,
)


def test_chat_session_index_records_first_message_and_updates_timestamp(tmp_path) -> None:
    record_chat_session_prompt(
        cwd=tmp_path,
        thread_id="thread-1",
        message="first prompt",
        model="alpha-model",
        created_at="2026-04-23T12:00:00+00:00",
    )
    record_chat_session_prompt(
        cwd=tmp_path,
        thread_id="thread-1",
        message="second prompt",
        model="beta-model",
        created_at="2026-04-23T12:00:00+00:00",
    )

    sessions = load_chat_session_summaries(cwd=tmp_path)

    assert len(sessions) == 1
    assert sessions[0].thread_id == "thread-1"
    assert sessions[0].first_user_message == "first prompt"
    assert sessions[0].model == "beta-model"


def test_format_relative_age_uses_min_hour_day_units() -> None:
    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)

    assert format_relative_age("2026-04-23T11:58:00+00:00", now=now) == "2m ago"
    assert format_relative_age("2026-04-23T09:00:00+00:00", now=now) == "3h ago"
    assert format_relative_age("2026-04-20T12:00:00+00:00", now=now) == "3d ago"


def test_chat_select_button_click_emits_button_pressed() -> None:
    class _App(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.pressed_id = ""

        def compose(self) -> ComposeResult:
            yield ChatSelectButton(id="chat-button")

        def on_button_pressed(self, event) -> None:
            self.pressed_id = str(event.button.id or "")

    async def _run() -> str:
        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#chat-button")
            await pilot.pause(0.2)
            return app.pressed_id

    assert asyncio.run(_run()) == "chat-button"


def test_chat_menu_screen_returns_resume_choice() -> None:
    class _App(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.selection: str | None = None

        def compose(self) -> ComposeResult:
            yield Button("Open", id="open")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "open":
                await self.push_screen(
                    ChatMenuScreen(),
                    callback=lambda selection: setattr(self, "selection", selection),
                )

    async def _run() -> str | None:
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#open")
            await pilot.pause(0.2)
            await pilot.click("#chat-menu-resume")
            await pilot.pause(0.2)
            return app.selection

    assert asyncio.run(_run()) == "resume"


def test_chat_session_screen_returns_selected_thread() -> None:
    class _App(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.selection: str | None = None

        def compose(self) -> ComposeResult:
            yield Button("Open", id="open")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "open":
                await self.push_screen(
                    ChatSessionSelectScreen(
                        sessions=[
                            ChatSessionSummary(
                                thread_id="thread-1",
                                first_user_message="first prompt",
                                created_at="2026-04-23T11:58:00+00:00",
                                updated_at="2026-04-23T12:00:00+00:00",
                                model="alpha-model",
                            )
                        ],
                        now=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
                    ),
                    callback=lambda selection: setattr(self, "selection", selection),
                )

    async def _run() -> str | None:
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#open")
            await pilot.pause(0.2)
            assert app.screen.sessions[0].first_user_message == "first prompt"
            await pilot.press("enter")
            await pilot.pause(0.2)
            return app.selection

    assert asyncio.run(_run()) == "thread-1"


def test_smallctl_app_chat_button_starts_new_chat(monkeypatch) -> None:
    create_calls = 0

    def _fake_create_harness(self: SmallctlApp) -> None:
        nonlocal create_calls
        create_calls += 1

        async def _teardown() -> None:
            return None

        self.harness = SimpleNamespace(
            client=SimpleNamespace(model="alpha-model"),
            state=SimpleNamespace(
                cwd=".",
                thread_id=f"thread-{create_calls}",
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

    async def _run() -> int:
        app = SmallctlApp({"endpoint": "http://example.test/v1", "model": "alpha-model"})
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.click("#chat-button")
            await pilot.pause(0.2)
            assert isinstance(app.screen, ChatMenuScreen)
            await pilot.click("#chat-menu-new")
            await pilot.pause(0.3)
            return create_calls

    assert asyncio.run(_run()) == 2


def test_smallctl_app_chat_resume_lists_sessions_and_restores_selection(monkeypatch, tmp_path) -> None:
    restored: list[str] = []

    record_chat_session_prompt(
        cwd=tmp_path,
        thread_id="thread-1",
        message="first saved prompt",
        model="alpha-model",
        created_at="2026-04-23T12:00:00+00:00",
    )

    def _fake_create_harness(self: SmallctlApp) -> None:
        async def _teardown() -> None:
            return None

        self.harness = SimpleNamespace(
            client=SimpleNamespace(model="alpha-model"),
            state=SimpleNamespace(
                cwd=str(tmp_path),
                thread_id="thread-current",
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

    async def _fake_resume(self: SmallctlApp, thread_id: str) -> None:
        restored.append(thread_id)

    monkeypatch.setattr(SmallctlApp, "_create_harness", _fake_create_harness)
    monkeypatch.setattr(SmallctlApp, "_resume_chat_session", _fake_resume)

    async def _run() -> list[str]:
        app = SmallctlApp({"endpoint": "http://example.test/v1", "model": "alpha-model"})
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.click("#chat-button")
            await pilot.pause(0.2)
            await pilot.click("#chat-menu-resume")
            await pilot.pause(0.3)
            assert isinstance(app.screen, ChatSessionSelectScreen)
            assert app.screen.sessions[0].first_user_message == "first saved prompt"
            await pilot.press("enter")
            await pilot.pause(0.3)
            return list(restored)

    assert asyncio.run(_run()) == ["thread-1"]


def test_smallctl_app_chat_resume_falls_back_to_saved_state(monkeypatch, tmp_path) -> None:
    restored_states: list[str] = []

    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-1"
    state.scratchpad["_model_name"] = "alpha-model"
    state.append_message(ConversationMessage(role="user", content="saved hello"))

    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-1",
        state_payload=state.to_dict(),
        model="alpha-model",
    )

    def _fake_create_harness(self: SmallctlApp) -> None:
        async def _teardown() -> None:
            return None

        harness = SimpleNamespace(
            client=SimpleNamespace(model="alpha-model"),
            state=SimpleNamespace(
                cwd=str(tmp_path),
                thread_id="thread-current",
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
            restore_graph_state=lambda thread_id=None: False,
            _sync_run_logger_session_id=lambda: None,
            teardown=_teardown,
        )
        self.harness = harness

    async def _fake_render_restored_chat(
        self: SmallctlApp, messages: list[dict[str, str]] | None = None
    ) -> None:
        restored_states.append(str(self.harness.state.thread_id))

    monkeypatch.setattr(SmallctlApp, "_create_harness", _fake_create_harness)
    monkeypatch.setattr(SmallctlApp, "_render_restored_chat", _fake_render_restored_chat)

    async def _run() -> list[str]:
        app = SmallctlApp({"endpoint": "http://example.test/v1", "model": "alpha-model"})
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await app._resume_chat_session("thread-1")
            await pilot.pause(0.2)
            return list(restored_states)

    assert asyncio.run(_run()) == ["thread-1"]
