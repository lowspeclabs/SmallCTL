from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import smallctl.client.model_listing as model_listing
import smallctl.ui.model_selector as model_selector
from smallctl.client.model_listing import (
    ModelListResult,
    ProviderModel,
    fetch_available_models,
    parse_lmstudio_models,
    parse_ollama_models,
    parse_openai_models,
)
from smallctl.harness import Harness
from smallctl.ui.app import SmallctlApp
from smallctl.ui.app_actions import SmallctlAppActionsMixin
from smallctl.ui.app_flow import SmallctlAppFlowMixin
from smallctl.ui.model_selector import ModelSelectButton, ModelSelectScreen


def test_openai_compatible_model_list_parsing_returns_data_ids() -> None:
    models = parse_openai_models(
        {
            "data": [
                {"id": "gpt-5.4", "context_length": 128000},
                {"id": "wrench-9b"},
                {"object": "ignored"},
            ]
        }
    )

    assert [model.id for model in models] == ["gpt-5.4", "wrench-9b"]
    assert models[0].context_length == 128000
    assert models[0].metadata["id"] == "gpt-5.4"


def test_lmstudio_model_list_parsing_returns_keys_and_loaded_metadata() -> None:
    models = parse_lmstudio_models(
        {
            "models": [
                {
                    "key": "wrench-9b",
                    "display_name": "Wrench 9B",
                    "loaded_instances": [{"id": "instance-1"}],
                    "max_context_length": 32768,
                }
            ]
        }
    )

    assert len(models) == 1
    assert models[0].id == "wrench-9b"
    assert models[0].display_name == "Wrench 9B"
    assert models[0].loaded is True
    assert models[0].context_length == 32768
    assert models[0].metadata["loaded_instances"] == [{"id": "instance-1"}]


def test_ollama_model_list_parsing_returns_model_names() -> None:
    models = parse_ollama_models(
        {
            "models": [
                {"name": "qwen3.5:4b", "size": 123},
                {"model": "llama3.2:latest"},
            ]
        }
    )

    assert [model.id for model in models] == ["qwen3.5:4b", "llama3.2:latest"]
    assert models[0].metadata["size"] == 123


def test_provider_failure_returns_error(monkeypatch) -> None:
    class _Response:
        status_code = 503

        def json(self) -> dict[str, Any]:
            return {}

    class _AsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_AsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]) -> _Response:
            return _Response()

    monkeypatch.setattr(model_listing.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        fetch_available_models(
            base_url="http://example.test/v1",
            api_key="test-key",
            provider_profile="generic",
        )
    )

    assert result.ok is False
    assert result.models == []
    assert "HTTP 503" in result.error
    assert result.source_url == "http://example.test/v1/models"


def test_lmstudio_fetch_prefers_native_endpoint(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class _Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"models": [{"key": "native-model"}]}

    class _AsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_AsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]) -> _Response:
            calls.append((url, headers.get("Authorization")))
            return _Response()

    monkeypatch.setattr(model_listing.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        fetch_available_models(
            base_url="http://example.test/v1",
            api_key="test-key",
            provider_profile="lmstudio",
        )
    )

    assert result.ok is True
    assert [model.id for model in result.models] == ["native-model"]
    assert calls == [("http://example.test/api/v1/models", "Bearer test-key")]


def test_model_select_button_set_model_updates_label() -> None:
    button = ModelSelectButton("old-model")

    button.set_model("new-model")

    assert "model: new-model v" in str(button.label)


def test_model_select_button_click_emits_button_pressed() -> None:
    from textual.app import App, ComposeResult

    class _App(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.pressed_id = ""

        def compose(self) -> ComposeResult:
            yield ModelSelectButton("alpha-model", id="model-button")

        def on_button_pressed(self, event) -> None:
            self.pressed_id = str(event.button.id or "")

    async def _run() -> str:
        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#model-button")
            await pilot.pause(0.2)
            return app.pressed_id

    assert asyncio.run(_run()) == "model-button"


def test_model_select_screen_filter_input_selects_matching_model(monkeypatch) -> None:
    from textual.app import App, ComposeResult
    from textual.widgets import Button

    async def _fake_fetch(**kwargs: Any) -> ModelListResult:
        return ModelListResult(
            True,
            [
                ProviderModel("alpha-model", "alpha-model"),
                ProviderModel("beta-model", "beta-model"),
            ],
            "fake://models",
        )

    monkeypatch.setattr(model_selector, "fetch_available_models", _fake_fetch)

    class _App(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.selection: str | None = None

        def compose(self) -> ComposeResult:
            yield Button("Open", id="open")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "open":
                await self.push_screen(
                    ModelSelectScreen(
                        base_url="http://example.test/v1",
                        api_key=None,
                        provider_profile="generic",
                        current_model="alpha-model",
                    ),
                    callback=lambda selection: setattr(self, "selection", selection),
                )

    async def _run() -> str | None:
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#open")
            await pilot.pause(0.2)
            input_widget = app.screen.query_one("#model-select-filter")
            assert input_widget.has_focus
            await pilot.press("b", "e", "t", "a")
            await pilot.pause(0.2)
            assert len(app.screen.query_one("#model-select-list").children) == 1
            await pilot.click("#model-select-confirm")
            await pilot.pause(0.2)
            return app.selection

    assert asyncio.run(_run()) == "beta-model"


def test_model_select_screen_uses_typed_text_when_no_models_match(monkeypatch) -> None:
    from textual.app import App, ComposeResult
    from textual.widgets import Button

    async def _fake_fetch(**kwargs: Any) -> ModelListResult:
        return ModelListResult(True, [ProviderModel("alpha-model", "alpha-model")], "fake://models")

    monkeypatch.setattr(model_selector, "fetch_available_models", _fake_fetch)

    class _App(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.selection: str | None = None

        def compose(self) -> ComposeResult:
            yield Button("Open", id="open")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "open":
                await self.push_screen(
                    ModelSelectScreen(
                        base_url="http://example.test/v1",
                        api_key=None,
                        provider_profile="generic",
                        current_model="alpha-model",
                    ),
                    callback=lambda selection: setattr(self, "selection", selection),
                )

    async def _run() -> str | None:
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#open")
            await pilot.pause(0.2)
            await pilot.press("m", "a", "n", "u", "a", "l")
            await pilot.click("#model-select-confirm")
            await pilot.pause(0.2)
            return app.selection

    assert asyncio.run(_run()) == "manual"


def test_model_select_screen_allows_manual_entry_while_models_are_loading(monkeypatch) -> None:
    from textual.app import App, ComposeResult
    from textual.widgets import Button

    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def _slow_fetch(**kwargs: Any) -> ModelListResult:
        fetch_started.set()
        await release_fetch.wait()
        return ModelListResult(True, [ProviderModel("alpha-model", "alpha-model")], "fake://models")

    monkeypatch.setattr(model_selector, "fetch_available_models", _slow_fetch)

    class _App(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.selection: str | None = None

        def compose(self) -> ComposeResult:
            yield Button("Open", id="open")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "open":
                await self.push_screen(
                    ModelSelectScreen(
                        base_url="http://example.test/v1",
                        api_key=None,
                        provider_profile="generic",
                        current_model="alpha-model",
                    ),
                    callback=lambda selection: setattr(self, "selection", selection),
                )

    async def _run() -> str | None:
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#open")
            await pilot.pause(0.2)
            await asyncio.wait_for(fetch_started.wait(), timeout=1)
            input_widget = app.screen.query_one("#model-select-filter")
            assert input_widget.has_focus
            await pilot.press("m", "a", "n", "u", "a", "l")
            await pilot.click("#model-select-confirm")
            await pilot.pause(0.2)
            release_fetch.set()
            return app.selection

    assert asyncio.run(_run()) == "manual"


def test_smallctl_app_model_button_opens_selector_and_accepts_text(monkeypatch) -> None:
    async def _fake_fetch(**kwargs: Any) -> ModelListResult:
        return ModelListResult(
            True,
            [
                ProviderModel("alpha-model", "alpha-model"),
                ProviderModel("beta-model", "beta-model"),
            ],
            "fake://models",
        )

    monkeypatch.setattr(model_selector, "fetch_available_models", _fake_fetch)

    def _fake_create_harness(self: SmallctlApp) -> None:
        async def _teardown() -> None:
            return None

        harness = SimpleNamespace(
            client=SimpleNamespace(
                base_url="http://example.test/v1",
                model="alpha-model",
                api_key="test-key",
                provider_profile="generic",
            ),
            provider_profile="generic",
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
        def _switch_model(model: str) -> None:
            harness.client.model = model
            harness.state.scratchpad["_model_name"] = model

        harness.switch_model = _switch_model
        self.harness = harness

    monkeypatch.setattr(SmallctlApp, "_create_harness", _fake_create_harness)

    async def _run() -> str | None:
        app = SmallctlApp(
            {
                "endpoint": "http://example.test/v1",
                "model": "alpha-model",
                "api_key": "test-key",
                "provider_profile": "generic",
            }
        )
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.click("#model-button")
            await pilot.pause(0.3)
            assert isinstance(app.screen, ModelSelectScreen)
            assert app.screen.query_one("#model-select-filter").has_focus
            await pilot.press("b", "e", "t", "a")
            await pilot.click("#model-select-confirm")
            await pilot.pause(0.3)
            return app.harness_kwargs.get("model")

    assert asyncio.run(_run()) == "beta-model"


def test_clicking_model_button_while_running_does_not_switch_model() -> None:
    class _RunningTask:
        def done(self) -> bool:
            return False

    class _Actions(SmallctlAppActionsMixin):
        def __init__(self) -> None:
            self.active_task = _RunningTask()
            self.switched: list[str] = []
            self.messages: list[str] = []

        def notify(self, message: str, **kwargs: Any) -> None:
            self.messages.append(message)

        async def _append_system_line(self, text: str, *, force: bool = False) -> None:
            self.messages.append(text)

        def _switch_model(self, model: str) -> None:
            self.switched.append(model)

    actions = _Actions()

    asyncio.run(actions.action_open_model_selector())

    assert actions.switched == []
    assert "Model can be changed after the active task finishes." in actions.messages


def test_switching_model_updates_harness_kwargs() -> None:
    state = SimpleNamespace(scratchpad={})

    class _Harness:
        def __init__(self) -> None:
            self.state = state
            self.provider_profile = "lmstudio"
            self.client = SimpleNamespace(model="old-model")
            self.config = SimpleNamespace(model="old-model")

        def switch_model(self, model: str) -> None:
            self.client.model = model
            self.config.model = model
            self.state.scratchpad["_model_name"] = model

        def build_status_snapshot(self, *, activity: str = "", api_errors: int = 0) -> dict[str, object]:
            return {
                "model": self.client.model,
                "phase": "explore",
                "activity": activity,
                "api_errors": api_errors,
            }

    class _Flow(SmallctlAppFlowMixin):
        def __init__(self) -> None:
            self.harness = _Harness()
            self.harness_kwargs = {"model": "old-model", "provider_profile": "lmstudio"}
            self.refreshed = False
            self._api_error_count = 0
            self._latest_status_snapshot = None
            self._status_activity = ""
            self.active_task = None

        def _refresh_status(self, step_override: int | str | None = None) -> None:
            self.refreshed = True

    flow = _Flow()

    asyncio.run(flow._switch_model("new-model"))

    assert flow.harness_kwargs["model"] == "new-model"
    assert flow.harness.client.model == "new-model"
    assert flow.harness.config.model == "new-model"
    assert flow.harness.state is state
    assert flow.refreshed is True


def test_harness_switch_model_preserves_conversation_state() -> None:
    harness = Harness.__new__(Harness)
    state = SimpleNamespace(scratchpad={"conversation": "kept"})
    harness.state = state
    harness.run_logger = None
    harness.provider_profile = "generic"
    harness.client = SimpleNamespace(
        base_url="http://example.test/v1",
        model="old-model",
        api_key="test-key",
        chat_endpoint="/chat/completions",
        provider_profile="generic",
    )
    harness.config = SimpleNamespace(
        model="old-model",
        provider_profile="generic",
        context_limit=8192,
    )
    harness._harness_kwargs = {
        "endpoint": "http://example.test/v1",
        "model": "old-model",
        "api_key": None,
        "chat_endpoint": "/chat/completions",
        "provider_profile": "generic",
        "first_token_timeout_sec": None,
        "runtime_context_probe": True,
        "context_limit": 8192,
    }
    harness.discovered_server_context_limit = 8192
    harness.server_context_limit = 8192
    harness._runtime_context_probe_attempted = True

    async def _recover_backend_wedge(details: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ignored", "details": details}

    harness.recover_backend_wedge = _recover_backend_wedge

    harness.switch_model("new-model")

    assert harness.state is state
    assert harness.state.scratchpad["conversation"] == "kept"
    assert harness.state.scratchpad["_model_name"] == "new-model"
    assert harness.client.model == "new-model"
    assert harness.config.model == "new-model"
    assert harness._harness_kwargs["model"] == "new-model"
    assert harness.server_context_limit is None
    assert harness.discovered_server_context_limit is None
    assert harness._runtime_context_probe_attempted is False
