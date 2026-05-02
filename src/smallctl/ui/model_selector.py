from __future__ import annotations

import asyncio
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from ..client.model_listing import ModelListResult, ProviderModel, fetch_available_models


class ModelSelectButton(Button):
    def __init__(self, model: str = "n/a", **kwargs: Any) -> None:
        super().__init__("", compact=True, flat=True, **kwargs)
        self._model = ""
        self.set_model(model)

    def set_model(self, model: str) -> None:
        self._model = str(model or "n/a").strip() or "n/a"
        self.label = f"model: {self._short_model(self._model)} v"

    def set_busy(self, busy: bool) -> None:
        self.set_class(bool(busy), "model-button-busy")

    @staticmethod
    def _short_model(model: str, *, limit: int = 24) -> str:
        value = str(model or "n/a").strip() or "n/a"
        if len(value) <= limit:
            return value
        return value[: max(1, limit - 1)] + "~"


class ModelSelectScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "submit", "Select"),
    ]

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        provider_profile: str,
        current_model: str,
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.provider_profile = provider_profile
        self.current_model = str(current_model or "").strip()
        self._models: list[ProviderModel] = []
        self._visible_models: list[ProviderModel] = []
        self._load_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        with Container(id="model-select-shell"):
            with Vertical(id="model-select"):
                yield Static("Select Model", id="model-select-title")
                yield Static("Loading models...", id="model-select-message", markup=False)
                yield Input(
                    placeholder="Filter models or enter model id",
                    id="model-select-filter",
                )
                yield ListView(id="model-select-list")
                with Horizontal(id="model-select-buttons"):
                    yield Button("Select", id="model-select-confirm", variant="success", compact=True)
                    yield Button("Use Current", id="model-select-current", variant="primary", compact=True)
                    yield Button("Cancel", id="model-select-cancel", variant="error", compact=True)

    def on_mount(self) -> None:
        self.query_one("#model-select-confirm", Button).disabled = True
        self.query_one("#model-select-message", Static).update(
            "Loading models... type a model id to use manual entry."
        )
        self.query_one("#model-select-filter", Input).focus()
        self._load_task = asyncio.create_task(self._load_models())

    def on_unmount(self) -> None:
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()

    async def _load_models(self) -> None:
        try:
            result = await fetch_available_models(
                base_url=self.base_url,
                api_key=self.api_key,
                provider_profile=self.provider_profile,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = ModelListResult(False, [], self.base_url, str(exc))

        if not self.is_mounted:
            return

        self._models = result.models
        filter_value = self.query_one("#model-select-filter", Input).value.strip()

        if result.ok:
            if self._models:
                if filter_value:
                    await self._apply_filter(filter_value)
                else:
                    await self._render_models(self._models, prefer_model=self.current_model)
                return
            self.query_one("#model-select-message", Static).update(
                f"No models returned by provider. Current: {self.current_model or 'n/a'}. Type a model id manually."
            )
        else:
            self.query_one("#model-select-message", Static).update(
                f"Could not fetch models. Current: {self.current_model or 'n/a'}\n{result.error}\nType a model id manually."
            )
        self._update_select_button_for_text(filter_value)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        filter_text = self.query_one("#model-select-filter", Input).value.strip()
        exact_match = self._find_exact_model(filter_text)
        if exact_match is not None:
            self.dismiss(exact_match.id)
            return
        list_view = self.query_one("#model-select-list", ListView)
        selected = list_view.highlighted_child
        model_id = str(getattr(selected, "model_id", "") or "").strip()
        if model_id:
            self.dismiss(model_id)
            return
        if filter_text:
            self.dismiss(filter_text)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-select-filter":
            self.action_submit()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "model-select-filter":
            return
        await self._apply_filter(event.value)

    def _update_select_button_for_text(self, text: str) -> None:
        has_text = bool(str(text or "").strip())
        has_selection = bool(getattr(self.query_one("#model-select-list", ListView).highlighted_child, "model_id", ""))
        self.query_one("#model-select-confirm", Button).disabled = not (has_text or has_selection)

    async def _apply_filter(self, value: str) -> None:
        query = value.strip().lower()
        if not query:
            await self._render_models(self._models, prefer_model=self.current_model)
            self._update_select_button_for_text(value)
            return
        filtered = [
            model
            for model in self._models
            if query in model.id.lower()
            or query in model.display_name.lower()
        ]
        await self._render_models(filtered, prefer_model=value.strip())
        if not filtered:
            self.query_one("#model-select-confirm", Button).disabled = False
            self.query_one("#model-select-message", Static).update(
                "No provider models match. Press Enter or Select to use the typed model id."
            )
        else:
            self._update_select_button_for_text(value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        model_id = str(getattr(event.item, "model_id", "") or "").strip()
        if model_id:
            self.dismiss(model_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "model-select-confirm":
            self.action_submit()
            event.stop()
            return
        if event.button.id == "model-select-current":
            self.dismiss(self.current_model or None)
            event.stop()
            return
        if event.button.id == "model-select-cancel":
            self.dismiss(None)
            event.stop()

    def _format_model_label(self, model: ProviderModel) -> str:
        parts = [model.display_name or model.id]
        if model.id == self.current_model:
            parts.append("current")
        if model.loaded is True:
            parts.append("loaded")
        elif model.loaded is False:
            parts.append("not loaded")
        if model.context_length:
            parts.append(f"{model.context_length:,} ctx")
        if model.id != model.display_name:
            parts.append(model.id)
        return " | ".join(parts)

    async def _render_models(
        self,
        models: list[ProviderModel],
        *,
        prefer_model: str = "",
    ) -> None:
        self._visible_models = list(models)
        list_view = self.query_one("#model-select-list", ListView)
        await list_view.clear()
        preferred = str(prefer_model or "").strip().lower()
        selected_index: int | None = None
        for index, model in enumerate(self._visible_models):
            item = ListItem(Label(self._format_model_label(model), markup=False))
            setattr(item, "model_id", model.id)
            await list_view.append(item)
            if selected_index is None and (
                model.id.lower() == preferred
                or model.display_name.lower() == preferred
            ):
                selected_index = index
        if self._visible_models:
            list_view.index = selected_index if selected_index is not None else 0
        else:
            list_view.index = None
        self.query_one("#model-select-confirm", Button).disabled = not bool(self._visible_models)
        if self._visible_models:
            self.query_one("#model-select-message", Static).update(
                f"{len(self._visible_models)} matching model(s). Type to filter, Enter to select."
            )

    def _find_exact_model(self, model_id: str) -> ProviderModel | None:
        target = str(model_id or "").strip().lower()
        if not target:
            return None
        for model in self._models:
            if model.id.lower() == target or model.display_name.lower() == target:
                return model
        return None
