from __future__ import annotations

from typing import Any

from textual.widgets import Static


class StatusBar(Static):
    _BAR_WIDTH = 20

    def __init__(
        self,
        model: str = "n/a",
        phase: str = "explore",
        step: int | str = 0,
        mode: str = "execution",
        plan: str = "",
        active_step: str = "",
        activity: str = "",
        contract_flow_ui: bool = False,
        contract_phase: str = "",
        acceptance_progress: str = "",
        latest_verdict: str = "",
        show_model: bool = True,
        vertical: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._show_model = show_model
        self._vertical = vertical
        self._phase = phase
        self._step = step
        self._mode = mode
        self._plan = plan
        self._active_step = active_step
        self._activity = activity
        self._contract_flow_ui = contract_flow_ui
        self._contract_phase = contract_phase
        self._acceptance_progress = acceptance_progress
        self._latest_verdict = latest_verdict
        self._token_usage = 0  # Current prompt estimate
        self._token_limit = 0  # Effective prompt budget (free)
        self._token_total = 0  # Cumulative session tokens
        self._context_window = 0  # Total server context window
        self._api_errors = 0
        self._fama_off = False
        self._refresh_display()

    def set_status(self, text: str) -> None:
        # Backward-compatible fallback for any legacy callers.
        self.update(text)

    def set_fama_off(self, off: bool = True) -> None:
        self._fama_off = off
        self._refresh_display()

    def set_state(
        self,
        *,
        model: str,
        phase: str,
        step: int | str,
        mode: str,
        plan: str,
        active_step: str,
        activity: str,
        contract_flow_ui: bool,
        contract_phase: str,
        acceptance_progress: str,
        latest_verdict: str,
        token_usage: int, # prompt estimate
        token_total: int, # cumulative
        token_limit: int, # effective prompt budget (free)
        context_window: int = 0, # total server context window
        api_errors: int = 0,
        fama_off: bool = False,
    ) -> None:
        self._model = model
        self._phase = phase
        self._step = step
        self._mode = mode
        self._plan = plan
        self._active_step = active_step
        self._activity = activity
        self._contract_flow_ui = contract_flow_ui
        self._contract_phase = contract_phase
        self._acceptance_progress = acceptance_progress
        self._latest_verdict = latest_verdict
        self._token_usage = max(0, token_usage)
        self._token_total = max(0, token_total)
        self._token_limit = max(0, token_limit)
        self._context_window = max(0, context_window)
        self._api_errors = api_errors
        self._fama_off = fama_off
        self._refresh_display()

    def _build_status_text(self) -> str:
        if getattr(self, "_vertical", False):
            return self._build_vertical_status_text()

        parts = []
        if getattr(self, "_show_model", True):
            parts.append(f"[bold #9ca3af] model: {self._model} [/]")
        parts.append(f"[bold #9ca3af] chat [/]")
        parts.extend(
            [
                f"phase: {self._phase}",
                f"step: {self._step}",
                f"mode: {self._mode}",
            ]
        )
        if self._activity:
            parts.append(f"activity: {self._activity}")
        
        if self._api_errors > 0:
            parts.append(f"[bold red]API ERRORS: {self._api_errors}[/]")
        if getattr(self, "_fama_off", False):
            parts.append("[bold #ffb000]FAMA:OFF[/]")

        # Cumulative token progress bar
        ratio = 0.0
        if self._token_limit > 0:
            ratio = min(1.0, self._token_total / (self._token_limit * 2))
        
        filled = int(round(ratio * self._BAR_WIDTH))
        filled_segment = "█" * filled
        bar_markup = f"[bold #fca5a5]{filled_segment}[/]"

        parts.append(f"cumulative: {bar_markup} {self._token_total:,}")

        return " | ".join(parts)

    def _build_vertical_status_text(self) -> str:
        lines = []
        if getattr(self, "_show_model", True):
            lines.extend(["[bold #9ca3af]Model[/]", f"{self._model}", ""])

        lines.extend(
            [
                "[bold #9ca3af]Run[/]",
                f"phase: {self._phase}",
                f"step: {self._step}",
                f"mode: {self._mode}",
            ]
        )
        if self._activity:
            lines.append(f"activity: {self._activity}")
        if self._api_errors > 0:
            lines.append(f"[bold red]API ERRORS: {self._api_errors}[/]")
        if getattr(self, "_fama_off", False):
            lines.append("[bold #ffb000]FAMA:OFF[/]")

        # Cumulative token progress bar
        ratio = 0.0
        if self._token_limit > 0:
            ratio = min(1.0, self._token_total / (self._token_limit * 2))
        filled = int(round(ratio * self._BAR_WIDTH))
        bar_markup = f"[bold #fca5a5]{'█' * filled}[/]"

        lines.extend(["", "[bold #9ca3af]Usage[/]", f"cumulative: {bar_markup} {self._token_total:,}"])
        return "\n".join(lines)

    def _refresh_display(self) -> None:
        self.update(self._build_status_text())
