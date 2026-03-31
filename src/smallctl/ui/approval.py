from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


@dataclass(frozen=True)
class ShellApprovalDecision:
    approved: bool
    remember_session: bool = False

    def __bool__(self) -> bool:
        return self.approved


@dataclass(frozen=True)
class PlanApprovalDecision:
    choice: str

    def __bool__(self) -> bool:
        return self.choice == "yes"


class ApprovePromptScreen(ModalScreen[ShellApprovalDecision]):
    BINDINGS = [
        ("left", "focus_previous", "Previous"),
        ("right", "focus_next", "Next"),
        ("y", "confirm_once", "Yes"),
        ("s", "confirm_session", "Yes for Session"),
        ("n", "cancel", "No"),
        ("escape", "cancel", "No"),
    ]

    def __init__(
        self,
        *,
        approval_id: str,
        command: str,
        cwd: str,
        timeout_sec: int,
    ) -> None:
        super().__init__()
        self.approval_id = approval_id
        self.command = command
        self.cwd = cwd
        self.timeout_sec = timeout_sec

    def compose(self) -> ComposeResult:
        with Container(id="approve-prompt-shell"):
            with Vertical(id="approve-prompt"):
                yield Static("Approve shell command?", id="approve-prompt-title")
                yield Static(self._build_body(), id="approve-prompt-body")
                with Horizontal(id="approve-prompt-buttons"):
                    yield Button("Yes", id="approve-yes", variant="success")
                    yield Button("Yes for Session", id="approve-yes-session", variant="success")
                    yield Button("No", id="approve-no", variant="error")

    async def on_mount(self) -> None:
        self.query_one("#approve-yes", Button).focus()

    def action_focus_previous(self) -> None:
        self._cycle_focus(-1)

    def action_focus_next(self) -> None:
        self._cycle_focus(1)

    def action_confirm_once(self) -> None:
        self.dismiss(ShellApprovalDecision(True, False))

    def action_confirm_session(self) -> None:
        self.dismiss(ShellApprovalDecision(True, True))

    def action_cancel(self) -> None:
        self.dismiss(ShellApprovalDecision(False, False))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve-yes":
            self.dismiss(ShellApprovalDecision(True, False))
            return
        if event.button.id == "approve-yes-session":
            self.dismiss(ShellApprovalDecision(True, True))
            return
        if event.button.id == "approve-no":
            self.dismiss(ShellApprovalDecision(False, False))

    def _cycle_focus(self, direction: int) -> None:
        button_ids = ("approve-yes", "approve-yes-session", "approve-no")
        buttons = {
            button.id: button
            for button in self.query(Button)
            if button.id in button_ids
        }
        ordered = [buttons[button_id] for button_id in button_ids if button_id in buttons]
        if not ordered:
            return
        focused_index = next((index for index, button in enumerate(ordered) if button.has_focus), 0)
        ordered[(focused_index + direction) % len(ordered)].focus()

    def _build_body(self) -> str:
        parts = [
            f"Approval ID: {self.approval_id}",
            f"CWD: {self.cwd}",
            f"Timeout: {self.timeout_sec}s",
            "",
            "Command:",
            self.command,
            "",
            "Use left/right to choose, then Enter.",
            "Y approves once, S selects Yes for Session, N denies.",
        ]
        return "\n".join(parts)


class PlanApprovalScreen(ModalScreen[PlanApprovalDecision]):
    BINDINGS = [
        ("left", "focus_previous", "Previous"),
        ("right", "focus_next", "Next"),
        ("y", "approve", "Yes"),
        ("n", "deny", "No"),
        ("r", "revise", "Revise"),
        ("escape", "revise", "Revise"),
    ]

    def __init__(
        self,
        *,
        question: str,
        plan_id: str,
        response_mode: str,
    ) -> None:
        super().__init__()
        self.question = question
        self.plan_id = plan_id
        self.response_mode = response_mode

    def compose(self) -> ComposeResult:
        with Container(id="approve-prompt-plan"):
            with Vertical(id="approve-prompt"):
                yield Static("Approve plan?", id="approve-prompt-title")
                yield Static(self._build_body(), id="approve-prompt-body")
                with Horizontal(id="approve-prompt-buttons"):
                    yield Button("Yes", id="approve-plan-yes", variant="success")
                    yield Button("No", id="approve-plan-no", variant="error")
                    yield Button("Revise", id="approve-plan-revise", variant="primary")

    async def on_mount(self) -> None:
        self.query_one("#approve-plan-yes", Button).focus()

    def action_focus_previous(self) -> None:
        self._cycle_focus(-1)

    def action_focus_next(self) -> None:
        self._cycle_focus(1)

    def action_approve(self) -> None:
        self.dismiss(PlanApprovalDecision("yes"))

    def action_deny(self) -> None:
        self.dismiss(PlanApprovalDecision("no"))

    def action_revise(self) -> None:
        self.dismiss(PlanApprovalDecision("revise"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve-plan-yes":
            self.dismiss(PlanApprovalDecision("yes"))
            return
        if event.button.id == "approve-plan-no":
            self.dismiss(PlanApprovalDecision("no"))
            return
        if event.button.id == "approve-plan-revise":
            self.dismiss(PlanApprovalDecision("revise"))

    def _cycle_focus(self, direction: int) -> None:
        button_ids = ("approve-plan-yes", "approve-plan-no", "approve-plan-revise")
        buttons = {
            button.id: button
            for button in self.query(Button)
            if button.id in button_ids
        }
        ordered = [buttons[button_id] for button_id in button_ids if button_id in buttons]
        if not ordered:
            return
        focused_index = next((index for index, button in enumerate(ordered) if button.has_focus), 0)
        ordered[(focused_index + direction) % len(ordered)].focus()

    def _build_body(self) -> str:
        parts = [
            f"Plan ID: {self.plan_id or 'pending'}",
            f"Response mode: {self.response_mode or 'yes/no/revise'}",
            "",
            self.question or "Plan ready. Execute it now?",
            "",
            "Use left/right to choose, then Enter.",
            "Y approves, N declines, R requests revision.",
        ]
        return "\n".join(parts)
