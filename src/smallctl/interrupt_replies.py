from __future__ import annotations

from typing import Any


_APPROVE_REPLIES = {
    "yes",
    "y",
    "approve",
    "approved",
    "execute",
    "go ahead",
    "run it",
}
_REJECT_REPLIES = {"no", "n", "reject", "decline", "cancel"}
_REVISE_REPLIES = {"revise", "revise it", "change it", "modify it"}
_SKIP_REPLIES = {"skip", "skip it"}
_RETRY_REPLIES = {"retry", "try again", "rerun"}
_CONTINUE_REPLIES = {"continue", "keep going", "proceed", "resume"}
_PLAN_APPROVAL_REPLIES = _APPROVE_REPLIES | _CONTINUE_REPLIES | {
    "do it",
    "please proceed",
    "proceed with it",
}

_CHOICE_ALIASES = {
    "yes": _APPROVE_REPLIES,
    "approve": _APPROVE_REPLIES,
    "approved": _APPROVE_REPLIES,
    "execute": _APPROVE_REPLIES,
    "no": _REJECT_REPLIES,
    "reject": _REJECT_REPLIES,
    "decline": _REJECT_REPLIES,
    "revise": _REVISE_REPLIES,
    "skip": _SKIP_REPLIES,
    "retry": _RETRY_REPLIES,
    "continue": _CONTINUE_REPLIES,
    "proceed": _CONTINUE_REPLIES,
}

_REPLY_ACTIONS = {
    **{reply: "approve" for reply in _APPROVE_REPLIES},
    **{reply: "reject" for reply in _REJECT_REPLIES},
    **{reply: "revise" for reply in _REVISE_REPLIES},
    **{reply: "skip" for reply in _SKIP_REPLIES},
    **{reply: "retry" for reply in _RETRY_REPLIES},
    **{reply: "continue" for reply in _CONTINUE_REPLIES},
}


def normalize_interrupt_reply(task: str) -> str:
    return " ".join(str(task or "").strip().lower().split())


def _response_choices(interrupt: dict[str, Any]) -> set[str]:
    response_mode = str(interrupt.get("response_mode") or "").strip().lower()
    kind = str(interrupt.get("kind") or "").strip()
    if not response_mode:
        if kind == "plan_execute_approval":
            response_mode = "yes/no/revise"
        elif kind == "staged_step_blocked":
            response_mode = "revise/skip/retry"
        else:
            response_mode = "continue"

    choices: set[str] = set()
    for raw_choice in response_mode.replace(",", "/").split("/"):
        choice = normalize_interrupt_reply(raw_choice)
        if not choice:
            continue
        if kind == "plan_execute_approval" and choice in {"yes", "approve", "execute"}:
            choices.update(_PLAN_APPROVAL_REPLIES)
        choices.update(_CHOICE_ALIASES.get(choice, {choice}))
    return choices


def interrupt_response_action(interrupt: dict[str, Any] | None, task: str) -> str | None:
    if not isinstance(interrupt, dict) or not interrupt:
        return None
    normalized = normalize_interrupt_reply(task)
    if not normalized:
        return None
    if normalized not in _response_choices(interrupt):
        return None
    if str(interrupt.get("kind") or "").strip() == "plan_execute_approval" and normalized in _PLAN_APPROVAL_REPLIES:
        return "approve"
    return _REPLY_ACTIONS.get(normalized, normalized)


def is_interrupt_response(interrupt: dict[str, Any] | None, task: str) -> bool:
    return interrupt_response_action(interrupt, task) is not None


def is_interrupt_affirmative_response(interrupt: dict[str, Any] | None, task: str) -> bool:
    return interrupt_response_action(interrupt, task) in {"approve", "continue", "retry"}


def is_plan_approval_reply(task: str) -> bool:
    return is_interrupt_response({"kind": "plan_execute_approval", "response_mode": "yes/no/revise"}, task)
