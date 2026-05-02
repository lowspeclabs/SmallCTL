from __future__ import annotations

import re
from typing import Any, Iterable

_AFFIRMATIVE_EXACT_PHRASES = {
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "do it",
    "please do",
    "go ahead",
    "run it",
    "execute",
    "approve",
    "approved",
}
_AFFIRMATIVE_LEAD_PHRASES = (
    ("yes",),
    ("y",),
    ("yeah",),
    ("yep",),
    ("sure",),
    ("ok",),
    ("okay",),
    ("do", "it"),
    ("go", "ahead"),
    ("run", "it"),
    ("execute",),
    ("approve",),
    ("approved",),
)
_AFFIRMATIVE_CONTINUATION_TOKENS = {
    "and",
    "apply",
    "approved",
    "changes",
    "continue",
    "do",
    "execution",
    "fix",
    "fixes",
    "go",
    "going",
    "implementation",
    "implement",
    "it",
    "now",
    "please",
    "proceed",
    "remote",
    "run",
    "step",
    "steps",
    "the",
    "these",
    "this",
    "update",
    "updates",
    "with",
}
_IMPLEMENTATION_ACTION_RE = re.compile(
    r"\b(?:apply|build|change|configure|continue|create|deploy|execute|fix|implement|install|patch|proceed|restart|run|update|write)\b",
    re.IGNORECASE,
)
_IMPLEMENTATION_PROPOSAL_PATTERNS = (
    "i can ",
    "i could ",
    "i will ",
    "i'm ready to ",
    "im ready to ",
    "ready for me to ",
    "next step is to ",
    "next steps are to ",
    "my next step is to ",
    "proposed implementation",
    "proposed change",
    "proposed changes",
    "proposed fix",
    "proposed fixes",
    "implementation steps",
)


def normalize_followup_text(value: str) -> str:
    return " ".join(token for token in re.split(r"[^a-z0-9]+", str(value or "").strip().lower()) if token)


def followup_tokens(value: str, *, fillers: Iterable[str] = ()) -> list[str]:
    tokens = [token for token in normalize_followup_text(value).split() if token]
    while tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    filler_set = {str(token).strip().lower() for token in fillers if str(token).strip()}
    if not filler_set:
        return tokens
    return [token for token in tokens if token not in filler_set]


def is_affirmative_followup(value: str, *, fillers: Iterable[str] = ()) -> bool:
    normalized = normalize_followup_text(value)
    if not normalized:
        return False
    if normalized in _AFFIRMATIVE_EXACT_PHRASES:
        return True

    tokens = followup_tokens(value, fillers=fillers)
    if not tokens:
        return False

    collapsed = " ".join(tokens)
    if collapsed in _AFFIRMATIVE_EXACT_PHRASES:
        return True

    for lead in _AFFIRMATIVE_LEAD_PHRASES:
        if tokens[: len(lead)] != list(lead):
            continue
        trailing = tokens[len(lead) :]
        if not trailing:
            return True
        if all(token in _AFFIRMATIVE_CONTINUATION_TOKENS for token in trailing):
            return True
    return False


def assistant_message_proposes_concrete_implementation(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized or not _IMPLEMENTATION_ACTION_RE.search(normalized):
        return False
    return any(pattern in normalized for pattern in _IMPLEMENTATION_PROPOSAL_PATTERNS)


def recent_assistant_requested_action_confirmation(
    messages: list[Any],
    *,
    prompts: Iterable[str],
) -> bool:
    prompt_values = tuple(str(prompt or "").strip().lower() for prompt in prompts if str(prompt or "").strip())
    for message in reversed(messages[-8:]):
        if getattr(message, "role", "") != "assistant":
            continue
        text = str(getattr(message, "content", "") or "").strip().lower()
        if not text:
            continue
        if any(prompt in text for prompt in prompt_values):
            return True
        if assistant_message_proposes_concrete_implementation(text):
            return True
    return False
