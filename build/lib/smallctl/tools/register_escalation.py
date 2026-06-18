from __future__ import annotations

from typing import Any, Callable

from . import escalation
from .base import build_tool_schema


def register_escalation_tools(
    *,
    state_provider: Any,
    register: Callable[[list[Any]], None],
    make_registration: Callable[..., Any],
    inject_state_and_harness: Callable[..., Any],
    core_profile: str,
) -> None:
    config = getattr(state_provider, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return
    if not bool(getattr(config, "escalation_expose_tool", True)):
        return

    register([
        make_registration(
            name="escalate_to_bigger_model",
            description=(
                "Ask the configured larger model for bounded recovery advice. "
                "Call ONLY when: (a) you have at least 2 failed tool attempts or verifier failures, "
                "AND (b) you have gathered concrete evidence (tool outputs, artifact reads, or verifier results), "
                "AND (c) you are still stuck after trying repair. "
                "Do NOT call for simple path corrections or missing tool availability. "
                "The larger model cannot execute tools."
            ),
            schema=build_tool_schema(
                required=["reason", "question", "requested_output"],
                properties={
                    "reason": {"type": "string"},
                    "question": {"type": "string"},
                    "requested_output": {
                        "type": "string",
                        "enum": [
                            "next_action",
                            "repair_plan",
                            "patch_review",
                            "root_cause",
                            "tool_call_repair",
                            "final_answer_review",
                        ],
                    },
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            ),
            handler=inject_state_and_harness(escalation.escalate_to_bigger_model),
            category="control",
            risk="low",
            allowed_modes={"loop", "planning"},
            profiles={core_profile},
        )
    ])
