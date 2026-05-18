from __future__ import annotations

from typing import Any

from ..context.policy import estimate_text_tokens
from ..graph.solver_refine import (
    SolverRefineResult,
    build_critique_prompt,
    parse_critique_response,
)


class RefineService:
    """Runs a single-pass bounded critique on solver draft output."""

    def __init__(self, harness: Any) -> None:
        self._harness = harness

    async def run_bounded_refine(
        self,
        draft: str,
        observations_text: str,
        *,
        active_subtask: str | None = None,
        verifier_signals: dict[str, Any] | None = None,
        context_frame: str = "",
    ) -> SolverRefineResult | None:
        config = getattr(self._harness, "config", None)
        if not bool(getattr(config, "solver_refine_enabled", False)):
            return None

        budget = int(getattr(config, "solver_refine_token_budget", 700) or 700)
        prompt = build_critique_prompt(
            draft=draft,
            observations_text=observations_text,
            active_subtask=active_subtask,
            verifier_signals=verifier_signals,
            context_frame=context_frame,
        )
        if estimate_text_tokens(prompt) > budget:
            if not context_frame:
                return None
            prompt = build_critique_prompt(
                draft=draft,
                observations_text=observations_text,
                active_subtask=active_subtask,
                verifier_signals=verifier_signals,
                context_frame="",
            )
            if estimate_text_tokens(prompt) > budget:
                return None

        model_call_fn = getattr(self._harness, "_model_call", None) or getattr(
            self._harness, "_run_model_call", None
        )
        if not callable(model_call_fn):
            return None

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a concise critique verifier."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await model_call_fn(messages=messages, tools=[])
        except Exception:
            return None

        response_text = ""
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = str(response.get("content") or response.get("text") or "")
        elif hasattr(response, "content"):
            response_text = str(response.content or "")

        return parse_critique_response(response_text)
