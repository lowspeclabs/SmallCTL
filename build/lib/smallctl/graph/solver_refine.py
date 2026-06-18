from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SolverRefineResult:
    verdict: str  # "pass", "revise", or "block"
    issues: list[str]
    revised_output: str


_CRITIQUE_INSTRUCTION = """You are a bounded critique verifier. Review the solver draft below against the tool plan observations.

Check exactly these four items:
1. Is the answer supported by TOOL PLAN OBSERVATIONS?
2. Did it ignore a failed step?
3. Is it claiming completion without verifier success?
4. Is the next action smaller and safer than any failed action?

Return ONLY JSON in this shape:
{"verdict": "pass" | "revise" | "block", "issues": ["..."], "revised_output": "..."}

- "pass" if the draft looks correct and well-supported.
- "revise" if the draft is mostly right but needs a small fix; provide the full corrected text in "revised_output".
- "block" if the draft is unsupported, ignores failures, or is unsafe; provide a brief explanation in "issues".
"""


def build_critique_prompt(
    draft: str,
    observations_text: str,
    active_subtask: str | None = None,
    verifier_signals: dict[str, Any] | None = None,
    context_frame: str = "",
) -> str:
    evidence_text = context_frame or observations_text
    evidence_label = "REWOO CONTEXT FRAME" if context_frame or "REWOO " in evidence_text else "TOOL PLAN OBSERVATIONS"
    parts = [
        _CRITIQUE_INSTRUCTION,
        "",
        f"=== {evidence_label} ===",
        evidence_text or "(no observations)",
        "",
        "=== SOLVER DRAFT ===",
        draft or "(empty draft)",
    ]
    if active_subtask:
        parts.extend(["", f"=== ACTIVE SUBTASK ===\n{active_subtask}"])
    if verifier_signals:
        parts.extend(["", f"=== VERIFIER SIGNALS ===\n{json.dumps(verifier_signals, indent=2)}"])
    return "\n".join(parts)


def parse_critique_response(text: str) -> SolverRefineResult | None:
    """Extract JSON critique from model output."""
    if not text:
        return None
    raw = text.strip()
    # Strip fenced code blocks
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    # Try balanced-brace extraction if plain JSON parse fails
    payload = _try_json(raw)
    if payload is None:
        payload = _extract_balanced_json(raw)
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in {"pass", "revise", "block"}:
        return None
    issues = payload.get("issues")
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    issues = [str(i) for i in issues if i]
    revised = str(payload.get("revised_output") or "").strip()
    return SolverRefineResult(verdict=verdict, issues=issues, revised_output=revised)


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_balanced_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None
