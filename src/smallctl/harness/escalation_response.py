from __future__ import annotations

import json
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..recovery_metrics import increment_metric

ALLOWED_VERDICTS = {
    "continue",
    "need_more_evidence",
    "reject_current_plan",
    "propose_patch",
    "final_answer_ok",
    "ask_human",
    "abort",
    "next_action",
}
ALLOWED_ACTION_TYPES = {"tool_call", "repair_plan", "patch", "answer_review", "ask_human", "none"}
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class EscalationValidationResult:
    ok: bool
    response: dict[str, Any] | None
    error: str = ""


def parse_and_validate_escalation_response(text: str, *, harness: Any) -> EscalationValidationResult:
    try:
        payload = json.loads(_extract_json_text(text))
    except Exception as exc:
        increment_metric(harness.state, "escalation_invalid_json")
        return EscalationValidationResult(False, None, f"Invalid escalation JSON: {exc}")

    if not isinstance(payload, dict):
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, None, "Escalation response must be a JSON object.")

    verdict = str(payload.get("verdict") or "").strip()
    if verdict not in ALLOWED_VERDICTS:
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, payload, f"Unknown escalation verdict: {verdict}")
    if verdict == "final_answer_ok" and _latest_verifier_failed(harness):
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(
            False,
            payload,
            "Escalation cannot mark final answer OK while the latest verifier is failing.",
        )

    confidence = payload.get("confidence", 0)
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, payload, "Escalation confidence must be numeric.")
    if confidence_float < 0 or confidence_float > 1:
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, payload, "Escalation confidence must be between 0 and 1.")
    payload["confidence"] = confidence_float

    action = payload.get("recommended_next_action")
    if action is None:
        payload["recommended_next_action"] = {"type": "none", "reason": "No next action supplied."}
        action = payload["recommended_next_action"]
    if not isinstance(action, dict):
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, payload, "recommended_next_action must be an object.")

    action_type = str(action.get("type") or "none").strip()
    if action_type not in ALLOWED_ACTION_TYPES:
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, payload, f"Unknown recommended action type: {action_type}")

    if action_type == "tool_call":
        tool_name = str(action.get("tool") or "").strip()
        args_error = _validate_tool_call_args(action.get("args"))
        if args_error:
            increment_metric(harness.state, "escalation_validator_rejections")
            return EscalationValidationResult(False, payload, args_error)
        spec = getattr(harness, "registry", None).get(tool_name) if getattr(harness, "registry", None) else None
        if spec is None:
            increment_metric(harness.state, "escalation_validator_rejections")
            return EscalationValidationResult(False, payload, f"Recommended tool is not registered: {tool_name}")
        mode = str(getattr(getattr(harness, "config", None), "run_mode", "loop") or "loop")
        if not spec.mode_allowed(mode if mode != "auto" else "loop"):
            increment_metric(harness.state, "escalation_validator_rejections")
            return EscalationValidationResult(False, payload, f"Recommended tool is not allowed in mode: {mode}")
        visibility_error = _validate_runtime_visibility(tool_name, harness, mode=mode)
        if visibility_error:
            increment_metric(harness.state, "escalation_validator_rejections")
            return EscalationValidationResult(False, payload, visibility_error)
        path_error = _validate_local_paths(action.get("args"), harness)
        if path_error:
            increment_metric(harness.state, "escalation_validator_rejections")
            return EscalationValidationResult(False, payload, path_error)
    elif action_type == "patch" and bool(action.get("apply") or action.get("auto_apply")):
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, payload, "Patch recommendations are advice only and cannot request auto-apply.")

    forbidden = _forbidden_advice_text(payload)
    if forbidden:
        increment_metric(harness.state, "escalation_validator_rejections")
        return EscalationValidationResult(False, payload, forbidden)

    return EscalationValidationResult(True, payload)


def _extract_json_text(text: str) -> str:
    raw = str(text or "").strip()
    match = _FENCE_RE.search(raw)
    if match:
        raw = match.group(1).strip()
    if raw.startswith("{"):
        return raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def _validate_local_paths(args: Any, harness: Any) -> str:
    if not isinstance(args, dict):
        return ""
    cwd = Path(str(getattr(harness.state, "cwd", "") or ".")).resolve()
    for key in ("path", "file", "target_path"):
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (cwd / path).resolve()
        if cwd != resolved and cwd not in resolved.parents:
            return f"Recommended local path is outside cwd: {value}"
    return ""


def _validate_tool_call_args(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    forbidden = {"endpoint", "base_url", "model", "api_key", "api_key_env", "authorization", "token"}
    found = sorted(_find_forbidden_keys(args, forbidden))
    if found:
        return f"Recommended tool arguments contain forbidden escalation/provider keys: {', '.join(found)}"
    return ""


def _find_forbidden_keys(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key or "").strip().lower()
            if normalized in forbidden:
                found.add(normalized)
            found.update(_find_forbidden_keys(item, forbidden))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_forbidden_keys(item, forbidden))
    return found


def _validate_runtime_visibility(tool_name: str, harness: Any, *, mode: str) -> str:
    normalized_mode = mode if str(mode or "").strip() != "auto" else "loop"
    scratchpad_snapshot = None
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad_snapshot = copy.deepcopy(scratchpad)
    try:
        from .tool_visibility import resolve_turn_tool_exposure

        exposure = resolve_turn_tool_exposure(harness, normalized_mode)
    except Exception:
        return ""
    finally:
        if scratchpad_snapshot is not None:
            try:
                harness.state.scratchpad = scratchpad_snapshot
            except Exception:
                pass
    names = exposure.get("names") if isinstance(exposure, dict) else None
    if isinstance(names, list) and tool_name not in {str(name) for name in names}:
        return f"Recommended tool is currently hidden by runtime visibility policy: {tool_name}"
    return ""


def _forbidden_advice_text(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True).lower()
    forbidden = (
        "bypass verifier",
        "ignore approval",
        "ignore approvals",
        "expose secret",
        "print secret",
        "mark task complete without",
    )
    for phrase in forbidden:
        if phrase in text:
            return f"Escalation advice contained forbidden instruction: {phrase}"
    return ""


def _latest_verifier_failed(harness: Any) -> bool:
    verifier = getattr(getattr(harness, "state", None), "last_verifier_verdict", None)
    if not isinstance(verifier, dict):
        return False
    verdict = str(verifier.get("verdict") or "").strip().lower()
    return verdict in {"fail", "failed", "error"}
