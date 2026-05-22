from __future__ import annotations

from typing import Any

from .common import fail


async def escalate_to_bigger_model(
    *,
    reason: str,
    question: str,
    requested_output: str,
    risk_level: str = "medium",
    state: Any = None,
    harness: Any = None,
) -> dict[str, Any]:
    del state
    if harness is None:
        return {"success": False, "status": "error", "error": "Harness is required for escalation."}
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=reason,
        question=question,
        requested_output=requested_output,
        risk_level=risk_level,
    )
    if isinstance(result, dict) and result.get("success") is False:
        message = str(result.get("error") or result.get("reason") or result.get("status") or "Escalation failed.")
        metadata = {key: value for key, value in result.items() if key not in {"error", "reason"}}
        metadata["escalation_result"] = result
        return fail(message, metadata=metadata)
    return result
