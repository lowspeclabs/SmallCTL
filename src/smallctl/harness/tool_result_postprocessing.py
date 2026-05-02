from __future__ import annotations

import logging
from typing import Any

from ..models.tool_result import ToolEnvelope
from .tool_result_failure import record_failure_outcome as _record_failure_outcome
from .tool_result_artifact_updates import apply_artifact_success_outcome as _apply_artifact_success_outcome

logger = logging.getLogger("smallctl.harness.tool_results")


async def apply_persisted_artifact_outcome(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
) -> Any:
    if artifact:
        _apply_artifact_success_outcome(
            service,
            tool_name=tool_name,
            result=result,
            artifact=artifact,
            arguments=arguments,
            operation_id=operation_id,
        )

    _record_failure_outcome(service, tool_name=tool_name, result=result)
