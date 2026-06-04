from __future__ import annotations

from typing import Any


def _remember_web_search_results(
    service: Any,
    *,
    result: Any,
    artifact: Any,
) -> None:
    output = result.output if isinstance(result.output, dict) else {}
    results = output.get("results")
    if not isinstance(results, list):
        return

    result_ids = [
        str(item.get("result_id") or "").strip()
        for item in results
        if isinstance(item, dict) and str(item.get("result_id") or "").strip()
    ]
    fetch_ids = [
        str(item.get("fetch_id") or "").strip()
        for item in results
        if isinstance(item, dict) and str(item.get("fetch_id") or "").strip()
    ]
    if not result_ids:
        return

    scratchpad = getattr(service.harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        service.harness.state.scratchpad = scratchpad

    by_artifact = scratchpad.get("_web_search_artifact_results")
    if not isinstance(by_artifact, dict):
        by_artifact = {}
        scratchpad["_web_search_artifact_results"] = by_artifact

    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip()
    if artifact_id:
        by_artifact[artifact_id] = list(result_ids)
        scratchpad["_web_last_search_artifact_id"] = artifact_id
        if isinstance(getattr(artifact, "metadata", None), dict):
            artifact.metadata["web_result_ids"] = list(result_ids)
            artifact.metadata["web_fetch_ids"] = list(fetch_ids)
            artifact.metadata["web_result_count"] = len(result_ids)

    scratchpad["_web_last_search_result_ids"] = list(result_ids)
    scratchpad["_web_last_search_fetch_ids"] = list(fetch_ids)
