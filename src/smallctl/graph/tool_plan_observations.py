from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..context.policy import estimate_text_tokens
from ..state import EvidenceRecord, LoopState, json_safe_value
from .state import ToolExecutionRecord
from .tool_plan_schema import ToolPlan


@dataclass(slots=True)
class ToolPlanObservation:
    step_id: str
    tool: str
    success: bool
    summary: str
    excerpt: str = ""
    artifact_id: str = ""
    operation_id: str = ""
    path: str = ""
    query: str = ""
    error: str = ""
    duplicate_of: str = ""


@dataclass(slots=True)
class ToolPlanObservationStats:
    requested_steps: int = 0
    executed_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    missing_record_count: int = 0
    duplicate_read_count: int = 0
    artifact_yield_count: int = 0
    success_rate: float | None = None
    tool_failure_classes: list[str] = field(default_factory=list)


def _step_id_from_record(record: ToolExecutionRecord) -> str:
    call_id = str(record.tool_call_id or "")
    if call_id.startswith("toolplan:"):
        return call_id.split(":", 1)[1]
    metadata = getattr(record, "result", None).metadata if getattr(record, "result", None) is not None else {}
    return str(metadata.get("tool_plan_step_id") or "").strip()


def _artifact_id(record: ToolExecutionRecord) -> str:
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    for key in ("artifact_id", "artifact", "fetch_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    output = record.result.output
    if isinstance(output, dict):
        value = output.get("artifact_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _bounded_text(value: Any, *, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(json_safe_value(value), ensure_ascii=True, sort_keys=True)
    text = " ".join(text.strip().split())
    if len(text) > limit:
        return f"{text[: max(0, limit - 15)].rstrip()}... [truncated]"
    return text


def _metadata_summary(metadata: dict[str, Any], *, max_chars: int) -> str:
    pieces: list[str] = []
    for key in ("path", "query", "pattern", "matches", "count", "line_count", "truncated", "url", "title"):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            pieces.append(f"{key}={_bounded_text(value, limit=140)}")
    if pieces:
        return _bounded_text("; ".join(pieces), limit=max_chars)
    return ""


def _hint_terms(text: str) -> list[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or ""):
        pieces = [raw]
        pieces.extend(part for part in raw.split("_") if part)
        pieces.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", raw))
        for piece in pieces:
            term = piece.lower().strip("_")
            if len(term) < 4:
                continue
            terms.add(term)
            for suffix in ("ing", "ed", "es", "s"):
                if term.endswith(suffix) and len(term) > len(suffix) + 3:
                    terms.add(term[: -len(suffix)])
                    break
    return sorted(terms, key=lambda value: (-len(value), value))[:32]


def _relevant_text_excerpt(text: str, *, hint_text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return _bounded_text(text, limit=max_chars)

    terms = _hint_terms(hint_text)
    if not terms:
        return _bounded_text(text, limit=max_chars)

    selected: list[str] = []
    seen_lines: set[int] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        normalized = line.lower()
        if not any(term in normalized for term in terms):
            continue
        rendered = f"L{line_no}: {line.strip()}"
        if line_no in seen_lines or not rendered.strip():
            continue
        selected.append(rendered)
        seen_lines.add(line_no)
        if len(" ".join(selected)) >= max_chars:
            break

    if not selected:
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not re.match(r"^(async\s+def|def|class)\s+", stripped):
                continue
            selected.append(f"L{line_no}: {stripped}")
            if len(" ".join(selected)) >= max_chars:
                break

    if selected:
        return _bounded_text("\n".join(selected), limit=max_chars)
    return _bounded_text(text, limit=max_chars)


def _output_excerpt(record: ToolExecutionRecord, *, max_chars: int, hint_text: str = "") -> str:
    result = record.result
    output = result.output
    if output in (None, "", [], {}):
        return ""
    if isinstance(output, dict) and str(output.get("status") or "").strip().lower() == "cached":
        summary = output.get("summary")
        if summary not in (None, "", [], {}):
            return _bounded_text(summary, limit=max_chars)
    if isinstance(output, str) and hint_text:
        return _relevant_text_excerpt(output, hint_text=hint_text, max_chars=max_chars)
    return _bounded_text(output, limit=max_chars)


def _summary_from_record(record: ToolExecutionRecord, *, max_chars: int) -> str:
    result = record.result
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    metadata_text = _metadata_summary(metadata, max_chars=max_chars)
    if metadata_text:
        return metadata_text
    if result.error:
        return _bounded_text(result.error, limit=max_chars)
    if result.output not in (None, "", [], {}):
        return _bounded_text(result.output, limit=max_chars)
    status = result.status or ("success" if result.success else "failed")
    return _bounded_text(status, limit=max_chars)


def _dedupe_key(tool: str, args: dict[str, Any]) -> tuple[str, str]:
    return (
        str(tool or "").strip(),
        json.dumps(json_safe_value(args or {}), ensure_ascii=True, sort_keys=True),
    )


def _render_observation_lines(observation: ToolPlanObservation) -> list[str]:
    target = observation.path or observation.query
    header = f"{observation.step_id} {observation.tool}"
    if target:
        header = f"{header} {target}"
    lines = [header, "- success" if observation.success else "- failed"]
    if observation.duplicate_of:
        lines.append(f"- duplicate_of: {observation.duplicate_of}")
    if observation.artifact_id:
        lines.append(f"- artifact: {observation.artifact_id}")
    if observation.operation_id:
        lines.append(f"- operation: {observation.operation_id}")
    if observation.summary:
        lines.append(f"- summary: {observation.summary}")
    if observation.excerpt:
        lines.append(f"- excerpt: {observation.excerpt}")
    if not observation.success:
        lines.append(f"- error: {observation.error or observation.summary}")
    elif not observation.summary and not observation.excerpt:
        lines.append("- summary: success")
    lines.append("")
    return lines


def _fit_observations_to_token_limit(
    observations: list[ToolPlanObservation],
    *,
    token_limit: int,
    max_chars_per_step: int,
) -> list[ToolPlanObservation]:
    if token_limit <= 0:
        return observations
    rendered = "\n".join(line for observation in observations for line in _render_observation_lines(observation))
    if estimate_text_tokens(rendered) <= token_limit:
        return observations

    min_summary_chars = 80
    current_limit = max(min_summary_chars, min(max_chars_per_step, int(token_limit * 4 / max(1, len(observations)))))
    fitted: list[ToolPlanObservation] = []
    for observation in observations:
        summary = observation.summary
        excerpt = observation.excerpt
        error = observation.error
        if len(summary) > current_limit:
            summary = _bounded_text(summary, limit=current_limit)
        if len(excerpt) > current_limit:
            excerpt = _bounded_text(excerpt, limit=current_limit)
        if len(error) > current_limit:
            error = _bounded_text(error, limit=current_limit)
        fitted.append(
            ToolPlanObservation(
                step_id=observation.step_id,
                tool=observation.tool,
                success=observation.success,
                summary=summary,
                excerpt=excerpt,
                artifact_id=observation.artifact_id,
                operation_id=observation.operation_id,
                path=observation.path,
                query=observation.query,
                error=error,
                duplicate_of=observation.duplicate_of,
            )
        )

    while fitted and estimate_text_tokens(
        "\n".join(line for observation in fitted for line in _render_observation_lines(observation))
    ) > token_limit:
        largest_index = max(
            range(len(fitted)),
            key=lambda index: len(fitted[index].summary) + len(fitted[index].excerpt) + len(fitted[index].error),
        )
        largest = fitted[largest_index]
        if (
            len(largest.summary) <= min_summary_chars
            and len(largest.excerpt) <= min_summary_chars
            and len(largest.error) <= min_summary_chars
        ):
            break
        fitted[largest_index] = ToolPlanObservation(
            step_id=largest.step_id,
            tool=largest.tool,
            success=largest.success,
            summary=_bounded_text(largest.summary, limit=min_summary_chars),
            excerpt=_bounded_text(largest.excerpt, limit=min_summary_chars),
            artifact_id=largest.artifact_id,
            operation_id=largest.operation_id,
            path=largest.path,
            query=largest.query,
            error=_bounded_text(largest.error, limit=min_summary_chars),
            duplicate_of=largest.duplicate_of,
        )
    return fitted


def build_tool_plan_observations(
    plan: ToolPlan,
    records: list[ToolExecutionRecord],
    *,
    token_limit: int,
    max_chars_per_step: int,
) -> list[ToolPlanObservation]:
    by_step = {_step_id_from_record(record): record for record in records}
    observations: list[ToolPlanObservation] = []
    seen_calls: dict[tuple[str, str], ToolPlanObservation] = {}
    for step in plan.steps:
        record = by_step.get(step.id)
        path = str(step.args.get("path") or step.args.get("target_path") or "").strip()
        query = str(step.args.get("query") or step.args.get("pattern") or step.args.get("url") or "").strip()
        call_key = _dedupe_key(step.tool, step.args)
        duplicate = seen_calls.get(call_key)
        if duplicate is not None:
            observations.append(
                ToolPlanObservation(
                    step_id=step.id,
                    tool=step.tool,
                    success=duplicate.success,
                    summary=f"Duplicate of {duplicate.step_id}; reused prior observation.",
                    excerpt=duplicate.excerpt,
                    artifact_id=duplicate.artifact_id,
                    operation_id=duplicate.operation_id,
                    path=path,
                    query=query,
                    error="" if duplicate.success else duplicate.error,
                    duplicate_of=duplicate.step_id,
                )
            )
            continue
        if record is None:
            observation = ToolPlanObservation(
                step_id=step.id,
                tool=step.tool,
                success=False,
                summary="No execution record was produced.",
                path=path,
                query=query,
                error="missing execution record",
            )
            observations.append(observation)
            seen_calls[call_key] = observation
            continue
        observation = ToolPlanObservation(
            step_id=step.id,
            tool=record.tool_name or step.tool,
            success=bool(record.result.success),
            summary=_summary_from_record(record, max_chars=max_chars_per_step),
            excerpt=_output_excerpt(
                record,
                max_chars=max_chars_per_step,
                hint_text=" ".join(
                    part
                    for part in (plan.objective, step.reason, path, query)
                    if isinstance(part, str) and part.strip()
                ),
            ),
            artifact_id=_artifact_id(record),
            operation_id=str(record.operation_id or ""),
            path=path,
            query=query,
            error=str(record.result.error or ""),
        )
        observations.append(observation)
        seen_calls[call_key] = observation
    return _fit_observations_to_token_limit(
        observations,
        token_limit=token_limit,
        max_chars_per_step=max_chars_per_step,
    )


def summarize_tool_plan_observations(
    plan: ToolPlan,
    observations: list[ToolPlanObservation],
) -> ToolPlanObservationStats:
    failure_classes: set[str] = set()
    successful_steps = 0
    failed_steps = 0
    missing_record_count = 0
    duplicate_read_count = 0
    artifact_yield_count = 0

    for observation in observations:
        if observation.success:
            successful_steps += 1
        else:
            failed_steps += 1
            failure_classes.add(_normalize_failure_class(observation.error or observation.summary))
        if observation.error == "missing execution record":
            missing_record_count += 1
        if observation.duplicate_of:
            duplicate_read_count += 1
        if observation.artifact_id:
            artifact_yield_count += 1

    executed_steps = len(observations)
    success_rate = None
    if executed_steps > 0:
        success_rate = round(successful_steps / executed_steps, 3)

    return ToolPlanObservationStats(
        requested_steps=len(plan.steps),
        executed_steps=executed_steps,
        successful_steps=successful_steps,
        failed_steps=failed_steps,
        missing_record_count=missing_record_count,
        duplicate_read_count=duplicate_read_count,
        artifact_yield_count=artifact_yield_count,
        success_rate=success_rate,
        tool_failure_classes=sorted(failure_classes),
    )


def render_tool_plan_observations(objective: str, observations: list[ToolPlanObservation]) -> str:
    lines = ["TOOL PLAN OBSERVATIONS"]
    if objective:
        lines.append(f"Objective: {objective}")
        lines.append("")
    for observation in observations:
        lines.extend(_render_observation_lines(observation))
    return "\n".join(lines).rstrip()


def _normalize_failure_class(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return "unknown"
    if "missing execution record" in normalized:
        return "missing_execution_record"
    if "duplicate" in normalized:
        return "duplicate_read"
    if "permission" in normalized:
        return "permission_error"
    if "not found" in normalized or "missing" in normalized:
        return "missing"
    if "timeout" in normalized:
        return "timeout"
    if "denied" in normalized:
        return "permission_denied"
    return normalized[:64].replace(" ", "_")


def observation_to_evidence_record(
    observation: ToolPlanObservation,
    *,
    objective: str,
    step_index: int,
    created_at_step: int,
) -> EvidenceRecord:
    source = observation.path or observation.query or f"tool_plan:{observation.step_id}"
    statement = observation.excerpt or (observation.summary if observation.success else (observation.error or observation.summary))
    return EvidenceRecord(
        evidence_id=f"TP-E{created_at_step}-{observation.step_id}",
        kind="tool_plan_observation" if observation.success else "tool_plan_negative_observation",
        statement=statement,
        phase="tool_plan",
        tool_name=observation.tool,
        operation_id=observation.operation_id,
        artifact_id=observation.artifact_id,
        source=source,
        evidence_type="direct_observation",
        confidence=0.8 if observation.success else 0.4,
        negative=not observation.success,
        metadata={
            "tool_plan_step_id": observation.step_id,
            "tool_plan_step_index": step_index,
            "path": observation.path,
            "query": observation.query,
            "error": observation.error,
            "excerpt": observation.excerpt,
            "duplicate_of": observation.duplicate_of,
            "objective": objective,
            "observation_adapter": "tool_plan_observation",
            "observation_kind": "tool_plan_observation"
            if observation.success
            else "tool_plan_negative_observation",
        },
        created_at_step=created_at_step,
    )


def attach_tool_plan_observation_evidence(
    state: LoopState,
    *,
    objective: str,
    observations: list[ToolPlanObservation],
) -> list[str]:
    graph = state.reasoning_graph
    created_at_step = int(getattr(state, "step_count", 0) or 0)
    existing = {record.evidence_id for record in graph.evidence_records}
    attached_ids: list[str] = []
    for index, observation in enumerate(observations, start=1):
        record = observation_to_evidence_record(
            observation,
            objective=objective,
            step_index=index,
            created_at_step=created_at_step,
        )
        attached_ids.append(record.evidence_id)
        if record.evidence_id in existing:
            continue
        graph.evidence_records.append(record)
        existing.add(record.evidence_id)
    graph.touch_ids()
    return attached_ids
