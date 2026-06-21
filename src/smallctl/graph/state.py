from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..durable_tool_results import compact_tool_result_for_durable_state
from ..models.tool_result import ToolEnvelope
from ..state import LOOP_STATE_SCHEMA_VERSION, LoopState, json_safe_value


log = logging.getLogger(__name__)


def _normalize_write_session_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in {"file_write", "file_append", "file_patch", "ast_patch"}:
        return args
    if not isinstance(args, dict):
        return {}

    normalized = dict(args)

    session_id = str(normalized.get("session_id") or "").strip()
    write_session_id = str(normalized.get("write_session_id") or "").strip()
    if session_id and not write_session_id:
        normalized["write_session_id"] = session_id
    if session_id:
        normalized.pop("session_id", None)

    return normalized


@dataclass
class PendingToolCall:
    tool_name: str
    args: dict[str, Any]
    tool_call_id: str | None = None
    raw_arguments: str = ""
    source: str = "model"
    parser_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "PendingToolCall | None":
        if not isinstance(payload, dict):
            return None
        function = _coerce_dict_payload(payload.get("function"))
        raw_args = function.get("arguments", "")
        tool_name, signature_args = cls._parse_tool_signature(function.get("name", ""))
        args, parser_metadata = cls._parse_args_with_metadata(raw_args)
        if signature_args:
            if not args:
                args = signature_args
            else:
                # Preserve explicit provider arguments but backfill any fields that were
                # encoded into the tool name itself.
                for key, value in signature_args.items():
                    args.setdefault(key, value)
        if not tool_name:
            return None
        args = _normalize_write_session_tool_args(tool_name, args)
        tool_call_id = payload.get("id")
        return cls(
            tool_name=tool_name,
            args=args,
            tool_call_id=None if tool_call_id is None else str(tool_call_id),
            raw_arguments=raw_args,
            parser_metadata=parser_metadata,
        )

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        args, _metadata = PendingToolCall._parse_args_with_metadata(raw)
        return args

    @staticmethod
    def _parse_args_with_metadata(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata: dict[str, Any] = {}
        if not isinstance(raw, str):
            metadata["arguments_empty"] = True
            metadata["arguments_parse_error"] = {
                "kind": "non_string_arguments",
                "message": f"Expected string arguments payload, got {type(raw).__name__}.",
            }
            return {}, metadata
        if not raw:
            metadata["arguments_empty"] = True
            return {}, metadata

        metadata["raw_arguments_preview"] = raw[:500]

        def _parse_mapping(candidate: str) -> tuple[dict[str, Any], dict[str, str] | None]:
            try:
                parsed = json.loads(candidate)
            except Exception as json_exc:
                try:
                    parsed = ast.literal_eval(candidate)
                except Exception as ast_exc:
                    return {}, {
                        "kind": "malformed_json_arguments",
                        "message": str(json_exc),
                        "fallback_message": str(ast_exc),
                    }
            if not isinstance(parsed, dict):
                return {}, {
                    "kind": "non_object_arguments",
                    "message": f"Parsed arguments as {type(parsed).__name__}, expected object.",
                }
            return parsed, None

        # Step 1: Basic cleanup
        cleaned = raw.strip()
        if not cleaned:
            metadata["arguments_empty"] = True
            return {}, metadata
        last_error: dict[str, str] | None = None

        # Step 2: Fix trailing commas in objects and arrays
        # This regex looks for commas followed by any whitespace and then a closing brace or bracket.
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

        # Step 3: Attempt to fix missing closing braces
        # If the string starts with '{' but is missing the balancing '}', we try to append it.
        if cleaned.startswith('{') and not cleaned.endswith('}'):
            # Only try this if it's missing at most 5 closing braces (guard against crazy strings)
            for _ in range(5):
                cleaned += '}'
                # Clean AGAIN after adding brace to catch trailing commas that now have a bracket following them
                repaired = re.sub(r',\s*([\]}])', r'\1', cleaned)
                parsed, parse_error = _parse_mapping(repaired)
                if parse_error is None:
                    return parsed, metadata
                last_error = parse_error

        parsed, parse_error = _parse_mapping(cleaned)
        if parse_error is None:
            return parsed, metadata
        if parse_error is not None:
            last_error = parse_error
        if not parsed:
            # Step 4: Final fallback for very common 'code-block' wrap halluciations
            if "```json" in cleaned:
                # Pre-clean the whole thing for trailing commas if we're desperate
                desperate = re.sub(r',\s*([\]}])', r'\1', cleaned)
                match = re.search(r"```json\s*(\{.*?\})\s*```", desperate, re.DOTALL)
                if match:
                    parsed, parse_error = _parse_mapping(match.group(1))
                    if parse_error is not None:
                        last_error = parse_error
                else:
                    parsed = {}
        if isinstance(parsed, dict) and parsed:
            return parsed, metadata
        metadata["arguments_empty"] = False
        metadata["arguments_parse_error"] = last_error or {
            "kind": "malformed_json_arguments",
            "message": "Unable to parse tool-call arguments as a JSON object.",
        }
        return {}, metadata

    @staticmethod
    def _parse_tool_signature(raw_name: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw_name, str):
            return "", {}

        name = raw_name.strip()
        if not name:
            return "", {}

        match = re.match(r"^\s*([A-Za-z0-9_.:-]+)\((.*)\)\s*$", name, re.DOTALL)
        if not match:
            return name, {}

        tool_name = match.group(1).strip()
        args_text = match.group(2).strip()
        if not tool_name:
            return name, {}
        if not args_text:
            return tool_name, {}

        parsed_args = PendingToolCall._parse_function_arguments(args_text)
        return tool_name, parsed_args

    @staticmethod
    def _parse_function_arguments(raw: str) -> dict[str, Any]:
        if not isinstance(raw, str):
            return {}

        cleaned = raw.strip().rstrip(",")
        if not cleaned:
            return {}

        # Reuse the JSON repair path first when the content is JSON-like.
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed

        try:
            call = ast.parse(f"_tool__({cleaned})", mode="eval")
        except Exception:
            return {}

        expr = call.body
        if not isinstance(expr, ast.Call):
            return {}

        args: dict[str, Any] = {}
        for keyword in expr.keywords:
            if keyword.arg is None:
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except Exception:
                continue
            args[keyword.arg] = value
        return args


@dataclass
class ToolExecutionRecord:
    operation_id: str
    tool_name: str
    args: dict[str, Any]
    tool_call_id: str | None
    result: ToolEnvelope
    replayed: bool = False
    def to_summary_dict(self) -> dict[str, Any]:
        summary = {
            "tool_name": self.tool_name,
            "success": self.result.success,
            "replayed": self.replayed,
        }
        if self.result.error:
            summary["error"] = self.result.error
        elif isinstance(self.result.output, dict):
            summary["output"] = {
                key: value
                for key, value in self.result.output.items()
                if key in {"status", "message", "question"}
            }
        return summary

@dataclass
class GraphRunState:
    loop_state: LoopState
    thread_id: str
    run_mode: str
    pending_tool_calls: list[PendingToolCall] = field(default_factory=list)
    last_assistant_text: str = ""
    last_thinking_text: str = ""
    last_usage: dict[str, Any] = field(default_factory=dict)
    last_tool_results: list[ToolExecutionRecord] = field(default_factory=list)
    final_result: dict[str, Any] | None = None
    interrupt_payload: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    latency_metrics: dict[str, float | int] = field(default_factory=dict)
    recorded_tool_call_ids: list[str] = field(default_factory=list)


def build_operation_id(
    *,
    thread_id: str,
    step_count: int,
    tool_call_id: str | None,
    tool_name: str,
) -> str:
    safe_tool_call_id = tool_call_id or "no_tool_call_id"
    return f"{thread_id}:{step_count}:{safe_tool_call_id}:{tool_name}"


def serialize_graph_state(graph_state: GraphRunState, *, artifact_store: Any = None) -> dict[str, Any]:
    last_tool_results = [_serialize_tool_execution_record(item) for item in graph_state.last_tool_results]
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "graph_state_serialization thread_id=%s last_tool_results_count=%d last_tool_results_bytes=%d",
            graph_state.thread_id,
            len(last_tool_results),
            len(json.dumps(last_tool_results, ensure_ascii=False, default=str).encode("utf-8")),
        )
    return {
        "graph_state_schema_version": 1,
        "loop_state_schema_version": LOOP_STATE_SCHEMA_VERSION,
        "loop_state": graph_state.loop_state.to_dict(artifact_store=artifact_store),
        "thread_id": graph_state.thread_id,
        "run_mode": graph_state.run_mode,
        "pending_tool_calls": [_serialize_pending_tool_call(item) for item in graph_state.pending_tool_calls],
        "last_assistant_text": graph_state.last_assistant_text,
        "last_thinking_text": graph_state.last_thinking_text,
        "last_usage": _coerce_dict_payload(graph_state.last_usage),
        "last_tool_results": last_tool_results,
        "final_result": _coerce_optional_dict_payload(graph_state.final_result),
        "interrupt_payload": _coerce_optional_dict_payload(graph_state.interrupt_payload),
        "error": _coerce_optional_dict_payload(graph_state.error),
        "latency_metrics": _coerce_dict_payload(graph_state.latency_metrics),
        "recorded_tool_call_ids": graph_state.recorded_tool_call_ids,
    }


def inflate_graph_state(payload: dict[str, Any]) -> GraphRunState:
    loop_state = LoopState.from_dict(_coerce_dict_payload(payload.get("loop_state")))
    pending_tool_calls = [
        pending
        for item in _coerce_list_payload(payload.get("pending_tool_calls"))
        if (pending := _coerce_pending_tool_call(item)) is not None
    ]
    last_tool_results = [
        record
        for item in _coerce_list_payload(payload.get("last_tool_results"))
        if (record := _coerce_tool_execution_record(item)) is not None
    ]
    last_usage = _coerce_dict_payload(payload.get("last_usage"))
    final_result = _coerce_optional_dict_payload(payload.get("final_result"))
    interrupt_payload = _coerce_optional_dict_payload(payload.get("interrupt_payload"))
    error = _coerce_optional_dict_payload(payload.get("error"))
    recorded_tool_call_ids = _coerce_list_payload(payload.get("recorded_tool_call_ids"))
    return GraphRunState(
        loop_state=loop_state,
        thread_id=str(payload.get("thread_id", loop_state.thread_id or "")),
        run_mode=str(payload.get("run_mode", "loop")),
        pending_tool_calls=pending_tool_calls,
        last_assistant_text=str(payload.get("last_assistant_text", "")),
        last_thinking_text=str(payload.get("last_thinking_text", "")),
        last_usage=last_usage,
        last_tool_results=last_tool_results,
        final_result=final_result,
        interrupt_payload=interrupt_payload,
        error=error,
        latency_metrics=_coerce_dict_payload(payload.get("latency_metrics")),
        recorded_tool_call_ids=[str(item) for item in recorded_tool_call_ids],
    )


def _tool_envelope_from_dict(payload: dict[str, Any]) -> ToolEnvelope:
    metadata = _coerce_dict_payload(payload.get("metadata"))
    return ToolEnvelope(
        success=bool(payload.get("success")),
        status=payload.get("status"),
        output=json_safe_value(payload.get("output")),
        error=None if payload.get("error") is None else str(payload.get("error")),
        metadata=metadata,
    )


def _serialize_pending_tool_call(item: PendingToolCall) -> dict[str, Any]:
    return {
        "tool_name": item.tool_name,
        "args": _coerce_dict_payload(item.args),
        "tool_call_id": item.tool_call_id,
        "source": str(item.source or "model"),
        "raw_arguments": str(item.raw_arguments or ""),
        "parser_metadata": _coerce_dict_payload(item.parser_metadata),
    }


def _serialize_tool_execution_record(item: ToolExecutionRecord) -> dict[str, Any]:
    artifact_id = ""
    metadata = item.result.metadata if isinstance(item.result.metadata, dict) else {}
    if metadata:
        artifact_id = str(metadata.get("artifact_id") or "").strip()
    return {
        "operation_id": item.operation_id,
        "tool_name": item.tool_name,
        "args": _coerce_dict_payload(item.args),
        "tool_call_id": item.tool_call_id,
        "result": compact_tool_result_for_durable_state(
            item.result.to_dict(),
            tool_name=item.tool_name,
            artifact_id=artifact_id,
        ),
        "replayed": item.replayed,
    }


def _coerce_pending_tool_call(value: Any) -> PendingToolCall | None:
    if not isinstance(value, dict):
        return None
    tool_call_id = value.get("tool_call_id")
    source = str(value.get("source", "model") or "model").strip().lower()
    if source not in {"model", "system", "tool_plan"}:
        source = "model"
    return PendingToolCall(
        tool_name=str(value.get("tool_name", "")),
        args=_coerce_dict_payload(value.get("args")),
        tool_call_id=None if tool_call_id is None else str(tool_call_id),
        raw_arguments=str(value.get("raw_arguments", "")),
        source=source,
        parser_metadata=_coerce_dict_payload(value.get("parser_metadata")),
    )


def _coerce_tool_execution_record(value: Any) -> ToolExecutionRecord | None:
    if not isinstance(value, dict):
        return None
    tool_call_id = value.get("tool_call_id")
    return ToolExecutionRecord(
        operation_id=str(value.get("operation_id", "")),
        tool_name=str(value.get("tool_name", "")),
        args=_coerce_dict_payload(value.get("args")),
        tool_call_id=None if tool_call_id is None else str(tool_call_id),
        result=_tool_envelope_from_dict(_coerce_dict_payload(value.get("result"))),
        replayed=bool(value.get("replayed")),
    )


def _coerce_list_payload(value: Any) -> list[Any]:
    normalized = json_safe_value(value)
    return normalized if isinstance(normalized, list) else []


def _coerce_optional_dict_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    normalized = json_safe_value(value)
    return normalized if isinstance(normalized, dict) else None


def _coerce_dict_payload(value: Any) -> dict[str, Any]:
    normalized = json_safe_value(value or {})
    return normalized if isinstance(normalized, dict) else {}
