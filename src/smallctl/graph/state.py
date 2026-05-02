from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..models.tool_result import ToolEnvelope
from ..state import LOOP_STATE_SCHEMA_VERSION, LoopState, json_safe_value


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

    # Providers sometimes emit Python literals such as `None` for optional
    # write-session metadata. Treat those as omitted fields instead of letting
    # schema validation fail after we've already recovered the real content.
    for key, value in list(normalized.items()):
        if value is None:
            normalized.pop(key, None)
    for key in (
        "path",
        "write_session_id",
        "section_name",
        "next_section_name",
        "replace_strategy",
        "target_text",
        "replacement_text",
        "language",
        "operation",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
            normalized.pop(key, None)

    return normalized


@dataclass
class PendingToolCall:
    tool_name: str
    args: dict[str, Any]
    tool_call_id: str | None = None
    raw_arguments: str = ""
    source: str = "model"

    @classmethod
    def from_payload(cls, payload: Any) -> "PendingToolCall | None":
        if not isinstance(payload, dict):
            return None
        function = _coerce_dict_payload(payload.get("function"))
        raw_args = function.get("arguments", "")
        tool_name, signature_args = cls._parse_tool_signature(function.get("name", ""))
        args = cls._parse_args(raw_args)
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
        )

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str) or not raw:
            return {}

        def _parse_mapping(candidate: str) -> dict[str, Any]:
            try:
                parsed = json.loads(candidate)
            except Exception:
                try:
                    parsed = ast.literal_eval(candidate)
                except Exception:
                    return {}
            return parsed if isinstance(parsed, dict) else {}

        # Step 1: Basic cleanup
        cleaned = raw.strip()
        
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
                parsed = _parse_mapping(repaired)
                if parsed:
                    return parsed

        parsed = _parse_mapping(cleaned)
        if not parsed:
            # Step 4: Final fallback for very common 'code-block' wrap halluciations
            if "```json" in cleaned:
                # Pre-clean the whole thing for trailing commas if we're desperate
                desperate = re.sub(r',\s*([\]}])', r'\1', cleaned)
                match = re.search(r"```json\s*(\{.*?\})\s*```", desperate, re.DOTALL)
                if match:
                    parsed = _parse_mapping(match.group(1))
                else:
                    return {}
        return parsed if isinstance(parsed, dict) else {}

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


def serialize_graph_state(graph_state: GraphRunState) -> dict[str, Any]:
    return {
        "graph_state_schema_version": 1,
        "loop_state_schema_version": LOOP_STATE_SCHEMA_VERSION,
        "loop_state": graph_state.loop_state.to_dict(),
        "thread_id": graph_state.thread_id,
        "run_mode": graph_state.run_mode,
        "pending_tool_calls": [_serialize_pending_tool_call(item) for item in graph_state.pending_tool_calls],
        "last_assistant_text": graph_state.last_assistant_text,
        "last_thinking_text": graph_state.last_thinking_text,
        "last_usage": _coerce_dict_payload(graph_state.last_usage),
        "last_tool_results": [_serialize_tool_execution_record(item) for item in graph_state.last_tool_results],
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
    }


def _serialize_tool_execution_record(item: ToolExecutionRecord) -> dict[str, Any]:
    return {
        "operation_id": item.operation_id,
        "tool_name": item.tool_name,
        "args": _coerce_dict_payload(item.args),
        "tool_call_id": item.tool_call_id,
        "result": json_safe_value(item.result.to_dict()),
        "replayed": item.replayed,
    }


def _coerce_pending_tool_call(value: Any) -> PendingToolCall | None:
    if not isinstance(value, dict):
        return None
    tool_call_id = value.get("tool_call_id")
    source = str(value.get("source", "model") or "model").strip().lower()
    if source not in {"model", "system"}:
        source = "model"
    return PendingToolCall(
        tool_name=str(value.get("tool_name", "")),
        args=_coerce_dict_payload(value.get("args")),
        tool_call_id=None if tool_call_id is None else str(tool_call_id),
        source=source,
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
