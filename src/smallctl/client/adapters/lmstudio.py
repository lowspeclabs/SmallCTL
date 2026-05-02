"""LM Studio provider adapter."""

from __future__ import annotations

import copy
from typing import Any

from .base import StreamPolicy
from .common import sanitize_messages_for_lmstudio
from .common import should_retry_without_stream_options as _retry_without_stream_options


class LMStudioAdapter:
    name = "lmstudio"
    stream_policy = StreamPolicy(
        supports_stream_options=False,
        first_token_timeout_sec=45.0,
        tool_call_continuation_timeout_sec=90.0,
    )

    def sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sanitize_messages_for_lmstudio(messages)

    def mutate_headers(self, headers: dict[str, str]) -> dict[str, str]:
        return dict(headers)

    def mutate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        mutated = dict(payload)
        raw_tools = mutated.get("tools")
        if not isinstance(raw_tools, list):
            return mutated
        normalized_tools: list[dict[str, Any]] = []
        for tool in raw_tools:
            if not isinstance(tool, dict):
                normalized_tools.append(tool)
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                normalized_tools.append(tool)
                continue
            parameters = function.get("parameters")
            normalized_function = dict(function)
            normalized_function["parameters"] = _normalize_lmstudio_tool_parameters(parameters)
            normalized_tool = dict(tool)
            normalized_tool["function"] = normalized_function
            normalized_tools.append(normalized_tool)
        mutated["tools"] = normalized_tools
        return mutated

    def should_retry_without_stream_options(self, exc: Any) -> bool:
        return _retry_without_stream_options(exc)


def _normalize_lmstudio_tool_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "additionalProperties": False}
    return _flatten_schema_for_lmstudio(parameters)


def _flatten_schema_for_lmstudio(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(schema)
    combinator_keys = ("oneOf", "anyOf", "allOf")
    for key in combinator_keys:
        variants = normalized.pop(key, None)
        if not isinstance(variants, list):
            continue
        merged = _merge_object_schema_variants(variants)
        normalized = _merge_object_schema_variants([normalized, merged])
    properties = normalized.get("properties")
    if isinstance(properties, dict):
        normalized["properties"] = {
            str(name): _flatten_schema_for_lmstudio(value)
            if isinstance(value, dict)
            else value
            for name, value in properties.items()
        }
    items = normalized.get("items")
    if isinstance(items, dict):
        normalized["items"] = _flatten_schema_for_lmstudio(items)
    if str(normalized.get("type") or "").strip() != "object" and "properties" in normalized:
        normalized["type"] = "object"
    return normalized


def _merge_object_schema_variants(variants: list[Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        if str(variant.get("type") or "").strip() == "object" or isinstance(variant.get("properties"), dict):
            merged_properties = merged.setdefault("properties", {})
            if isinstance(merged_properties, dict):
                for name, value in dict(variant.get("properties") or {}).items():
                    merged_properties[str(name)] = copy.deepcopy(value)
            if variant.get("additionalProperties") is True:
                merged["additionalProperties"] = True
    return merged
