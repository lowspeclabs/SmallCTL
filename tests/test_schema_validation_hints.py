from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.node_support import schema_validation_repair_decision
from smallctl.graph.state import PendingToolCall
from smallctl.state import LoopState


class _Spec:
    def __init__(self, schema: dict[str, object]) -> None:
        self._schema = schema

    def openai_schema(self) -> dict[str, object]:
        return self._schema


class _Registry:
    def __init__(self, schemas: dict[str, dict[str, object]]) -> None:
        self._schemas = schemas

    def get(self, name: str) -> _Spec | None:
        schema = self._schemas.get(name)
        return _Spec(schema) if schema else None


def _harness(schema: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        state=LoopState(step_count=5),
        registry=_Registry({str(schema["function"]["name"]): schema}),
    )


def test_schema_validation_repair_appends_compact_schema_for_artifact_read() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "artifact_read",
            "parameters": {
                "type": "object",
                "required": ["artifact_id"],
                "properties": {
                    "artifact_id": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
            },
        },
    }
    harness = _harness(schema)

    decision = schema_validation_repair_decision(
        harness,
        PendingToolCall(tool_name="artifact_read", args={}),
        "missing required field",
        {"required_fields": ["artifact_id"]},
    )

    assert "Required fields: artifact_id" in decision.repair_message
    assert "start_line:integer" in decision.repair_message
    assert "end_line:integer" in decision.repair_message
    assert "max_chars:integer" in decision.repair_message
    assert '"artifact_id": "..."' in decision.repair_message
    assert harness.state.scratchpad["_last_schema_validation_hint"]["tool_name"] == "artifact_read"


def test_file_patch_schema_repair_keeps_tactical_hint_and_appends_schema() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "file_patch",
            "parameters": {
                "type": "object",
                "required": ["path", "target_text", "replacement_text"],
                "properties": {
                    "path": {"type": "string"},
                    "target_text": {"type": "string"},
                    "replacement_text": {"type": "string"},
                },
            },
        },
    }

    decision = schema_validation_repair_decision(
        _harness(schema),
        PendingToolCall(tool_name="file_patch", args={}),
        "missing required field",
        {"required_fields": ["path", "target_text", "replacement_text"]},
    )

    assert "Use exact target text and replacement text including whitespace" in decision.repair_message
    assert "Compact schema for `file_patch`" in decision.repair_message
    assert "target_text:string" in decision.repair_message
