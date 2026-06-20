from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.node_support import schema_validation_repair_decision
from smallctl.graph.state import PendingToolCall
from smallctl.graph.tool_call_parser_support import _detect_missing_required_tool_arguments
from smallctl.state import LoopState


class _Spec:
    def __init__(self, schema: dict[str, object]) -> None:
        self._schema = schema
        self.schema = schema

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


def test_schema_validation_repair_reports_malformed_arguments_as_system_nudge() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "ssh_exec",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string"},
                    "target": {"type": "string"},
                },
            },
        },
    }
    harness = _harness(schema)
    raw_arguments = '{"command": "cd /opt/qwen-compose-medium && docker compose up -d"{}'
    pending = PendingToolCall.from_payload(
        {
            "id": "call_step4",
            "type": "function",
            "function": {"name": "ssh_exec", "arguments": raw_arguments},
        }
    )
    assert pending is not None

    decision = schema_validation_repair_decision(
        harness,
        pending,
        "missing required field",
        {"required_fields": ["command"]},
    )

    assert "malformed JSON arguments" in decision.repair_message
    assert "without arguments" not in decision.repair_message
    assert decision.details["arguments_malformed"] is True
    assert decision.details["arguments_empty"] is False
    assert decision.alert_data["arguments_malformed"] is True
    assert decision.runlog_data["arguments_malformed"] is True
    assert decision.conversation_message is not None
    assert decision.conversation_message.role == "system"
    assert decision.conversation_message.metadata["is_recovery_nudge"] is True
    assert decision.conversation_message.metadata["recovery_kind"] == "schema_validation"
    assert decision.conversation_message.metadata["raw_arguments_preview"] == raw_arguments
    traces = harness.state.scratchpad["_tool_call_parse_failures"]
    assert traces == [
        {
            "tool_name": "ssh_exec",
            "tool_call_id": "call_step4",
            "step_count": 5,
            "parse_error": decision.details["arguments_parse_error"],
            "raw_arguments_preview": raw_arguments,
            "source": "model",
            "replayable": False,
        }
    ]


def test_empty_replacement_text_allowed_for_patch_tools() -> None:
    patch_schemas = {
        "file_patch": {
            "type": "object",
            "required": ["path", "target_text", "replacement_text"],
            "properties": {
                "path": {"type": "string"},
                "target_text": {"type": "string"},
                "replacement_text": {"type": "string"},
            },
        },
        "ssh_file_patch": {
            "type": "object",
            "required": ["path", "target_text", "replacement_text"],
            "properties": {
                "path": {"type": "string"},
                "target_text": {"type": "string"},
                "replacement_text": {"type": "string"},
            },
        },
        "ssh_file_replace_between": {
            "type": "object",
            "required": ["path", "start_text", "end_text", "replacement_text"],
            "properties": {
                "path": {"type": "string"},
                "start_text": {"type": "string"},
                "end_text": {"type": "string"},
                "replacement_text": {"type": "string"},
            },
        },
    }

    for tool_name, schema in patch_schemas.items():
        harness = SimpleNamespace(
            state=LoopState(step_count=5),
            registry=_Registry({tool_name: schema}),
        )
        args: dict[str, object] = {
            "path": "/tmp/webmin-compose.yml",
            "replacement_text": "",
        }
        if tool_name == "ssh_file_replace_between":
            args["start_text"] = "version: '3'"
            args["end_text"] = "services:"
        else:
            args["target_text"] = "version: '3'"

        pending = PendingToolCall(tool_name=tool_name, args=args)
        assert _detect_missing_required_tool_arguments(harness, pending) is None, (
            f"empty replacement_text should be allowed for {tool_name}"
        )


def test_empty_non_replacement_text_still_reported_missing() -> None:
    schema = {
        "type": "object",
        "required": ["path", "target_text", "replacement_text"],
        "properties": {
            "path": {"type": "string"},
            "target_text": {"type": "string"},
            "replacement_text": {"type": "string"},
        },
    }
    harness = SimpleNamespace(
        state=LoopState(step_count=5),
        registry=_Registry({"file_patch": schema}),
    )
    pending = PendingToolCall(
        tool_name="file_patch",
        args={"path": "/tmp/webmin-compose.yml", "target_text": "", "replacement_text": ""},
    )
    result = _detect_missing_required_tool_arguments(harness, pending)
    assert result is not None
    message, details = result
    assert "target_text" in details["required_fields"]
    assert "replacement_text" not in details["required_fields"]
    assert "missing required fields" in message.lower()
