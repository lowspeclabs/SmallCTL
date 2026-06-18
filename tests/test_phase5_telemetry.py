from __future__ import annotations

import logging

from smallctl.chat_sessions import (
    _largest_crossed_threshold,
    persist_chat_session_state,
    persist_chat_session_ui_transcript,
)
from smallctl.graph.state import GraphRunState, ToolExecutionRecord, serialize_graph_state
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def test_chat_session_write_logs_size_and_timing(tmp_path, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="smallctl.chat_sessions")

    persist_chat_session_state(
        cwd=tmp_path,
        thread_id="thread-telemetry",
        state_payload={"thread_id": "thread-telemetry", "recent_messages": []},
        model="test-model",
    )
    persist_chat_session_ui_transcript(
        cwd=tmp_path,
        thread_id="thread-telemetry",
        ui_transcript=[{"event_type": "system", "content": "note", "data": {}}],
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("chat_runtime_state_write" in message for message in messages)
    assert any("chat_ui_transcript_write" in message for message in messages)
    assert all("payload_bytes" in message for message in messages if "chat_session_persistence" in message)
    assert all("elapsed_ms" in message for message in messages if "chat_session_persistence" in message)


def test_chat_session_size_threshold_helper() -> None:
    assert _largest_crossed_threshold(1024) == 0
    assert _largest_crossed_threshold(5 * 1024 * 1024) == 5 * 1024 * 1024
    assert _largest_crossed_threshold(26 * 1024 * 1024) == 25 * 1024 * 1024


def test_loop_state_to_dict_logs_serialization_metrics(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="smallctl.state")
    state = LoopState(thread_id="thread-telemetry")
    state.tool_execution_records["op-1"] = {
        "operation_id": "op-1",
        "tool_name": "grep",
        "result": {"success": True, "status": "ok", "output": "short", "metadata": {}},
    }

    state.to_dict()

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "loop_state_serialization" in message
    assert "tool_execution_records_bytes" in message


def test_graph_state_serialization_logs_last_tool_result_metrics(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="smallctl.graph.state")
    graph_state = GraphRunState(
        loop_state=LoopState(thread_id="thread-telemetry"),
        thread_id="thread-telemetry",
        run_mode="loop",
        last_tool_results=[
            ToolExecutionRecord(
                operation_id="op-1",
                tool_name="grep",
                args={},
                tool_call_id="call-1",
                result=ToolEnvelope(success=True, status="ok", output="short", metadata={}),
            )
        ],
    )

    serialize_graph_state(graph_state)

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "graph_state_serialization" in message
    assert "last_tool_results_bytes" in message
