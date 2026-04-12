from __future__ import annotations

import asyncio
import json
from pathlib import Path

from smallctl.graph.state import inflate_graph_state, serialize_graph_state
from smallctl.state import LOOP_STATE_SCHEMA_VERSION, LoopState
from smallctl.tools.memory import checkpoint


def test_loop_state_to_dict_includes_schema_version() -> None:
    state = LoopState(cwd="/tmp")
    payload = state.to_dict()
    assert payload["schema_version"] == LOOP_STATE_SCHEMA_VERSION


def test_loop_state_strategy_round_trip_preserves_staged_reasoning_toggle() -> None:
    state = LoopState(cwd="/tmp", strategy={"thought_architecture": "staged_reasoning"})
    payload = state.to_dict()

    restored = LoopState.from_dict(payload)

    assert restored.strategy == {"thought_architecture": "staged_reasoning"}


def test_loop_state_from_dict_migrates_legacy_write_session_aliases() -> None:
    legacy_payload = {
        "current_phase": "execute",
        "thread_id": "t1",
        "write_session": {
            "session_id": "ws-1",
            "mode": "chunked_author",
            "lifecycle_status": "open",
            "write_target_path": "a.py",
        },
    }
    state = LoopState.from_dict(legacy_payload)
    assert state.schema_version == LOOP_STATE_SCHEMA_VERSION
    assert state.write_session is not None
    assert state.write_session.write_session_id == "ws-1"
    assert state.write_session.write_session_mode == "chunked_author"
    assert state.write_session.status == "open"


def test_checkpoint_payload_includes_schema_metadata(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    result = asyncio.run(checkpoint(state=state, label="schema-test"))
    assert result["success"] is True
    path = Path(result["output"]["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_schema_version"] == 1
    assert payload["loop_state_schema_version"] == LOOP_STATE_SCHEMA_VERSION
    assert payload["state"]["schema_version"] == LOOP_STATE_SCHEMA_VERSION


def test_legacy_graph_state_payload_migrates_loop_state_schema() -> None:
    legacy_graph_payload = {
        "thread_id": "th1",
        "run_mode": "loop",
        "loop_state": {
            "current_phase": "execute",
            "thread_id": "th1",
            "write_session": {
                "session_id": "ws_legacy",
                "mode": "chunked_author",
                "lifecycle_status": "open",
                "write_target_path": "legacy.py",
            },
        },
    }
    graph_state = inflate_graph_state(legacy_graph_payload)
    assert graph_state.loop_state.schema_version == LOOP_STATE_SCHEMA_VERSION
    assert graph_state.loop_state.write_session is not None
    assert graph_state.loop_state.write_session.write_session_id == "ws_legacy"
    serialized = serialize_graph_state(graph_state)
    assert serialized["loop_state"]["schema_version"] == LOOP_STATE_SCHEMA_VERSION
    assert serialized["graph_state_schema_version"] == 1


def test_graph_state_round_trip_preserves_staged_reasoning_toggle() -> None:
    state = LoopState(cwd="/tmp", strategy={"thought_architecture": "staged_reasoning"})
    graph_state = inflate_graph_state(
        {
            "thread_id": "th2",
            "run_mode": "loop",
            "loop_state": state.to_dict(),
        }
    )

    assert graph_state.loop_state.strategy == {"thought_architecture": "staged_reasoning"}
    serialized = serialize_graph_state(graph_state)
    assert serialized["loop_state"]["strategy"] == {"thought_architecture": "staged_reasoning"}
