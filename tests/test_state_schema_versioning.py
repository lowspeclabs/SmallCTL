from __future__ import annotations

import asyncio
import json
from pathlib import Path

from smallctl.graph.state import inflate_graph_state, serialize_graph_state
from smallctl.state import ExperienceMemory, LOOP_STATE_SCHEMA_VERSION, LoopState, PromptBudgetSnapshot, TurnBundle
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


def test_loop_state_round_trip_preserves_task_mode() -> None:
    state = LoopState(cwd="/tmp", task_mode="analysis")

    restored = LoopState.from_dict(state.to_dict())

    assert restored.task_mode == "analysis"


def test_loop_state_round_trip_preserves_warm_experience_namespace() -> None:
    state = LoopState(cwd="/tmp")
    state.warm_experiences.append(
        ExperienceMemory(
            memory_id="mem-namespace",
            intent="requested_ssh_exec",
            namespace="ssh_remote",
            tool_name="ssh_exec",
            outcome="success",
            notes="Run whoami on the remote host.",
        )
    )

    restored = LoopState.from_dict(state.to_dict())

    assert restored.warm_experiences[0].namespace == "ssh_remote"


def test_loop_state_from_dict_infers_namespace_for_legacy_experience_payload() -> None:
    state = LoopState.from_dict(
        {
            "cwd": "/tmp",
            "warm_experiences": [
                {
                    "memory_id": "mem-legacy",
                    "intent": "requested_ssh_exec",
                    "tool_name": "ssh_exec",
                    "outcome": "success",
                    "notes": "Successfully called ssh_exec.",
                }
            ],
        }
    )

    assert state.warm_experiences[0].namespace == "ssh_remote"


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


def test_loop_state_round_trip_preserves_turn_bundles_and_compaction_levels() -> None:
    state = LoopState(cwd="/tmp")
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB42",
            created_at="2026-04-18T00:00:00+00:00",
            step_range=(4, 8),
            phase="author",
            intent="requested_file_patch",
            summary_lines=["Patched src/app.py", "Reran pytest"],
            files_touched=["src/app.py"],
            artifact_ids=["A42"],
            evidence_refs=["E42"],
            source_message_count=5,
        )
    ]
    state.prompt_budget = PromptBudgetSnapshot(
        estimated_prompt_tokens=1200,
        included_compaction_levels=["L0", "L1", "L2"],
        dropped_compaction_levels=["L4"],
    )

    restored = LoopState.from_dict(state.to_dict())

    assert restored.turn_bundles
    assert restored.turn_bundles[0].bundle_id == "TB42"
    assert restored.turn_bundles[0].summary_lines == ["Patched src/app.py", "Reran pytest"]
    assert restored.prompt_budget.included_compaction_levels == ["L0", "L1", "L2"]
    assert restored.prompt_budget.dropped_compaction_levels == ["L4"]


def test_loop_state_from_legacy_payload_defaults_new_rollout_fields() -> None:
    restored = LoopState.from_dict(
        {
            "schema_version": 1,
            "current_phase": "explore",
            "thread_id": "legacy-thread",
            "recent_messages": [],
            "run_brief": {"original_task": "Read README"},
            "working_memory": {"current_goal": "Read README"},
            "prompt_budget": {"estimated_prompt_tokens": 100},
        }
    )

    assert restored.turn_bundles == []
    assert restored.prompt_budget.included_compaction_levels == []
    assert restored.prompt_budget.dropped_compaction_levels == []
