from __future__ import annotations

from smallctl.context import ContextPolicy, PromptAssembler, build_observation_packets
from smallctl.state import EvidenceRecord, LoopState


def test_build_observation_packets_normalizes_file_verifier_and_replay_records() -> None:
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-file",
            tool_name="file_read",
            statement="file_read: README.md text",
            metadata={"path": "README.md"},
            confidence=0.95,
        ),
        EvidenceRecord(
            evidence_id="E-verifier",
            tool_name="shell_exec",
            statement="shell_exec failed: tests failed",
            negative=True,
            metadata={"command": "pytest -q", "exit_code": 1},
            confidence=0.9,
        ),
        EvidenceRecord(
            evidence_id="E-replay",
            tool_name="artifact_read",
            statement="artifact_read: reused A100",
            replayed=True,
            evidence_type="replayed_or_cached",
            metadata={"artifact_id": "A100"},
            confidence=0.7,
        ),
    ]

    packets = build_observation_packets(state, limit=8)

    assert [packet.observation_id for packet in packets] == ["E-file", "E-verifier", "E-replay"]
    assert packets[0].kind == "file_fact"
    assert packets[1].kind == "verifier_verdict"
    assert packets[2].kind == "artifact_replay"


def test_observation_packets_mark_file_fact_stale_after_file_invalidation() -> None:
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-file",
            tool_name="file_read",
            statement="file_read: src/app.py",
            metadata={"path": "src/app.py"},
        )
    ]
    state.scratchpad["_context_invalidations"] = [
        {"reason": "file_changed", "paths": ["src/app.py"]}
    ]

    packets = build_observation_packets(state, limit=4)

    assert packets
    assert packets[0].stale is True


def test_prompt_assembler_renders_normalized_observation_lane() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Inspect observations"
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-file",
            tool_name="file_read",
            statement="file_read: README.md text",
            metadata={"path": "README.md"},
        )
    ]

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048)).build_messages(
        state=state,
        system_prompt="SYSTEM",
    )
    rendered = "\n".join(str(message.get("content") or "") for message in assembly.messages)

    assert "Normalized observations:" in rendered
    assert "E-file" in rendered
    assert "file_fact" in rendered
