from __future__ import annotations

from smallctl.context import ContextPolicy, PromptAssembler, build_observation_packets
from smallctl.evidence import normalize_tool_result
from smallctl.models.tool_result import ToolEnvelope
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


def test_observation_packets_use_adapter_driven_observation_list_without_statement() -> None:
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-search",
            tool_name="shell_exec",
            statement="",
            metadata={
                "observation_adapter": "shell_observation_list",
                "command": "rg TODO src",
                "query": "TODO",
                "observation_items": [
                    "src/app.py:12: TODO tighten parser budget",
                    "src/utils.py:9: TODO add regression guard",
                ],
            },
        )
    ]

    packets = build_observation_packets(state, limit=4)

    assert len(packets) == 1
    assert packets[0].kind == "observation_list"
    assert packets[0].query == "TODO"
    assert "Observation list [TODO]" in packets[0].summary
    assert "src/app.py:12: TODO tighten parser budget" in packets[0].summary


def test_observation_packets_fall_back_when_adapter_payload_is_malformed() -> None:
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-malformed",
            tool_name="shell_exec",
            statement="shell_exec: fallback summary should still render",
            metadata={
                "observation_adapter": "shell_observation_list",
                "observation_items": {"not": "a list"},
            },
        )
    ]

    packets = build_observation_packets(state, limit=4)

    assert packets
    assert packets[0].kind == "observation_list"
    assert "fallback summary should still render" in packets[0].summary.lower()


def test_normalize_tool_result_sets_observation_adapter_metadata() -> None:
    file_evidence = normalize_tool_result(
        tool_name="file_read",
        result=ToolEnvelope(success=True, output="hello", metadata={"path": "src/app.py"}),
        evidence_context={"args": {"path": "src/app.py"}},
    )
    verifier_evidence = normalize_tool_result(
        tool_name="shell_exec",
        result=ToolEnvelope(
            success=False,
            output={"exit_code": 1, "stderr": "assert failed"},
            metadata={"command": "pytest -q"},
        ),
        evidence_context={"args": {"command": "pytest -q"}},
    )
    search_evidence = normalize_tool_result(
        tool_name="shell_exec",
        result=ToolEnvelope(
            success=True,
            output={"stdout": "src/app.py:3: TODO improve parser"},
            metadata={"command": "rg TODO src"},
        ),
        evidence_context={"args": {"command": "rg TODO src"}},
    )

    assert file_evidence.metadata["observation_adapter"] == "file_read_fact"
    assert verifier_evidence.metadata["observation_adapter"] == "verifier_verdict"
    assert verifier_evidence.metadata["verdict"] == "fail"
    assert search_evidence.metadata["observation_adapter"] == "shell_observation_list"
    assert search_evidence.metadata["observation_kind"] == "observation_list"


def test_observation_packets_render_web_search_findings_compactly() -> None:
    search_evidence = normalize_tool_result(
        tool_name="web_search",
        result=ToolEnvelope(
            success=True,
            output={
                "query": "best self hosted blog docker",
                "provider": "duckduckgo",
                "results": [
                    {
                        "result_id": "webres-1",
                        "title": "Ghost Docker Install",
                        "url": "https://example.com/ghost",
                        "domain": "example.com",
                        "snippet": "Official Docker-based install guide for Ghost.",
                    },
                    {
                        "result_id": "webres-2",
                        "title": "WriteFreely Compose Setup",
                        "url": "https://example.com/writefreely",
                        "domain": "example.com",
                        "snippet": "Compose-based self-hosted blogging setup.",
                    },
                ],
            },
        ),
        evidence_context={"args": {"query": "best self hosted blog docker"}},
    )
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.evidence_records = [search_evidence]

    packets = build_observation_packets(state, limit=4)

    assert len(packets) == 1
    assert packets[0].kind == "observation_list"
    assert packets[0].tool_name == "web_search"
    assert "Observation list [best self hosted blog docker]" in packets[0].summary
    assert "Ghost Docker Install" in packets[0].summary
    assert "(+1 more)" in packets[0].summary


def test_observation_packets_render_web_fetch_findings_compactly() -> None:
    fetch_evidence = normalize_tool_result(
        tool_name="web_fetch",
        result=ToolEnvelope(
            success=True,
            output={
                "source_id": "webres-1",
                "title": "Ghost Docker Install",
                "url": "https://example.com/ghost",
                "canonical_url": "https://example.com/ghost",
                "domain": "example.com",
                "text_excerpt": "Official Docker-based install guide for Ghost.",
                "untrusted_text": "Official Docker-based install guide for Ghost.",
            },
        ),
        evidence_context={"args": {"result_id": "webres-1"}},
    )
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.evidence_records = [fetch_evidence]

    packets = build_observation_packets(state, limit=4)

    assert len(packets) == 1
    assert packets[0].kind == "web_observation"
    assert packets[0].tool_name == "web_fetch"
    assert packets[0].summary.startswith("Web finding: Ghost Docker Install")
    assert "Official Docker-based install guide for Ghost." in packets[0].summary


def test_observation_packets_legacy_negative_verifier_defaults_to_fail() -> None:
    state = LoopState(cwd="/tmp")
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-legacy-verifier",
            tool_name="shell_exec",
            statement="shell_exec failed: parser test failed",
            negative=True,
            metadata={"command": "pytest -q", "exit_code": 1},
        )
    ]

    packets = build_observation_packets(state, limit=4)

    assert packets
    assert packets[0].kind == "verifier_verdict"
    assert "Verifier verdict (fail)" in packets[0].summary
