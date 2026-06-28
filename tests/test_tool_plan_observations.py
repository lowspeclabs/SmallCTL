from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.runtime_tool_plan import _tool_plan_observation_budget
from smallctl.graph.state import ToolExecutionRecord
from smallctl.graph.tool_plan_observations import (
    attach_tool_plan_observation_evidence,
    build_tool_plan_observations,
    observation_to_evidence_record,
    summarize_tool_plan_observations,
    render_tool_plan_observations,
)
from smallctl.graph.tool_plan_schema import ToolPlan, ToolPlanStep
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def test_tool_plan_observation_budget_uses_config_defaults_without_wide_dag() -> None:
    state = LoopState()
    deps = SimpleNamespace(
        harness=SimpleNamespace(
            state=state,
            config=SimpleNamespace(
                tool_plan_observation_token_limit=1000,
                tool_plan_max_observation_chars_per_step=500,
            ),
        )
    )

    assert _tool_plan_observation_budget(deps) == (1000, 500)


def test_tool_plan_observation_budget_tightens_after_wide_dag_batch() -> None:
    state = LoopState()
    state.scratchpad["_recovery_metrics"] = {"tool_plan_dag_max_batch_size": 3}
    deps = SimpleNamespace(
        harness=SimpleNamespace(
            state=state,
            config=SimpleNamespace(
                tool_plan_observation_token_limit=1000,
                tool_plan_max_observation_chars_per_step=500,
            ),
        )
    )

    assert _tool_plan_observation_budget(deps) == (850, 400)


def test_tool_plan_observations_render_compact_success_and_failure() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect runtime",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E2", "grep", {"path": "src", "pattern": "missing"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(success=True, output="large body", metadata={"artifact_id": "A0001", "path": "src/app.py"}),
        ),
        ToolExecutionRecord(
            operation_id="op-2",
            tool_name="grep",
            args={"path": "src", "pattern": "missing"},
            tool_call_id="toolplan:E2",
            result=ToolEnvelope(success=False, error="no matches", metadata={"pattern": "missing"}),
        ),
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=200, max_chars_per_step=80)
    rendered = render_tool_plan_observations(plan.objective, observations)

    assert "TOOL PLAN OBSERVATIONS" in rendered
    assert "E1 file_read src/app.py" in rendered
    assert "- artifact: A0001" in rendered
    assert "E2 grep src" in rendered
    assert "- error: no matches" in rendered

    stats = summarize_tool_plan_observations(plan, observations)
    assert stats.requested_steps == 2
    assert stats.executed_steps == 2
    assert stats.successful_steps == 1
    assert stats.failed_steps == 1
    assert stats.artifact_yield_count == 1
    assert stats.tool_failure_classes == ["no_matches"]


def test_tool_plan_observations_render_output_excerpt_even_with_metadata() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect runtime routing",
        steps=[
            ToolPlanStep("E1", "grep", {"path": "src", "pattern": "ToolPlanRuntime"}),
            ToolPlanStep("E2", "file_read", {"path": "src/smallctl/graph/runtime_auto.py"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="grep",
            args={"path": "src", "pattern": "ToolPlanRuntime"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(
                success=True,
                output=[
                    {
                        "path": "src/smallctl/graph/runtime_auto.py",
                        "line": 22,
                        "text": "from .runtime_tool_plan import ToolPlanRuntime",
                    }
                ],
                metadata={"truncated": True, "count": 200},
            ),
        ),
        ToolExecutionRecord(
            operation_id="op-2",
            tool_name="file_read",
            args={"path": "src/smallctl/graph/runtime_auto.py"},
            tool_call_id="toolplan:E2",
            result=ToolEnvelope(
                success=True,
                output="if config.tool_plan_runtime_enabled:\n    return ToolPlanRuntime(...)",
                metadata={"path": "src/smallctl/graph/runtime_auto.py", "truncated": False},
            ),
        ),
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=400, max_chars_per_step=240)
    rendered = render_tool_plan_observations(plan.objective, observations)

    assert "- summary: count=200; truncated=true" in rendered
    assert "- excerpt:" in rendered
    assert "ToolPlanRuntime" in rendered
    assert "tool_plan_runtime_enabled" in rendered
    assert observations[0].excerpt
    assert observations[1].excerpt


def test_tool_plan_file_read_observation_prefers_relevant_long_file_lines() -> None:
    long_source = "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import os",
            "",
            "def unrelated_helper():",
            "    return 'noise'",
            "",
            "class ToolExecutionRecord:",
            "    pass",
            "",
            "async def dispatch_tool(name, args):",
            "    return await run_tool(name, args)",
            "",
            "def persist_artifact(record):",
            "    return artifact_store.save(record)",
        ]
    )
    plan = ToolPlan(
        mode="tool_plan",
        objective="Find where tool calls are dispatched and artifacts are persisted.",
        steps=[
            ToolPlanStep(
                "E1",
                "file_read",
                {"path": "src/smallctl/harness/tool_dispatch.py"},
                reason="Inspect dispatch_tool, ToolExecutionRecord, and artifact persistence.",
            ),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_read",
            args={"path": "src/smallctl/harness/tool_dispatch.py"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(
                success=True,
                output=long_source,
                metadata={"path": "src/smallctl/harness/tool_dispatch.py", "total_lines": 15},
            ),
        )
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=300, max_chars_per_step=260)

    assert "dispatch_tool" in observations[0].excerpt
    assert "ToolExecutionRecord" in observations[0].excerpt
    assert "persist_artifact" in observations[0].excerpt
    assert "L" in observations[0].excerpt
    assert "from __future__" not in observations[0].excerpt


def test_tool_plan_observations_dedupe_repeated_reads_and_respect_budget() -> None:
    long_output = "alpha " * 400
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect runtime",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E2", "file_read", {"path": "src/app.py"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(success=True, output=long_output, metadata={}),
        ),
        ToolExecutionRecord(
            operation_id="op-2",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E2",
            result=ToolEnvelope(success=True, output=long_output, metadata={}),
        ),
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=80, max_chars_per_step=600)
    rendered = render_tool_plan_observations(plan.objective, observations)

    assert observations[1].duplicate_of == "E1"
    assert "duplicate_of: E1" in rendered
    assert "Duplicate of E1" in rendered
    assert len(observations[0].summary) < len(long_output)


def test_tool_plan_observations_dedupe_before_tight_token_fitting() -> None:
    long_output = "beta " * 500
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect repeated reads",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E2", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E3", "file_read", {"path": "src/other.py"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(success=True, output=long_output, metadata={}),
        ),
        ToolExecutionRecord(
            operation_id="op-2",
            tool_name="file_read",
            args={"path": "src/app.py"},
            tool_call_id="toolplan:E2",
            result=ToolEnvelope(success=True, output=long_output, metadata={}),
        ),
        ToolExecutionRecord(
            operation_id="op-3",
            tool_name="file_read",
            args={"path": "src/other.py"},
            tool_call_id="toolplan:E3",
            result=ToolEnvelope(success=True, output=long_output, metadata={}),
        ),
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=60, max_chars_per_step=120)

    assert observations[1].duplicate_of == "E1"
    assert observations[1].operation_id == "op-1"
    assert "Duplicate of E1" in observations[1].summary


def test_tool_plan_observations_extracts_content_from_dict_output_without_json_dump() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect remote backup script",
        steps=[
            ToolPlanStep("E1", "ssh_file_read", {"path": "backup.sh"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="ssh_file_read",
            args={"path": "backup.sh"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(
                success=True,
                output={
                    "bytes": 376,
                    "content": "#!/bin/bash\nSOURCE_DIR=\"/root/source\"\nBACKUP_DIR=\"/root/backups\"",
                    "encoding": "utf-8",
                    "host": "192.168.1.64",
                    "path": "backup.sh",
                    "sha256": "deadbeef",
                    "truncated": False,
                },
                metadata={"path": "backup.sh", "host": "192.168.1.64"},
            ),
        )
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=400, max_chars_per_step=240)
    rendered = render_tool_plan_observations(plan.objective, observations)

    assert observations[0].excerpt
    assert "#!/bin/bash" in observations[0].excerpt
    assert "SOURCE_DIR" in observations[0].excerpt
    assert '"bytes":' not in observations[0].excerpt
    assert '"sha256":' not in observations[0].excerpt
    assert "deadbeef" not in rendered


def test_tool_plan_observations_extracts_stdout_from_shell_dict_output() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="check remote directory",
        steps=[
            ToolPlanStep("E1", "ssh_exec", {"command": "ls -la /root"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="ssh_exec",
            args={"command": "ls -la /root"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(
                success=True,
                output={"stdout": "total 12\ndrwxr-xr-x 3 root root 4096 Jun 27 00:00 .\n", "stderr": "", "exit_code": 0},
                metadata={"command": "ls -la /root"},
            ),
        )
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=400, max_chars_per_step=240)

    assert "total 12" in observations[0].excerpt
    assert "stdout" not in observations[0].excerpt
    assert "exit_code" not in observations[0].excerpt


def test_tool_plan_observations_flattens_list_of_match_dicts() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="find references",
        steps=[
            ToolPlanStep("E1", "grep", {"path": "src", "pattern": "ToolPlan"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="grep",
            args={"path": "src", "pattern": "ToolPlan"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(
                success=True,
                output=[
                    {"path": "src/a.py", "line": 10, "text": "class ToolPlan:"},
                    {"path": "src/b.py", "line": 20, "text": "def tool_plan():"},
                ],
                metadata={"count": 2},
            ),
        )
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=400, max_chars_per_step=240)

    assert "class ToolPlan:" in observations[0].excerpt
    assert "def tool_plan():" in observations[0].excerpt
    assert "{" not in observations[0].excerpt


def test_tool_plan_observations_does_not_json_dump_failed_dict_result() -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect remote source directory",
        steps=[
            ToolPlanStep("E1", "ssh_file_read", {"path": "/root/source"}),
        ],
    )
    records = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="ssh_file_read",
            args={"path": "/root/source"},
            tool_call_id="toolplan:E1",
            result=ToolEnvelope(
                success=False,
                error="Remote path is not a regular file.",
                output={"path": "/root/source", "error": "Remote path is not a regular file."},
                metadata={"path": "/root/source"},
            ),
        )
    ]

    observations = build_tool_plan_observations(plan, records, token_limit=400, max_chars_per_step=240)
    rendered = render_tool_plan_observations(plan.objective, observations)

    assert "Remote path is not a regular file." in rendered
    assert '"path":' not in observations[0].excerpt

def test_tool_plan_observation_to_evidence_preserves_fields_and_dedupes() -> None:
    observation = build_tool_plan_observations(
        ToolPlan(
            mode="tool_plan",
            objective="inspect runtime",
            steps=[
                ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
                ToolPlanStep("E2", "file_read", {"path": "src/app.py"}),
            ],
        ),
        [
            ToolExecutionRecord(
                operation_id="op-1",
                tool_name="file_read",
                args={"path": "src/app.py"},
                tool_call_id="toolplan:E1",
                result=ToolEnvelope(success=True, output="body", metadata={"artifact_id": "A0001"}),
            )
        ],
        token_limit=200,
        max_chars_per_step=80,
    )[1]

    record = observation_to_evidence_record(
        observation,
        objective="inspect runtime",
        step_index=2,
        created_at_step=7,
    )
    assert record.evidence_id == "TP-E7-E2"
    assert record.operation_id == "op-1"
    assert record.artifact_id == "A0001"
    assert record.source == "src/app.py"
    assert record.metadata["duplicate_of"] == "E1"
    assert record.metadata["objective"] == "inspect runtime"

    state = LoopState(step_count=7)
    ids = attach_tool_plan_observation_evidence(state, objective="inspect runtime", observations=[observation])
    ids_again = attach_tool_plan_observation_evidence(state, objective="inspect runtime", observations=[observation])
    assert ids == ids_again == ["TP-E7-E2"]
    assert len(state.reasoning_graph.evidence_records) == 1
