from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smallctl.context.retrieval import (
    RetrievalBundle,
    LexicalRetriever,
    build_refined_retrieval_query,
    build_retrieval_query,
)
from smallctl.context.policy import ContextPolicy
from smallctl.guards import is_over_twenty_b_model_name
from smallctl.harness.task_classifier import (
    classify_runtime_intent,
    classify_task_mode,
    looks_like_complex_task,
    runtime_policy_for_intent,
)
from smallctl.harness.tool_dispatch import chat_mode_tools
from smallctl.harness.task_intent import (
    derive_task_contract,
    extract_intent_state,
    memory_fact_hint,
    preserve_promoted_active_intent,
    promote_active_intent_for_tool_call,
)
from smallctl.memory_store import ExperienceStore
from smallctl.state import ArtifactRecord, EpisodicSummary, ExperienceMemory, LoopState, WriteSession
from smallctl.task_targets import extract_task_target_paths
from smallctl.tools.control_task_complete_gates import (
    task_complete_gate_command_backed_file_creation,
    task_complete_gate_docker_compose_lifecycle_report,
    task_complete_gate_remote_service_readiness,
    task_complete_gate_sysadmin_report_consistency,
)


def test_classify_task_mode_covers_chat_analysis_and_execution_shapes() -> None:
    cases = {
        "hello": "chat",
        "explain this error": "analysis",
        "make a plan first": "plan_only",
        "run pytest locally": "local_execute",
        "use pubkey auth": "local_execute",
        "run apt-get on remote host 192.168.1.63": "remote_execute",
        "inspect this log and tell me what failed": "debug_inspect",
        "create a detailed report of your findings": "local_execute",
    }

    for task, expected in cases.items():
        assert classify_task_mode(task) == expected


def test_local_only_file_create_with_remote_prohibition_stays_local() -> None:
    task = """Create a file named ./temp/scope-tests/local-only.txt containing:

LOCAL_SCOPE_OK

Do not connect to the remote host.
After creating it, report whether the file exists locally."""

    assert classify_task_mode(task) == "local_execute"
    assert looks_like_complex_task(task) is False
    assert extract_task_target_paths(task) == ["./temp/scope-tests/local-only.txt"]


def test_local_temp_log_recovery_task_ignores_config_hosts_for_remote_mode() -> None:
    task = """You are working only inside ./temp.

Create this directory:

./temp/log_recovery

Create these three files.

./temp/log_recovery/app.log

2026-06-05T10:00:04Z ERROR failed to connect to db host=db.internal port=5432

./temp/log_recovery/config.yaml

server:
  host: 0.0.0.0
  port: 8000

database:
  host: db.internal
  port: 5432
  timeout: 2

./temp/log_recovery/inventory.txt

db-primary.internal 5432 online
db-replica.internal 5432 online

Then patch config.yaml and write ./temp/log_recovery/fix_plan.md."""

    assert classify_task_mode(task) == "local_execute"


def test_local_and_remote_system_timestamp_task_is_hybrid_execute() -> None:
    task = """Create a local timestamp file:

./temp/scope-tests/time-local.txt

using the local system's date command.

Create a remote timestamp file:

/tmp/smallctl-scope-tests/time-remote.txt

using the remote system's date command.

Each file must contain:
hostname=<hostname>
epoch=<unix timestamp>
timezone=<timezone if available>

Report both files. Remote is 192.168.1.163 password is "Temp@Pass"""

    assert classify_task_mode(task) == "hybrid_execute"


def test_local_remote_file_merge_task_is_hybrid_execute() -> None:
    task = """Create a local file:

./temp/scope-tests/mixed-direction/data.txt

with contents:

LOCAL_INPUT_DATA

Create a remote file:

/tmp/smallctl-scope-tests/mixed-direction/data.txt

with contents:

REMOTE_INPUT_DATA

Now read both files.

Create a local merged file:

./temp/scope-tests/mixed-direction/merged.txt

with contents:

local=<local data.txt contents>
remote=<remote data.txt contents>

Do not modify either original data.txt file.
Do not create merged.txt on the remote host. Remote is root@192.168.1.163 with password "Temp@Pass"""

    assert classify_task_mode(task) == "hybrid_execute"


def test_command_backed_timestamp_task_blocks_direct_file_tools() -> None:
    state = LoopState()
    state.run_brief.original_task = """Create a local timestamp file:

./temp/scope-tests/time-local.txt

using the local system's date command.

Create a remote timestamp file:

/tmp/smallctl-scope-tests/time-remote.txt

using the remote system's date command.

Each file must contain:
hostname=<hostname>
epoch=<unix timestamp>
timezone=<timezone if available>

Report both files. Remote is root@192.168.1.163 password is "Temp@Pass"""
    state.artifacts = {
        "A-local-write": ArtifactRecord(
            artifact_id="A-local-write",
            kind="file_write",
            source="./temp/scope-tests/time-local.txt",
            created_at="2026-06-05T13:06:40+00:00",
            size_bytes=53,
            summary="time-local.txt written",
            tool_name="file_write",
            metadata={"success": True, "arguments": {"path": "./temp/scope-tests/time-local.txt"}},
        ),
        "A-remote-write": ArtifactRecord(
            artifact_id="A-remote-write",
            kind="ssh_file_write",
            source="/tmp/smallctl-scope-tests/time-remote.txt",
            created_at="2026-06-05T13:06:41+00:00",
            size_bytes=44,
            summary="time-remote.txt written",
            tool_name="ssh_file_write",
            metadata={"success": True, "arguments": {"path": "/tmp/smallctl-scope-tests/time-remote.txt"}},
        ),
    }

    result = task_complete_gate_command_backed_file_creation(state)

    assert result is not None
    assert result["metadata"]["reason"] == "command_backed_file_creation_required"
    pending = result["metadata"]["pending_command_backed_file_requirements"]
    assert {item["tool_name"] for item in pending} == {"shell_exec", "ssh_exec"}


def test_command_backed_timestamp_task_accepts_shell_and_ssh_writes() -> None:
    state = LoopState()
    state.run_brief.original_task = """Create a local timestamp file:

./temp/scope-tests/time-local.txt

using the local system's date command.

Create a remote timestamp file:

/tmp/smallctl-scope-tests/time-remote.txt

using the remote system's date command.

Each file must contain:
hostname=<hostname>
epoch=<unix timestamp>
timezone=<timezone if available>

Report both files. Remote is root@192.168.1.163 password is "Temp@Pass"""
    state.artifacts = {
        "A-local-shell": ArtifactRecord(
            artifact_id="A-local-shell",
            kind="shell_exec",
            source="{ hostname; date +%s; date +%Z; } > ./temp/scope-tests/time-local.txt",
            created_at="2026-06-05T13:06:40+00:00",
            size_bytes=53,
            summary="shell_exec SUCCESS",
            tool_name="shell_exec",
            metadata={
                "success": True,
                "arguments": {"command": "{ hostname; date +%s; date +%Z; } > ./temp/scope-tests/time-local.txt"},
            },
        ),
        "A-remote-ssh": ArtifactRecord(
            artifact_id="A-remote-ssh",
            kind="ssh_exec",
            source="{ hostname; date +%s; date +%Z; } > /tmp/smallctl-scope-tests/time-remote.txt",
            created_at="2026-06-05T13:06:41+00:00",
            size_bytes=44,
            summary="ssh_exec SUCCESS",
            tool_name="ssh_exec",
            metadata={
                "success": True,
                "arguments": {"command": "{ hostname; date +%s; date +%Z; } > /tmp/smallctl-scope-tests/time-remote.txt"},
            },
        ),
    }

    assert task_complete_gate_command_backed_file_creation(state) is None


def test_listing_output_result_files_require_command_evidence() -> None:
    state = LoopState()
    state.run_brief.original_task = """Create these local files:

./temp/scope-tests/glob-test/local-a.log
./temp/scope-tests/glob-test/local-b.log

Create these remote files:

/tmp/smallctl-scope-tests/glob-test/remote-a.log
/tmp/smallctl-scope-tests/glob-test/remote-b.log

Now create:

Local:
./temp/scope-tests/glob-local-result.txt

Remote:
/tmp/smallctl-scope-tests/glob-remote-result.txt

Each result file must contain the output of listing *.log in that scope's glob-test directory."""
    state.artifacts = {
        "A-local-write": ArtifactRecord(
            artifact_id="A-local-write",
            kind="file_write",
            source="./temp/scope-tests/glob-local-result.txt",
            created_at="2026-06-05T13:06:40+00:00",
            size_bytes=24,
            summary="glob-local-result.txt written",
            tool_name="file_write",
            metadata={"success": True, "arguments": {"path": "./temp/scope-tests/glob-local-result.txt"}},
        ),
        "A-remote-write": ArtifactRecord(
            artifact_id="A-remote-write",
            kind="ssh_file_write",
            source="/tmp/smallctl-scope-tests/glob-remote-result.txt",
            created_at="2026-06-05T13:06:41+00:00",
            size_bytes=26,
            summary="glob-remote-result.txt written",
            tool_name="ssh_file_write",
            metadata={"success": True, "arguments": {"path": "/tmp/smallctl-scope-tests/glob-remote-result.txt"}},
        ),
    }

    result = task_complete_gate_command_backed_file_creation(state)

    assert result is not None
    pending = result["metadata"]["pending_command_backed_file_requirements"]
    assert {item["tool_name"] for item in pending} == {"shell_exec", "ssh_exec"}


def test_listing_output_result_files_accept_shell_and_ssh_listing_commands() -> None:
    state = LoopState()
    state.run_brief.original_task = """Local:
./temp/scope-tests/glob-local-result.txt

Remote:
/tmp/smallctl-scope-tests/glob-remote-result.txt

Each result file must contain the output of listing *.log in that scope's glob-test directory."""
    state.artifacts = {
        "A-local-shell": ArtifactRecord(
            artifact_id="A-local-shell",
            kind="shell_exec",
            source="ls ./temp/scope-tests/glob-test/*.log > ./temp/scope-tests/glob-local-result.txt",
            created_at="2026-06-05T13:06:40+00:00",
            size_bytes=24,
            summary="shell_exec SUCCESS",
            tool_name="shell_exec",
            metadata={"success": True, "arguments": {"command": "ls ./temp/scope-tests/glob-test/*.log > ./temp/scope-tests/glob-local-result.txt"}},
        ),
        "A-remote-ssh": ArtifactRecord(
            artifact_id="A-remote-ssh",
            kind="ssh_exec",
            source="ls /tmp/smallctl-scope-tests/glob-test/*.log > /tmp/smallctl-scope-tests/glob-remote-result.txt",
            created_at="2026-06-05T13:06:41+00:00",
            size_bytes=26,
            summary="ssh_exec SUCCESS",
            tool_name="ssh_exec",
            metadata={"success": True, "arguments": {"command": "ls /tmp/smallctl-scope-tests/glob-test/*.log > /tmp/smallctl-scope-tests/glob-remote-result.txt"}},
        ),
    }

    assert task_complete_gate_command_backed_file_creation(state) is None


def test_sysadmin_report_gate_blocks_inconsistent_report_completion() -> None:
    state = LoopState()
    state.task_mode = "remote_execute"
    state.run_brief.original_task = """Sysadmin Challenge: Disk, Process, Network, and Configuration RCA

Target host: root@192.168.1.89
Connect to the remote Linux host over SSH.
List all listening TCP and UDP ports and map each listening port to the owning process.
Create a report at:

/root/rca-health-investigation-report.txt
"""
    state.tool_execution_records = {
        "net": {
            "tool_name": "ssh_exec",
            "args": {"command": "ss -tuln"},
            "result": {"success": True, "output": {"exit_code": 0}},
        }
    }
    state.artifacts = {
        "report": ArtifactRecord(
            artifact_id="report",
            kind="ssh_file_write",
            source="/root/rca-health-investigation-report.txt",
            created_at="2026-06-06T13:10:25+00:00",
            size_bytes=1000,
            summary="rca-health-investigation-report.txt written",
            tool_name="ssh_file_write",
            preview_text="""Report Generated: 2026-02-23

NO SUSPICIOUS NETWORK EXPOSURE: No services listening on 0.0.0.0 outside expected ports.

Top 10 Largest Files Under /var/log:
Unable to obtain list.

Listening UDP Services:
* Port 69/tcp - TFTP - 0.0.0.0
""",
            metadata={"success": True, "arguments": {"path": "/root/rca-health-investigation-report.txt"}},
        )
    }

    result = task_complete_gate_sysadmin_report_consistency(
        state,
        "Top 3 Risks: MySQL exposed, TFTP/FTP listening, Root SSH login permitted.",
    )

    assert result is not None
    assert result["metadata"]["reason"] == "sysadmin_report_consistency_required"
    issues = result["metadata"]["sysadmin_report_issues"]
    assert any("report date" in issue for issue in issues)
    assert any("contradicts" in issue for issue in issues)
    assert any("UDP listener" in issue for issue in issues)
    assert any("owning process" in issue for issue in issues)
    assert any("/var/log" in issue for issue in issues)


def test_sysadmin_report_gate_accepts_consistent_report_with_process_mapping() -> None:
    state = LoopState()
    state.task_mode = "remote_execute"
    state.run_brief.original_task = """Sysadmin Challenge: Disk, Process, Network, and Configuration RCA

Target host: root@192.168.1.89
Connect to the remote Linux host over SSH.
List all listening TCP and UDP ports and map each listening port to the owning process.
Create a report at:

/root/rca-health-investigation-report.txt
"""
    state.tool_execution_records = {
        "net": {
            "tool_name": "ssh_exec",
            "args": {"command": "ss -tlnp && ss -ulnp"},
            "result": {"success": True, "output": {"exit_code": 0}},
        }
    }
    state.artifacts = {
        "report": ArtifactRecord(
            artifact_id="report",
            kind="ssh_file_write",
            source="/root/rca-health-investigation-report.txt",
            created_at="2026-06-06T13:10:25+00:00",
            size_bytes=1000,
            summary="rca-health-investigation-report.txt written",
            tool_name="ssh_file_write",
            preview_text="""Report Generated: 2026-06-06

Network Exposure Assessment: Suspicious services found.

Top 10 Largest Files Under /var/log:
1. /var/log/ganesha/ganesha.log

Listening UDP Services:
* Port 69/udp - TFTP - 0.0.0.0 - in.tftpd
""",
            metadata={"success": True, "arguments": {"path": "/root/rca-health-investigation-report.txt"}},
        )
    }

    assert task_complete_gate_sysadmin_report_consistency(state, "Task completed.") is None


def test_local_stdout_then_remote_report_task_is_hybrid_execute() -> None:
    task = """Run this command locally:

printf 'LOCAL_STDOUT_ONLY\n'

Do not save it to a file.

Then create a remote report:

/tmp/smallctl-scope-tests/local-stdout-observed.txt

The remote report must contain:

LOCAL_COMMAND_RAN_BUT_OUTPUT_NOT_PERSISTED_LOCALLY

Do not create a local file named local-stdout-observed.txt. Remote is root@192.168.1.163 password is "Temp@Pass"""

    assert classify_task_mode(task) == "hybrid_execute"


def test_extract_intent_state_does_not_infer_scripts_from_cwd() -> None:
    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="execute",
            cwd="/home/stephen/Scripts/Harness-Redo",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: False,
    )

    primary, secondary, tags = extract_intent_state(harness, "hello")

    assert primary == "general_task"
    assert secondary == []
    assert "scripts" not in tags
    assert "execute" not in tags
    assert "lmstudio" not in tags
    assert tags == ["phase_execute"]


def test_extract_intent_state_uses_author_write_for_plain_language_report_requests() -> None:
    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="execute",
            cwd="/home/stephen/Scripts/Harness-Redo",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: False,
    )

    primary, secondary, tags = extract_intent_state(harness, "create a detailed report of your findings")

    assert primary == "author_write"
    assert "mutate_repo" in secondary
    assert "complete_validation_task" in secondary
    assert "write_file" in tags
    assert "phase_execute" in tags


def test_looping_does_not_trigger_memory_contract_from_pin_substring() -> None:
    task = "your looping esclate to bigger model for help"

    assert memory_fact_hint(task) == ""
    assert derive_task_contract(task) == "general"


def test_runtime_intent_routes_plain_language_report_requests_to_loop() -> None:
    intent = classify_runtime_intent("create a detailed report of your findings", recent_messages=[])

    assert intent.label == "author_write"
    assert intent.task_mode == "local_execute"
    assert runtime_policy_for_intent(intent).route_mode == "loop"


def test_file_improvement_listing_is_readonly_analysis_not_write_file() -> None:
    task = "read ./temp/pony.py and list improvements you would make to that file"

    assert classify_task_mode(task) == "analysis"

    intent = classify_runtime_intent(task, recent_messages=[])
    assert intent.label in {"content_lookup", "readonly_lookup"}
    assert intent.task_mode == "analysis"

    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="execute",
            cwd="/home/stephen/Scripts/Harness-Redo",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: False,
    )
    primary, secondary, tags = extract_intent_state(harness, task)
    assert primary == "inspect_repo"
    assert "read_artifacts" in secondary
    assert "write_file" not in tags


def test_file_propose_fixes_is_readonly_analysis_not_patch_intent() -> None:
    task = "read, then run ./temp/vikunja-9b.py, then propose fixes/improvemnts to the script"

    assert classify_task_mode(task) == "local_execute"

    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="execute",
            cwd="/home/stephen/Scripts/Harness-Redo",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: False,
    )
    primary, secondary, tags = extract_intent_state(harness, task)
    assert primary == "inspect_repo"
    assert "read_artifacts" in secondary
    assert "write_file" not in tags


def test_over_twenty_b_model_name_helper_is_strictly_greater_than_twenty_b() -> None:
    assert is_over_twenty_b_model_name("gpt-oss-120b") is True
    assert is_over_twenty_b_model_name("openai/gpt-oss-20b") is False
    assert is_over_twenty_b_model_name("wrench-9b") is False


def test_refined_retrieval_query_filters_legacy_generic_memory_tags() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "phase_execute"]
    state.run_brief.original_task = "ssh into the remote host and run whoami"
    state.working_memory.current_goal = state.run_brief.original_task

    legacy_memory = ExperienceMemory(
        memory_id="mem-legacy",
        tool_name="ssh_exec",
        intent="requested_ssh_exec",
        intent_tags=["ssh_exec", "lmstudiogoogle/gemma-4-31b-it:free", "execute", "scripts"],
        outcome="success",
        notes="Successfully called ssh_exec. Key pattern: ['host', 'user', 'auth', 'command'].",
    )

    query = build_refined_retrieval_query(
        state,
        base_query=state.run_brief.original_task,
        bundle=RetrievalBundle(
            query=state.run_brief.original_task,
            summaries=[],
            artifacts=[],
            experiences=[legacy_memory],
        ),
    )

    assert "Prior outcome: requested_ssh_exec / ssh_exec / success" in query
    assert "Memory tags: ssh_exec" in query
    assert "scripts" not in query
    assert "lmstudiogoogle/gemma-4-31b-it:free" not in query


def test_refined_retrieval_query_skips_generic_task_complete_memory_for_vague_prompt() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_intent = "general_task"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "hello"
    state.working_memory.current_goal = "hello"

    memory = ExperienceMemory(
        memory_id="mem-generic",
        tool_name="task_complete",
        intent="general_task",
        intent_tags=["phase_execute"],
        outcome="success",
        notes="chat_completed",
    )

    query = build_refined_retrieval_query(
        state,
        base_query="hello",
        bundle=RetrievalBundle(
            query="hello",
            summaries=[],
            artifacts=[],
            experiences=[memory],
        ),
    )

    assert "Prior outcome:" not in query


def test_retrieval_penalizes_generic_task_complete_memories_for_vague_prompts() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_intent = "general_task"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "hello"
    state.working_memory.current_goal = "hello"

    generic_complete = ExperienceMemory(
        memory_id="mem-task-complete",
        tool_name="task_complete",
        intent="general_task",
        intent_tags=["phase_execute"],
        outcome="success",
        confidence=0.95,
        notes="chat_completed",
    )
    relevant_fact = ExperienceMemory(
        memory_id="mem-greeting",
        tool_name="memory_update",
        intent="general_task",
        intent_tags=["phase_execute", "task_greeting"],
        outcome="success",
        confidence=0.8,
        notes="hello maps to a greeting response, not script execution",
    )
    state.warm_experiences = [generic_complete, relevant_fact]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query="hello",
        include_experiences=True,
    )

    assert bundle.experiences
    assert bundle.experiences[0].memory_id == "mem-greeting"


def test_retrieval_bundle_reports_ranked_candidates_and_miss_reasons() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "inspect alpha beta"
    state.working_memory.current_goal = "inspect alpha beta"
    state.artifacts = {
        "A1": ArtifactRecord(
            artifact_id="A1",
            kind="tool_result",
            source="/tmp/alpha.txt",
            created_at="2026-01-01T00:00:00Z",
            size_bytes=100,
            summary="alpha beta primary file",
            keywords=["alpha", "beta"],
            tool_name="file_read",
            inline_content="alpha beta selected content",
        ),
        "A2": ArtifactRecord(
            artifact_id="A2",
            kind="tool_result",
            source="/tmp/beta.txt",
            created_at="2026-01-01T00:00:01Z",
            size_bytes=100,
            summary="alpha beta secondary file",
            keywords=["alpha", "beta"],
            tool_name="file_read",
            inline_content="alpha beta unselected content",
        ),
    }

    retriever = LexicalRetriever(policy=ContextPolicy(max_artifact_snippets=1))
    bundle = retriever.retrieve_bundle(
        state=state,
        query="inspect alpha beta",
        include_experiences=False,
    )

    assert [snippet.artifact_id for snippet in bundle.artifacts] == ["A2"]
    assert [candidate["artifact_id"] for candidate in bundle.ranked_candidates["artifacts"][:2]] == ["A2", "A1"]
    assert bundle.ranked_candidates["artifacts"][0]["tool_name"] == "file_read"
    assert bundle.miss_reasons["experiences"] == ["experiences_disabled"]


def test_chat_task_mode_suppresses_execution_biased_retrieval_and_chat_tools() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.task_mode = "chat"
    state.active_intent = "general_task"
    state.intent_tags = ["phase_execute"]
    state.run_brief.original_task = "hello"
    state.run_brief.current_phase_objective = "execute: hello"
    state.working_memory.current_goal = "hello"
    state.working_memory.next_actions = ["Run shell_exec(command='pytest -q') to confirm the fix."]
    shell_memory = ExperienceMemory(
        memory_id="mem-shell",
        tool_name="shell_exec",
        intent="requested_shell_exec",
        intent_tags=["shell_exec", "scripts"],
        outcome="success",
        confidence=0.95,
        notes="Successfully called shell_exec. Key pattern: ['command'].",
    )
    greeting_memory = ExperienceMemory(
        memory_id="mem-greeting",
        tool_name="memory_update",
        intent="general_task",
        intent_tags=["phase_execute", "task_greeting"],
        outcome="success",
        confidence=0.8,
        notes="hello maps to a greeting response, not script execution",
    )
    state.warm_experiences = [shell_memory, greeting_memory]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query="hello",
        include_experiences=True,
    )

    assert "shell_exec" not in bundle.query
    assert bundle.experiences
    assert [memory.memory_id for memory in bundle.experiences] == ["mem-greeting"]

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "hello",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                {"type": "function", "function": {"name": "file_read", "description": "", "parameters": {}}},
                {"type": "function", "function": {"name": "ssh_exec", "description": "", "parameters": {}}},
            ],
            get=lambda name: None,
        ),
    )

    assert chat_mode_tools(harness) == []


def _artifact(
    artifact_id: str,
    *,
    kind: str,
    path: str,
    summary: str,
    metadata: dict[str, object] | None = None,
) -> ArtifactRecord:
    payload = {"path": path, **(metadata or {})}
    return ArtifactRecord(
        artifact_id=artifact_id,
        kind=kind,
        source=path,
        created_at="2026-04-30T00:00:00+00:00",
        size_bytes=128,
        summary=summary,
        keywords=[Path(path).name, path],
        path_tags=[Path(path).name],
        tool_name=kind,
        inline_content=f"{path}\n{summary}",
        preview_text=f"{path}\n{summary}",
        metadata=payload,
    )


def test_remote_multifile_retrieval_keeps_files_mutation_and_verifier() -> None:
    state = LoopState(cwd="/tmp")
    state.task_mode = "remote_execute"
    state.run_brief.original_task = (
        "ssh into root@192.168.1.63 and update /var/www/html/index.html "
        "and /var/www/html/style.css, then verify"
    )
    state.working_memory.next_actions = ["ssh_file_patch(path='/var/www/html/style.css')"]
    state.artifacts = {
        "A0001": _artifact(
            "A0001",
            kind="file_read",
            path="/var/www/html/index.html",
            summary="index.html full file",
            metadata={"complete_file": True, "total_lines": 20},
        ),
        "A0002": _artifact(
            "A0002",
            kind="file_read",
            path="/var/www/html/style.css",
            summary="style.css full file",
            metadata={"complete_file": True, "total_lines": 12},
        ),
        "A0003": _artifact(
            "A0003",
            kind="ssh_file_write",
            path="/var/www/html/style.css",
            summary="style.css written",
            metadata={
                "host": "192.168.1.63",
                "changed": True,
                "bytes_written": 120,
                "new_sha256": "new",
                "readback_sha256": "new",
            },
        ),
        "A0004": _artifact(
            "A0004",
            kind="ssh_exec",
            path="/var/www/html/style.css",
            summary="remote verifier pass",
            metadata={
                "verifier_verdict": "pass",
                "verifier_target": "grep -q theme /var/www/html/style.css",
                "verifier_exit_code": 0,
            },
        ),
    }

    snippets = LexicalRetriever(ContextPolicy(max_artifact_snippets=4)).retrieve_artifacts(
        state=state,
        query=state.run_brief.original_task,
        token_budget=4000,
    )

    selected = {snippet.artifact_id for snippet in snippets}
    assert {"A0001", "A0002", "A0003", "A0004"} <= selected


def test_remote_failure_artifact_is_downranked_after_same_target_success() -> None:
    state = LoopState(cwd="/tmp")
    state.task_mode = "remote_execute"
    state.current_phase = "execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "phase_execute"]
    state.run_brief.original_task = 'is the "demo-site" enabled in the nginx config on the remote server?'
    state.working_memory.current_goal = state.run_brief.original_task
    state.working_memory.failures = ["ssh_file_read: ssh: connect to host 192.168.1.89 port 22: Connection timed out"]
    state.artifacts = {
        "A0002": _artifact(
            "A0002",
            kind="ssh_file_read",
            path="/etc/nginx/sites-enabled/default",
            summary="ssh: connect to host 192.168.1.89 port 22: Connection timed out",
            metadata={
                "host": "192.168.1.89",
                "path": "/etc/nginx/sites-enabled/default",
                "success": False,
                "arguments": {
                    "host": "192.168.1.89",
                    "path": "/etc/nginx/sites-enabled/default",
                },
            },
        ),
        "A0003": _artifact(
            "A0003",
            kind="ssh_file_read",
            path="/etc/nginx/sites-enabled/default",
            summary="default nginx site config",
            metadata={
                "host": "192.168.1.89",
                "path": "/etc/nginx/sites-enabled/default",
                "success": True,
                "complete_file": True,
                "arguments": {
                    "host": "192.168.1.89",
                    "path": "/etc/nginx/sites-enabled/default",
                },
            },
        ),
    }

    query = build_retrieval_query(state)
    query_tokens = set(query.lower().split())
    failure_score = LexicalRetriever._score_artifact(
        artifact=state.artifacts["A0002"],
        query=query,
        query_tokens=query_tokens,
        recency=2,
        state=state,
    )
    success_score = LexicalRetriever._score_artifact(
        artifact=state.artifacts["A0003"],
        query=query,
        query_tokens=query_tokens,
        recency=1,
        state=state,
    )

    assert success_score > failure_score


def test_remote_repair_retrieval_prefers_interactive_ssh_failure_over_meta_failure() -> None:
    state = LoopState(cwd="/tmp")
    state.task_mode = "remote_execute"
    state.current_phase = "repair"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "phase_repair"]
    state.last_failure_class = "verifier_failed"
    state.run_brief.original_task = "install FOG over SSH on root@192.168.1.89"
    state.working_memory.current_goal = "Continue remote task over SSH and restart the installation"
    state.artifacts = {
        "A0019": _artifact(
            "A0019",
            kind="artifact_grep",
            path="artifact_grep",
            summary="Query 'y/N|yes|n/no' looks like a regex, but artifact_grep uses literal substring matching",
            metadata={
                "success": False,
                "error": "Query 'y/N|yes|n/no' looks like a regex",
                "intent": "requested_ssh_exec",
                "phase": "repair",
            },
        ),
        "A0025": _artifact(
            "A0025",
            kind="ssh_exec",
            path="/root/fogproject/bin/installfog.sh",
            summary="Remote SSH command exited with code 1",
            metadata={
                "success": False,
                "host": "192.168.1.89",
                "command": "cd /root/fogproject/bin && echo \"n\" | ./installfog.sh",
                "failure_kind": "remote_command",
                "ssh_transport_succeeded": True,
                "output_received": True,
                "intent": "requested_ssh_exec",
                "phase": "repair",
                "output": {
                    "stdout": "Should the installer try to disable the local firewall for you now? (y/N)\n"
                    "Are you sure you wish to continue (Y/N)\n"
                    "Sorry, answer not recognized",
                    "stderr": "",
                    "exit_code": 1,
                },
            },
        ),
    }

    snippets = LexicalRetriever(ContextPolicy(max_artifact_snippets=4)).retrieve_artifacts(
        state=state,
        query=build_retrieval_query(state),
        token_budget=4000,
    )

    assert snippets
    assert snippets[0].artifact_id == "A0025"
    assert "A0019" not in {snippet.artifact_id for snippet in snippets[:1]}


def test_single_file_non_detail_retrieval_keeps_one_primary_file() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Update /workspace/app.py"
    state.artifacts = {
        "A0001": _artifact(
            "A0001",
            kind="file_read",
            path="/workspace/app.py",
            summary="app.py full file",
            metadata={"complete_file": True, "total_lines": 20},
        ),
        "A0002": _artifact(
            "A0002",
            kind="file_read",
            path="/workspace/other.py",
            summary="other.py full file",
            metadata={"complete_file": True, "total_lines": 12},
        ),
    }

    snippets = LexicalRetriever(ContextPolicy(max_artifact_snippets=4)).retrieve_artifacts(
        state=state,
        query="Update /workspace/app.py",
        token_budget=4000,
    )

    primary_ids = [
        snippet.artifact_id
        for snippet in snippets
        if state.artifacts[snippet.artifact_id].kind == "file_read"
    ]
    assert primary_ids == ["A0001"]


def test_experience_store_rewrites_legacy_generic_tags(tmp_path) -> None:
    store = ExperienceStore(tmp_path / "warm-experiences.jsonl")

    memory = ExperienceMemory(
        memory_id="mem-legacy-tags",
        tool_name="task_complete",
        intent="general_task",
        intent_tags=["lmstudio", "execute", "scripts", "phase_execute", "python"],
        environment_tags=["lmstudio", "repair", "phase_repair"],
        entity_tags=["python", "openrouter"],
        outcome="success",
        notes="chat_completed",
    )

    store.upsert(memory)
    rewritten = store.list()

    assert len(rewritten) == 1
    assert rewritten[0].intent_tags == ["phase_execute", "python"]
    assert rewritten[0].environment_tags == ["phase_repair"]
    assert rewritten[0].entity_tags == ["python"]


def test_retrieval_bundle_exposes_lane_routes_for_frame_packets() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_intent = "requested_file_read"
    state.intent_tags = ["file_read", "phase_execute"]
    state.run_brief.original_task = "inspect README.md"
    state.working_memory.current_goal = state.run_brief.original_task
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="S-lane",
            created_at="2026-04-18T00:00:00+00:00",
            decisions=["Read README first"],
            files_touched=["README.md"],
        )
    ]
    state.artifacts["A-lane"] = ArtifactRecord(
        artifact_id="A-lane",
        kind="file_read",
        source="README.md",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=20,
        summary="README contents",
        tool_name="file_read",
    )
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-lane",
            intent="requested_file_read",
            tool_name="file_read",
            outcome="success",
            confidence=0.8,
            notes="Read README.md and extract key facts.",
        )
    ]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=True,
    )

    assert bundle.lane_routes["artifact_packet"] == ["A-lane"]
    assert bundle.lane_routes["experience_packet"] == ["mem-lane"]
    assert bundle.lane_routes["evidence_packet"] == ["S-lane"]


def test_retrieval_prefers_artifact_matching_active_write_target_path() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "author"
    state.active_intent = "requested_file_patch"
    state.intent_tags = ["file_patch", "phase_author"]
    state.run_brief.original_task = "patch app module logic"
    state.working_memory.current_goal = state.run_brief.original_task
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="src/app.py",
        write_session_intent="patch_existing",
        write_session_mode="chunked_author",
        status="open",
    )

    state.artifacts["A-app"] = ArtifactRecord(
        artifact_id="A-app",
        kind="file_read",
        source="src/app.py",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=120,
        summary="module patch candidate",
        tool_name="file_read",
    )
    state.artifacts["A-other"] = ArtifactRecord(
        artifact_id="A-other",
        kind="file_read",
        source="src/other.py",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=120,
        summary="module patch candidate",
        tool_name="file_read",
    )

    artifacts = LexicalRetriever().retrieve_artifacts(
        state=state,
        query="patch src/app.py module logic",
    )

    assert artifacts
    assert artifacts[0].artifact_id == "A-app"


def test_large_models_skip_weak_artifact_packets_without_explicit_detail_request() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_model_name"] = "gpt-oss-120b"
    state.current_phase = "author"
    state.active_intent = "requested_file_patch"
    state.intent_tags = ["file_patch", "phase_author"]
    state.run_brief.original_task = "patch module logic"
    state.working_memory.current_goal = state.run_brief.original_task
    state.artifacts["A-weak"] = ArtifactRecord(
        artifact_id="A-weak",
        kind="file_read",
        source="src/app.py",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=120,
        summary="module patch candidate",
        tool_name="file_read",
    )

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=False,
    )

    assert bundle.artifacts == []
    assert bundle.lane_routes["artifact_packet"] == []


def test_large_models_allow_artifact_packets_for_explicit_prior_evidence_queries() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_model_name"] = "gpt-oss-120b"
    state.current_phase = "author"
    state.active_intent = "requested_file_patch"
    state.intent_tags = ["file_patch", "phase_author"]
    state.run_brief.original_task = "show prior evidence for patch module logic"
    state.working_memory.current_goal = state.run_brief.original_task
    state.artifacts["A-detail"] = ArtifactRecord(
        artifact_id="A-detail",
        kind="file_read",
        source="src/app.py",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=120,
        summary="module patch candidate",
        tool_name="file_read",
    )

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=False,
    )

    assert [artifact.artifact_id for artifact in bundle.artifacts] == ["A-detail"]
    assert bundle.lane_routes["artifact_packet"] == ["A-detail"]


def test_large_models_keep_verifier_artifacts_even_when_signal_is_below_large_model_threshold() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_model_name"] = "gpt-oss-120b"
    state.current_phase = "repair"
    state.active_intent = "requested_shell_exec"
    state.intent_tags = ["shell_exec", "phase_repair"]
    state.run_brief.original_task = "repair failing verifier"
    state.working_memory.current_goal = state.run_brief.original_task
    state.artifacts["A-verifier"] = ArtifactRecord(
        artifact_id="A-verifier",
        kind="verification",
        source="pytest -q",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=120,
        summary="failing verifier transcript",
        tool_name="shell_exec",
        metadata={"verifier_verdict": "fail", "verifier_target": "src/app.py"},
    )

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=False,
    )

    assert [artifact.artifact_id for artifact in bundle.artifacts] == ["A-verifier"]
    assert bundle.lane_routes["artifact_packet"] == ["A-verifier"]


def test_retrieval_uses_body_excerpt_when_preview_text_is_missing(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "explore"
    state.run_brief.original_task = "review the fetched article"
    state.working_memory.current_goal = state.run_brief.original_task
    content_path = tmp_path / "A-web.txt"
    content_path.write_text(
        "TurboQuant body details appear here with the important benchmark summary.\nSecond line.\n",
        encoding="utf-8",
    )
    state.artifacts["A-web"] = ArtifactRecord(
        artifact_id="A-web",
        kind="web_fetch",
        source="https://example.com/article",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=content_path.stat().st_size,
        summary="Example fetched article",
        tool_name="web_fetch",
        content_path=str(content_path),
        metadata={"render_mode": "body_with_preview"},
    )

    artifacts = LexicalRetriever().retrieve_artifacts(
        state=state,
        query="example fetched article",
    )

    assert artifacts
    assert artifacts[0].artifact_id == "A-web"
    assert "TurboQuant body details" in artifacts[0].text


def test_retrieval_pins_recent_research_artifact_for_followup_turns() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_intent = "requested_ssh_exec"
    state.run_brief.original_task = "ssh into the remote host and fix nginx"
    state.working_memory.current_goal = "Continue remote task over SSH on root@192.168.1.63. User follow-up: debug nginx and do a websearch first"
    state.scratchpad["_task_boundary_previous_task"] = "ssh into the remote host and fix nginx"
    state.scratchpad["_last_task_handoff"] = {
        "effective_task": state.working_memory.current_goal,
        "current_goal": state.working_memory.current_goal,
        "recent_research_artifact_ids": ["A-web"],
    }
    state.artifacts["A-web"] = ArtifactRecord(
        artifact_id="A-web",
        kind="web_fetch",
        source="https://example.com/nginx-fix",
        created_at="2026-05-01T00:00:00+00:00",
        size_bytes=128,
        summary="Fetched article covering the nginx upstream fix",
        tool_name="web_fetch",
        inline_content="The nginx upstream fix is documented here.",
        metadata={"intent": "requested_ssh_exec", "phase": "execute"},
    )
    state.artifacts["A-ssh"] = ArtifactRecord(
        artifact_id="A-ssh",
        kind="ssh_exec",
        source="ssh://root@192.168.1.63/var/log/nginx/error.log",
        created_at="2026-05-01T00:00:00+00:00",
        size_bytes=128,
        summary="SSH log excerpt about the nginx upstream failure",
        tool_name="ssh_exec",
        inline_content="Nginx upstream failure observed in the remote log.",
        metadata={"intent": "requested_ssh_exec", "phase": "execute"},
    )

    artifacts = LexicalRetriever().retrieve_artifacts(
        state=state,
        query="continue fixing the nginx upstream issue",
    )

    assert artifacts
    assert artifacts[0].artifact_id == "A-web"


def test_retrieval_prefers_experience_matching_last_failure_mode() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.active_intent = "requested_shell_exec"
    state.intent_tags = ["shell_exec", "phase_repair"]
    state.last_failure_class = "syntax"
    state.run_brief.original_task = "fix failing verifier"
    state.working_memory.current_goal = state.run_brief.original_task
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-syntax",
            tool_name="shell_exec",
            intent="requested_shell_exec",
            outcome="failure",
            failure_mode="syntax",
            confidence=0.72,
            notes="Previous syntax failure during verifier run.",
        ),
        ExperienceMemory(
            memory_id="mem-import",
            tool_name="shell_exec",
            intent="requested_shell_exec",
            outcome="failure",
            failure_mode="import",
            confidence=0.72,
            notes="Previous import failure during verifier run.",
        ),
    ]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=True,
    )

    assert bundle.experiences
    assert bundle.experiences[0].memory_id == "mem-syntax"


def test_retrieval_prefers_summary_matching_write_target_and_failure_mode() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.active_intent = "requested_file_patch"
    state.secondary_intents = ["requested_shell_exec"]
    state.intent_tags = ["file_patch", "phase_repair"]
    state.last_failure_class = "syntax"
    state.run_brief.original_task = "repair failing app module patch"
    state.working_memory.current_goal = state.run_brief.original_task
    state.write_session = WriteSession(
        write_session_id="ws-summary",
        write_target_path="repo/src/app.py",
        write_session_intent="patch_existing",
        write_session_mode="chunked_author",
        status="open",
    )
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="S-app",
            created_at="2026-04-18T00:00:00+00:00",
            decisions=["Continue repair loop for requested_file_patch"],
            files_touched=["src/app.py"],
            failed_approaches=["syntax mismatch from previous patch"],
            remaining_plan=["apply focused patch and rerun verifier"],
            notes=["module patch hot path for app.py"],
        ),
        EpisodicSummary(
            summary_id="S-other",
            created_at="2026-04-18T00:00:00+00:00",
            decisions=["Continue repair loop for requested_file_patch"],
            files_touched=["src/other.py"],
            failed_approaches=["network timeout during unrelated operation"],
            remaining_plan=["patch another module"],
            notes=["secondary file was changed while exploring"],
        ),
    ]

    summaries = LexicalRetriever().retrieve_summaries(
        state=state,
        query=state.run_brief.original_task,
    )

    assert summaries
    assert summaries[0].summary_id == "S-app"


def test_retrieval_decay_repeated_experiences_after_several_turns() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "phase_repair"]
    state.last_failure_class = "verifier_failed"
    state.run_brief.original_task = "install FOG on Debian"
    state.working_memory.current_goal = state.run_brief.original_task
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-ssh-retry",
            tool_name="ssh_exec",
            intent="requested_ssh_exec",
            outcome="failure",
            failure_mode="verifier_failed",
            confidence=0.8,
            notes="Retried ssh_exec verifier failed.",
        ),
        ExperienceMemory(
            memory_id="mem-interactive",
            tool_name="ssh_session_start",
            intent="requested_ssh_exec",
            outcome="success",
            failure_mode="",
            confidence=0.75,
            notes="Interactive SSH session succeeded for installer.",
        ),
    ]
    state.scratchpad["_retrieved_experience_history"] = [
        {"memory_id": "mem-ssh-retry", "retrieved_at_step": i} for i in range(5)
    ]

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=True,
    )

    assert bundle.experiences
    assert bundle.experiences[0].memory_id == "mem-interactive"
    assert "mem-ssh-retry" != bundle.experiences[0].memory_id


def test_retrieval_history_is_recorded_after_selection() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_intent = "requested_ssh_exec"
    state.run_brief.original_task = "deploy app on remote host"
    state.working_memory.current_goal = state.run_brief.original_task
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-deploy",
            tool_name="ssh_exec",
            intent="requested_ssh_exec",
            outcome="success",
            confidence=0.8,
            notes="Deploy succeeded.",
        ),
    ]

    LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=True,
    )

    history = state.scratchpad.get("_retrieved_experience_history")
    assert isinstance(history, list)
    assert any(entry.get("memory_id") == "mem-deploy" for entry in history)
    assert state.retrieved_experience_ids == ["mem-deploy"]

    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.active_intent = "requested_file_patch"
    state.intent_tags = ["file_patch", "phase_repair"]
    state.run_brief.original_task = "patch src/app.py"
    state.working_memory.current_goal = state.run_brief.original_task
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-stale",
            tool_name="file_patch",
            intent="requested_file_patch",
            outcome="success",
            confidence=0.8,
            notes="Successfully patched src/app.py",
        ),
        ExperienceMemory(
            memory_id="mem-fresh",
            tool_name="file_patch",
            intent="requested_file_patch",
            outcome="success",
            confidence=0.8,
            notes="Successfully patched docs/readme.md",
        ),
    ]
    state.scratchpad["_experience_staleness"] = {
        "mem-stale": {
            "stale": True,
            "reason": "file_changed",
            "reasons": ["file_changed"],
            "paths": ["src/app.py"],
            "updated_at": "2026-04-19T00:00:00+00:00",
            "phase": "repair",
        }
    }

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=state.run_brief.original_task,
        include_experiences=True,
    )

    assert bundle.experiences
    assert [memory.memory_id for memory in bundle.experiences] == ["mem-fresh"]


def test_retrieval_skips_durably_stale_summaries_and_artifacts() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "author"
    state.active_intent = "requested_file_patch"
    state.intent_tags = ["file_patch", "phase_author"]
    state.run_brief.original_task = "patch src/app.py"
    state.working_memory.current_goal = state.run_brief.original_task
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="S-stale",
            created_at="2026-04-18T00:00:00+00:00",
            decisions=["Patch src/app.py"],
            files_touched=["src/app.py"],
            notes=["stale summary"],
        ),
        EpisodicSummary(
            summary_id="S-fresh",
            created_at="2026-04-18T00:00:00+00:00",
            decisions=["Patch docs/readme.md"],
            files_touched=["docs/readme.md"],
            notes=["fresh summary"],
        ),
    ]
    state.artifacts["A-stale"] = ArtifactRecord(
        artifact_id="A-stale",
        kind="file_read",
        source="src/app.py",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=120,
        summary="stale artifact",
        tool_name="file_read",
    )
    state.artifacts["A-fresh"] = ArtifactRecord(
        artifact_id="A-fresh",
        kind="file_read",
        source="docs/readme.md",
        created_at="2026-04-18T00:00:00+00:00",
        size_bytes=120,
        summary="fresh artifact",
        tool_name="file_read",
    )
    state.scratchpad["_summary_staleness"] = {"S-stale": {"stale": True}}
    state.scratchpad["_artifact_staleness"] = {"A-stale": {"stale": True}}

    summaries = LexicalRetriever().retrieve_summaries(
        state=state,
        query="patch",
    )
    artifacts = LexicalRetriever().retrieve_artifacts(
        state=state,
        query="patch docs/readme.md",
    )

    assert [summary.summary_id for summary in summaries] == ["S-fresh"]
    assert artifacts
    assert [artifact.artifact_id for artifact in artifacts] == ["A-fresh"]


def test_build_retrieval_query_includes_touched_symbols() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.task_mode = "local_execute"
    state.active_intent = "requested_file_patch"
    state.intent_tags = ["file_patch", "phase_repair"]
    state.run_brief.original_task = "continue repair loop"
    state.working_memory.current_goal = "continue repair loop"
    state.scratchpad["_touched_symbols"] = ["parse_config", "ParserState"]

    query = LexicalRetriever().retrieve_bundle(
        state=state,
        query="",
        include_experiences=False,
    ).query

    assert "Touched symbols: parse_config ParserState" in query


def test_retrieval_prefers_artifact_matching_touched_symbols_for_implicit_query() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.task_mode = "local_execute"
    state.active_intent = "requested_file_patch"
    state.intent_tags = ["file_patch", "phase_repair"]
    state.run_brief.original_task = "continue repair loop"
    state.working_memory.current_goal = "continue repair loop"
    state.scratchpad["_touched_symbols"] = ["parse_config"]
    state.artifacts["A-parse"] = ArtifactRecord(
        artifact_id="A-parse",
        kind="file_read",
        source="src/helpers.py",
        created_at="2026-04-19T00:00:00+00:00",
        size_bytes=80,
        summary="function parse_config updated for repair",
        tool_name="file_read",
    )
    state.artifacts["A-render"] = ArtifactRecord(
        artifact_id="A-render",
        kind="file_read",
        source="src/ui.py",
        created_at="2026-04-19T00:00:00+00:00",
        size_bytes=80,
        summary="function render_widget updated for repair",
        tool_name="file_read",
    )

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query="",
        include_experiences=False,
    )

    assert bundle.artifacts
    assert bundle.artifacts[0].artifact_id == "A-parse"


def test_traceback_does_not_trigger_ssh_exec_intent() -> None:
    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="execute",
            cwd="/home/stephen/Scripts/Harness-Redo",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: False,
    )
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "/home/stephen/Scripts/Harness-Redo/src/smallctl/tools/dispatcher.py", line 392, in normalize_tool_request\n'
        "    ...\n"
        '  File "/home/stephen/Scripts/Harness-Redo/pong.py", line 1, in <module>\n'
        "    import pygame\n"
        "ModuleNotFoundError: No module named 'pygame'\n"
        "stephen@Ubuntu-Devbox:~/Scripts/Harness-Redo$ "
    )
    primary, secondary, tags = extract_intent_state(harness, traceback_text)
    assert primary != "requested_ssh_exec"
    assert "ssh_exec" not in tags


def test_explicit_local_shell_override() -> None:
    task = "use shell exec no ssh exec"
    assert classify_task_mode(task) == "local_execute"

    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="execute",
            cwd="/home/stephen/Scripts/Harness-Redo",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: True,
    )
    primary, secondary, tags = extract_intent_state(harness, task)
    assert primary == "requested_shell_exec"
    assert "shell_exec" in tags


def test_accepted_ssh_exec_promotes_readonly_lookup_intent() -> None:
    state = LoopState()
    state.current_phase = "execute"
    state.active_intent = "readonly_lookup"
    state.secondary_intents = ["answer_only"]
    state.intent_tags = ["research"]

    promoted = promote_active_intent_for_tool_call(state, "ssh_exec")

    assert promoted is True
    assert state.active_intent == "requested_ssh_exec"
    assert "ssh_exec" in state.intent_tags
    assert "execute" in state.intent_tags
    assert "phase_execute" in state.intent_tags
    assert state.scratchpad["_active_intent_promoted_by_tool"]["tool_name"] == "ssh_exec"


def test_readonly_tool_does_not_promote_active_intent() -> None:
    state = LoopState()
    state.current_phase = "execute"
    state.active_intent = "readonly_lookup"
    state.intent_tags = ["research"]

    promoted = promote_active_intent_for_tool_call(state, "file_read")

    assert promoted is False
    assert state.active_intent == "readonly_lookup"
    assert state.intent_tags == ["research"]
    assert "_active_intent_promoted_by_tool" not in state.scratchpad


def test_refresh_preserves_observed_tool_intent_over_derived_readonly() -> None:
    state = LoopState()
    state.current_phase = "execute"
    promote_active_intent_for_tool_call(state, "ssh_exec")

    primary, secondary, tags = preserve_promoted_active_intent(
        state,
        "readonly_lookup",
        ["answer_only"],
        ["research"],
    )

    assert primary == "requested_ssh_exec"
    assert secondary == ["complete_validation_task"]
    assert "ssh_exec" in tags


def test_refresh_keeps_stronger_derived_intent_over_promoted_tool_intent() -> None:
    state = LoopState()
    promote_active_intent_for_tool_call(state, "ssh_exec")

    primary, secondary, tags = preserve_promoted_active_intent(
        state,
        "author_write",
        ["mutate_repo"],
        ["write_file"],
    )

    assert primary == "author_write"
    assert secondary == ["mutate_repo"]
    assert tags == ["write_file"]


def _docker_lifecycle_state() -> LoopState:
    state = LoopState()
    state.task_mode = "remote_execute"
    state.run_brief.original_task = (
        "Create and verify a Docker Compose service named qwen-whoami on the remote host. "
        "Then tear down the stack with docker compose down, recreate it, verify it again, "
        "and write a report to /tmp/qwen-compose-medium-report.txt."
    )
    return state


def _ssh_record(command: str, *, success: bool = True, stdout: str = "ok") -> dict[str, object]:
    return {
        "tool_name": "ssh_exec",
        "args": {"command": command},
        "result": {"success": success, "output": {"exit_code": 0 if success else 1, "stdout": stdout, "stderr": ""}},
    }


def test_docker_compose_lifecycle_report_gate_blocks_unsupported_offline_claim() -> None:
    state = _docker_lifecycle_state()
    state.tool_execution_records = {
        "up1": _ssh_record("cd /opt/qwen-compose-medium && docker compose up -d"),
        "verify1": _ssh_record("curl -fsS http://127.0.0.1:8091"),
        "down": _ssh_record("cd /opt/qwen-compose-medium && docker compose down"),
        "up2": _ssh_record("cd /opt/qwen-compose-medium && docker compose up -d"),
        "verify2": _ssh_record("curl -fsS http://127.0.0.1:8091"),
    }
    state.artifacts = {
        "report": ArtifactRecord(
            artifact_id="report",
            kind="ssh_file_write",
            source="/tmp/qwen-compose-medium-report.txt",
            created_at="2026-06-17T18:20:00+00:00",
            size_bytes=200,
            summary="qwen-compose-medium-report.txt written",
            tool_name="ssh_file_write",
            preview_text="The service was stopped and removed, then recreated and verified.",
            metadata={"success": True, "arguments": {"path": "/tmp/qwen-compose-medium-report.txt"}},
        )
    }

    result = task_complete_gate_docker_compose_lifecycle_report(state, "Task completed.")

    assert result is not None
    assert result["metadata"]["reason"] == "docker_compose_lifecycle_report_grounding_required"
    issues = result["metadata"]["docker_compose_lifecycle_issues"]
    assert any("post-down" in issue for issue in issues)
    assert not any("recreated service" in issue for issue in issues)


def test_docker_compose_lifecycle_report_gate_accepts_direct_teardown_and_recreate_evidence() -> None:
    state = _docker_lifecycle_state()
    state.tool_execution_records = {
        "up1": _ssh_record("cd /opt/qwen-compose-medium && docker compose up -d"),
        "verify1": _ssh_record("curl -fsS http://127.0.0.1:8091"),
        "down": _ssh_record("cd /opt/qwen-compose-medium && docker compose down"),
        "verify_down": _ssh_record("cd /opt/qwen-compose-medium && docker compose ps"),
        "up2": _ssh_record("cd /opt/qwen-compose-medium && docker compose up -d"),
        "verify2": _ssh_record("curl -fsS http://127.0.0.1:8091"),
    }
    state.artifacts = {
        "report": ArtifactRecord(
            artifact_id="report",
            kind="ssh_file_write",
            source="/tmp/qwen-compose-medium-report.txt",
            created_at="2026-06-17T18:20:00+00:00",
            size_bytes=200,
            summary="qwen-compose-medium-report.txt written",
            tool_name="ssh_file_write",
            preview_text="The service was stopped and removed, then recreated and verified.",
            metadata={"success": True, "arguments": {"path": "/tmp/qwen-compose-medium-report.txt"}},
        )
    }

    assert task_complete_gate_docker_compose_lifecycle_report(state, "Task completed.") is None


def test_docker_compose_lifecycle_report_gate_ignores_non_lifecycle_reports() -> None:
    state = LoopState()
    state.task_mode = "remote_execute"
    state.run_brief.original_task = "Inspect Docker Compose status on a remote host and write a report."

    assert task_complete_gate_docker_compose_lifecycle_report(state, "Report complete.") is None


def _remote_service_state() -> LoopState:
    state = LoopState()
    state.task_mode = "remote_execute"
    state.run_brief.original_task = "SSH into root@192.168.1.161 and install NetBox as a Docker container."
    state.working_memory.current_goal = state.run_brief.original_task
    return state


def _ssh_step(step: int, command: str, *, stdout: str = "ok", stderr: str = "") -> dict[str, object]:
    return {
        "operation_id": f"op-{step}",
        "step_count": step,
        "tool_name": "ssh_exec",
        "args": {"host": "192.168.1.161", "user": "root", "command": command},
        "result": {
            "success": True,
            "output": {"exit_code": 0, "stdout": stdout, "stderr": stderr},
        },
    }


def test_remote_service_readiness_gate_rejects_docker_ps_only_after_detached_run() -> None:
    state = _remote_service_state()
    state.tool_execution_records = {
        "run": _ssh_step(1, "docker run -d --name netbox -p 8000:8080 netboxcommunity/netbox:latest"),
        "ps": _ssh_step(2, "docker ps --filter name=netbox"),
    }

    result = task_complete_gate_remote_service_readiness(state)

    assert result is not None
    assert result["metadata"]["reason"] == "remote_service_readiness_required"
    assert any("docker ps" in issue for issue in result["metadata"]["remote_service_readiness_issues"])


def test_remote_service_readiness_gate_rejects_unhealthy_post_start_logs() -> None:
    state = _remote_service_state()
    state.tool_execution_records = {
        "run": _ssh_step(1, "docker run -d --name netbox -p 8000:8080 netboxcommunity/netbox:latest"),
        "logs": _ssh_step(2, "docker logs netbox", stdout="Waiting on DB...\nWaited 30s or more for the DB to become ready"),
    }

    result = task_complete_gate_remote_service_readiness(state)

    assert result is not None
    assert "post-start logs" in result["metadata"]["remote_service_readiness_issues"][0]


def test_remote_service_readiness_gate_accepts_http_probe_after_detached_run() -> None:
    state = _remote_service_state()
    state.tool_execution_records = {
        "run": _ssh_step(1, "docker run -d --name netbox -p 8000:8080 netboxcommunity/netbox:latest"),
        "curl": _ssh_step(2, "curl -fsS http://127.0.0.1:8000/login/", stdout="<html>NetBox</html>"),
    }

    assert task_complete_gate_remote_service_readiness(state) is None
