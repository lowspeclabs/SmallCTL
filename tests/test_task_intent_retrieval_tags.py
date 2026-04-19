from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.retrieval import RetrievalBundle, LexicalRetriever, build_refined_retrieval_query
from smallctl.harness.task_classifier import classify_task_mode
from smallctl.harness.tool_dispatch import chat_mode_tools
from smallctl.harness.task_intent import extract_intent_state
from smallctl.memory_store import ExperienceStore
from smallctl.state import ArtifactRecord, EpisodicSummary, ExperienceMemory, LoopState, WriteSession


def test_classify_task_mode_covers_chat_analysis_and_execution_shapes() -> None:
    cases = {
        "hello": "chat",
        "explain this error": "analysis",
        "make a plan first": "plan_only",
        "run pytest locally": "local_execute",
        "run apt-get on remote host 192.168.1.63": "remote_execute",
        "inspect this log and tell me what failed": "debug_inspect",
    }

    for task, expected in cases.items():
        assert classify_task_mode(task) == expected


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


def test_retrieval_skips_durably_stale_experiences() -> None:
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
