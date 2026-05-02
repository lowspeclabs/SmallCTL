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
    runtime_policy_for_intent,
)
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
        "use pubkey auth": "local_execute",
        "run apt-get on remote host 192.168.1.63": "remote_execute",
        "inspect this log and tell me what failed": "debug_inspect",
        "create a detailed report of your findings": "local_execute",
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


def test_runtime_intent_routes_plain_language_report_requests_to_loop() -> None:
    intent = classify_runtime_intent("create a detailed report of your findings", recent_messages=[])

    assert intent.label == "author_write"
    assert intent.task_mode == "local_execute"
    assert runtime_policy_for_intent(intent).route_mode == "loop"


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
