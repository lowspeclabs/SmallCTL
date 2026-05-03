from __future__ import annotations

from types import SimpleNamespace

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.harness.conversation_logging import record_assistant_message
from smallctl.harness.task_boundary import _collapse_task_chain
from smallctl.context.retrieval import build_retrieval_query
from smallctl.harness import Harness
from smallctl.models.conversation import ConversationMessage
from smallctl.state import ArtifactRecord, EpisodicSummary, ExecutionPlan, LoopState, MemoryEntry
from smallctl.task_targets import primary_task_target_path


def _make_harness(state: LoopState) -> SimpleNamespace:
    harness = SimpleNamespace(
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="explore",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
    )
    harness._refresh_task_handoff_action_options = lambda text: Harness._refresh_task_handoff_action_options(
        harness, text
    )
    return harness


def test_new_task_replaces_old_goal_in_run_brief_and_retrieval_query() -> None:
    state = LoopState(cwd="/tmp")
    old_task = "run nmap against 192.168.1.0/24 and list all hosts found in scan"
    new_task = (
        "Build a Python script at `./temp/dead_letter_queue.py` that simulates a tiny "
        "message processor with retry, backoff, and dead-letter queue behavior."
    )
    state.run_brief.original_task = old_task
    state.working_memory.current_goal = old_task

    harness = _make_harness(state)

    Harness._maybe_reset_for_new_task(harness, new_task)
    Harness._initialize_run_brief(harness, new_task, raw_task=new_task)

    assert state.run_brief.original_task == new_task
    assert state.working_memory.current_goal == new_task

    query = build_retrieval_query(state)
    assert new_task in query
    assert old_task not in query


def test_continue_collapses_legacy_followup_chain_to_current_task() -> None:
    state = LoopState(cwd="/tmp")
    old_task = "run nmap against 192.168.1.0/24 and list all hosts found in scan"
    new_task = (
        "Build a Python script at `./temp/dead_letter_queue.py` that simulates a tiny "
        "message processor with retry, backoff, and dead-letter queue behavior."
    )
    contaminated = f"{old_task}\nFollow-up: {new_task}"
    state.run_brief.original_task = contaminated
    state.working_memory.current_goal = contaminated
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": new_task,
        "effective_task": contaminated,
        "current_goal": contaminated,
    }

    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(harness, "continue")
    Harness._initialize_run_brief(harness, resolved, raw_task="continue")

    assert resolved == new_task
    assert state.run_brief.original_task == new_task
    assert state.working_memory.current_goal == new_task
    assert "Follow-up:" not in state.run_brief.original_task

    query = build_retrieval_query(state)
    assert new_task in query
    assert old_task not in query


def test_bare_continue_preserves_last_effective_task_wrapper() -> None:
    state = LoopState(cwd="/tmp")
    prior = (
        "Continue current task: ssh into root@192.168.1.63 and install nginx. "
        "User follow-up: debug the nginx config and do a websearch first"
    )
    state.run_brief.original_task = "ssh into root@192.168.1.63 and install nginx"
    state.working_memory.current_goal = prior
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": "debug the nginx config and do a websearch first",
        "effective_task": prior,
        "current_goal": prior,
    }

    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(harness, "continue")
    Harness._initialize_run_brief(harness, resolved, raw_task="continue")

    assert resolved == prior
    assert state.run_brief.original_task == prior
    assert state.working_memory.current_goal == prior


def test_store_task_handoff_tracks_recent_research_artifacts() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Debug nginx on the remote host and do a websearch first"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.artifacts["A2001"] = ArtifactRecord(
        artifact_id="A2001",
        kind="web_fetch",
        source="https://example.com/nginx-fix",
        created_at="2026-05-01T00:00:00+00:00",
        size_bytes=256,
        summary="Fetched nginx fix article",
        tool_name="web_fetch",
    )
    state.tool_execution_records = {
        "op-1": {
            "operation_id": "op-1",
            "step_count": 2,
            "tool_name": "web_fetch",
            "result": {
                "success": True,
                "metadata": {"artifact_id": "A2001"},
            },
        }
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    handoff = state.scratchpad["_last_task_handoff"]
    assert handoff["recent_research_artifact_ids"] == ["A2001"]


def test_collapse_task_chain_canonicalizes_nested_inline_followup_wrapper() -> None:
    nested = (
        "Continue current task: Continue current task: what port does the compose stack publish. "
        "User follow-up: check compose. User follow-up: check compose"
    )

    assert _collapse_task_chain(nested) == (
        "Continue current task: what port does the compose stack publish. User follow-up: check compose"
    )


def test_continue_preserves_active_plan_goal() -> None:
    state = LoopState(cwd="/tmp")
    task = (
        "Build a Python script at `./temp/dead_letter_queue.py` that simulates a tiny "
        "message processor with retry, backoff, and dead-letter queue behavior."
    )
    plan_goal = "Implement the retry loop and dead-letter queue tests."
    state.run_brief.original_task = task
    state.working_memory.current_goal = plan_goal
    state.active_plan = ExecutionPlan(plan_id="plan-1", goal=plan_goal)

    harness = _make_harness(state)

    Harness._initialize_run_brief(harness, task, raw_task="continue")

    assert state.run_brief.original_task == task
    assert state.working_memory.current_goal == plan_goal


def test_assistant_numbered_options_are_persisted_in_last_task_handoff() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/file_deduper.py and propose upgrades"
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)
    record_assistant_message(
        harness,
        assistant_text="Recommended upgrades:\n1. Streaming MD5 calculation\n2. Skip unreadable files",
        tool_calls=[],
    )

    handoff = state.scratchpad["_last_task_handoff"]
    assert handoff["action_options"][0] == {
        "index": 1,
        "title": "Streaming MD5 calculation",
        "target_paths": ["temp/file_deduper.py"],
    }


def test_assistant_inline_numbered_options_are_persisted_in_last_task_handoff() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/file_deduper.py and propose upgrades"
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)
    record_assistant_message(
        harness,
        assistant_text=(
            "Recommended upgrades: 1. Patch temp/file_deduper.py to use streaming MD5 "
            "2. Add skip handling for unreadable files"
        ),
        tool_calls=[],
    )

    handoff = state.scratchpad["_last_task_handoff"]
    assert [option["index"] for option in handoff["action_options"]] == [1, 2]
    assert handoff["action_options"][0]["target_paths"] == ["temp/file_deduper.py"]


def test_task_boundary_reset_preserves_action_options_for_next_turn() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": "read temp/file_deduper.py and propose upgrades",
        "effective_task": "read temp/file_deduper.py and propose upgrades",
        "current_goal": "read temp/file_deduper.py and propose upgrades",
        "target_paths": ["temp/file_deduper.py"],
        "action_options": [
            {"index": 1, "title": "Streaming MD5 calculation", "target_paths": ["temp/file_deduper.py"]}
        ],
    }
    harness = _make_harness(state)

    Harness._reset_task_boundary_state(
        harness,
        reason="run_task",
        new_task="start with 1) patch the script",
    )

    assert state.scratchpad["_last_task_handoff"]["action_options"][0]["index"] == 1


def test_task_boundary_reset_preserves_web_search_scratchpad_state() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_web_result_index"] = {"r1": {"url": "https://example.com/article"}}
    state.scratchpad["_web_search_artifact_results"] = {"A0001": ["webres-test-1"]}
    state.scratchpad["_web_last_search_result_ids"] = ["webres-test-1"]
    state.scratchpad["_web_last_search_fetch_ids"] = ["r1"]
    state.scratchpad["_web_last_search_artifact_id"] = "A0001"
    state.scratchpad["_web_fetch_id_counter"] = 3
    state.scratchpad["_web_budget"] = {"searches_used": 1, "fetches_used": 2, "total_fetched_chars": 99}
    harness = _make_harness(state)

    Harness._reset_task_boundary_state(
        harness,
        reason="task_switch",
        new_task="follow up on the recent search result",
    )

    assert state.scratchpad["_web_result_index"]["r1"]["url"] == "https://example.com/article"
    assert state.scratchpad["_web_search_artifact_results"]["A0001"] == ["webres-test-1"]
    assert state.scratchpad["_web_last_search_result_ids"] == ["webres-test-1"]
    assert state.scratchpad["_web_last_search_fetch_ids"] == ["r1"]
    assert state.scratchpad["_web_last_search_artifact_id"] == "A0001"
    assert state.scratchpad["_web_fetch_id_counter"] == 3
    assert state.scratchpad["_web_budget"]["fetches_used"] == 2


def test_task_boundary_reset_preserves_mission_anchor_in_recent_tail() -> None:
    state = LoopState(cwd="/tmp")
    state.recent_message_limit = 3
    state.recent_messages = [
        ConversationMessage(role="user", content="update remote files with darkmode across the whole site"),
        ConversationMessage(role="assistant", content="I found the current CSS and page templates."),
        ConversationMessage(role="tool", name="ssh_exec", content="cat /var/www/html/index.html"),
        ConversationMessage(role="assistant", content="The home page still hardcodes light colors."),
        ConversationMessage(role="user", content="also make the footer consistent on every page"),
        ConversationMessage(role="tool", name="ssh_exec", content="cat /var/www/html/footer.html"),
    ]
    harness = _make_harness(state)

    Harness._reset_task_boundary_state(
        harness,
        reason="task_soft_switch",
        preserve_recent_tail=True,
    )

    assert [message.role for message in state.recent_messages] == ["user", "assistant", "user"]
    assert state.recent_messages[0].content == "update remote files with darkmode across the whole site"
    assert state.recent_messages[-1].content == "also make the footer consistent on every page"


def test_store_task_handoff_records_recent_remote_target_paths() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "Continue remote task over SSH on root@192.168.1.63. User follow-up: update llm-explainer.html"
    command = "tail -3 /var/www/html/llm-explainer.html && grep -c 'HTMLEOF' /var/www/html/llm-explainer.html"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.tool_execution_records = {
        "op-ssh-tail": {
            "operation_id": "op-ssh-tail",
            "step_count": 5,
            "tool_name": "ssh_exec",
            "args": {"host": "192.168.1.63", "user": "root", "command": command},
            "result": {"success": True, "metadata": {"command": command}},
        }
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    assert state.scratchpad["_last_task_handoff"]["remote_target_paths"] == [
        "/var/www/html/llm-explainer.html"
    ]


def test_ordinal_followup_resolves_structured_action_option_with_path() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/file_deduper.py and propose upgrades"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/file_deduper.py"],
        "action_options": [
            {
                "index": 1,
                "title": "Streaming MD5 calculation",
                "target_paths": ["temp/file_deduper.py"],
            }
        ],
    }
    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(
        harness,
        "start with 1) patch the script, test the new functionality then report back",
    )

    assert "Streaming MD5 calculation" in resolved
    assert "temp/file_deduper.py" in resolved
    assert "test the new functionality" in resolved.lower()
    assert state.scratchpad["_resolved_followup"]["option_index"] == 1


def test_ordinal_followup_effective_task_is_used_before_task_reset() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/file_deduper.py and propose upgrades"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/file_deduper.py"],
        "action_options": [
            {"index": 1, "title": "Streaming MD5 calculation", "target_paths": ["temp/file_deduper.py"]}
        ],
    }
    harness = _make_harness(state)

    raw = "start with 1) patch the script, test the new functionality then report back"
    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert "Streaming MD5 calculation" in state.run_brief.original_task
    assert "temp/file_deduper.py" in state.run_brief.original_task


def test_first_post_restore_task_switch_preserves_recent_messages_once() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Summarize the current harness status"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.recent_messages = [
        ConversationMessage(role="user", content=prior),
        ConversationMessage(role="assistant", content="The restore path currently reloads recent messages."),
    ]
    state.scratchpad["_session_restored"] = True
    harness = _make_harness(state)

    Harness._maybe_reset_for_new_task(harness, "List the repository files")

    assert [message.content for message in state.recent_messages] == [
        prior,
        "The restore path currently reloads recent messages.",
    ]
    assert "_session_restored" not in state.scratchpad
    Harness._initialize_run_brief(
        harness,
        "List the repository files",
        raw_task="List the repository files",
    )

    Harness._maybe_reset_for_new_task(harness, "Open README.md")

    assert state.recent_messages == []


def test_begin_task_scope_clears_session_restored_flag_without_reset() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_session_restored"] = True
    harness = _make_harness(state)

    Harness._begin_task_scope(harness, raw_task="continue", effective_task="continue")

    assert "_session_restored" not in state.scratchpad


def test_same_scope_resteer_soft_resets_without_losing_live_evidence() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/logwatch.py and identify the required fix"
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"explore: {prior}"
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["artifact_read(A0010) already showed the relevant function"]
    state.working_memory.decisions = ["Patch temp/logwatch.py in place"]
    state.working_memory.failures = ["Repeating artifact_read(A0010) is not useful"]
    state.active_plan = ExecutionPlan(plan_id="plan-1", goal=prior)
    state.recent_messages = [
        ConversationMessage(role="tool", name="artifact_read", content="A0010: reset code excerpt"),
        ConversationMessage(role="assistant", content="I have read the file and found the reset path."),
    ]
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="S-prior",
            created_at="2026-04-22T16:20:00Z",
            notes=["The artifact has already been read."],
        )
    ]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "artifact_read", "fingerprint": "artifact_read|A0010"}
    ]
    state.scratchpad["_chunk_write_loop_guard"] = {"version": 1, "paths": {}, "events": ["blocked"]}
    harness = _make_harness(state)

    raw = "you've read the file enough; make the change in the same file"
    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)

    assert state.step_count == 0
    assert state.active_plan is None
    assert state.working_memory.known_facts == ["artifact_read(A0010) already showed the relevant function"]
    assert state.working_memory.decisions == ["Patch temp/logwatch.py in place"]
    assert state.working_memory.failures == ["Repeating artifact_read(A0010) is not useful"]
    assert state.recent_messages[-1].content == "I have read the file and found the reset path."
    assert state.episodic_summaries[-1].summary_id == "S-prior"
    assert "_tool_attempt_history" not in state.scratchpad
    assert "_chunk_write_loop_guard" not in state.scratchpad


def test_loop_guard_nudge_preserves_goal_and_memory_after_soft_reset() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Fix theming across all three files in the demo app"
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"explore: {prior}"
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["The three themed files have already been inspected"]
    state.recent_errors = ["Guard tripped: repeated tool call loop on file_read"]
    state.recent_messages = [
        ConversationMessage(role="assistant", content="I keep rereading the same file."),
    ]
    harness = _make_harness(state)

    raw = "your looping make a decision now"
    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)

    assert resolved == f"Continue current task: {prior}. User follow-up: {raw}"
    assert state.run_brief.original_task == prior
    assert state.working_memory.known_facts == ["The three themed files have already been inspected"]
    assert state.recent_errors == ["Guard tripped: repeated tool call loop on file_read"]

    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert prior in state.run_brief.original_task
    assert raw in state.run_brief.original_task


def test_quality_followup_with_existing_target_stays_in_same_scope() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Fix theming across all three files in the demo app"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["app/a.css and app/b.css use different accent tokens"]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["app/a.css", "app/b.css", "app/c.css"],
    }
    harness = _make_harness(state)

    raw = "theming across all three file still inconsistent"
    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)

    assert resolved == f"Continue current task: {prior}. User follow-up: {raw}"
    assert state.run_brief.original_task == prior
    assert state.working_memory.known_facts == ["app/a.css and app/b.css use different accent tokens"]


def test_guard_context_does_not_make_unrelated_explicit_path_same_scope() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Fix temp/logwatch.py"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["temp/logwatch.py has a stale timeout"]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    state.recent_errors = ["Guard tripped: repeated tool call loop on file_read"]
    harness = _make_harness(state)

    new_task = "fix temp/other.py"
    Harness._maybe_reset_for_new_task(harness, new_task, raw_task=new_task)

    assert state.run_brief.original_task == ""
    assert state.working_memory.known_facts == []


def test_same_scope_soft_reset_preserves_only_semantic_visible_recent_context() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/logwatch.py and identify the required fix"
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"explore: {prior}"
    state.working_memory.current_goal = prior
    state.recent_messages = [
        ConversationMessage(role="user", content=prior),
        ConversationMessage(role="tool", name="artifact_read", content="A0010: code excerpt"),
        ConversationMessage(
            role="user",
            content="### SYSTEM ALERT: emit the JSON tool call now.",
            metadata={"is_recovery_nudge": True, "recovery_kind": "action_stall"},
        ),
        ConversationMessage(role="assistant", content="I found the reset path and recommend patching the same file."),
    ]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    harness = _make_harness(state)

    raw = "use file_patch instead"
    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)

    assert [message.role for message in state.recent_messages] == ["user", "assistant"]
    assert state.recent_messages[0].content == prior
    assert "recommend patching the same file" in str(state.recent_messages[1].content)


def test_store_task_handoff_captures_structured_continuation_anchor() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Patch temp/logwatch.py to preserve the active goal during soft switches"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.retrieval_cache = ["A1002"]
    state.tool_execution_records = {
        "op-1": {
            "operation_id": "op-1",
            "step_count": 3,
            "tool_name": "file_patch",
            "result": {
                "success": False,
                "error": "file_patch: expected exact target text",
                "metadata": {
                    "next_required_tool": {
                        "tool_name": "file_patch",
                        "required_arguments": {"path": "temp/logwatch.py"},
                    }
                },
            },
        },
        "op-2": {
            "operation_id": "op-2",
            "step_count": 2,
            "tool_name": "file_read",
            "result": {
                "success": True,
                "output": {"artifact_id": "A1001"},
                "metadata": {"artifact_id": "A1001"},
            },
        },
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    handoff = state.scratchpad["_last_task_handoff"]
    assert handoff["effective_task"] == prior
    assert handoff["last_good_artifact_ids"] == ["A1002", "A1001"]
    assert handoff["next_required_tool"]["tool_name"] == "file_patch"
    assert handoff["last_failed_tool"]["tool_name"] == "file_patch"


def test_corrective_tool_resteer_preserves_goal_and_working_memory_fields() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Patch temp/logwatch.py to preserve the active goal during soft switches"
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"execute: {prior}"
    state.working_memory.current_goal = prior
    state.working_memory.plan = ["Patch reset_task_boundary_state"]
    state.working_memory.open_questions = ["Should hard switches still clear task-local plans?"]
    state.working_memory.next_actions = ["Use a narrow patch"]
    state.working_memory.next_action_meta = [MemoryEntry(content="Use a narrow patch", created_at_step=2)]
    state.working_memory.known_facts = ["The target reset path is in task_boundary.py"]
    state.recent_messages = [
        ConversationMessage(role="assistant", content="I found the reset path and was about to edit it."),
    ]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    harness = _make_harness(state)

    raw = "use file_patch instead"
    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == f"Continue current task: {prior}. User correction: {raw}"
    assert state.run_brief.original_task == resolved
    assert state.working_memory.current_goal == prior
    assert state.working_memory.plan == []
    assert state.working_memory.open_questions == ["Should hard switches still clear task-local plans?"]
    assert "Use a narrow patch" not in state.working_memory.next_actions
    assert state.working_memory.next_action_meta == []
    assert state.working_memory.known_facts == ["The target reset path is in task_boundary.py"]


def test_corrective_tool_resteer_can_resolve_from_handoff_after_run_task_reset() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Patch temp/logwatch.py to preserve the active goal during soft switches"
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(harness, "switch to ast_patch now")

    assert resolved == f"Continue current task: {prior}. User correction: switch to ast_patch now"


def test_contextual_resteer_resolution_keeps_instruction_and_prior_target() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/logwatch.py and identify the required fix"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    state.recent_messages = [ConversationMessage(role="tool", name="artifact_read", content="A0010")]
    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(
        harness,
        "you've read the file enough; make the change",
    )

    assert "temp/logwatch.py" in resolved
    assert "you've read the file enough; make the change" in resolved


def test_repeated_inline_contextual_followups_do_not_keep_growing_resolved_task_text() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/logwatch.py and identify the required fix"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    state.recent_messages = [ConversationMessage(role="tool", name="artifact_read", content="A0010")]
    harness = _make_harness(state)

    raw = "you've read the file enough; make the change"
    first = Harness._resolve_followup_task(harness, raw)
    Harness._initialize_run_brief(harness, first, raw_task=raw)
    query_after_first = build_retrieval_query(state)

    second = Harness._resolve_followup_task(harness, raw)
    Harness._initialize_run_brief(harness, second, raw_task=raw)
    query_after_second = build_retrieval_query(state)

    assert first == second
    assert second.count("Continue current task:") == 1
    assert second.count("User follow-up:") == 1
    assert query_after_second == query_after_first


def test_unrelated_task_hard_resets_guard_state_and_durable_memory() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/logwatch.py and identify the required fix"
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"explore: {prior}"
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["prior file was read"]
    state.recent_messages = [ConversationMessage(role="tool", name="artifact_read", content="A0010")]
    state.artifacts = {"A0010": {"artifact_id": "A0010", "summary": "prior file"}}
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "artifact_read", "fingerprint": "artifact_read|A0010"}
    ]
    harness = _make_harness(state)

    new_task = "write a fresh README section about deployment"
    Harness._maybe_reset_for_new_task(harness, new_task, raw_task=new_task)

    assert state.recent_messages == []
    assert state.working_memory.known_facts == []
    assert "_tool_attempt_history" not in state.scratchpad
    assert "A0010" in state.artifacts


def test_finalize_task_scope_adds_prompt_visible_episodic_summary() -> None:
    state = LoopState(cwd="/tmp")
    task = "patch temp/logwatch.py"
    harness = _make_harness(state)

    Harness._begin_task_scope(harness, raw_task=task, effective_task=task)
    Harness._finalize_task_scope(
        harness,
        terminal_event="task_complete",
        status="complete",
        result={"status": "complete", "message": "Patched the reset boundary."},
    )

    assert state.episodic_summaries
    summary = state.episodic_summaries[-1]
    assert summary.summary_id == "task-0001-summary"
    assert "Patched the reset boundary." in summary.notes


def test_ordinal_followup_blocks_inherited_path_when_user_changes_target_language() -> None:
    state = LoopState(cwd="/tmp")
    prior = "read temp/file_deduper.py and propose upgrades"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/file_deduper.py"],
        "action_options": [
            {
                "index": 1,
                "title": "Streaming MD5 calculation",
                "target_paths": ["temp/file_deduper.py"],
            }
        ],
    }
    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(
        harness,
        "Start with 1, but do it in Rust instead of modifying the Python file",
    )

    assert "Streaming MD5 calculation" in resolved
    assert "Rust" in resolved
    assert "Do not assume temp/file_deduper.py is the edit target" in resolved
    assert state.scratchpad["_resolved_followup"]["target_paths"] == []
    assert state.scratchpad["_resolved_followup"]["blocked_target_paths"] == ["temp/file_deduper.py"]
    assert state.scratchpad["_resolved_followup"]["target_inheritance"] == "blocked_by_user_constraint"


def test_affirmative_followup_resolves_to_prior_task_after_action_confirmation() -> None:
    state = LoopState(cwd="/tmp")
    task = "read ./portainer_cli.py and recommend updates to that script"
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="Would you like me to create an updated version incorporating these recommendations?",
        )
    ]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": task,
        "effective_task": task,
        "current_goal": task,
    }

    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(harness, "yes")
    Harness._initialize_run_brief(harness, resolved, raw_task="yes")

    assert resolved == task
    assert state.run_brief.original_task == task
    assert state.working_memory.current_goal == task

    query = build_retrieval_query(state)
    assert task in query


def test_affirmative_followup_resolves_after_concrete_implementation_proposal() -> None:
    state = LoopState(cwd="/tmp")
    task = "read ./portainer_cli.py and recommend updates to that script"
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="I can create an updated version incorporating these recommendations next.",
        )
    ]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": task,
        "effective_task": task,
        "current_goal": task,
    }

    harness = _make_harness(state)

    resolved = Harness._resolve_followup_task(harness, "approved, proceed with implementation")

    assert resolved == task


def test_primary_task_target_path_falls_back_to_last_task_handoff() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "run and confirm functionality"
    state.working_memory.current_goal = "run and confirm functionality"
    state.scratchpad["_task_target_paths"] = []
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": "Build a Python script at `./temp/mini_indexer.py`.",
        "effective_task": "Build a Python script at `./temp/mini_indexer.py`.",
        "current_goal": "Verify `./temp/mini_indexer.py`.",
        "target_paths": ["./temp/mini_indexer.py"],
    }

    harness = _make_harness(state)

    assert primary_task_target_path(harness) == "./temp/mini_indexer.py"


def test_primary_task_target_path_skips_blocked_handoff_fallback() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_target_paths"] = []
    state.scratchpad["_resolved_followup"] = {
        "target_inheritance": "blocked_by_user_constraint",
        "blocked_target_paths": ["./temp/mini_indexer.py"],
        "target_paths": [],
    }
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": "Build a Python script at `./temp/mini_indexer.py`.",
        "effective_task": "Build a Python script at `./temp/mini_indexer.py`.",
        "current_goal": "Verify `./temp/mini_indexer.py`.",
        "target_paths": ["./temp/mini_indexer.py"],
    }

    harness = _make_harness(state)

    assert primary_task_target_path(harness) is None


def test_run_brief_renders_resolved_followup_option() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Patch temp/file_deduper.py to implement proposal #1: Streaming MD5 calculation."
    state.scratchpad["_resolved_followup"] = {
        "option_index": 1,
        "option_title": "Streaming MD5 calculation",
        "target_paths": ["temp/file_deduper.py"],
    }

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    rendered = "\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "Resolved follow-up: proposal #1 = Streaming MD5 calculation in temp/file_deduper.py." in rendered


def test_same_scope_remote_directory_transition_preserves_memory() -> None:
    state = LoopState(cwd="/tmp")
    prior = (
        "ssh root@192.168.1.89 and look for /var/www/html/llm-explainer.html "
        "and other explainer html files, make sure all files have the same theme"
    )
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["Google Minimal theme uses #1A73E8 primary"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.89": {"host": "192.168.1.89", "user": "root", "confirmed": True}
    }
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "remote_target_paths": ["/var/www/html/llm-explainer.html"],
    }
    harness = _make_harness(state)

    new_task = "now do `/var/www/html/llm-explainer-page-004.html`"
    Harness._maybe_reset_for_new_task(harness, new_task, raw_task=new_task)

    assert state.run_brief.original_task == prior
    assert state.working_memory.known_facts == ["Google Minimal theme uses #1A73E8 primary"]


def test_sequential_remote_followup_is_same_scope() -> None:
    state = LoopState(cwd="/tmp")
    prior = "ssh root@192.168.1.89 and redesign /var/www/html/page-001.html"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["theme vars extracted"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.89": {"host": "192.168.1.89", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    new_task = "next do /var/www/html/page-002.html"
    Harness._maybe_reset_for_new_task(harness, new_task, raw_task=new_task)

    assert state.run_brief.original_task == prior
    assert state.working_memory.known_facts == ["theme vars extracted"]


def test_unrelated_local_file_in_same_directory_still_hard_resets() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Fix temp/logwatch.py"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["temp/logwatch.py has a stale timeout"]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": prior,
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/logwatch.py"],
    }
    harness = _make_harness(state)

    new_task = "fix temp/other.py"
    Harness._maybe_reset_for_new_task(harness, new_task, raw_task=new_task)

    assert state.run_brief.original_task == ""
    assert state.working_memory.known_facts == []
