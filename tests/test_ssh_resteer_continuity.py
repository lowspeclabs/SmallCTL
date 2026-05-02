from __future__ import annotations

from types import SimpleNamespace

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.context.retrieval import LexicalRetriever, build_retrieval_query
from smallctl.harness import Harness
from smallctl.harness.task_boundary import TaskBoundaryService
from smallctl.models.conversation import ConversationMessage
from smallctl.state import ArtifactRecord, LoopState


def _make_harness(state: LoopState) -> SimpleNamespace:
    harness = SimpleNamespace(
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="execute",
        _configured_planning_mode=False,
        _configured_tool_profiles=None,
        _runlog=lambda *args, **kwargs: None,
    )
    harness._task_boundary_service = TaskBoundaryService(harness)
    return harness


def test_remote_operational_followup_resolves_to_active_ssh_target() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into 192.168.1.63 and pull vikunja"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "launch the container and configure it to autolaunch on startup"
    resolved = Harness._resolve_followup_task(harness, raw)

    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"
    assert state.scratchpad["_resolved_remote_followup"]["user"] == "root"
    assert state.scratchpad["_resolved_remote_followup"]["target_status"] == "resolved"

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: launch the container and configure it to autolaunch on startup"
    )
    assert state.task_mode == "remote_execute"
    assert state.run_brief.original_task == prior
    assert state.working_memory.current_goal == prior
    assert state.run_brief.current_phase_objective == f"execute: {resolved}"


def test_remote_clarification_followup_preserves_ssh_context_and_network_profile() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into 192.168.1.63 and install a task tracker docker container"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.active_tool_profiles = ["core", "network"]
    state.recent_messages = [
        ConversationMessage(role="assistant", content="I could not find an exact task tracker image yet.")
    ]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "the image does not have to be called exactly task tracker, any app used to task track will do"
    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: the image does not have to be called exactly task tracker, any app used to task track will do"
    )
    assert state.scratchpad["_resolved_remote_followup"]["target_status"] == "resolved"

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)
    Harness._activate_tool_profiles(harness, resolved)

    assert state.task_mode == "remote_execute"
    assert "network" in state.active_tool_profiles
    assert state.recent_messages[-1].content == "I could not find an exact task tracker image yet."


def test_remote_research_install_task_activates_network_and_web_profiles() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    harness = _make_harness(state)
    task = (
        'do reserach on self hosted docker containers you can use for project management, '
        'choose one of them and spin up that docker container on 192.168.1.63 '
        'username "root" password "@S02v1735"'
    )

    Harness._initialize_run_brief(harness, task, raw_task=task)
    Harness._activate_tool_profiles(harness, task)

    assert state.task_mode == "remote_execute"
    assert "network" in state.active_tool_profiles
    assert "network_read" in state.active_tool_profiles


def test_bare_remote_continue_preserves_network_read_profile_after_research_followup() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: debug nginx and do a websearch first"
    )
    state.run_brief.original_task = "ssh into root@192.168.1.63 and install nginx"
    state.working_memory.current_goal = prior
    state.active_tool_profiles = ["core", "network", "network_read"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, "continue")
    Harness._initialize_run_brief(harness, resolved, raw_task="continue")
    Harness._activate_tool_profiles(harness, resolved)

    assert resolved == prior
    assert "network" in state.active_tool_profiles
    assert "network_read" in state.active_tool_profiles


def test_remote_operational_followup_is_explicitly_ambiguous_with_multiple_sessions() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into the remote host and inspect the vikunja deployment"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True},
        "10.0.0.5": {"host": "10.0.0.5", "user": "ubuntu", "confirmed": True},
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(
        harness,
        "restart the service and verify the container comes back cleanly",
    )

    assert "Continue remote task over SSH." in resolved
    assert "root@192.168.1.63" in resolved
    assert "ubuntu@10.0.0.5" in resolved
    assert "Resolve which host before executing remote commands." in resolved
    assert state.scratchpad["_resolved_remote_followup"]["target_status"] == "ambiguous"


def test_remote_operational_followup_with_same_explicit_host_reuses_active_session() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and install a docker task tracker"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root"}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "vikunja should be getting installed on the remote host 192.168.1.63"
    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: vikunja should be getting installed on the remote host 192.168.1.63"
    )
    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"
    assert state.scratchpad["_resolved_remote_followup"]["user"] == "root"


def test_implicit_remote_operational_followup_requires_confirmed_ssh_session() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and install a docker task tracker"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root"}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "spin up a vikunja container instead"
    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved == raw
    assert "_resolved_remote_followup" not in state.scratchpad


def test_remote_operational_followup_with_different_explicit_host_does_not_inherit_session() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and install a docker task tracker"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root"}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "vikunja should be getting installed on the remote host 10.0.0.5"
    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved == raw
    assert "_resolved_remote_followup" not in state.scratchpad


def test_remote_spin_up_followup_soft_switches_and_preserves_working_memory() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and deploy a task tracker container"
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"execute: {prior}"
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["docker is already installed on the remote host"]
    state.working_memory.decisions = ["Use the existing SSH session instead of working locally"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "spin up a vikunja container instead"
    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: spin up a vikunja container instead"
    )
    assert state.task_mode == "remote_execute"
    assert state.working_memory.known_facts == ["docker is already installed on the remote host"]
    assert state.working_memory.decisions == ["Use the existing SSH session instead of working locally"]


def test_remote_404_followup_stays_scoped_to_active_ssh_target() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and publish split explainer pages under /var/www/html"
    raw = "pages are not live in nginx, 404 error"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)
    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == "Continue remote task over SSH on root@192.168.1.63. User follow-up: pages are not live in nginx, 404 error"
    assert state.task_mode == "remote_execute"


def test_remote_nginx_config_followup_stays_scoped_to_active_ssh_target() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and split the site into multiple HTML pages under /var/www/html"
    raw = "has the nginx config been update to reflect the new site structure?"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)
    assert state.scratchpad["_resolved_remote_followup"]["target_status"] == "resolved"
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: has the nginx config been update to reflect the new site structure?"
    )
    assert state.task_mode == "remote_execute"
    assert state.run_brief.original_task == prior
    assert state.working_memory.current_goal == prior
    assert state.run_brief.current_phase_objective == f"execute: {resolved}"


def test_remote_nginx_followup_after_validated_ssh_file_read_session_stays_remote() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh root@192.168.1.89 and read the nginx config, what is the url for the demo-site website at /var/www/?"
    raw = 'is the "demo-site" enabled in the nginx config?'
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.89": {
            "host": "192.168.1.89",
            "user": "root",
            "confirmed": True,
            "validated_tools": ["ssh_file_read"],
            "last_success_tool": "ssh_file_read",
            "last_validated_path": "/etc/nginx/sites-enabled/default",
        }
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)
    assert state.scratchpad["_resolved_remote_followup"]["target_status"] == "resolved"

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == (
        'Continue remote task over SSH on root@192.168.1.89. '
        'User follow-up: is the "demo-site" enabled in the nginx config?'
    )
    assert state.task_mode == "remote_execute"
    assert state.run_brief.original_task == prior
    assert state.working_memory.current_goal == prior


def test_remote_theme_followup_with_matching_remote_page_names_stays_on_active_ssh_target() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = (
        'ssh into 192.168.1.63, username "root" password "@S02v1735" update the site at /var/www/, '
        "its supposed to be a presentation about small llms, do derearch on small language models "
        "and updated the pages as needed"
    )
    raw = "all pages should have the same theme including the home page llm-explainer"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.active_tool_profiles = ["core", "network"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.tool_execution_records = {
        "op-home": {
            "operation_id": "op-home",
            "step_count": 2,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": "cat /var/www/html/llm-explainer.html",
            },
            "result": {
                "success": True,
                "metadata": {"command": "cat /var/www/html/llm-explainer.html"},
            },
        },
        "op-page": {
            "operation_id": "op-page",
            "step_count": 3,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": "cat /var/www/html/llm-explainer-page-005.html",
            },
            "result": {
                "success": True,
                "metadata": {"command": "cat /var/www/html/llm-explainer-page-005.html"},
            },
        },
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: all pages should have the same theme including the home page llm-explainer"
    )
    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)
    Harness._activate_tool_profiles(harness, resolved)

    assert state.task_mode == "remote_execute"
    assert "network" in state.active_tool_profiles


def test_remote_site_structure_correction_preserves_ssh_context_and_network_profile() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and split the site into multiple HTML pages under /var/www/html"
    raw = "correciton there should be 1 main page and 4-5 subpages"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.active_tool_profiles = ["core", "network"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: correciton there should be 1 main page and 4-5 subpages"
    )
    assert state.scratchpad["_resolved_remote_followup"]["target_status"] == "resolved"

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)
    Harness._activate_tool_profiles(harness, resolved)

    assert state.task_mode == "remote_execute"
    assert "network" in state.active_tool_profiles


def test_affirmative_remote_confirmation_resolves_to_remote_execution_continuation() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and fix the nginx site routing"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="I found the remote fixes. Would you like me to apply these fixes now?",
        )
    ]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, "yes")

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: proceed with the approved remote execution steps now"
    )
    assert state.scratchpad["_resolved_remote_followup"]["target_status"] == "resolved"
    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task="yes")
    Harness._initialize_run_brief(harness, resolved, raw_task="yes")

    assert state.task_mode == "remote_execute"


def test_affirmative_remote_proposal_followup_preserves_remote_execution_context() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and fix the nginx site routing"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="I can apply the remote fixes next: update the nginx config and restart the service.",
        )
    ]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, "1) approved, proceed with implementation")

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: proceed with the approved remote execution steps now"
    )

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task="1) approved, proceed with implementation")
    Harness._initialize_run_brief(
        harness,
        resolved,
        raw_task="1) approved, proceed with implementation",
    )
    Harness._activate_tool_profiles(harness, resolved)

    assert state.task_mode == "remote_execute"
    assert "network" in state.active_tool_profiles


def test_run_brief_renders_active_ssh_sessions_for_remote_followups() -> None:
    state = LoopState(cwd="/tmp")
    state.task_mode = "remote_execute"
    state.run_brief.original_task = "Continue remote task over SSH on root@192.168.1.63. User follow-up: restart vikunja"
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root"}
    }
    state.scratchpad["_resolved_remote_followup"] = {
        "host": "192.168.1.63",
        "user": "root",
        "target_status": "resolved",
    }

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    rendered = "\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "Active SSH sessions: root@192.168.1.63" in rendered


def test_remote_followup_preserves_original_mission_under_tight_prompt_window() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and update remote files with darkmode across the whole site"
    raw = "has the nginx config been updated for the new darkmode layout?"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.recent_messages = [
        ConversationMessage(role="user", content=prior),
        ConversationMessage(role="assistant", content="I found the live site files under /var/www/html."),
        ConversationMessage(role="tool", name="ssh_exec", content="cat /var/www/html/index.html"),
    ]
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)
    state.append_message(ConversationMessage(role="user", content=raw))

    assembly = PromptAssembler(
        ContextPolicy(max_prompt_tokens=2048, recent_message_limit=2)
    ).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )
    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)

    assert state.run_brief.original_task == prior
    assert state.working_memory.current_goal == prior
    assert state.scratchpad["_resolved_remote_followup"]["mission_task"] == prior
    assert "darkmode" in rendered
    assert raw in rendered
    assert "root@192.168.1.63" in rendered


def test_remote_correction_keeps_tail_and_prefers_fresh_ssh_evidence() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and install a docker task tracker"
    correction = "actually use ssh, do not rely on past records"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.recent_messages = [
        ConversationMessage(role="user", content=prior),
        ConversationMessage(role="assistant", content="I can continue through the existing SSH session."),
        ConversationMessage(role="tool", name="task_complete", content="Task completed successfully."),
    ]
    state.artifacts = {
        "A-terminal": ArtifactRecord(
            artifact_id="A-terminal",
            kind="task_complete",
            source="model_terminal_claim",
            created_at="2026-04-24T12:00:00+00:00",
            size_bytes=120,
            summary="Task completed successfully on the remote host.",
            tool_name="task_complete",
            metadata={"source": "model_terminal_claim"},
        ),
        "A-ssh": ArtifactRecord(
            artifact_id="A-ssh",
            kind="ssh_exec",
            source="ssh_exec",
            created_at="2026-04-24T12:01:00+00:00",
            size_bytes=220,
            summary="Fresh ssh_exec run against 192.168.1.63 verified the host directly.",
            tool_name="ssh_exec",
            metadata={"command": "ssh root@192.168.1.63"},
        ),
    }
    harness = _make_harness(state)
    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    service = harness._task_boundary_service
    reset_kwargs: dict[str, object] = {}
    original_reset = service.reset_task_boundary_state

    def _reset_spy(**kwargs: object) -> None:
        reset_kwargs.update(kwargs)
        original_reset(**kwargs)

    service.reset_task_boundary_state = _reset_spy

    resolved = Harness._resolve_followup_task(harness, correction)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=correction)
    Harness._initialize_run_brief(harness, resolved, raw_task=correction)
    state.append_message(ConversationMessage(role="user", content=correction))

    bundle = LexicalRetriever().retrieve_bundle(
        state=state,
        query=build_retrieval_query(state),
        include_experiences=True,
    )

    assert resolved.startswith("Continue remote task over SSH on root@192.168.1.63.")
    assert correction in resolved
    assert reset_kwargs["preserve_recent_tail"] is True
    assert state.recent_messages[0].content == prior
    assert state.scratchpad["_session_ssh_targets"]["192.168.1.63"]["confirmed"] is True
    assert state.run_brief.original_task == prior
    assert state.working_memory.current_goal == prior
    assert state.run_brief.current_phase_objective == f"execute: {resolved}"
    assert state.task_mode == "remote_execute"
    assert bundle.artifacts[0].artifact_id == "A-ssh"
    assert bundle.artifacts[0].text.startswith("ssh_exec | Fresh ssh_exec run against 192.168.1.63")
    assert state.retrieval_cache[0] == "A-ssh"


def test_remote_check_again_correction_stays_scoped_to_ssh_target() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and inspect the nginx config for the split explainer pages"
    raw = "check again, the target host does have nginx installed"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)
    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: check again, the target host does have nginx installed"
    )
    assert state.task_mode == "remote_execute"


def test_remote_here_doc_cleanup_followup_soft_switches_and_stays_remote() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = (
        "Continue current task: Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: explainer.html already exists configure nginx. "
        "User follow-up: update llm-explainer.html, it should have a modern minimilistic design"
    )
    raw = (
        '\'HTMLEOF && echo "Written $(wc -l < /var/www/html/llm-explainer.html) lines."\' '
        "at the very end of the page, please fix"
    )
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"execute: {prior}"
    state.working_memory.current_goal = prior
    state.working_memory.known_facts = ["The page was just written on the remote host over SSH."]
    state.working_memory.decisions = ["Keep editing the live remote artifact over SSH."]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    ssh_command = (
        "cat > /var/www/html/llm-explainer.html << 'HTMLEOF'\n"
        "<html>\n"
        "</html>'HTMLEOF && echo \"Written $(wc -l < /var/www/html/llm-explainer.html) lines.\""
    )
    state.tool_execution_records = {
        "op-ssh-write": {
            "operation_id": "op-ssh-write",
            "step_count": 3,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": ssh_command,
            },
            "result": {"success": True, "metadata": {"command": ssh_command}},
        },
        "op-ssh-tail": {
            "operation_id": "op-ssh-tail",
            "step_count": 4,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": "tail -3 /var/www/html/llm-explainer.html",
            },
            "result": {
                "success": True,
                "metadata": {"command": "tail -3 /var/www/html/llm-explainer.html"},
            },
        },
    }
    harness = _make_harness(state)
    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    service = harness._task_boundary_service
    reset_kwargs: dict[str, object] = {}
    original_reset = service.reset_task_boundary_state

    def _reset_spy(**kwargs: object) -> None:
        reset_kwargs.update(kwargs)
        original_reset(**kwargs)

    service.reset_task_boundary_state = _reset_spy

    resolved = Harness._resolve_followup_task(harness, raw)
    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)

    assert resolved == f"Continue remote task over SSH on root@192.168.1.63. User follow-up: {raw}"
    assert reset_kwargs["reason"] == "task_soft_switch"
    assert reset_kwargs["preserve_memory"] is True
    assert reset_kwargs["preserve_summaries"] is True
    assert reset_kwargs["preserve_recent_tail"] is True
    assert state.task_mode == "remote_execute"
    assert state.run_brief.original_task == prior
    assert state.working_memory.current_goal == prior
    assert state.run_brief.current_phase_objective == f"execute: {resolved}"
    assert state.working_memory.known_facts == ["The page was just written on the remote host over SSH."]
    assert state.working_memory.decisions == ["Keep editing the live remote artifact over SSH."]
    assert "/var/www/html/llm-explainer.html" in state.scratchpad["_last_task_handoff"]["remote_target_paths"]


def test_unrelated_local_absolute_path_task_does_not_inherit_remote_session() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and inspect the remote website"
    raw = "patch /home/stephen/Scripts/Harness-Redo/README.md locally to document the fix"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.tool_execution_records = {
        "op-ssh-tail": {
            "operation_id": "op-ssh-tail",
            "step_count": 2,
            "tool_name": "ssh_exec",
            "args": {"host": "192.168.1.63", "user": "root", "command": "tail -3 /var/www/html/llm-explainer.html"},
            "result": {"success": True, "metadata": {}},
        }
    }
    harness = _make_harness(state)
    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved == raw
    assert "_resolved_remote_followup" not in state.scratchpad


def test_upstream_theme_followup_survives_intervening_chat_handoff() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = (
        'ssh root@192.168.1.63 password "@S02v1735" and look for '
        "/var/www/html/llm-explainer.html and other explainer files, "
        "make sure all files have the same google minimal theme and color scheme"
    )
    intervening_chat = "did you copy the save files upstream?"
    raw = "the html files upstream are not lading the css theme"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.task_mode = "remote_execute"
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.tool_execution_records = {
        "op-ssh-list": {
            "operation_id": "op-ssh-list",
            "step_count": 1,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": "find /var/www/html -type f -name 'llm-explainer*.html'",
            },
            "result": {
                "success": True,
                "metadata": {"command": "find /var/www/html -type f -name 'llm-explainer*.html'"},
            },
        },
        "op-ssh-theme": {
            "operation_id": "op-ssh-theme",
            "step_count": 2,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": "cat > /var/www/html/llm-explainer-theme.css << 'EOF'\nbody{}\nEOF",
            },
            "result": {
                "success": True,
                "metadata": {"command": "cat > /var/www/html/llm-explainer-theme.css << 'EOF'\nbody{}\nEOF"},
            },
        },
    }
    harness = _make_harness(state)
    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    state.run_brief.original_task = intervening_chat
    state.working_memory.current_goal = intervening_chat
    state.task_mode = "chat"
    Harness._store_task_handoff(harness, raw_task=intervening_chat, effective_task=intervening_chat)

    reset_kwargs: dict[str, object] = {}
    service = harness._task_boundary_service
    original_reset = service.reset_task_boundary_state

    def _reset_spy(**kwargs: object) -> None:
        reset_kwargs.update(kwargs)
        original_reset(**kwargs)

    service.reset_task_boundary_state = _reset_spy

    resolved = Harness._resolve_followup_task(harness, raw)
    assert resolved == (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: the html files upstream are not lading the css theme"
    )
    assert state.scratchpad["_resolved_remote_followup"]["host"] == "192.168.1.63"

    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)
    Harness._activate_tool_profiles(harness, resolved)

    assert reset_kwargs["reason"] == "task_soft_switch"
    assert reset_kwargs["preserve_memory"] is True
    assert reset_kwargs["preserve_recent_tail"] is True
    assert state.task_mode == "remote_execute"
    assert "network" in state.active_tool_profiles


def test_local_followup_after_remote_task_does_not_inherit_network_profile() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and inspect the remote website"
    raw = "update /home/stephen/Scripts/Harness-Redo/README.md locally to document the fix"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.active_tool_profiles = ["core", "network"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.tool_execution_records = {
        "op-ssh-read": {
            "operation_id": "op-ssh-read",
            "step_count": 2,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": "cat /var/www/html/llm-explainer.html",
            },
            "result": {
                "success": True,
                "metadata": {"command": "cat /var/www/html/llm-explainer.html"},
            },
        }
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, raw)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=raw)
    Harness._initialize_run_brief(harness, resolved, raw_task=raw)
    Harness._activate_tool_profiles(harness, resolved)

    assert resolved == raw
    assert state.task_mode == "local_execute"
    assert state.active_tool_profiles == ["core"]
    assert "_resolved_remote_followup" not in state.scratchpad


def test_remote_web_style_followup_keeps_network_profile_with_prior_ssh_target() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and update /var/www/html/llm-explainer.html"
    raw = "dont just update the font family, update the colors buttons background and other details"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.active_tool_profiles = ["core"]
    state.task_mode = "chat"
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.tool_execution_records = {
        "op-ssh-read": {
            "operation_id": "op-ssh-read",
            "step_count": 2,
            "tool_name": "ssh_exec",
            "args": {
                "host": "192.168.1.63",
                "user": "root",
                "command": "cat /var/www/html/llm-explainer-page-001.html",
            },
            "result": {
                "success": True,
                "metadata": {"command": "cat /var/www/html/llm-explainer-page-001.html"},
            },
        }
    }
    harness = _make_harness(state)

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)
    Harness._activate_tool_profiles(harness, raw)

    assert "network" in state.active_tool_profiles


def test_remote_followup_after_recovery_nudge_preserves_mission_and_ignores_alert_text() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    prior = "ssh into root@192.168.1.63 and find the vikunja docker compose file"
    correction = "use ssh on the same host and find the actual vikunja compose file, not the portainer one"
    state.run_brief.original_task = prior
    state.run_brief.current_phase_objective = f"execute: {prior}"
    state.working_memory.current_goal = prior
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec"]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.recent_messages = [
        ConversationMessage(role="user", content=prior),
        ConversationMessage(role="assistant", content="I found a compose file under docker overlay storage."),
        ConversationMessage(
            role="user",
            content="### SYSTEM ALERT: You identified or described a tool action, but you did not emit the JSON tool call.",
            metadata={"is_recovery_nudge": True, "recovery_kind": "action_stall"},
        ),
        ConversationMessage(role="assistant", content="That compose file appears to belong to Portainer rather than Vikunja."),
    ]
    harness = _make_harness(state)
    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, correction)
    Harness._maybe_reset_for_new_task(harness, resolved, raw_task=correction)
    Harness._initialize_run_brief(harness, resolved, raw_task=correction)
    state.append_message(ConversationMessage(role="user", content=correction))

    assembly = PromptAssembler(
        ContextPolicy(max_prompt_tokens=2048, recent_message_limit=2)
    ).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )
    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)
    query = build_retrieval_query(state)

    assert resolved.startswith("Continue remote task over SSH on root@192.168.1.63.")
    assert correction in resolved
    assert "root@192.168.1.63" in rendered
    assert correction in rendered
    assert "### SYSTEM ALERT" not in rendered
    assert correction in query
    assert "### SYSTEM ALERT" not in query
    assert "emit the JSON tool call" not in query


def test_run_brief_renders_active_ssh_sessions_after_non_remote_task_reset() -> None:
    state = LoopState(cwd="/tmp")
    state.task_mode = "local_execute"
    state.run_brief.original_task = "inspect the local project files"
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": "inspect the local project files",
        "effective_task": "inspect the local project files",
        "current_goal": "inspect the local project files",
        "task_mode": "local_execute",
    }
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root"}
    }

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    rendered = "\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "Active SSH sessions: root@192.168.1.63" in rendered
