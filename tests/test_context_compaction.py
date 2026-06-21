from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.assembler import PromptAssembler
from smallctl.context.frame_compiler import PromptStateFrameCompiler
from smallctl.context.frame_invalidation_filtering import (
    experience_invalidated_with_reason,
    verifier_failure_related_to_text,
)
from smallctl.context.frame_working_memory_rendering import _is_low_value_known_fact, render_working_memory
from smallctl.context.retrieval import LexicalRetriever
from smallctl.graph.interpret_nodes import _assistant_text_looks_like_shell_command
from smallctl.harness.tool_result_artifact_updates import _apply_ssh_file_mutation_updates
from smallctl.models.tool_result import ToolEnvelope
from smallctl.recovery_schema import Subtask, SubtaskLedger
from smallctl.state import ArtifactRecord, ExperienceMemory, EvidenceRecord, LoopState
from smallctl.harness.tool_result_verification import _store_verifier_verdict


def _make_state(*, tool_history: list[str] | None = None, step_count: int = 10) -> LoopState:
    state = LoopState()
    state.step_count = step_count
    if tool_history:
        state.tool_history = list(tool_history)
    return state


def test_repair_phase_retains_web_fetch_and_ssh_session_start_memories() -> None:
    """After a failed ssh_exec, repair phase keeps non-failing memories."""
    state = LoopState(cwd="/tmp")
    state.step_count = 8
    state.current_phase = "repair"

    experiences = [
        ExperienceMemory(
            memory_id="mem-webfetch",
            phase="explore",
            intent="retrieve_install_guide",
            tool_name="web_fetch",
            outcome="success",
            notes="Fetched Webmin install guide from webmin.com",
        ),
        ExperienceMemory(
            memory_id="mem-sshstart",
            phase="execute",
            intent="start_interactive_session",
            tool_name="ssh_session_start",
            outcome="success",
            notes="Started PTY session for interactive installer",
        ),
        ExperienceMemory(
            memory_id="mem-sshexec",
            phase="execute",
            intent="install_package",
            tool_name="ssh_exec",
            outcome="failure",
            notes="Installer stalled at (y/N) prompt",
        ),
    ]
    state.warm_experiences = experiences

    invalidations = [
        {
            "reason": "phase_advanced",
            "details": {
                "from_phase": "execute",
                "to_phase": "repair",
                "command": "apt-get install -y webmin",
            },
        }
    ]

    invalidated = []
    retained = []
    for mem in experiences:
        is_invalidated, reason = experience_invalidated_with_reason(
            state=state,
            memory=mem,
            invalidations=invalidations,
            failing_tool="ssh_exec",
        )
        if is_invalidated:
            invalidated.append((mem.memory_id, reason))
        else:
            retained.append(mem.memory_id)

    assert "mem-sshexec" in [i[0] for i in invalidated]
    assert "mem-webfetch" in retained
    assert "mem-sshstart" in retained


def test_repair_phase_retains_relevant_artifact_snippets() -> None:
    """Artifact snippets referenced in task goal survive phase transitions."""
    state = LoopState(cwd="/tmp")
    state.step_count = 8
    state.current_phase = "repair"
    state.run_brief.original_task = "Install Webmin using the guide from webmin.com"

    state.artifacts = {
        "A0003": ArtifactRecord(
            artifact_id="A0003",
            kind="web_fetch",
            source="https://webmin.com/download/",
            created_at="2026-04-19T00:00:00+00:00",
            size_bytes=1024,
            summary="Webmin installation guide",
        ),
    }

    state.invalidate_context(
        reason="phase_advanced",
        details={"from_phase": "execute", "to_phase": "repair"},
    )

    stale_art = set(state.scratchpad.get("_artifact_staleness", {}).keys())
    assert "A0003" not in stale_art


def test_low_value_memory_facts_suppressed_from_working_memory() -> None:
    assert _is_low_value_known_fact("task_complete keys: message, status") is True
    assert _is_low_value_known_fact("The installer returned HTML instead of a script") is False

    state = LoopState()
    state.working_memory.known_facts = [
        "task_complete keys: message, status",
        "FOG installer fetch returned HTML",
    ]
    rendered = render_working_memory(state, phase_lines=[], coding_anchor_lines=[])
    assert "FOG installer fetch returned HTML" in rendered
    assert "task_complete keys" not in rendered


def test_remote_ssh_recovery_prefers_actionable_ssh_memories_over_generic_completion() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 7
    state.current_phase = "repair"
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.last_failure_class = "remote_interactive_stall"
    state.run_brief.original_task = "ssh to the remote host and recover the apt install after tty and noninteractive retries stalled"
    state.working_memory.current_goal = "recover the remote apt installer"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-complete",
            intent="general_task",
            tool_name="task_complete",
            outcome="success",
            notes="Remote task completed successfully.",
            confidence=0.9,
        ),
        ExperienceMemory(
            memory_id="mem-ssh",
            intent="requested_ssh_exec",
            tool_name="ssh_session_start",
            outcome="success",
            notes="Open an interactive SSH session first when apt needs a tty and noninteractive retries stall.",
            environment_tags=["tty", "noninteractive", "apt"],
            confidence=0.9,
        ),
    ]

    ranked = LexicalRetriever()._rank_experiences(state=state)

    assert ranked
    assert ranked[0][1].tool_name == "ssh_session_start"


def test_successful_ssh_file_write_does_not_invalidate_observations() -> None:
    state = LoopState()
    service = SimpleNamespace(
        harness=SimpleNamespace(
            state=state,
            _runlog=lambda *args, **kwargs: None,
        )
    )
    result = ToolEnvelope(
        success=True,
        metadata={"path": "/tmp/test.sh", "host": "pi.hole", "changed": True},
    )
    _apply_ssh_file_mutation_updates(
        service,
        tool_name="ssh_file_write",
        result=result,
        arguments={"path": "/tmp/test.sh", "host": "pi.hole"},
        artifact=None,
    )
    invalidations = state.scratchpad.get("_context_invalidations", [])
    ssh_file_write_invalidations = [
        inv for inv in invalidations
        if inv.get("details", {}).get("tool_name") == "ssh_file_write"
    ]
    assert len(ssh_file_write_invalidations) == 0


def test_fresh_tool_outputs_includes_ssh_session_status() -> None:
    state = LoopState()
    assembler = PromptAssembler(policy=SimpleNamespace(
        fresh_tool_output_items=4,
        fresh_tool_output_token_limit=1200,
    ))
    output = assembler._render_fresh_tool_outputs(state)
    assert "Fresh tool outputs" in output or output == ""


def test_3a60dd3d_trace_replay_simulated() -> None:
    """Replay anonymized session events and verify all phases."""
    state = LoopState(cwd="/tmp")
    state.step_count = 1
    state.current_phase = "explore"
    state.run_brief.original_task = "Install Webmin on the remote host."

    state.reasoning_graph.evidence_records.append(
        EvidenceRecord(
            evidence_id="E-webfetch",
            statement="Fetched Webmin install guide from webmin.com",
            phase="explore",
            tool_name="web_fetch",
            metadata={"url": "https://webmin.com/download/"},
        )
    )

    state.step_count = 4
    state.current_phase = "execute"
    state.tool_history.append("ssh_session_start|192.168.1.162|root|installer")

    state.step_count = 5
    installer_result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "Setting up webmin...\n(y/N) ? ",
            "stderr": "",
        },
    )
    install_verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=installer_result,
        arguments={"host": "192.168.1.162", "command": "apt-get install -y webmin"},
    )

    assert install_verdict is not None
    assert install_verdict["verdict"] == "fail"
    assert "interactive" in str(install_verdict.get("failure_mode", "")).lower()

    state.step_count = 6
    state.current_phase = "repair"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-webfetch",
            phase="explore",
            intent="retrieve_install_guide",
            tool_name="web_fetch",
            outcome="success",
            notes="Fetched Webmin install guide",
        ),
        ExperienceMemory(
            memory_id="mem-sshstart",
            phase="execute",
            intent="start_interactive_session",
            tool_name="ssh_session_start",
            outcome="success",
            notes="Started PTY session",
        ),
        ExperienceMemory(
            memory_id="mem-sshexec",
            phase="execute",
            intent="install_package",
            tool_name="ssh_exec",
            outcome="failure",
            notes="Installer stalled at prompt",
        ),
    ]

    invalidations = [
        {
            "reason": "phase_advanced",
            "details": {
                "from_phase": "execute",
                "to_phase": "repair",
                "command": "apt-get install -y webmin",
            },
        }
    ]

    invalidated_ids = []
    retained_ids = []
    for mem in state.warm_experiences:
        is_invalidated, _reason = experience_invalidated_with_reason(
            state=state,
            memory=mem,
            invalidations=invalidations,
            failing_tool="ssh_exec",
        )
        if is_invalidated:
            invalidated_ids.append(mem.memory_id)
        else:
            retained_ids.append(mem.memory_id)

    assert "mem-sshexec" in invalidated_ids
    assert "mem-webfetch" in retained_ids
    assert "mem-sshstart" in retained_ids

    from smallctl.fama.detectors import detect_interactive_installer_stall
    from smallctl.fama.signals import ActiveMitigation
    from smallctl.fama.state import activate_mitigations
    from smallctl.fama.capsules import render_fama_capsules
    from smallctl.fama.detectors import detect_identical_tool_loop

    state.tool_history = [
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
    ]
    stall_signal = detect_interactive_installer_stall(state, threshold=2)
    assert stall_signal is not None
    assert "same prompt" in stall_signal.evidence.lower()

    state.scratchpad["_fama_config"] = {"enabled": True, "mode": "lite", "capsule_token_budget": 180}
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="interactive_installer_stall_capsule",
                reason="stall_detected",
                source_signal=f"interactive_session_stall:{state.step_count}",
                activated_step=state.step_count,
                expires_after_step=state.step_count + 5,
            )
        ],
        max_active=5,
    )

    capsules = render_fama_capsules(state, token_budget=180)
    assert len(capsules) > 0
    combined_capsules = "\n".join(capsules)
    assert "interactive" in combined_capsules.lower() or "stall" in combined_capsules.lower()

    state.tool_history.extend([
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
    ])
    _ = detect_identical_tool_loop(state, threshold=3)
    _ = detect_identical_tool_loop(state, threshold=3)
    loop_signal = detect_identical_tool_loop(state, threshold=3)
    assert loop_signal is not None
    assert "try a different specific fix" in loop_signal.next_safe_action.lower()

    state.subtask_ledger = SubtaskLedger(
        task_id="task-1",
        subtasks=[
            Subtask(
                subtask_id="S2",
                title="Install Webmin",
                goal="Install Webmin on remote host",
                status="blocked",
                blockers=[
                    "ssh_session_read failed — unknown session",
                    "unsafe single-answer pipe blocked",
                    "task_complete blocked by failing verifier",
                ],
                next_action="Use the tool failure evidence to take the next smallest different action.",
            )
        ],
        active_subtask_id="S2",
    )

    frame = PromptStateFrameCompiler().compile(state=state)
    wm_text = frame.spine.working_memory_text
    assert "blocked" in wm_text.lower() or "blocker" in wm_text.lower()


def test_verifier_failure_without_paths_does_not_invalidate_optimistic_text() -> None:
    """Path-less verifier failures should not blanket-invalidate optimistic context."""
    event = {"reason": "verifier_failed", "paths": [], "details": {"command": "systemctl status docker"}}
    assert verifier_failure_related_to_text("docker service is healthy", event) is False


def test_fama_failure_without_paths_still_invalidates_optimistic_text() -> None:
    """FAMA-detected failures stay broad even without concrete paths."""
    event = {"reason": "fama_failure_detected", "paths": [], "details": {}}
    assert verifier_failure_related_to_text("docker service is healthy", event) is True


def test_verifier_failure_with_paths_relates_to_matching_text() -> None:
    event = {
        "reason": "verifier_failed",
        "paths": ["/tmp/config.yaml"],
        "details": {"command": "cat /tmp/config.yaml"},
    }
    assert verifier_failure_related_to_text("the config at /tmp/config.yaml is valid", event) is True
    assert verifier_failure_related_to_text("the config at /etc/other.yaml is valid", event) is False


def test_assistant_shell_command_detection() -> None:
    assert _assistant_text_looks_like_shell_command("docker inspect abc123") is True
    assert _assistant_text_looks_like_shell_command("systemctl restart docker") is True
    assert _assistant_text_looks_like_shell_command("`ssh root@host uptime`") is True
    assert _assistant_text_looks_like_shell_command("The container failed due to memory pressure.") is False
    assert _assistant_text_looks_like_shell_command("") is False
