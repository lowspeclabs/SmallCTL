from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context.frame_compiler import PromptStateFrameCompiler
from smallctl.context.frame_invalidation_filtering import experience_invalidated_with_reason
from smallctl.fama.capsules import render_fama_capsules
from smallctl.fama.detectors import (
    detect_interactive_installer_stall,
    detect_weak_verifier_logic,
    detect_identical_tool_loop,
)
from smallctl.fama.signals import FamaFailureKind
from smallctl.harness.tool_result_verification import _store_verifier_verdict
from smallctl.models.tool_result import ToolEnvelope
from smallctl.recovery_schema import Subtask, SubtaskLedger
from smallctl.state import (
    ArtifactRecord,
    ContextBrief,
    EpisodicSummary,
    EvidenceRecord,
    ExperienceMemory,
    FailureEvent,
    LoopState,
    TurnBundle,
)


def _make_state(*, tool_history: list[str] | None = None, step_count: int = 10) -> LoopState:
    state = LoopState()
    state.step_count = step_count
    if tool_history:
        state.tool_history = list(tool_history)
    return state


def test_interactive_installer_stall_detected() -> None:
    """Phase 4: Detect repeated ssh_session_read with same prompt after send."""
    state = _make_state(
        tool_history=[
            "ssh_session_send|sess_abc|y",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_send|sess_abc|y",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_send|sess_abc|y",
            "ssh_session_read|sess_abc|prompt1|waiting",
        ]
    )
    signal = detect_interactive_installer_stall(state, threshold=2)
    assert signal is not None
    assert signal.kind == FamaFailureKind.INTERACTIVE_SESSION_STALL
    assert "same prompt" in signal.evidence.lower()


def test_interactive_installer_stall_not_detected_without_send() -> None:
    """Phase 4: No stall if no sends between reads."""
    state = _make_state(
        tool_history=[
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
        ]
    )
    signal = detect_interactive_installer_stall(state, threshold=2)
    assert signal is None


def test_weak_verifier_logic_detected() -> None:
    """Phase 4: Detect verifier pass when ssh_exec output shows interactive prompt."""
    state = _make_state()
    state.scratchpad["_last_verifier_verdict"] = {"verdict": "pass"}
    result = SimpleNamespace(
        success=True,
        metadata={"command": "apt-get install webmin"},
        output={"stdout": "Setting up...\n(y/N) ? ", "stderr": ""},
        error="",
    )
    signal = detect_weak_verifier_logic(state, tool_name="ssh_exec", result=result)
    assert signal is not None
    assert signal.kind == FamaFailureKind.EARLY_STOP
    assert "interactive prompt" in signal.evidence.lower()


def test_weak_verifier_logic_not_detected_when_verifier_fails() -> None:
    """Phase 4: No signal when verifier already failed."""
    state = _make_state()
    state.scratchpad["_last_verifier_verdict"] = {"verdict": "fail"}
    result = SimpleNamespace(
        success=True,
        metadata={"command": "apt-get install webmin"},
        output={"stdout": "Setting up...\n(y/N) ? ", "stderr": ""},
        error="",
    )
    signal = detect_weak_verifier_logic(state, tool_name="ssh_exec", result=result)
    assert signal is None


def test_identical_tool_loop_detected() -> None:
    """Phase 4: Detect same tool with same arguments called repeatedly."""
    state = _make_state(
        tool_history=[
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt1|waiting",
        ]
    )
    # Call 3 times to reach threshold
    _ = detect_identical_tool_loop(state, threshold=3)
    _ = detect_identical_tool_loop(state, threshold=3)
    signal = detect_identical_tool_loop(state, threshold=3)
    assert signal is not None
    assert signal.kind == FamaFailureKind.LOOPING
    assert "ssh_session_read" in signal.evidence


def test_identical_tool_loop_not_detected_with_variation() -> None:
    """Phase 4: No loop signal when arguments differ."""
    state = _make_state(
        tool_history=[
            "ssh_session_read|sess_abc|prompt1|waiting",
            "ssh_session_read|sess_abc|prompt2|waiting",
            "ssh_session_read|sess_abc|prompt3|waiting",
        ]
    )
    signal = detect_identical_tool_loop(state, threshold=3)
    assert signal is None


# --- Phase 1: Verifier Hardening ---

def test_verifier_fails_ssh_exec_with_interactive_prompt() -> None:
    """Phase 1: An ssh_exec installer command that returns a (y/N) prompt must fail verification."""
    state = LoopState()
    state.run_brief.original_task = "Install Webmin on the remote host."
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "Setting up webmin...\n(y/N) ? ",
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.162", "command": "apt-get install -y webmin"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert "interactive" in str(verdict.get("failure_mode", "")).lower()


def test_verifier_passes_install_followed_by_verification_command() -> None:
    """Phase 1: A successful install followed by dpkg -l verification passes."""
    state = LoopState()
    state.run_brief.original_task = "Install Webmin on the remote host."
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "ii  webmin  2.000-1  all  web-based administration interface\n",
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.162", "command": "dpkg -l | grep -w webmin"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"


def test_non_install_ssh_tasks_unaffected_by_interactive_prompt_check() -> None:
    """Phase 1: Non-install SSH commands are not blocked by interactive prompt detection."""
    state = LoopState()
    state.run_brief.original_task = "Check disk space on remote host."
    result = ToolEnvelope(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       20G   10G   10G  50% /\n",
            "stderr": "",
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"host": "192.168.1.162", "command": "df -h /"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "pass"


# --- Phase 3: Fine-Grained Context Invalidation ---

def test_repair_phase_retains_web_fetch_and_ssh_session_start_memories() -> None:
    """Phase 3: After a failed ssh_exec, repair phase keeps non-failing memories."""
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

    # Use phase_advanced invalidation with command in details to pass severity check
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
    """Phase 3: Artifact snippets referenced in task goal survive phase transitions."""
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

    event = state.invalidate_context(
        reason="phase_advanced",
        details={"from_phase": "execute", "to_phase": "repair"},
    )

    stale_art = set(state.scratchpad.get("_artifact_staleness", {}).keys())
    assert "A0003" not in stale_art


# --- Phase 4: FAMA Capsules ---

def test_fama_renders_interactive_installer_stall_capsule_by_turn_10() -> None:
    """Phase 4: At least one FAMA capsule appears in the prompt after stall detection."""
    from smallctl.fama.state import activate_mitigations
    from smallctl.fama.signals import ActiveMitigation

    state = LoopState()
    state.step_count = 10
    state.scratchpad["_fama_config"] = {"enabled": True, "mode": "lite", "capsule_token_budget": 180}
    state.tool_history = [
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
        "ssh_session_send|sess_abc|y",
        "ssh_session_read|sess_abc|Continue? (y/N)|waiting",
    ]

    signal = detect_interactive_installer_stall(state, threshold=2)
    assert signal is not None

    # Register the capsule as an active mitigation (simulating runtime behaviour)
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
    combined = "\n".join(capsules)
    assert "interactive" in combined.lower() or "stall" in combined.lower() or "prompt" in combined.lower()


def test_loop_guard_context_aware_nudge_not_generic() -> None:
    """Phase 4: Loop guard for repeated ssh_session_read gives context-aware advice, not generic text."""
    state = LoopState()
    state.step_count = 10
    state.tool_history = [
        "ssh_session_read|sess_abc|prompt1|waiting",
        "ssh_session_read|sess_abc|prompt1|waiting",
        "ssh_session_read|sess_abc|prompt1|waiting",
    ]

    # Call 3 times to reach threshold
    _ = detect_identical_tool_loop(state, threshold=3)
    _ = detect_identical_tool_loop(state, threshold=3)
    signal = detect_identical_tool_loop(state, threshold=3)
    assert signal is not None
    assert signal.kind == FamaFailureKind.LOOPING
    assert "explain the blocker" in signal.next_safe_action.lower() or "try a different" in signal.next_safe_action.lower()
    assert "do not call it again" in signal.next_safe_action.lower()


# --- Phase 7: Trace Replay ---

def test_3a60dd3d_trace_replay_simulated() -> None:
    """Phase 7: Replay anonymized session 3a60dd3d events and verify all phases."""
    # Turn 1-3: Explore phase - web_fetch of install guide
    state = LoopState(cwd="/tmp")
    state.step_count = 1
    state.current_phase = "explore"
    state.run_brief.original_task = "Install Webmin on the remote host."

    # Simulate web_fetch success (web_fetch does not use _store_verifier_verdict)
    state.reasoning_graph.evidence_records.append(
        EvidenceRecord(
            evidence_id="E-webfetch",
            statement="Fetched Webmin install guide from webmin.com",
            phase="explore",
            tool_name="web_fetch",
            metadata={"url": "https://webmin.com/download/"},
        )
    )

    # Turn 4: ssh_session_start to prepare for interactive installer
    state.step_count = 4
    state.current_phase = "execute"
    state.tool_history.append("ssh_session_start|192.168.1.162|root|installer")

    # Turn 5: ssh_exec that hits interactive prompt
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

    # Verify Phase 1: installer step does not pass verification
    assert install_verdict is not None
    assert install_verdict["verdict"] == "fail"
    assert "interactive" in str(install_verdict.get("failure_mode", "")).lower()

    # Simulate Phase 3 transition: execute -> repair
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

    # Use phase_advanced invalidation with command in details to pass severity check
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

    # Verify Phase 3: only failing tool memories are invalidated
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

    # Verify Phase 4: FAMA detects stall if model repeats ssh_session_read
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

    # Verify Phase 4: FAMA capsule appears
    from smallctl.fama.state import activate_mitigations
    from smallctl.fama.signals import ActiveMitigation
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

    # Verify Phase 4: loop guard is context-aware
    # Need last 3 tool_history entries to be identical for the detector to fire
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

    # Verify Phase 6: blocked subtask surfaces blockers
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

    # Check that blocked subtask directive is rendered (via frame compiler)
    frame = PromptStateFrameCompiler().compile(state=state)
    wm_text = frame.spine.working_memory_text
    assert "blocked" in wm_text.lower() or "blocker" in wm_text.lower()
