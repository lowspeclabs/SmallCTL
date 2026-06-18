"""
Test for interrupt approval handling fixes.

This test verifies that the fixes for the session f75c1c57 failure are working correctly.
The original failure occurred when a user responded "yes" to a plan approval prompt,
but the system misclassified it as a new chat task, causing catastrophic context loss.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

from smallctl.harness.task_classifier import (
    classify_runtime_intent,
    runtime_policy_for_intent,
)
from smallctl.harness.runtime_facade import (
    get_pending_interrupt,
    has_pending_interrupt,
    run_auto_with_events,
    run_task_with_events,
)
from smallctl.harness.run_mode import (
    ModeDecisionService,
    _has_plan_execution_approval_context,
    has_active_remote_handoff,
    is_contextual_affirmative_execution_continuation,
)
from smallctl.harness.task_boundary import TaskBoundaryService
from smallctl.interrupt_replies import is_interrupt_response, interrupt_response_action
from smallctl.state_schema import ExecutionPlan
from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.lifecycle_nodes import resume_planning_run
from smallctl.graph.state import GraphRunState


class TestInterruptApprovalFixes:
    """Test suite for verifying interrupt approval handling fixes."""

    def test_ask_human_accepts_freeform_password_reply(self):
        """Free-form ask_human replies should resume the interrupt, not replace the task."""
        interrupt = {
            "kind": "ask_human",
            "question": "What is the SSH password for root@192.168.1.89?",
            "thread_id": "bcd21692",
        }

        assert is_interrupt_response(interrupt, "Temp@Pass")
        assert interrupt_response_action(interrupt, "Temp@Pass") == "answer"

    @pytest.mark.asyncio
    async def test_fix_1_mode_decision_pending_interrupt_check(self):
        """Test Fix #1: ModeDecisionService checks pending interrupts first."""
        # Setup mock harness with pending interrupt
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        mock_harness.client = Mock()
        mock_harness.client.model = "test-model"
        mock_harness.state.cwd = "/test"
        mock_harness._runlog = Mock()
        mock_harness._emit = Mock()
        
        # Create ModeDecisionService
        service = ModeDecisionService(mock_harness)
        
        # Test "yes" response with pending interrupt
        result = await service.decide("yes")
        
        # Should return "loop" mode, not "chat"
        assert result == "loop"
        # Should log the interrupt-based decision
        mock_harness._runlog.assert_called_with(
            "mode_decision",
            "selected run mode",
            mode="loop",
            raw="pending_interrupt_response",
            interrupt_kind="plan_execute_approval",
            raw_task="yes",
        )

    def test_fix_2_task_boundary_preserves_interrupt_context(self):
        """Test Fix #2: TaskBoundaryService preserves context when pending interrupt exists."""
        # Setup mock harness with pending interrupt
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        mock_harness.state.run_brief = Mock()
        mock_harness.state.run_brief.original_task = "SSH operations on 192.168.1.89"
        mock_harness.state.scratchpad = {}
        
        # Create TaskBoundaryService
        service = TaskBoundaryService(mock_harness)
        
        # Mock methods that should NOT be called due to pending interrupt
        service.store_task_handoff = Mock()
        service.finalize_task_scope = Mock()
        service.reset_task_boundary_state = Mock()
        
        # Call maybe_reset_for_new_task with "yes"
        service.maybe_reset_for_new_task("yes")
        
        # Should NOT reset context when pending interrupt exists
        service.store_task_handoff.assert_not_called()
        service.finalize_task_scope.assert_not_called()
        service.reset_task_boundary_state.assert_not_called()

    def test_begin_task_scope_reuses_active_scope_for_interrupt_reply(self):
        """A plan approval reply should not create a replacement task scope."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }

        current_scope = {
            "task_id": "task-0001",
            "raw_task": "ssh root@192.168.1.89 and edit /var/www/demo-site/index.html",
            "effective_task": "ssh root@192.168.1.89 and edit /var/www/demo-site/index.html",
        }
        service = TaskBoundaryService(mock_harness)
        service._consume_session_restored_flag = Mock()
        service._active_task_scope_payload = Mock(return_value=current_scope)
        service.finalize_task_scope = Mock()

        result = service.begin_task_scope(
            raw_task="yes",
            effective_task="Continue remote task over SSH on root@192.168.1.89. User follow-up: proceed",
        )

        assert result == current_scope
        service.finalize_task_scope.assert_not_called()

    def test_task_boundary_does_not_preserve_unrelated_task_with_pending_interrupt(self):
        """Pending interrupts only preserve context for recognized interrupt replies."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        mock_harness.state.run_brief = Mock()
        mock_harness.state.run_brief.original_task = "SSH operations on 192.168.1.89"
        mock_harness.state.scratchpad = {}

        service = TaskBoundaryService(mock_harness)
        service.has_task_local_context = Mock(return_value=True)
        service._consume_session_restored_flag = Mock(return_value=False)
        service._is_remote_correction_followup = Mock(return_value=False)
        service._is_same_scope_transition = Mock(return_value=False)
        service.store_task_handoff = Mock()
        service.finalize_task_scope = Mock()
        service.reset_task_boundary_state = Mock()

        service.maybe_reset_for_new_task("write a local README")

        service.store_task_handoff.assert_called_once()
        service.finalize_task_scope.assert_called_once()
        service.reset_task_boundary_state.assert_called_once()

    def test_fix_3_interrupt_aware_intent_classification(self):
        """Test Fix #3: classify_runtime_intent considers pending interrupts."""
        # Test without pending interrupt - "yes" should be classified as chat/smalltalk
        intent_without_interrupt = classify_runtime_intent(
            "yes",
            recent_messages=[],
            pending_interrupt=None,
        )
        assert intent_without_interrupt.label in ["chat_only", "smalltalk"]
        assert intent_without_interrupt.task_mode == "chat"
        
        # Test with pending interrupt - "yes" should be classified as interrupt continuation
        intent_with_interrupt = classify_runtime_intent(
            "yes",
            recent_messages=[],
            pending_interrupt={
                "kind": "plan_execute_approval",
                "question": "Plan ready. Execute it now?",
                "response_mode": "yes/no/revise",
            },
        )
        assert intent_with_interrupt.label == "interrupt_continuation"
        assert intent_with_interrupt.task_mode == "loop"
        
        # Verify runtime policy for interrupt_continuation
        policy = runtime_policy_for_intent(intent_with_interrupt)
        assert policy.route_mode == "loop"
        assert policy.chat_requires_tools is True

    def test_fix_4_context_affirmative_with_pending_interrupt(self):
        """Test Fix #4: is_contextual_affirmative_execution_continuation checks pending interrupts."""
        # Setup mock harness with pending interrupt
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        mock_harness.state.recent_messages = []
        mock_harness.state.scratchpad = {}
        
        # Test "yes" with pending interrupt - should return True
        result = is_contextual_affirmative_execution_continuation(
            mock_harness,
            raw_task="yes",
            resolved_task="yes",
        )
        assert result is True
        
        # Test "no" with pending interrupt - should return False (not affirmative)
        result = is_contextual_affirmative_execution_continuation(
            mock_harness,
            raw_task="no",
            resolved_task="no",
        )
        assert result is False

    def test_context_affirmative_uses_remote_plan_handoff_if_pending_interrupt_was_lost(self):
        """Plan approval recovery should not depend only on live pending_interrupt state."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = None
        mock_harness.state.planner_interrupt = None
        mock_harness.state.active_plan = ExecutionPlan(
            plan_id="plan-remote",
            goal="ssh root@192.168.1.89 and edit /var/www/demo-site/index.html",
            status="awaiting_approval",
        )
        mock_harness.state.draft_plan = None
        mock_harness.state.recent_messages = [
            Mock(role="user", content="yes", metadata={"resumed_from_interrupt": True})
        ]
        mock_harness.state.scratchpad = {
            "_last_task_handoff": {
                "task_mode": "remote_execute",
                "ssh_target": {"host": "192.168.1.89", "user": "root"},
            }
        }

        result = is_contextual_affirmative_execution_continuation(
            mock_harness,
            raw_task="yes",
            resolved_task="yes",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_fix_approved_proceed_preserves_remote_handoff_without_live_interrupt(self):
        """Blended approval phrases should continue the active SSH task, not fall into chat."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = None
        mock_harness.state.planner_interrupt = None
        mock_harness.state.active_plan = None
        mock_harness.state.draft_plan = None
        mock_harness.state.planning_mode_enabled = False
        mock_harness.state.recent_messages = []
        mock_harness.state.cwd = "/home/stephen/Scripts/Harness-Redo"
        mock_harness.state.task_mode = ""
        mock_harness.state.active_tool_profiles = ["core"]
        mock_harness.state.scratchpad = {
            "_last_task_handoff": {
                "task_mode": "remote_execute",
                "ssh_target": {"host": "192.168.1.89", "user": "root"},
            }
        }
        mock_harness.client = Mock()
        mock_harness.client.model = "test-model"
        mock_harness._runlog = Mock()
        mock_harness._emit = Mock()

        assert has_active_remote_handoff(mock_harness) is True
        assert is_contextual_affirmative_execution_continuation(
            mock_harness,
            raw_task="fix approved proceed",
            resolved_task="fix approved proceed",
        ) is True

        mode = await ModeDecisionService(mock_harness).decide("fix approved proceed")

        assert mode == "loop"
        assert "network" in mock_harness.state.active_tool_profiles

    @pytest.mark.asyncio
    async def test_original_scenario_prevented(self):
        """Test that the original failure scenario is now prevented."""
        # This test simulates the exact sequence from session f75c1c57
        
        # Setup mock harness in the state it would be after Task 2 completion
        mock_harness = Mock()
        mock_harness.state = Mock()
        
        # Simulate pending plan approval interrupt (what should exist after plan_set)
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        
        # Mock other required attributes
        mock_harness.state.recent_messages = []
        mock_harness.state.run_brief = Mock()
        mock_harness.state.run_brief.original_task = "Remove Google branding from /var/www/demo-site/index.html on 192.168.1.89"
        mock_harness.state.scratchpad = {
            "_last_task_handoff": {
                "task_mode": "remote_execute",
                "ssh_target": {"host": "192.168.1.89", "user": "root"},
            }
        }
        mock_harness.client = Mock()
        mock_harness.client.model = "test-model"
        mock_harness.state.cwd = "/home/stephen/Scripts/Harness-Redo"
        mock_harness._runlog = Mock()
        mock_harness._emit = Mock()
        
        # Create services
        mode_service = ModeDecisionService(mock_harness)
        boundary_service = TaskBoundaryService(mock_harness)
        
        # Mock boundary service methods to track if they're called
        boundary_service.store_task_handoff = Mock()
        boundary_service.finalize_task_scope = Mock()
        boundary_service.reset_task_boundary_state = Mock()
        
        # Step 1: User responds with "yes" (the original failure point)
        mode_result = await mode_service.decide("yes")
        
        # Should NOT return "chat" mode (which caused the original failure)
        assert mode_result != "chat", "ModeDecisionService should not return 'chat' for 'yes' with pending interrupt"
        assert mode_result == "loop", "ModeDecisionService should return 'loop' to maintain continuity"
        
        # Step 2: Task boundary should NOT reset context
        boundary_service.maybe_reset_for_new_task("yes")
        
        # Should NOT call reset methods (which caused context loss)
        boundary_service.store_task_handoff.assert_not_called()
        boundary_service.finalize_task_scope.assert_not_called()
        boundary_service.reset_task_boundary_state.assert_not_called()
        
        # Step 3: Intent classification should recognize interrupt continuation
        intent = classify_runtime_intent(
            "yes",
            recent_messages=[],
            pending_interrupt=mock_harness.state.pending_interrupt,
        )
        assert intent.label == "interrupt_continuation"
        assert intent.task_mode == "loop"
        
        # Step 4: Context affirmative detection should return True
        is_continuation = is_contextual_affirmative_execution_continuation(
            mock_harness,
            raw_task="yes",
            resolved_task="yes",
        )
        assert is_continuation is True

    @pytest.mark.asyncio
    async def test_facade_resumes_plan_interrupt_before_runtime_start(self):
        """run_* entrypoints should resume valid interrupt replies before normal initialization."""
        mock_harness = Mock()
        mock_harness.get_pending_interrupt.return_value = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        expected = {"status": "plan_approved"}

        async def fake_resume(human_input, event_handler=None):
            assert human_input == "yes"
            assert event_handler == "handler"
            return expected

        mock_harness.resume_task_with_events = fake_resume

        assert await run_auto_with_events(mock_harness, "yes", event_handler="handler") == expected
        assert await run_task_with_events(mock_harness, "yes", event_handler="handler") == expected

    @pytest.mark.asyncio
    async def test_resume_planning_run_persists_playbook_via_node_support(self, monkeypatch):
        """Approving a plan should persist the playbook without a missing-symbol crash."""
        plan = ExecutionPlan(plan_id="plan-1", goal="inspect")
        state = Mock()
        state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "plan_id": "plan-1",
            "response_mode": "yes/no/revise",
        }
        state.active_plan = plan
        state.draft_plan = plan
        state.planning_mode_enabled = True
        state.current_phase = "explore"
        state.planner_resume_target_mode = "loop"
        state.append_message = Mock()
        state.sync_plan_mirror = Mock()
        state.touch = Mock()

        harness = Mock()
        harness.state = state
        harness.get_pending_interrupt.return_value = state.pending_interrupt
        harness._log_conversation_state = Mock()
        harness._emit = AsyncMock()

        persisted: list[ExecutionPlan] = []
        monkeypatch.setattr(
            "smallctl.graph.lifecycle_nodes._nodes._persist_planning_playbook",
            lambda _harness, persisted_plan: persisted.append(persisted_plan),
        )

        graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="planning")
        await resume_planning_run(
            graph_state,
            GraphRuntimeDeps(harness=harness, event_handler=None),
            human_input="approve",
        )

        assert graph_state.final_result["status"] == "plan_approved"
        assert plan.approved is True
        assert plan.status == "approved"
        assert persisted == [plan]

    @pytest.mark.asyncio
    async def test_facade_does_not_resume_unrelated_reply(self):
        """A new task can still replace a pending interrupt when it is not a valid reply."""
        mock_harness = Mock()
        mock_harness.get_pending_interrupt.return_value = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        mock_harness.resume_task_with_events = Mock()
        mock_harness.event_handler = None
        mock_harness._indexer = False

        class Runtime:
            @classmethod
            def from_harness(cls, harness, *, event_handler=None):
                assert harness is mock_harness
                assert event_handler == "handler"
                return cls()

            async def run(self, task):
                assert task == "write a local README"
                return {"status": "normal_runtime"}

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("smallctl.graph.runtime.LoopGraphRuntime", Runtime)
            result = await run_task_with_events(
                mock_harness,
                "write a local README",
                event_handler="handler",
            )

        assert result == {"status": "normal_runtime"}
        mock_harness.resume_task_with_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_various_approval_responses(self):
        """Test that various approval responses are handled correctly."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        mock_harness.client = Mock()
        mock_harness.client.model = "test-model"
        mock_harness.state.cwd = "/test"
        mock_harness._runlog = Mock()
        mock_harness._emit = Mock()
        
        service = ModeDecisionService(mock_harness)
        
        # Test various affirmative responses
        affirmative_responses = [
            "yes",
            "y",
            "approve",
            "approved",
            "execute",
            "go ahead",
            "run it",
            "proceed",
            "continue",
            "resume",
        ]
        for response in affirmative_responses:
            result = await service.decide(response)
            assert result == "loop", f"Response '{response}' should trigger loop mode with pending interrupt"

    def test_negative_response_with_interrupt(self):
        """Test that negative responses don't trigger continuation."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "response_mode": "yes/no/revise",
        }
        mock_harness.state.recent_messages = []
        mock_harness.state.scratchpad = {}
        
        # Test "no" response - should not be treated as continuation
        result = is_contextual_affirmative_execution_continuation(
            mock_harness,
            raw_task="no",
            resolved_task="no",
        )
        assert result is False
        
        # Test "revise" response - should not be treated as continuation
        result = is_contextual_affirmative_execution_continuation(
            mock_harness,
            raw_task="revise",
            resolved_task="revise",
        )
        assert result is False

    def test_approved_plan_ignores_stale_planner_interrupt(self):
        """A leftover planner_interrupt must not keep an approved plan waiting forever."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = None
        mock_harness.state.active_plan = ExecutionPlan(
            plan_id="plan-approved",
            goal="do the thing",
            status="approved",
            approved=True,
        )
        mock_harness.state.draft_plan = None
        mock_harness.state.planner_interrupt = Mock(
            kind="plan_execute_approval",
            question="Plan ready. Execute it now?",
            plan_id="plan-approved",
            approved=False,
            response_mode="yes/no/revise",
        )

        assert has_pending_interrupt(mock_harness) is False
        assert get_pending_interrupt(mock_harness) is None
        assert _has_plan_execution_approval_context(mock_harness) is False

    def test_approved_plan_ignores_stale_pending_interrupt(self):
        """A serialized pending approval should be stale once the same plan is approved."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = {
            "kind": "plan_execute_approval",
            "question": "Plan ready. Execute it now?",
            "plan_id": "plan-approved",
            "approved": False,
            "response_mode": "yes/no/revise",
        }
        mock_harness.state.active_plan = ExecutionPlan(
            plan_id="plan-approved",
            goal="do the thing",
            status="approved",
            approved=True,
        )
        mock_harness.state.draft_plan = None
        mock_harness.state.planner_interrupt = None

        assert has_pending_interrupt(mock_harness) is False
        assert get_pending_interrupt(mock_harness) is None
        assert _has_plan_execution_approval_context(mock_harness) is False

    def test_unapproved_plan_still_exposes_planner_interrupt(self):
        """The stale guard should not hide a real unapproved plan approval prompt."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = None
        mock_harness.state.active_plan = ExecutionPlan(
            plan_id="plan-draft",
            goal="do the thing",
            status="draft",
            approved=False,
        )
        mock_harness.state.draft_plan = None
        mock_harness.state.planner_interrupt = Mock(
            kind="plan_execute_approval",
            question="Plan ready. Execute it now?",
            plan_id="plan-draft",
            approved=False,
            response_mode="yes/no/revise",
        )

        assert has_pending_interrupt(mock_harness) is True
        assert get_pending_interrupt(mock_harness)["plan_id"] == "plan-draft"
        assert _has_plan_execution_approval_context(mock_harness) is True

    @pytest.mark.asyncio
    async def test_mode_decision_does_not_use_plan_approval_fallback_for_approved_plan(self):
        """Session 4715ce37 regression: stale planner_interrupt is not approval context."""
        mock_harness = Mock()
        mock_harness.state = Mock()
        mock_harness.state.pending_interrupt = None
        mock_harness.state.active_plan = ExecutionPlan(
            plan_id="plan-e892d6fa",
            goal="convert pong",
            status="approved",
            approved=True,
        )
        mock_harness.state.draft_plan = None
        mock_harness.state.planner_interrupt = Mock(
            kind="plan_execute_approval",
            question="Plan ready. Execute it now?",
            plan_id="plan-e892d6fa",
            approved=False,
            response_mode="yes/no/revise",
        )
        mock_harness.state.recent_messages = []
        mock_harness.state.run_brief = Mock()
        mock_harness.state.run_brief.original_task = "read the script ./temp/pong.py update the script"
        mock_harness.state.scratchpad = {}
        mock_harness.state.cwd = "/home/stephen/Scripts/Harness-Redo"
        mock_harness.state.task_mode = ""
        mock_harness.state.active_tool_profiles = ["core"]
        mock_harness.client = Mock()
        mock_harness.client.model = "test-model"
        mock_harness._runlog = Mock()
        mock_harness._emit = Mock()

        mode = await ModeDecisionService(mock_harness).decide("approve")

        assert _has_plan_execution_approval_context(mock_harness) is False
        assert not any(
            kwargs.get("raw") == "plan_approval_fallback"
            for _args, kwargs in mock_harness._runlog.call_args_list
        )

    def test_vague_planning_prose_does_not_synthesize_empty_plan(self):
        """Approval prompts require an extractable plan, not just text mentioning a plan."""
        from smallctl.graph.planning_support import synthesize_plan_from_text
        from types import SimpleNamespace

        harness = SimpleNamespace(
            state=SimpleNamespace(
                run_brief=SimpleNamespace(original_task="Build a script"),
            )
        )

        text = (
            "Let's first explore the workspace to understand the context, then I will propose a plan.\n\n"
            "I'll explore the project structure, then list related files."
        )

        assert synthesize_plan_from_text(harness, text) is None


    def test_task_summary_postmortem_includes_interrupt_question(self, tmp_path) -> None:
        """task_summary.json for needs_human should include the interrupt question, not 'No reason provided'."""
        from smallctl.harness.core_facade import _finalize
        from types import SimpleNamespace

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        class MockHarness:
            def __init__(self):
                self.run_logger = SimpleNamespace(run_dir=run_dir)
                self.state = SimpleNamespace(
                    step_count=3,
                    inactive_steps=0,
                    token_usage={},
                    recent_errors=[],
                )
                self.checkpoint_on_exit = False
                self._cancel_requested = False
                self._active_dispatch_task = None

            def _finalize_task_scope(self, **kwargs):
                return None

            def _record_terminal_experience(self, result):
                pass

            def _rewrite_active_plan_export(self):
                pass

            def _persist_checkpoint(self, result):
                pass

            def _runlog(self, *args, **kwargs):
                pass

        harness = MockHarness()
        result = {
            "status": "needs_human",
            "message": "Plan ready. Execute it now?",
            "interrupt": {
                "kind": "plan_execute_approval",
                "question": "Plan ready. Execute it now?",
                "plan_id": "plan-test",
            },
        }
        _finalize(harness, result)

        summary_path = run_dir / "task_summary.json"
        assert summary_path.exists()
        import json
        payload = json.loads(summary_path.read_text())
        assert payload["postmortem_summary"] == "Plan ready. Execute it now?"

    def test_task_summary_postmortem_includes_message_dict_question(self, tmp_path) -> None:
        """task_summary.json should derive postmortem from message dict question when interrupt is absent."""
        from smallctl.harness.core_facade import _finalize
        from types import SimpleNamespace

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        class MockHarness:
            def __init__(self):
                self.run_logger = SimpleNamespace(run_dir=run_dir)
                self.state = SimpleNamespace(
                    step_count=1,
                    inactive_steps=0,
                    token_usage={},
                    recent_errors=[],
                )
                self.checkpoint_on_exit = False
                self._cancel_requested = False
                self._active_dispatch_task = None

            def _finalize_task_scope(self, **kwargs):
                return None

            def _record_terminal_experience(self, result):
                pass

            def _rewrite_active_plan_export(self):
                pass

            def _persist_checkpoint(self, result):
                pass

            def _runlog(self, *args, **kwargs):
                pass

        harness = MockHarness()
        result = {
            "status": "needs_human",
            "message": {"question": "Approve this change?", "status": "pending"},
        }
        _finalize(harness, result)

        summary_path = run_dir / "task_summary.json"
        import json
        payload = json.loads(summary_path.read_text())
        assert payload["postmortem_summary"] == "Approve this change?"

    def test_session_summary_distinguishes_prior_and_latest_cancelled_unverified(self, tmp_path) -> None:
        from smallctl.harness.core_facade import _finalize
        from types import SimpleNamespace
        import json

        run_dir = tmp_path / "run"
        prior_dir = run_dir / "tasks" / "task-0001"
        latest_dir = run_dir / "tasks" / "task-0002"
        latest_dir.mkdir(parents=True)
        prior_dir.mkdir(parents=True)
        (prior_dir / "task_summary.json").write_text(
            json.dumps({"status": "completed", "terminal_event": "task_completed"}),
            encoding="utf-8",
        )
        (latest_dir / "task_summary.json").write_text(
            json.dumps({"status": "interrupted", "terminal_event": "task_interrupted"}),
            encoding="utf-8",
        )

        class MockHarness:
            def __init__(self):
                self.run_logger = SimpleNamespace(run_dir=run_dir)
                self.state = SimpleNamespace(
                    step_count=3,
                    inactive_steps=0,
                    token_usage={},
                    recent_errors=[],
                    challenge_progress=SimpleNamespace(
                        task_category="coding",
                        challenge_id="",
                        required_output_paths=[],
                        phase="implement",
                        code_change_count=1,
                        last_code_change_step=2,
                        last_code_change_paths=["temp/example.py"],
                        last_verifier_artifact_paths=[],
                        last_verifier_step=0,
                        last_verifier_command="",
                        last_verifier_kind="",
                        last_verifier_verdict="",
                        verified_after_last_change=False,
                        redundant_verifier_count=0,
                        post_pass_nonterminal_steps=0,
                        no_change_steps_after_write=0,
                        successful_artifact_write_step=None,
                        first_post_write_verification_step=None,
                        nonterminal_steps_after_verified_write=0,
                    ),
                    run_brief=SimpleNamespace(original_task="edit ./temp/example.py"),
                    working_memory=SimpleNamespace(current_goal=""),
                )
                self.checkpoint_on_exit = False
                self._cancel_requested = False
                self._active_dispatch_task = None

            def _finalize_task_scope(self, **kwargs):
                return {"task_id": "task-0002", "summary_path": str(latest_dir / "task_summary.json")}

            def _record_terminal_experience(self, result):
                pass

            def _rewrite_active_plan_export(self):
                pass

            def _persist_checkpoint(self, result):
                pass

            def _runlog(self, *args, **kwargs):
                pass

        result = _finalize(MockHarness(), {"status": "cancelled", "reason": "cancel_requested"})

        assert result["unverified_change_warning"] == (
            "Task cancelled after modifying files to temp/example.py. Changes were not verified."
        )
        payload = json.loads((run_dir / "session_summary.json").read_text(encoding="utf-8"))
        assert payload["prior_task_completed"] is True
        assert payload["latest_task_cancelled"] is True
        assert payload["files_changed_after_latest_task_start"] is True
        assert payload["verification_after_latest_change"] is False

    def test_session_summary_aggregates_task_step_counts_including_current_task(self, tmp_path) -> None:
        from smallctl.harness.core_facade import _finalize
        from types import SimpleNamespace
        import json

        run_dir = tmp_path / "run"
        prior_dir = run_dir / "tasks" / "task-0001"
        latest_dir = run_dir / "tasks" / "task-0002"
        latest_dir.mkdir(parents=True)
        prior_dir.mkdir(parents=True)
        (prior_dir / "task_summary.json").write_text(
            json.dumps({"task_id": "task-0001", "status": "completed", "step_count": 4}),
            encoding="utf-8",
        )
        (latest_dir / "task_summary.json").write_text(
            json.dumps({"task_id": "task-0002", "status": "failed", "step_count": 7, "last_recent_error": "Guard tripped: max_consecutive_errors (5)"}),
            encoding="utf-8",
        )

        class MockHarness:
            def __init__(self):
                self.run_logger = SimpleNamespace(run_dir=run_dir)
                self.state = SimpleNamespace(
                    step_count=7,
                    inactive_steps=0,
                    token_usage={},
                    recent_errors=["Guard tripped: max_consecutive_errors (5)"],
                    scratchpad={},
                    challenge_progress=None,
                    run_brief=SimpleNamespace(original_task="install pihole"),
                    working_memory=SimpleNamespace(current_goal=""),
                )
                self.checkpoint_on_exit = False
                self._cancel_requested = False
                self._active_dispatch_task = None

            def _finalize_task_scope(self, **kwargs):
                return {"task_id": "task-0002", "summary_path": str(latest_dir / "task_summary.json")}

            def _record_terminal_experience(self, result):
                pass

            def _rewrite_active_plan_export(self):
                pass

            def _persist_checkpoint(self, result):
                pass

            def _runlog(self, *args, **kwargs):
                pass

        _finalize(MockHarness(), {"status": "failed", "reason": "Guard tripped: max_consecutive_errors (5)"})

        payload = json.loads((run_dir / "session_summary.json").read_text(encoding="utf-8"))
        assert payload["total_tool_calls"] == 11
        assert payload["guard_trips"] == 1
        assert payload["task_count"] == 2
        assert payload["local_task_status"] == "failed"
        assert payload["final_task_status"] == "failed"
        assert payload["overall_objective_status"] == "incomplete"
        assert payload["incomplete_task_ids"] == ["task-0002"]

    def test_session_summary_keeps_prior_failed_task_in_overall_status(self, tmp_path) -> None:
        from smallctl.harness.core_facade import _finalize
        from types import SimpleNamespace
        import json

        run_dir = tmp_path / "run"
        prior_dir = run_dir / "tasks" / "task-0001"
        latest_dir = run_dir / "tasks" / "task-0002"
        latest_dir.mkdir(parents=True)
        prior_dir.mkdir(parents=True)
        (prior_dir / "task_summary.json").write_text(
            json.dumps({"task_id": "task-0001", "status": "failed", "reason": "Guard tripped: max_consecutive_errors (5)", "step_count": 5}),
            encoding="utf-8",
        )
        (latest_dir / "task_summary.json").write_text(
            json.dumps({"task_id": "task-0002", "status": "completed", "step_count": 1}),
            encoding="utf-8",
        )

        class MockHarness:
            def __init__(self):
                self.run_logger = SimpleNamespace(run_dir=run_dir)
                self.state = SimpleNamespace(
                    step_count=1,
                    inactive_steps=0,
                    token_usage={},
                    recent_errors=[],
                    scratchpad={},
                    challenge_progress=None,
                    run_brief=SimpleNamespace(original_task="status?"),
                    working_memory=SimpleNamespace(current_goal=""),
                )
                self.checkpoint_on_exit = False
                self._cancel_requested = False
                self._active_dispatch_task = None

            def _finalize_task_scope(self, **kwargs):
                return {"task_id": "task-0002", "summary_path": str(latest_dir / "task_summary.json")}

            def _record_terminal_experience(self, result):
                pass

            def _rewrite_active_plan_export(self):
                pass

            def _persist_checkpoint(self, result):
                pass

            def _runlog(self, *args, **kwargs):
                pass

        _finalize(MockHarness(), {"status": "completed", "message": "container is running"})

        payload = json.loads((run_dir / "session_summary.json").read_text(encoding="utf-8"))
        assert payload["local_task_status"] == "completed"
        assert payload["final_task_status"] == "completed"
        assert payload["overall_objective_status"] == "incomplete"
        assert payload["incomplete_task_ids"] == ["task-0001"]
        assert payload["has_incomplete_prior_tasks"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
