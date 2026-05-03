"""
Test for interrupt approval handling fixes.

This test verifies that the fixes for the session f75c1c57 failure are working correctly.
The original failure occurred when a user responded "yes" to a plan approval prompt,
but the system misclassified it as a new chat task, causing catastrophic context loss.
"""

import pytest
import asyncio
from unittest.mock import Mock

from smallctl.harness.task_classifier import (
    classify_runtime_intent,
    runtime_policy_for_intent,
)
from smallctl.harness.runtime_facade import run_auto_with_events, run_task_with_events
from smallctl.harness.run_mode import (
    ModeDecisionService,
    is_contextual_affirmative_execution_continuation,
)
from smallctl.harness.task_boundary import TaskBoundaryService
from smallctl.state_schema import ExecutionPlan


class TestInterruptApprovalFixes:
    """Test suite for verifying interrupt approval handling fixes."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
