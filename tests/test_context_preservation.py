"""Tests for guard block context preservation through invalidation and compaction."""

from __future__ import annotations

from smallctl.state_schema import BlockedOperation, AptUpdateResult
from smallctl.state import LoopState
from smallctl.context.frame_invalidation_utils import guard_trip_preserved_ids
from smallctl.context.frame_invalidation_filtering import (
    filter_invalidated_turn_bundles,
    filter_invalidated_observations,
    filter_invalidated_summaries,
    filter_invalidated_artifact_snippets,
)


class TestGuardTripPreservedIds:
    def test_guard_trip_preserved_ids_empty_when_no_guard(self):
        state = LoopState()
        assert guard_trip_preserved_ids(state, "_guard_trip_preserved_observation_ids") == set()

    def test_guard_trip_preserved_ids_returns_set(self):
        state = LoopState()
        state.scratchpad["_guard_trip_preserved_observation_ids"] = ["obs-1", "obs-2"]
        result = guard_trip_preserved_ids(state, "_guard_trip_preserved_observation_ids")
        assert result == {"obs-1", "obs-2"}

    def test_guard_trip_preserved_ids_tolerates_non_list(self):
        state = LoopState()
        state.scratchpad["_guard_trip_preserved_observation_ids"] = "not-a-list"
        result = guard_trip_preserved_ids(state, "_guard_trip_preserved_observation_ids")
        assert result == set()


class TestBlockedOperationSurvivesState:
    def test_blocked_operation_serializes_to_dict(self):
        state = LoopState()
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install -y webmin",
            guard_reason="apt_deb822_validation_required",
            recovery_action={"tool_name": "ssh_exec", "required_arguments": {"command": "validate"}},
            timestamp="2024-01-01T00:00:00",
        )
        data = state.to_dict()
        assert "blocked_operation" in data
        assert data["blocked_operation"]["tool"] == "ssh_exec"
        assert data["blocked_operation"]["guard_reason"] == "apt_deb822_validation_required"

    def test_blocked_operation_deserializes_from_dict(self):
        state = LoopState()
        data = state.to_dict()
        data["blocked_operation"] = {
            "tool": "ssh_exec",
            "command": "apt install foo",
            "guard_reason": "apt_deb822_validation_required",
            "recovery_action": {"tool_name": "ssh_exec"},
            "timestamp": "2024-01-01T00:00:00",
        }
        state2 = LoopState.from_dict(data)
        assert state2.blocked_operation is not None
        assert state2.blocked_operation.tool == "ssh_exec"
        assert state2.blocked_operation.guard_reason == "apt_deb822_validation_required"

    def test_apt_update_results_survive_serialization(self):
        state = LoopState()
        state.apt_update_results["localhost|root"] = AptUpdateResult(
            host="localhost",
            user="root",
            attempted=True,
            succeeded=False,
            error="deb822 format error",
            timestamp="2024-01-01T00:00:00",
        )
        data = state.to_dict()
        assert "apt_update_results" in data
        assert "localhost|root" in data["apt_update_results"]
        assert data["apt_update_results"]["localhost|root"]["succeeded"] is False

    def test_guard_trip_count_survives_serialization(self):
        state = LoopState()
        state.guard_trip_count = 3
        data = state.to_dict()
        assert data.get("guard_trip_count") == 3
        state2 = LoopState.from_dict(data)
        assert state2.guard_trip_count == 3


class TestContextInvalidationPreservesGuard:
    def test_guard_blocked_observation_preserved(self):
        from types import SimpleNamespace
        state = LoopState()
        state.scratchpad["_guard_trip_preserved_observation_ids"] = ["obs-guard-1"]
        observations = [
            SimpleNamespace(observation_id="obs-guard-1", text="guard block context", tool_name="ssh_exec"),
            SimpleNamespace(observation_id="obs-normal-1", text="normal context", tool_name="shell_exec"),
        ]
        kept, dropped, _ = filter_invalidated_observations(state=state, observations=observations)
        kept_ids = {o.observation_id for o in kept}
        assert "obs-guard-1" in kept_ids

    def test_guard_blocked_summary_preserved(self):
        from smallctl.state_schema import EpisodicSummary
        state = LoopState()
        state.scratchpad["_guard_trip_preserved_summary_ids"] = ["sum-guard-1"]
        summaries = [
            EpisodicSummary(summary_id="sum-guard-1", created_at="2024-01-01T00:00:00"),
            EpisodicSummary(summary_id="sum-normal-1", created_at="2024-01-01T00:00:00"),
        ]
        kept, dropped = filter_invalidated_summaries(state=state, summaries=summaries)
        kept_ids = {s.summary_id for s in kept}
        assert "sum-guard-1" in kept_ids

    def test_guard_blocked_artifact_preserved(self):
        from smallctl.state_schema import ArtifactSnippet
        state = LoopState()
        state.scratchpad["_guard_trip_preserved_artifact_ids"] = ["art-guard-1"]
        snippets = [
            ArtifactSnippet(artifact_id="art-guard-1", text="guard artifact"),
            ArtifactSnippet(artifact_id="art-normal-1", text="normal artifact"),
        ]
        kept, dropped = filter_invalidated_artifact_snippets(state=state, snippets=snippets)
        kept_ids = {a.artifact_id for a in kept}
        assert "art-guard-1" in kept_ids

    def test_blocked_operation_cleared_on_successful_apt_update(self):
        from smallctl.tools.shell_support_apt_and_outcome import _mark_deb822_preflight_clean
        state = LoopState()
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install foo",
            guard_reason="apt_deb822_validation_required",
        )
        _mark_deb822_preflight_clean(state, host="remote", user="root")
        assert state.blocked_operation is None

    def test_blocked_operation_not_cleared_on_unrelated_success(self):
        from smallctl.tools.shell_support_apt_and_outcome import _mark_deb822_preflight_clean
        state = LoopState()
        state.blocked_operation = BlockedOperation(
            tool="shell_exec",
            command="rm -rf /",
            guard_reason="risk_policy_blocked",
        )
        _mark_deb822_preflight_clean(state, host="remote", user="root")
        # Should clear guard_trip_count but not blocked_operation since reason doesn't start with apt_deb822
        assert state.blocked_operation is not None
        assert state.blocked_operation.guard_reason == "risk_policy_blocked"
