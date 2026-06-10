"""Tests for model recovery after guard trip."""

from __future__ import annotations

import pytest

from smallctl.state_schema import BlockedOperation
from smallctl.state import LoopState
from smallctl.prompts import build_system_prompt


class TestGuardRecoveryPrompt:
    def test_guard_recovery_included_when_blocked_operation_set(self):
        state = LoopState()
        state.repair_cycle_id = "rc-1"
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install -y webmin",
            guard_reason="apt_deb822_validation_required",
            recovery_action={
                "tool_name": "ssh_exec",
                "required_arguments": {"command": "python3 -c 'validate deb822'"},
            },
            timestamp="2024-01-01T00:00:00",
        )
        prompt = build_system_prompt(state, phase="repair")
        assert "GUARD BLOCK RECOVERY" in prompt
        assert "apt_deb822_validation_required" in prompt
        assert "Do NOT stop or ask for help" in prompt

    def test_escalation_level_1(self):
        state = LoopState()
        state.repair_cycle_id = "rc-1"
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install foo",
            guard_reason="apt_deb822_validation_required",
            recovery_action={"tool_name": "ssh_exec", "required_arguments": {"command": "validate"}},
        )
        state.guard_trip_count = 1
        prompt = build_system_prompt(state, phase="repair")
        assert "Execute the recovery action provided in the guard message" in prompt

    def test_escalation_level_2(self):
        state = LoopState()
        state.repair_cycle_id = "rc-1"
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install foo",
            guard_reason="apt_deb822_validation_required",
            recovery_action={"tool_name": "ssh_exec", "required_arguments": {"command": "validate"}},
        )
        state.guard_trip_count = 2
        prompt = build_system_prompt(state, phase="repair")
        assert "You have not executed the recovery action" in prompt

    def test_escalation_level_3(self):
        state = LoopState()
        state.repair_cycle_id = "rc-1"
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install foo",
            guard_reason="apt_deb822_validation_required",
            recovery_action={"tool_name": "ssh_exec", "required_arguments": {"command": "validate"}},
        )
        state.guard_trip_count = 3
        prompt = build_system_prompt(state, phase="repair")
        assert "Escalating" in prompt

    def test_no_guard_recovery_when_no_blocked_operation(self):
        state = LoopState()
        prompt = build_system_prompt(state, phase="repair")
        assert "GUARD BLOCK RECOVERY" not in prompt

    def test_recovery_action_command_in_prompt(self):
        state = LoopState()
        state.repair_cycle_id = "rc-1"
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install -y webmin",
            guard_reason="apt_deb822_validation_required",
            recovery_action={
                "tool_name": "ssh_exec",
                "required_arguments": {"command": "python3 -c 'validate deb822 sources'"},
            },
        )
        prompt = build_system_prompt(state, phase="repair")
        assert "Recovery action: Run `ssh_exec`" in prompt
        assert "validate deb822 sources" in prompt


class TestGuardTripCountReset:
    def test_guard_trip_count_reset_on_deb822_clean(self):
        from smallctl.tools.shell_support_apt_and_outcome import _mark_deb822_preflight_clean
        state = LoopState()
        state.guard_trip_count = 5
        _mark_deb822_preflight_clean(state, host="remote", user="root")
        assert state.guard_trip_count == 0

    def test_guard_trip_count_increments_on_guard_fail(self):
        from smallctl.tools.shell_support_apt_and_outcome import _guard_fail
        state = LoopState()
        assert state.guard_trip_count == 0
        _guard_fail(
            "blocked",
            reason="apt_deb822_preflight_required",
            command="apt install foo",
            state=state,
            tool_name="ssh_exec",
        )
        assert state.guard_trip_count == 1
