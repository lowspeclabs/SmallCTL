"""Tests for reactive APT deb822 guard behavior in shell_support_apt_and_outcome.py."""

from __future__ import annotations

import pytest

from smallctl.state_schema import AptUpdateResult, BlockedOperation
from smallctl.state import LoopState
from smallctl.tools.shell_support_apt_and_outcome import (
    _apt_deb822_preflight_guard,
    _guard_fail,
    _mark_deb822_preflight_clean,
    _is_deb822_preflight_clean,
    validate_sources_file,
)


class TestValidateSourcesFile:
    def test_valid_deb822_content(self):
        content = (
            "Types: deb\n"
            "URIs: http://deb.debian.org/debian\n"
            "Suites: stable\n"
            "Components: main\n"
        )
        result = validate_sources_file(content)
        assert result["valid"] is True

    def test_missing_fields(self):
        content = "Types: deb\n"
        result = validate_sources_file(content)
        assert result["valid"] is False
        assert "missing" in result["error"].lower()


class TestReactiveGuard:
    def test_allows_apt_update_when_no_history(self):
        state = LoopState()
        result = _apt_deb822_preflight_guard(
            "apt-get update",
            tool_name="shell_exec",
            state=state,
            host="localhost",
            user="",
        )
        assert result is None

    def test_allows_apt_commands_after_successful_update(self):
        state = LoopState()
        state.apt_update_results["localhost|"] = AptUpdateResult(
            host="localhost",
            user="",
            attempted=True,
            succeeded=True,
        )
        result = _apt_deb822_preflight_guard(
            "apt install -y webmin",
            tool_name="shell_exec",
            state=state,
            host="localhost",
            user="",
        )
        assert result is None

    def test_blocks_apt_install_after_failed_update_with_deb822_error(self):
        state = LoopState()
        state.apt_update_results["remote|root"] = AptUpdateResult(
            host="remote",
            user="root",
            attempted=True,
            succeeded=False,
            error="deb822 format error in sources",
        )
        result = _apt_deb822_preflight_guard(
            "apt install -y webmin",
            tool_name="ssh_exec",
            state=state,
            host="remote",
            user="root",
        )
        assert result is not None
        assert result.get("success") is False
        metadata = result.get("metadata", {})
        assert "apt_deb822_validation_required" in str(metadata.get("reason", ""))
        assert "deb822" in str(metadata.get("next_required_action", {})).lower()

    def test_allows_apt_update_after_failed_update(self):
        state = LoopState()
        state.apt_update_results["localhost|"] = AptUpdateResult(
            host="localhost",
            user="",
            attempted=True,
            succeeded=False,
            error="deb822 format error",
        )
        result = _apt_deb822_preflight_guard(
            "apt-get update",
            tool_name="shell_exec",
            state=state,
            host="localhost",
            user="",
        )
        assert result is None

    def test_allows_non_apt_commands(self):
        state = LoopState()
        result = _apt_deb822_preflight_guard(
            "ls -la",
            tool_name="shell_exec",
            state=state,
            host="localhost",
            user="",
        )
        assert result is None

    def test_allows_apt_list_commands(self):
        state = LoopState()
        result = _apt_deb822_preflight_guard(
            "apt list --installed",
            tool_name="shell_exec",
            state=state,
            host="localhost",
            user="",
        )
        assert result is None


class TestGuardFailBlockedOperation:
    def test_sets_blocked_operation_on_state(self):
        state = LoopState()
        result = _guard_fail(
            "blocked",
            reason="apt_deb822_preflight_required",
            command="apt install foo",
            next_required_action={"tool_name": "ssh_exec", "required_arguments": {"command": "validate"}},
            state=state,
            tool_name="ssh_exec",
        )
        assert state.blocked_operation is not None
        assert state.blocked_operation.tool == "ssh_exec"
        assert state.blocked_operation.guard_reason == "apt_deb822_preflight_required"
        assert state.guard_trip_count == 1

    def test_increments_guard_trip_count(self):
        state = LoopState()
        _guard_fail(
            "blocked 1",
            reason="apt_deb822_preflight_required",
            command="apt install foo",
            state=state,
            tool_name="ssh_exec",
        )
        _guard_fail(
            "blocked 2",
            reason="apt_deb822_preflight_required",
            command="apt install bar",
            state=state,
            tool_name="ssh_exec",
        )
        assert state.guard_trip_count == 2


class TestMarkDeb822PreflightClean:
    def test_clears_blocked_operation_and_guard_trip_count(self):
        state = LoopState()
        state.blocked_operation = BlockedOperation(
            tool="ssh_exec",
            command="apt install foo",
            guard_reason="apt_deb822_validation_required",
        )
        state.guard_trip_count = 3
        _mark_deb822_preflight_clean(state, host="remote", user="root")
        assert state.blocked_operation is None
        assert state.guard_trip_count == 0

    def test_does_not_clear_non_apt_blocked_operation(self):
        state = LoopState()
        state.blocked_operation = BlockedOperation(
            tool="shell_exec",
            command="rm -rf /",
            guard_reason="risk_policy_blocked",
        )
        state.guard_trip_count = 1
        _mark_deb822_preflight_clean(state, host="remote", user="root")
        # Should clear guard_trip_count but not blocked_operation since reason doesn't start with apt_deb822
        assert state.blocked_operation is not None
        assert state.guard_trip_count == 0

    def test_records_apt_update_result(self):
        state = LoopState()
        _mark_deb822_preflight_clean(state, host="remote", user="root")
        key = "remote|root"
        assert key in state.apt_update_results
        assert state.apt_update_results[key].succeeded is True
        assert state.apt_update_results[key].attempted is True

    def test_sets_scratchpad_preflight_clean(self):
        state = LoopState()
        _mark_deb822_preflight_clean(state, host="remote", user="root")
        preflights = state.scratchpad.get("_deb822_preflight", {})
        assert "remote|root" in preflights
        assert preflights["remote|root"]["status"] == "clean"
