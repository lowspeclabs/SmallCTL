from __future__ import annotations

from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.harness.tool_result_artifact_updates import (
    _observe_remote_installer_preflight_check,
)
from smallctl.tools.shell_support import _REMOTE_INSTALLER_PREFLIGHT_KEY


class _FakeRunlog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, event: str, message: str, **kwargs) -> None:
        self.calls.append((event, message, kwargs))


class _FakeHarness:
    def __init__(self, state: LoopState) -> None:
        self.state = state
        self._runlog = _FakeRunlog()


class _FakeService:
    def __init__(self, state: LoopState | None = None) -> None:
        if state is None:
            state = LoopState()
            state.current_phase = "execute"
        self.harness = _FakeHarness(state)


def _make_preflight_entry(
    host: str = "192.0.2.10",
    user: str = "root",
    cwd: str = "/opt/fogproject/bin",
    script_path: str = "/opt/fogproject/bin/installfog.sh",
    step: int = 0,
) -> tuple[str, dict]:
    """Build a preflight scratchpad entry matching what the guard creates."""
    key = "|".join([host, user, cwd])
    checks = [
        "pwd",
        f"cd '{cwd}' && git rev-parse --show-toplevel",
        f"cd '{cwd}' && git status --short",
        f"test -x '{script_path}'",
    ]
    entry = {
        "host": host,
        "user": user,
        "cwd": cwd,
        "script_path": script_path,
        "checks": checks,
        "created_at_step": step,
        "status": "required",
    }
    return key, entry


def _ssh_result(*, exit_code: int = 0, stdout: str = "", success: bool = True) -> ToolEnvelope:
    return ToolEnvelope(
        success=success,
        output={"stdout": stdout, "stderr": "", "exit_code": exit_code},
        metadata={},
    )


# --- Tests ---


def test_observer_transitions_to_clean_after_all_checks_pass() -> None:
    state = LoopState()
    state.current_phase = "execute"
    state.step_count = 5
    key, entry = _make_preflight_entry()
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    checks = list(entry["checks"])

    # Run each check command as a separate ssh_exec success
    for check_cmd in checks:
        _observe_remote_installer_preflight_check(
            service,
            result=_ssh_result(),
            arguments={"command": check_cmd, "host": "192.0.2.10"},
        )

    # After all checks pass, status should be "clean"
    assert entry["status"] == "clean"
    assert entry["created_at_step"] == 5
    assert set(entry["completed_checks"]) == set(checks)

    # Verify runlog was emitted
    runlog_calls = service.harness._runlog.calls
    assert any(ev == "remote_installer_preflight_cleared" for ev, _, _ in runlog_calls)


def test_observer_does_not_transition_with_partial_checks() -> None:
    state = LoopState()
    state.current_phase = "execute"
    key, entry = _make_preflight_entry()
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    # Run only the first two checks
    for check_cmd in entry["checks"][:2]:
        _observe_remote_installer_preflight_check(
            service,
            result=_ssh_result(),
            arguments={"command": check_cmd, "host": "192.0.2.10"},
        )

    assert entry["status"] == "required"
    assert len(entry.get("completed_checks", [])) == 2


def test_observer_ignores_failed_checks() -> None:
    state = LoopState()
    state.current_phase = "execute"
    key, entry = _make_preflight_entry()
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    # Run a check that returns exit_code != 0
    _observe_remote_installer_preflight_check(
        service,
        result=_ssh_result(exit_code=1, success=True),
        arguments={"command": entry["checks"][0], "host": "192.0.2.10"},
    )

    assert entry["status"] == "required"
    assert entry.get("completed_checks") is None or len(entry.get("completed_checks", [])) == 0


def test_observer_ignores_unsuccessful_result() -> None:
    state = LoopState()
    state.current_phase = "execute"
    key, entry = _make_preflight_entry()
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    # Result with success=False (like a transport failure)
    _observe_remote_installer_preflight_check(
        service,
        result=_ssh_result(success=False),
        arguments={"command": entry["checks"][0], "host": "192.0.2.10"},
    )

    assert entry["status"] == "required"


def test_observer_ignores_wrong_host() -> None:
    state = LoopState()
    state.current_phase = "execute"
    key, entry = _make_preflight_entry(host="192.0.2.10")
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    # Run all checks against wrong host
    for check_cmd in entry["checks"]:
        _observe_remote_installer_preflight_check(
            service,
            result=_ssh_result(),
            arguments={"command": check_cmd, "host": "10.0.0.99"},
        )

    assert entry["status"] == "required"


def test_observer_matches_combined_command() -> None:
    """Agent may run multiple checks in a single ssh_exec using &&."""
    state = LoopState()
    state.current_phase = "execute"
    state.step_count = 3
    key, entry = _make_preflight_entry()
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    checks = entry["checks"]
    # Combine all checks into one command
    combined = " && ".join(checks)
    _observe_remote_installer_preflight_check(
        service,
        result=_ssh_result(),
        arguments={"command": combined, "host": "192.0.2.10"},
    )

    assert entry["status"] == "clean"
    assert set(entry["completed_checks"]) == set(checks)


def test_observer_skips_already_clean_entries() -> None:
    state = LoopState()
    state.current_phase = "execute"
    key, entry = _make_preflight_entry()
    entry["status"] = "clean"
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    _observe_remote_installer_preflight_check(
        service,
        result=_ssh_result(),
        arguments={"command": entry["checks"][0], "host": "192.0.2.10"},
    )

    # Should remain clean (no completed_checks added since it was already clean)
    assert entry["status"] == "clean"
    assert "completed_checks" not in entry


def test_observer_noop_when_no_preflights_in_scratchpad() -> None:
    state = LoopState()
    state.current_phase = "execute"
    service = _FakeService(state)
    original_scratchpad = dict(state.scratchpad)

    result = _observe_remote_installer_preflight_check(
        service,
        result=_ssh_result(),
        arguments={"command": "pwd", "host": "192.0.2.10"},
    )

    assert result is None
    assert state.scratchpad == original_scratchpad


def test_observer_idempotent_on_duplicate_checks() -> None:
    state = LoopState()
    state.current_phase = "execute"
    key, entry = _make_preflight_entry()
    state.scratchpad[_REMOTE_INSTALLER_PREFLIGHT_KEY] = {key: entry}
    service = _FakeService(state)

    first_check = entry["checks"][0]
    # Run same check twice
    for _ in range(3):
        _observe_remote_installer_preflight_check(
            service,
            result=_ssh_result(),
            arguments={"command": first_check, "host": "192.0.2.10"},
        )

    # Should only appear once in completed_checks
    assert entry["completed_checks"].count(first_check) == 1
    assert entry["status"] == "required"  # Still required — other checks not done
